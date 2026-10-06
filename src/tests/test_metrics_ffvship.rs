use super::*;

fn info() -> MediaInfo {
    MediaInfo {
        fps: Some(25.0),
        duration: Some(100.0),
        ..Default::default()
    }
}

/// Test-only first-score projection (the runner inlines this over
/// `parse_live_rows` so full rows survive for CSV detail).
fn parse_series(text: &str, expected_n: Option<i64>, kind: FfvshipKind) -> Option<Vec<f64>> {
    parse_live_rows(text, expected_n, kind.n_scores())
        .map(|rows| rows.into_iter().map(|r| r[0]).collect())
}

#[test]
fn series_happy_paths() {
    use FfvshipKind::{Butteraugli, Cvvdp, Ssimulacra2};
    let text = "3\n0 95.1\n1 95.2\n2 95.3\n";
    assert_eq!(
        parse_series(text, None, Ssimulacra2),
        Some(vec![95.1, 95.2, 95.3])
    );
    assert_eq!(parse_series(text, Some(3), Ssimulacra2).unwrap().len(), 3);
    // Butteraugli emits 3 scores per line; the first is used.
    let but = "2\n0 3.1 0.5 0.2\n1 3.3 0.6 0.1\n";
    assert_eq!(parse_series(but, None, Butteraugli), Some(vec![3.1, 3.3]));
    // Single-score text is not valid Butteraugli output.
    assert_eq!(parse_series(text, None, Butteraugli), None);
    assert_eq!(parse_series(text, None, Cvvdp).unwrap().len(), 3);
}

#[test]
fn series_rejects_garbage() {
    use FfvshipKind::Ssimulacra2 as S;
    assert_eq!(parse_series("", None, S), None);
    assert_eq!(parse_series("oops\n0 1.0\n", None, S), None);
    assert_eq!(parse_series("0\n", None, S), None);
    assert_eq!(parse_series("-2\n", None, S), None);
    // Count line must match the trim window when set.
    assert_eq!(parse_series("3\n0 1.0\n1 2.0\n2 3.0\n", Some(2), S), None);
    // Duplicate / missing / out-of-range indices.
    assert_eq!(parse_series("2\n0 1.0\n0 2.0\n", None, S), None);
    assert_eq!(parse_series("2\n0 1.0\n2 2.0\n", None, S), None);
    assert_eq!(parse_series("2\n0 1.0\n", None, S), None);
    // Non-finite scores and bad arity.
    assert_eq!(parse_series("1\n0 inf\n", None, S), None);
    assert_eq!(parse_series("1\n0 nan\n", None, S), None);
    assert_eq!(parse_series("1\n0 1.0 extra\n", None, S), None);
    assert_eq!(parse_series("1\n0\n", None, S), None);
    assert_eq!(parse_series("1\n0 abc\n", None, S), None);
    // Blank lines are ignored, like Python's stripped-lines filter.
    assert_eq!(
        parse_series("2\n\n0 1.0\n\n1 2.0\n", None, S),
        Some(vec![1.0, 2.0])
    );
}

#[test]
fn pooling_rules() {
    use FfvshipKind::{Butteraugli, Cvvdp, Ssimulacra2};
    let v = vec![90.0, 92.0, 94.0];
    assert_eq!(pooled_avg(&v, Ssimulacra2), Some(92.0));
    assert_eq!(pooled_avg(&v, Butteraugli), Some(92.0));
    assert_eq!(pooled_avg(&v, Cvvdp), Some(94.0));
    assert_eq!(pooled_avg(&[], Cvvdp), None);
}

#[test]
fn csv_column_shapes() {
    use FfvshipKind::{Butteraugli, Cvvdp, Ssimulacra2};
    assert_eq!(Ssimulacra2.csv_cols(), vec!["ssimulacra2".to_owned()]);
    assert_eq!(
        Butteraugli.csv_cols(),
        vec![
            "butteraugli_2norm".to_owned(),
            "butteraugli_3norm".to_owned(),
            "butteraugli_infnorm".to_owned()
        ]
    );
    assert_eq!(Cvvdp.csv_cols(), vec!["cvvdp".to_owned()]);
    // Live arity matches the CSV width (strict parser enforces it).
    assert_eq!(Ssimulacra2.n_scores(), 1);
    assert_eq!(Butteraugli.n_scores(), 3);
    assert_eq!(Cvvdp.n_scores(), 1);
}

#[test]
fn window_frames_match_python() {
    // No trim: full video, no expectation.
    assert_eq!(
        trim_window_frames(&info(), &info(), None, None),
        Ok((Vec::new(), None))
    );
    // Zero trim disables the window (falsy parity).
    assert_eq!(
        trim_window_frames(&info(), &info(), Some(0.0), Some(0.0)),
        Ok((Vec::new(), None))
    );
    // Skip only: --start, no expected count.
    let (w, e) = trim_window_frames(&info(), &info(), Some(4.0), None).unwrap();
    assert_eq!(w, vec!["--start", "100"]);
    assert_eq!(e, None);
    // Skip + clip: --start/--end with expected count.
    let (w, e) = trim_window_frames(&info(), &info(), Some(4.0), Some(2.0)).unwrap();
    assert_eq!(w, vec!["--start", "100", "--end", "150"]);
    assert_eq!(e, Some(50));
    // End clamps to the probed duration (100s × 25fps = 2500).
    let (w, e) = trim_window_frames(&info(), &info(), Some(99.0), Some(10.0)).unwrap();
    assert_eq!(w, vec!["--start", "2475", "--end", "2500"]);
    assert_eq!(e, Some(25));
    // Trim without any fps is a hard error.
    let no_fps = MediaInfo::default();
    assert_eq!(
        trim_window_frames(&no_fps, &no_fps, Some(1.0), None),
        Err("unknown fps for skip/duration".to_owned())
    );
}

#[test]
fn args_shape() {
    let a = build_args(
        FfvshipKind::Ssimulacra2,
        "ref.mp4",
        "dist.mp4",
        &["--start".to_owned(), "100".to_owned()],
        None,
    );
    assert_eq!(
        a,
        vec![
            "-s",
            "ref.mp4",
            "-e",
            "dist.mp4",
            "-m",
            "SSIMULACRA2",
            "--live-score-output",
            "--start",
            "100"
        ]
    );
}

#[test]
fn display_flags_only_off_default() {
    use FfvshipKind::Cvvdp;
    use std::path::Path;
    // Default display is the binary's own: omitted (argv parity).
    let a = build_args(Cvvdp, "r.mp4", "d.mp4", &[], None);
    assert!(!a.iter().any(|s| s == "--displayModel"));
    assert!(!a.iter().any(|s| s == "--displayConfig"));
    // Any other key rides `--displayConfig` + `--displayModel`.
    let cfg = Path::new("C:/tmp/rfmetrics-cvvdp-1-standard_4k.json");
    let a = build_args(Cvvdp, "r.mp4", "d.mp4", &[], Some(("standard_4k", cfg)));
    assert_eq!(
        &a[a.len() - 4..],
        [
            "--displayConfig",
            "C:/tmp/rfmetrics-cvvdp-1-standard_4k.json",
            "--displayModel",
            "standard_4k"
        ]
    );
    // Non-CVVDP metrics never take the flags, even off-default.
    let a = build_args(
        FfvshipKind::Ssimulacra2,
        "r.mp4",
        "d.mp4",
        &[],
        Some(("standard_4k", cfg)),
    );
    assert!(!a.iter().any(|s| s == "--displayModel"));
    assert!(!a.iter().any(|s| s == "--displayConfig"));
}

#[test]
fn describe_matches_reference_format() {
    let show = |key: &str| {
        let m = display_named(key).unwrap();
        describe_display(&m.key, &m.display)
    };
    assert_eq!(
        show("standard_fhd"),
        "24\" 1920x1080 SDR, 200 nits, 250 lux, 0.60 m (2.0 x screen height)"
    );
    assert_eq!(
        show("standard_hdr_pq"),
        "30\" 3840x2160 HDR, 1500 nits, 10 lux, 0.75 m (2.0 x screen height)"
    );
    assert_eq!(
        show("iphone_14_pro"),
        "6.1\" 2532x1170 SDR, 1025 nits, 250 lux, 0.51 m (7.8 x screen height)"
    );
}

#[test]
fn registry_holds_all_models_split_by_group() {
    let reg = display_registry();
    assert_eq!(reg.len(), 26);
    assert_eq!(reg.iter().filter(|m| m.vmlab).count(), 8);
    assert_eq!(reg.iter().filter(|m| !m.vmlab).count(), 18);
    // Unique keys; friendly names never empty or key-identical.
    let mut keys = std::collections::HashSet::new();
    for m in reg {
        assert!(keys.insert(m.key.clone()), "duplicate key {}", m.key);
        assert!(!m.name.is_empty());
    }
    // Required display fields on every entry (FFVship validation).
    for f in [
        "colorspace",
        "contrast",
        "diagonal_size_inches",
        "E_ambient",
        "max_luminance",
        "resolution",
        "viewing_distance_meters",
    ] {
        for m in reg {
            assert!(m.display.contains_key(f), "{} misses {f}", m.key);
        }
    }
    // Default + lookup.
    assert_eq!(DEFAULT_DISPLAY_KEY, "standard_fhd");
    assert_eq!(
        display_named("standard_4k").map(|m| m.name.as_str()),
        Some("30-inch 4K monitor, office")
    );
    assert_eq!(
        display_named("iphone_14_pro").map(|m| m.name.as_str()),
        Some("iPhone 14 Pro")
    );
    assert!(display_named("bogus_display_xyz").is_none());
}

/// Valid display object for validation tests (mirrors a registry entry).
fn valid_display() -> serde_json::Map<String, serde_json::Value> {
    serde_json::from_str(
        r#"{"colorspace":"SDR","resolution":[1920,1080],"viewing_distance_meters":0.6,
            "diagonal_size_inches":24,"max_luminance":200,"contrast":1000,
            "E_ambient":250,"k_refl":0.005,"exposure":1.0}"#,
    )
    .unwrap()
}

#[test]
fn custom_validation_accepts_good_rejects_bad() {
    let good = valid_display();
    let customs: Vec<CustomDisplay> = Vec::new();
    assert!(validate_custom_display("Mine", &good, None, &customs).is_ok());
    // Rename-own exception: keeping the name while overwriting passes.
    let customs = vec![CustomDisplay {
        name: "Mine".to_owned(),
        display: good.clone(),
    }];
    assert!(validate_custom_display("Mine", &good, Some("Mine"), &customs).is_ok());
    // Empty name, built-in collision, custom collision.
    assert!(validate_custom_display("", &good, None, &customs).is_err());
    assert!(validate_custom_display("standard_fhd", &good, None, &customs).is_err());
    assert!(validate_custom_display("Mine", &good, None, &customs).is_err());
    // Missing field, bad colorspace, out-of-range values.
    let mut bad = good.clone();
    bad.remove("contrast");
    assert!(validate_custom_display("Other", &bad, None, &customs).is_err());
    let mut bad = good.clone();
    bad.insert("colorspace".to_owned(), serde_json::Value::from("sRGB"));
    assert!(validate_custom_display("Other", &bad, None, &customs).is_err());
    for (field, value) in [
        ("diagonal_size_inches", 0.0),
        ("viewing_distance_meters", 0.0),
        ("max_luminance", 0.0),
        ("contrast", 0.0),
        ("E_ambient", -1.0),
        ("k_refl", 1.0),
        ("exposure", 0.0),
    ] {
        let mut bad = good.clone();
        bad.insert(field.to_owned(), serde_json::Value::from(value));
        assert!(
            validate_custom_display("Other", &bad, None, &customs).is_err(),
            "{field}={value} accepted"
        );
    }
    // reflectivity is a fraction (UI shows percent); resolution bounds.
    let mut bad = good.clone();
    bad.insert("resolution".to_owned(), serde_json::json!([1920]));
    assert!(validate_custom_display("Other", &bad, None, &customs).is_err());
    let mut bad = good.clone();
    bad.insert("resolution".to_owned(), serde_json::json!([8, 1080]));
    assert!(validate_custom_display("Other", &bad, None, &customs).is_err());
}

#[test]
fn lookup_prefers_customs_then_registry() {
    let customs = vec![CustomDisplay {
        name: "Mine".to_owned(),
        display: valid_display(),
    }];
    // Custom hit (name is the key).
    let (name, _) = lookup_display("Mine", &customs).unwrap();
    assert_eq!(name, "Mine");
    // Registry hit.
    let (name, _) = lookup_display("standard_4k", &customs).unwrap();
    assert_eq!(name, "30-inch 4K monitor, office");
    // Miss.
    assert!(lookup_display("bogus_display_xyz", &customs).is_none());
}

#[test]
fn heights_ratio_matches_reference() {
    // 30" 4K at 0.7472 m: the classic 2.0 x screen height.
    let r = heights_ratio(3840.0, 2160.0, 30.0, 0.7472);
    assert!((r - 2.0).abs() < 0.01, "{r}");
}

#[test]
fn display_config_guard_writes_and_cleans() {
    let customs = vec![CustomDisplay {
        name: "Mine".to_owned(),
        display: valid_display(),
    }];
    // Default and unknown keys: no file, no flags.
    assert!(
        DisplayConfig::new("standard_fhd", &customs)
            .unwrap()
            .is_none()
    );
    assert!(
        DisplayConfig::new("bogus_display_xyz", &customs)
            .unwrap()
            .is_none()
    );
    // Custom key: temp file lives while the guard does, gone after.
    let path = {
        let guard = DisplayConfig::new("Mine", &customs).unwrap().unwrap();
        let (key, path) = guard.argv();
        assert_eq!(key, "Mine");
        assert!(path.is_file());
        path.to_path_buf()
    };
    assert!(!path.exists());
}

#[test]
fn live_value_takes_first_score_of_clean_lines() {
    // Single-score metric: count line and short lines skipped.
    assert_eq!(live_value("300", 1), None);
    assert_eq!(live_value("12 48.5", 1), Some(48.5));
    assert_eq!(live_value("12 48.5 extra", 1), None);
    assert_eq!(live_value("xx 48.5", 1), None);
    assert_eq!(live_value("12 nan", 1), None);
    assert_eq!(live_value("12 inf", 1), None);
    // Multi-score metric: first score wins, rest must be finite.
    assert_eq!(live_value("7 9.5 10.0 2.0", 3), Some(9.5));
    assert_eq!(live_value("7 9.5 oops 2.0", 3), None);
    assert_eq!(live_value("7 9.5 10.0", 3), None);
}
