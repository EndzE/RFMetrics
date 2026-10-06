use super::*;

#[test]
fn round_trip() {
    let s = AppState {
        ref_path: "C:/vids/ref.mp4".to_owned(),
        skip: "5".to_owned(),
        duration: "00:10".to_owned(),
        files: Some(vec![
            FileEntry {
                path: "C:/vids/a.mp4".to_owned(),
                include: true,
            },
            FileEntry {
                path: "C:/vids/b.mp4".to_owned(),
                include: false,
            },
        ]),
        metrics: MetricsState {
            psnr: Some(true),
            vmaf: Some(false),
            ..Default::default()
        },
        vmaf: VmafState {
            model: Some("vmaf_v0.6.1.json".to_owned()),
            threads: Some("auto".to_owned()),
            ..Default::default()
        },
        cvvdp: CvvdpState {
            display: Some("standard_4k".to_owned()),
        },
        options: OptionsState {
            scaling: Some("Bicubic".to_owned()),
            fps_mode: Some("Reference rate on both".to_owned()),
            plot_at_start: Some(true),
            plot_size: Some("3200×800".to_owned()),
            csv_export: Some(true),
            csv_dir: Some("D:/csv".to_owned()),
            results_autosave: Some(true),
            results_path: Some("D:/r.csv".to_owned()),
            ..Default::default()
        },
    };
    let back: AppState = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back, s);
}

#[test]
fn python_shaped_files_load_included() {
    let s: AppState = serde_json::from_str(
            r#"{"ref": "x", "ref_path": "C:/r.mp4", "files": ["C:/a.mp4", {"path": "C:/b.mp4", "include": false}]}"#,
        )
        .unwrap();
    // Unknown keys ("ref") are ignored; strings load as included rows.
    assert_eq!(s.ref_path, "C:/r.mp4");
    assert_eq!(
        s.files,
        Some(vec![
            FileEntry {
                path: "C:/a.mp4".to_owned(),
                include: true,
            },
            FileEntry {
                path: "C:/b.mp4".to_owned(),
                include: false,
            },
        ])
    );
    // Absent sections stay None so live defaults survive.
    assert_eq!(s.metrics, MetricsState::default());
    assert_eq!(s.vmaf, VmafState::default());
    assert_eq!(s.options, OptionsState::default());
}

#[test]
fn options_scaling_round_trips_and_rejects_unknown() {
    let s: AppState = serde_json::from_str(r#"{"options": {"scaling": "Lanczos"}}"#).unwrap();
    assert_eq!(
        s.options.scaling,
        Some("Lanczos".to_owned()),
        "label persists verbatim; from_label validates on apply"
    );
    // Unknown section keys are ignored, like Python's unknown keys.
    let s: AppState =
        serde_json::from_str(r#"{"options": {"scaling": "Lanczos", "other": 1}}"#).unwrap();
    assert_eq!(s.options.scaling, Some("Lanczos".to_owned()));
}

#[test]
fn tmp_suffix_targets_state_file() {
    let tmp = PathBuf::from("ffmetrics-state.json").with_extension(format!("json{TMP_SUFFIX}"));
    assert_eq!(tmp.to_string_lossy(), "ffmetrics-state.json.tmp");
}

#[test]
fn candidate_dirs_lead_with_app_dir_deduped() {
    // Fallback order is exe → cwd → temp with no repeats, so a writable
    // exe dir keeps the file exactly where it always was.
    let dirs = crate::binaries::candidate_dirs();
    assert!(!dirs.is_empty());
    assert_eq!(dirs[0], crate::binaries::app_dir());
    let mut seen = std::collections::HashSet::new();
    for d in &dirs {
        assert!(seen.insert(d), "duplicate candidate dir {d:?}");
    }
}

#[test]
fn write_atomic_fails_cleanly_without_side_effects() {
    // Read-only install path: a failed write leaves no tmp behind so the
    // next candidate dir gets a clean shot.
    let dir = std::env::temp_dir().join(format!("rfmetrics-state-fail-{}", std::process::id()));
    let path = dir.join("no-such-dir").join("ffmetrics-state.json");
    assert!(!write_atomic("{}", &path));
    assert!(!path.with_extension(format!("json{TMP_SUFFIX}")).exists());
    assert!(!path.exists());
}

#[test]
fn salvage_keeps_good_sections_past_bad_ones() {
    // A null in files (untagged FileEntryRaw rejects it) plus a mistyped
    // toggle must not take down the surviving sections.
    let s = salvage(
        r#"{"ref_path": "C:/r.mp4", "files": ["C:/a.mp4", null], "metrics": {"psnr": "yes"}}"#,
    )
    .expect("partial state must survive");
    assert_eq!(s.ref_path, "C:/r.mp4");
    assert_eq!(s.files, None);
    assert_eq!(s.metrics, MetricsState::default());
}

#[test]
fn salvage_rejects_garbage_and_non_objects() {
    assert!(salvage("{oops").is_none());
    assert!(salvage("[1, 2]").is_none());
    assert!(salvage("null").is_none());
    // Clean files still take the exact-shape path through load_from.
    let dir = std::env::temp_dir().join(format!("rfmetrics-state-clean-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rfmetrics-state.json");
    std::fs::write(&path, r#"{"ref_path": "C:/r.mp4"}"#).unwrap();
    let back = load_from(std::slice::from_ref(&path)).expect("clean file loads");
    assert_eq!(back.ref_path, "C:/r.mp4");
    assert!(
        !dir.join("rfmetrics-state.json.bak").exists(),
        "clean file quarantined"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_from_skips_corrupt_candidates_with_bak() {
    let dir = std::env::temp_dir().join(format!("rfmetrics-state-rot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bad = dir.join("bad.json");
    let good = dir.join("good.json");
    std::fs::write(&bad, "{oops").unwrap();
    std::fs::write(&good, r#"{"ref_path": "C:/r.mp4"}"#).unwrap();
    let back = load_from(&[bad.clone(), good.clone()]).expect("valid candidate loads");
    assert_eq!(back.ref_path, "C:/r.mp4");
    assert!(!bad.exists(), "corrupt candidate left in place");
    assert!(
        dir.join("bad.json.bak").is_file(),
        "corrupt candidate not quarantined"
    );
    assert!(good.is_file(), "valid candidate touched");
    // Nothing usable anywhere: None, but the evidence survives.
    std::fs::write(&bad, "{oops").unwrap();
    assert!(load_from(std::slice::from_ref(&bad)).is_none());
    assert!(dir.join("bad.json.bak").is_file());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn save_round_trip_is_atomic() {
    let dir = std::env::temp_dir().join(format!("rfmetrics-state-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ffmetrics-state.json");
    let s = AppState {
        ref_path: "C:/r.mp4".to_owned(),
        metrics: MetricsState {
            vmaf: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    save_to(&s, &path);
    let back: AppState = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(back, s);
    // No temp file left behind.
    assert!(!path.with_extension(format!("json{TMP_SUFFIX}")).exists());
    std::fs::remove_dir_all(&dir).ok();
}
