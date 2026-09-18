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
        options: OptionsState {
            scaling: Some("Bicubic".to_owned()),
            plot_at_start: Some(true),
            plot_size: Some("3200×800".to_owned()),
            csv_export: Some(true),
            csv_dir: Some("D:/csv".to_owned()),
            results_autosave: Some(true),
            results_path: Some("D:/r.csv".to_owned()),
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
