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
