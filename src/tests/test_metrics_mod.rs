use super::*;

#[test]
fn stats_match_python_shape() {
    let v = vec![28.1, 33.97, 45.2, 30.2, 32.4];
    let t = stats_text(Some(33.977_123), 12.34, &v);
    assert!(t.starts_with("Avg: 33.977123\nExec time: 00:12.34\nFrames count: 5\n"));
    assert!(t.contains("\nFrames statistics\nMean: "));
    assert!(t.contains("Min: 28.100000\nMax: 45.200000\n"));
    assert!(t.ends_with("Percentile 25: 30.200000"));
}

#[test]
fn stats_empty_and_fallback_avg() {
    assert_eq!(stats_text(None, 0.0, &[]), "No data");
    let t = stats_text(None, 61.5, &[2.0, 4.0]);
    assert!(t.starts_with("Avg: 3.000000\nExec time: 01:01.50\n"));
    assert!(t.contains("Mean (harm): 2.666667"));
}

#[test]
fn harm_ignores_non_positive() {
    assert_eq!(harm_mean(&[]), 0.0);
    assert_eq!(harm_mean(&[0.0, -1.0]), 0.0);
    assert!((harm_mean(&[1.0, 2.0, 4.0]) - 12.0 / 7.0).abs() < 1e-9);
}

#[test]
fn percentile_is_ceil_rank() {
    let s: Vec<f64> = (1..=100).map(|v| v as f64).collect();
    assert_eq!(percentile(&s, 1.0), 1.0);
    assert_eq!(percentile(&s, 25.0), 25.0);
    let s5: Vec<f64> = vec![28.1, 30.2, 32.4, 33.97, 45.2];
    assert_eq!(percentile(&s5, 1.0), 28.1);
    assert_eq!(percentile(&s5, 25.0), 30.2);
}

#[test]
fn time_specs() {
    assert_eq!(parse_time_spec(""), None);
    assert_eq!(parse_time_spec("12.5"), Some(12.5));
    assert_eq!(parse_time_spec(" 90 "), Some(90.0));
    assert_eq!(parse_time_spec("01:02.5"), Some(62.5));
    assert_eq!(parse_time_spec("1:02:03"), Some(3723.0));
    assert_eq!(parse_time_spec("00:00.000"), Some(0.0));
    assert_eq!(parse_time_spec("-5"), None);
    assert_eq!(parse_time_spec("abc"), None);
    assert_eq!(parse_time_spec("1:2:3:4"), None);
    assert_eq!(parse_time_spec("inf"), None);
    assert_eq!(parse_time_spec("1:"), None);
    // Python-parity quirks:
    assert_eq!(parse_time_spec("1e-3"), None); // `-` anywhere rejects
    assert_eq!(parse_time_spec("1.5:02"), None); // heads must be integers
    assert_eq!(parse_time_spec("1:75"), Some(135.0)); // seconds may exceed 59
    assert_eq!(parse_time_spec("1: 02"), Some(62.0)); // inner space tolerated
    assert_eq!(parse_time_spec("0:75"), Some(75.0));
}

#[test]
fn rank_rules() {
    use super::StatRank;
    assert_eq!(rank(48.0, 46.0, 48.0), StatRank::Best);
    assert_eq!(rank(46.0, 46.0, 48.0), StatRank::Worst);
    assert_eq!(rank(47.0, 46.0, 48.0), StatRank::Plain);
    // Single row / all equal: tie, never best-vs-worst.
    assert_eq!(rank(30.0, 30.0, 30.0), StatRank::Tie);
    // NaN compares equal to nothing: no highlight.
    assert_eq!(rank(f64::NAN, 1.0, 2.0), StatRank::Plain);
}

#[test]
fn rank_low_inverts_best_worst() {
    use super::StatRank;
    // FFMetrics-original: StdDev 1.117 beats 1.326 (lower = greener).
    assert_eq!(rank_low(1.117, 1.117, 1.326), StatRank::Best);
    assert_eq!(rank_low(1.326, 1.117, 1.326), StatRank::Worst);
    assert_eq!(rank_low(1.2, 1.117, 1.326), StatRank::Plain);
    assert_eq!(rank_low(1.2, 1.2, 1.2), StatRank::Tie);
}

#[test]
fn done_stats_shape() {
    let s = DoneStats::new(&[28.0, 30.0, 32.0], 30.5, 61.5).unwrap();
    assert_eq!(s.frames, 3);
    assert_eq!(s.avg, 30.5);
    assert_eq!(s.comparable().len(), 10);
    assert_eq!(s.comparable()[0], ("Avg:", 30.5, false));
    assert!(s.comparable()[5].2); // only StdDev is lower-better
    assert_eq!(format_exec(s.exec_s), "01:01.50");
    assert!(DoneStats::new(&[], 0.0, 0.0).is_none());
}

#[test]
fn cell_text_and_tooltip() {
    assert_eq!(MetricCell::Idle.cell_text(), "N/A");
    assert_eq!(
        MetricCell::Running {
            frame: 7,
            values: vec![30.0],
        }
        .cell_text(),
        "Frame: 7"
    );
    assert_eq!(
        MetricCell::Done {
            values: vec![30.0],
            avg: 30.123_456,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: crate::metrics::ffmpeg::ScaleMethod::Bicubic,
        }
        .cell_text(),
        "30.1235"
    );
    assert!(
        MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: crate::metrics::ffmpeg::ScaleMethod::Bicubic,
        }
        .tooltip("PSNR")
        .starts_with("PSNR\nAvg: 30.000000")
    );
    assert_eq!(
        MetricCell::Error {
            msg: "probe failed".to_owned()
        }
        .tooltip("PSNR"),
        "Error: probe failed"
    );
}

/// Done-arrival budget check (2.5h @ 30fps ≈ 270k frames): `DoneStats::new`
/// (clone+sort+passes), `stats_text` (second sort + formats), and
/// `harm_mean` alone. Measurement only — decides whether any of the
/// one-shot paths deserve risk.
#[test]
fn done_arrival_budget() {
    use std::hint::black_box;
    let values: Vec<f64> = (0..270_000)
        .map(|i| 30.0 + (i as f64 * 0.01).sin() * 5.0)
        .collect();
    let n = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        black_box(DoneStats::new(&values, 30.0, 61.5));
    }
    let stats = t0.elapsed() / n;
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        black_box(stats_text(Some(30.0), 61.5, &values));
    }
    let text = t1.elapsed() / n;
    let t2 = std::time::Instant::now();
    for _ in 0..n {
        black_box(harm_mean(&values));
    }
    let harm = t2.elapsed() / n;
    eprintln!(
        "done arrival (270k frames): DoneStats::new {stats:?} | stats_text {text:?} | harm_mean {harm:?}"
    );
}
