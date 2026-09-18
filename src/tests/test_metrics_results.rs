use super::*;
use crate::metrics::ffmpeg::MetricKind;

fn stats() -> DoneStats {
    DoneStats {
        avg: 46.3144,
        mean: 46.380066,
        harm: 46.367635,
        min: 44.64,
        max: 48.69,
        stddev: 0.760418,
        p1: 44.96,
        p5: 45.2,
        p10: 45.46,
        p25: 45.78,
        exec_s: 1.0,
        frames: 300,
    }
}

fn empty_block() -> Block<'static> {
    Block {
        avg: None,
        stats: None,
        finished: None,
        options: String::new(),
    }
}

#[test]
fn header_matches_reference_order() {
    let h = header();
    assert!(h.starts_with("DateTime\tPSNR-DateTime\tPSNR-Value"));
    assert!(h.contains("\tVMAF-Value\t"));
    assert!(h.contains("\tXPSNR-Percentile25\tSSIM2-DateTime\t"));
    assert!(h.contains("\tCVVDP-Percentile25\tPSNR-Options\t"));
    assert!(h.ends_with("\tFrames\tFrame\tBitrate\tFileSpec\tRFMetrics\tFFMpeg"));
    // 1 + 7×11 + 7 + 6 columns.
    assert_eq!(h.split('\t').count(), 1 + 77 + 7 + 6);
}

#[test]
fn golden_row_matches_reference() {
    // Values transcribed from the reference file's PSNR block.
    let s = stats();
    let data = ResultsRow {
        blocks: [
            Block {
                avg: Some(46.3144),
                stats: Some(&s),
                finished: Some("2026-09-18 14:15:41"),
                options: "Duration=5".to_owned(),
            },
            empty_block(),
            empty_block(),
            empty_block(),
            empty_block(),
            empty_block(),
            empty_block(),
        ],
        frames: "300".to_owned(),
        frame: "1600x1080-60p, yuv420p10le (tv)".to_owned(),
        bitrate: "7233".to_owned(),
        path: "C:/vids/a.mkv",
    };
    assert_eq!(
        row("2026-09-18 14:15:47", &data, "0.1.0", "9.0.1"),
        "2026-09-18 14:15:47\t2026-09-18 14:15:41\t46.3144\t46.380066\t46.367635\t\
             44.640000\t48.690000\t0.760418\t44.960000\t45.200000\t45.460000\t45.780000\t\
             \t\t\t\t\t\t\t\t\t\t\t\
             \t\t\t\t\t\t\t\t\t\t\t\
             \t\t\t\t\t\t\t\t\t\t\t\
             \t\t\t\t\t\t\t\t\t\t\t\
             \t\t\t\t\t\t\t\t\t\t\t\
             \t\t\t\t\t\t\t\t\t\t\t\
             Duration=5\t\t\t\t\t\t\t\
             300\t1600x1080-60p, yuv420p10le (tv)\t7233\tC:/vids/a.mkv\t0.1.0\t9.0.1"
    );
}

fn blank_row() -> ResultsRow<'static> {
    ResultsRow {
        blocks: [
            empty_block(),
            empty_block(),
            empty_block(),
            empty_block(),
            empty_block(),
            empty_block(),
            empty_block(),
        ],
        frames: String::new(),
        frame: String::new(),
        bitrate: String::new(),
        path: "C:/vids/a.mkv",
    }
}

#[test]
fn append_stacks_rows_header_once() {
    let dir = std::env::temp_dir().join(format!("rfmetrics-results-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("RFMetrics.Results.csv");
    assert_eq!(
        append(
            &path,
            &[row("2026-09-18 14:15:47", &blank_row(), "0.1.0", "9.0.1")]
        )
        .unwrap(),
        1
    );
    assert_eq!(
        append(
            &path,
            &[row("2026-09-18 14:16:14", &blank_row(), "0.1.0", "9.0.1")]
        )
        .unwrap(),
        1
    );
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.split("\r\n").collect();
    // Header + 2 rows + trailing empty after final CRLF.
    assert!(lines[0].starts_with("DateTime\tPSNR-DateTime"));
    assert!(lines[1].starts_with("2026-09-18 14:15:47\t"));
    assert!(lines[2].starts_with("2026-09-18 14:16:14\t"));
    assert_eq!(lines[3], "");
    assert!(text.ends_with("\r\n"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn value_rounds_noise_keeps_summaries() {
    // Verbatim summaries pass through untouched.
    assert_eq!(fmt_value(46.313381), "46.313381");
    assert_eq!(fmt_value(0.988807), "0.988807");
    assert_eq!(fmt_value(100.0), "100");
    // Computed means/pools lose float noise beyond 6 dp.
    assert_eq!(fmt_value(41.191716666666665), "41.191717");
    assert_eq!(fmt_value(77.59046833333332), "77.590468");
    assert_eq!(fmt_value(0.9176235266666667), "0.917624");
}

#[test]
fn options_shapes() {
    assert_eq!(
        options_for(MetricKind::Psnr, None, Some(5.0), None),
        "Duration=5"
    );
    assert_eq!(options_for(MetricKind::Psnr, None, None, None), "");
    assert_eq!(
        options_for(MetricKind::Psnr, Some(2.5), Some(12.5), None),
        "Skip=2.5, Duration=12.5"
    );
    assert_eq!(
        options_for(
            MetricKind::Vmaf,
            None,
            Some(5.0),
            Some(("vmaf_v0.6.1.json", Pooling::Mean))
        ),
        "Duration=5, Model=vmaf_v0.6.1.json, Pool=mean"
    );
    assert_eq!(trim_num(5.0), "5");
    assert_eq!(trim_num(12.5), "12.5");
}

#[test]
fn default_path_joins_filename() {
    assert_eq!(
        default_results_path_in(Path::new("D:/out")),
        Path::new("D:/out").join(RESULTS_FILE_NAME)
    );
    assert_eq!(RESULTS_FILE_NAME, "RFMetrics.Results.csv");
    // Live default resolves somewhere (exe/current/temp dir).
    assert_eq!(
        default_results_path().file_name().and_then(|s| s.to_str()),
        Some(RESULTS_FILE_NAME)
    );
}
