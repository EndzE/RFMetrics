//! Results summary CSV (`RFMetrics.Results.csv`, original parity):
//! one TAB-separated row per queue row with pooled stats per metric.
//! DOT decimals, CRLF + trailing CRLF, append mode (header only when the
//! file is new/empty — repeat exports stack rows, as in the reference).
//! Written inline on the UI thread (KBs, like the state save); the worker
//! never touches it.

use std::path::{Path, PathBuf};

use crate::metrics::DoneStats;
use crate::metrics::ffmpeg::MetricKind;
use crate::metrics::vmaf::Pooling;

/// Default results filename (save dialog preset + exe-dir fallback).
pub const RESULTS_FILE_NAME: &str = "RFMetrics.Results.csv";

/// Default results path inside `dir` (pure helper; the live one resolves
/// `dir` through the exe-dir fallback chain below).
pub fn default_results_path_in(dir: &Path) -> PathBuf {
    dir.join(RESULTS_FILE_NAME)
}

/// Default results path: next to the exe (Python `app_dir` parity),
/// falling back like the state file when unresolvable.
pub fn default_results_path() -> PathBuf {
    let dir = crate::binaries::exe_dir()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(std::env::temp_dir);
    default_results_path_in(&dir)
}

/// Metrics in results-CSV column order (original PSNR/SSIM/VMAF/XPSNR
/// blocks first, FFVship appended).
pub const ORDER: [MetricKind; 7] = [
    MetricKind::Psnr,
    MetricKind::Ssim,
    MetricKind::Vmaf,
    MetricKind::Xpsnr,
    MetricKind::Ssim2,
    MetricKind::But,
    MetricKind::Cvvdp,
];

/// One metric's block: pooled value, cached stats, frozen completion
/// stamp, and the prebuilt `-Options` string. All-`None`/empty = an
/// unscored cell, which renders as empty fields.
pub struct Block<'a> {
    pub avg: Option<f64>,
    pub stats: Option<&'a DoneStats>,
    pub finished: Option<&'a str>,
    pub options: String,
}

/// One queue row: seven blocks in [`ORDER`], then the media columns.
pub struct ResultsRow<'a> {
    pub blocks: [Block<'a>; 7],
    pub frames: String,
    pub frame: String,
    pub bitrate: String,
    pub path: &'a str,
}

/// `Value` formatting: shortest repr of the 6-dp-rounded number. Verbatim
/// ffmpeg summaries (`46.313381`) pass through untouched; computed means
/// and weighted pools lose their float noise (`41.191716666666665` →
/// `41.191717`), matching the reference file's ≤6dp convention.
pub fn fmt_value(v: f64) -> String {
    format!("{}", (v * 1e6).round() / 1e6)
}

/// Compact number without a trailing `.0` (`5`, `12.5`) for `-Options`.
pub fn trim_num(v: f64) -> String {
    if v.is_finite() && v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// `-Options` string (`Duration=5`; VMAF appends `Model=…`, `Pool=…`).
/// Empty pieces are skipped, so a full-video run shows no `Duration`.
pub fn options_for(
    kind: MetricKind,
    skip: Option<f64>,
    clip_dur: Option<f64>,
    vmaf_cfg: Option<(&str, Pooling)>,
) -> String {
    let _ = kind;
    let mut parts = Vec::new();
    if let Some(s) = skip.filter(|&s| s != 0.0) {
        parts.push(format!("Skip={}", trim_num(s)));
    }
    if let Some(d) = clip_dur.filter(|&d| d != 0.0) {
        parts.push(format!("Duration={}", trim_num(d)));
    }
    if let Some((model, pooling)) = vmaf_cfg {
        parts.push(format!("Model={model}"));
        parts.push(format!("Pool={}", pooling.as_filter_str()));
    }
    parts.join(", ")
}

fn block_header(title: &str) -> Vec<String> {
    [
        "DateTime",
        "Value",
        "MeanA",
        "MeanH",
        "Min",
        "Max",
        "StdDevP",
        "Percentile01",
        "Percentile05",
        "Percentile10",
        "Percentile25",
    ]
    .into_iter()
    .map(|c| format!("{title}-{c}"))
    .collect()
}

/// Full header line (no line ending).
pub fn header() -> String {
    let mut cols = vec!["DateTime".to_owned()];
    for kind in ORDER {
        cols.extend(block_header(kind.name()));
    }
    for kind in ORDER {
        cols.push(format!("{}-Options", kind.name()));
    }
    cols.extend(
        [
            "Frames",
            "Frame",
            "Bitrate",
            "FileSpec",
            "RFMetrics",
            "FFMpeg",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    cols.join("\t")
}

fn fmt_stats(s: &DoneStats) -> [String; 9] {
    [
        format!("{:.6}", s.mean),
        format!("{:.6}", s.harm),
        format!("{:.6}", s.min),
        format!("{:.6}", s.max),
        format!("{:.6}", s.stddev),
        format!("{:.6}", s.p1),
        format!("{:.6}", s.p5),
        format!("{:.6}", s.p10),
        format!("{:.6}", s.p25),
    ]
}

/// One data row (no line ending); `now` is the export timestamp,
/// `app_version`/`ffmpeg_version` fill the trailing columns.
pub fn row(now: &str, data: &ResultsRow, app_version: &str, ffmpeg_version: &str) -> String {
    let mut cols = vec![now.to_owned()];
    for b in &data.blocks {
        cols.push(b.finished.unwrap_or_default().to_owned());
        cols.push(b.avg.map(fmt_value).unwrap_or_default());
        match b.stats {
            Some(s) => cols.extend(fmt_stats(s)),
            None => cols.extend(std::iter::repeat_with(String::new).take(9)),
        }
    }
    for b in &data.blocks {
        cols.push(b.options.clone());
    }
    cols.push(data.frames.clone());
    cols.push(data.frame.clone());
    cols.push(data.bitrate.clone());
    cols.push(data.path.to_owned());
    cols.push(app_version.to_owned());
    cols.push(ffmpeg_version.to_owned());
    cols.join("\t")
}

/// Append `rows` (no line endings) with CRLF; writes the header first
/// when the file is new or empty. Missing parents are created (a stale
/// custom dir must not fail the export). Returns rows appended.
pub fn append(path: &Path, rows: &[String]) -> std::io::Result<usize> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let fresh = !path.is_file() || path.metadata().map(|m| m.len() == 0).unwrap_or(true);
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    use std::io::Write as _;
    if fresh {
        writeln!(f, "{}\r", header())?;
    }
    for r in rows {
        writeln!(f, "{r}\r")?;
    }
    Ok(rows.len())
}

#[cfg(test)]
mod tests {
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
        let dir =
            std::env::temp_dir().join(format!("rfmetrics-results-test-{}", std::process::id()));
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
}
