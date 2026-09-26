//! Results summary CSV (`RFMetrics.Results.csv`, original parity):
//! one TAB-separated row per queue row with pooled stats per metric.
//! DOT decimals, CRLF + trailing CRLF, append mode (header only when the
//! file is new/empty — repeat exports stack rows, as in the reference).
//! Written inline on the UI thread (KBs, like the state save); the worker
//! never touches it.

use std::path::{Path, PathBuf};

use crate::metrics::DoneStats;
use crate::metrics::ffmpeg::{MetricKind, ensure_parent};
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
    default_results_path_in(&crate::binaries::app_dir())
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
    ensure_parent(path)?;
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
#[path = "../tests/test_metrics_results.rs"]
mod tests;
