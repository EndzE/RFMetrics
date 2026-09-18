//! Per-frame metrics CSV export (worker-side file writer).
//! Shape matches the reference `<dist>.<METRIC>.csv` files:
//! TAB-separated, UTF-8, CRLF line endings, comma decimals, `frame`
//! 0-based + `n` 1-based, no summary footer. Written by the metric worker
//! at `Done` time so the UI thread never touches the filesystem.

use std::path::{Path, PathBuf};

use crate::metrics::ffmpeg::{FrameDetail, MetricKind, RunOutcome};

/// Frozen-at-Start export setting (mid-run toggles must not half-apply).
#[derive(Debug, Clone, Default)]
pub struct CsvCfg {
    pub enabled: bool,
    /// Chosen folder; empty = beside the distorted file.
    pub dir: String,
}

/// `<dist basename>.<METRIC>.csv` in `dir`, or beside the distorted file
/// when `dir` is empty (e.g. `video.mkv.PSNR.csv`). The name is never
/// altered beyond the suffix — a dist file that already starts with
/// `output-` keeps it verbatim. A bare filename (no parent) lands in
/// the working directory.
pub fn csv_path_for(dir: &str, dist_path: &str, kind: MetricKind) -> PathBuf {
    let base = Path::new(dist_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dist_path.to_owned());
    let name = format!("{base}.{}.csv", kind.name());
    if dir.trim().is_empty() {
        match Path::new(dist_path).parent() {
            Some(p) if !p.as_os_str().is_empty() => p.join(&name),
            _ => PathBuf::from(&name),
        }
    } else {
        PathBuf::from(dir).join(&name)
    }
}

/// Comma-decimal number (`45.42` → `45,42`). Values are pre-sanitized,
/// so no `inf`/`NaN` can leak through (`{}` would print those raw).
pub fn fmt_num(v: f64) -> String {
    format!("{v}").replace('.', ",")
}

/// Header + body rows for one metric; `Err` when the outcome carries no
/// frame detail (errors, FFVship kinds). `has_n` mirrors the samples:
/// PSNR/SSIM/XPSNR carry the 1-based `n` column, VMAF does not. Row count
/// follows `detail` (zipped against `values` upstream); a mismatch warns.
#[allow(clippy::type_complexity)]
fn render(
    kind: MetricKind,
    detail: &FrameDetail,
    n_values: usize,
) -> Result<(String, Vec<Vec<f64>>, bool), String> {
    let (header, rows, has_n): (String, Vec<Vec<f64>>, bool) = match (kind, detail) {
        (MetricKind::Psnr, FrameDetail::Psnr(rows)) => (
            "frame\tn\tpsnr_avg\tpsnr_y\tpsnr_u\tpsnr_v".to_owned(),
            rows.iter().map(|r| r.to_vec()).collect(),
            true,
        ),
        (MetricKind::Ssim, FrameDetail::Ssim(rows)) => (
            "frame\tn\tY\tU\tV\tAll".to_owned(),
            rows.iter().map(|r| r.to_vec()).collect(),
            true,
        ),
        (MetricKind::Xpsnr, FrameDetail::Xpsnr(rows)) => (
            "frame\tn\tXPSNR y\tXPSNR u\tXPSNR v".to_owned(),
            rows.iter().map(|r| r.to_vec()).collect(),
            true,
        ),
        (MetricKind::Vmaf, FrameDetail::Vmaf { cols, rows }) => {
            (format!("frame\t{}", cols.join("\t")), rows.clone(), false)
        }
        // FFVship multi-score rows (headers from `FfvshipKind::csv_cols`);
        // like the ffmpeg metrics these carry the 1-based `n` column.
        (_, FrameDetail::Scores { cols, rows }) => {
            (format!("frame\tn\t{}", cols.join("\t")), rows.clone(), true)
        }
        _ => return Err("no frame data captured".to_owned()),
    };
    if rows.len() != n_values {
        log::warn!(
            target: "rfmetrics::csv",
            "{} detail rows ({}) != values ({n_values}); writing what fits",
            kind.name(),
            rows.len(),
        );
    }
    Ok((header, rows, has_n))
}

/// Write one metric CSV; the caller gates on `cfg.enabled`. Success
/// returns the path (for the log); errors carry it (for the toast).
pub fn write_metric_csv(
    cfg: &CsvCfg,
    kind: MetricKind,
    dist_path: &str,
    outcome: &RunOutcome,
) -> Result<PathBuf, String> {
    let (header, rows, has_n) = render(kind, &outcome.detail, outcome.values.len())?;
    let path = csv_path_for(&cfg.dir, dist_path, kind);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut text = String::with_capacity(header.len() + rows.len() * 50 + 2);
    text.push_str(&header);
    text.push_str("\r\n");
    for (i, r) in rows.iter().enumerate() {
        text.push_str(&i.to_string());
        if has_n {
            text.push('\t');
            text.push_str(&(i + 1).to_string());
        }
        for v in r {
            text.push('\t');
            text.push_str(&fmt_num(*v));
        }
        text.push_str("\r\n");
    }
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}
#[cfg(test)]
#[path = "../tests/test_metrics_csv.rs"]
mod tests;
