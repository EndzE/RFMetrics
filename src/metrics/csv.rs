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
mod tests {
    use super::*;
    use crate::metrics::ffmpeg::FrameDetail;

    fn test_dir(name: &str) -> PathBuf {
        let p =
            std::env::temp_dir().join(format!("rfmetrics-csv-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn outcome(detail: FrameDetail, n: usize) -> RunOutcome {
        RunOutcome {
            values: vec![0.0; n],
            avg: None,
            exec_s: 0.0,
            error: None,
            detail,
        }
    }

    #[test]
    fn num_comma_format() {
        assert_eq!(fmt_num(45.42), "45,42");
        assert_eq!(fmt_num(45.0), "45");
        assert_eq!(fmt_num(0.984065), "0,984065");
        assert_eq!(fmt_num(100.0), "100");
    }

    #[test]
    fn path_beside_dist_by_default() {
        // Joins go through `Path`, so separators are platform-native.
        // The basename is never altered: no added or stripped prefixes.
        let vids = Path::new("C:/vids");
        let p = csv_path_for("", "C:/vids/a.mkv", MetricKind::Psnr);
        assert_eq!(p, vids.join("a.mkv.PSNR.csv"));
        let p = csv_path_for("", "C:/vids/output-a.mkv", MetricKind::Psnr);
        assert_eq!(p, vids.join("output-a.mkv.PSNR.csv"));
        // Bare filename (no parent) stays relative.
        let p = csv_path_for("", "a.mkv", MetricKind::Vmaf);
        assert_eq!(p, PathBuf::from("a.mkv.VMAF.csv"));
        // Chosen dir wins; blank-padded empty falls back beside dist.
        let p = csv_path_for("D:/out", "C:/vids/a.mkv", MetricKind::Ssim);
        assert_eq!(p, Path::new("D:/out").join("a.mkv.SSIM.csv"));
        assert_eq!(
            csv_path_for("  ", "C:/vids/a.mkv", MetricKind::Xpsnr),
            vids.join("a.mkv.XPSNR.csv")
        );
    }

    #[test]
    fn golden_psnr_ssim_xpsnr() {
        let dir = test_dir("golden");
        let cfg = CsvCfg {
            enabled: true,
            dir: dir.to_string_lossy().into_owned(),
        };
        let dist = "C:/vids/a.mkv";
        let p = write_metric_csv(
            &cfg,
            MetricKind::Psnr,
            dist,
            &outcome(FrameDetail::Psnr(vec![[45.42, 43.93, 52.94, 52.73]]), 1),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "frame\tn\tpsnr_avg\tpsnr_y\tpsnr_u\tpsnr_v\r\n0\t1\t45,42\t43,93\t52,94\t52,73\r\n"
        );
        let p = write_metric_csv(
            &cfg,
            MetricKind::Ssim,
            dist,
            &outcome(
                FrameDetail::Ssim(vec![[0.984065, 0.995422, 0.995221, 0.987817]]),
                1,
            ),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "frame\tn\tY\tU\tV\tAll\r\n0\t1\t0,984065\t0,995422\t0,995221\t0,987817\r\n"
        );
        let p = write_metric_csv(
            &cfg,
            MetricKind::Xpsnr,
            dist,
            &outcome(FrameDetail::Xpsnr(vec![[48.1827, 56.2272, 55.9481]]), 1),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "frame\tn\tXPSNR y\tXPSNR u\tXPSNR v\r\n0\t1\t48,1827\t56,2272\t55,9481\r\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn golden_vmaf_vmaf_last() {
        let dir = test_dir("vmaf");
        let cfg = CsvCfg {
            enabled: true,
            dir: dir.to_string_lossy().into_owned(),
        };
        let p = write_metric_csv(
            &cfg,
            MetricKind::Vmaf,
            "C:/vids/a.mkv",
            &outcome(
                FrameDetail::Vmaf {
                    cols: vec!["integer_adm2".to_owned(), "vmaf".to_owned()],
                    rows: vec![vec![0.995258, 95.946422]],
                },
                1,
            ),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "frame\tinteger_adm2\tvmaf\r\n0\t0,995258\t95,946422\r\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Golden shapes with real measured live values (frames 0–1 of the
    /// sample pair): SSIMULACRA2 single score, Butteraugli's three norms,
    /// CVVDP single score — all with the `n` column.
    #[test]
    fn golden_ffvship_shapes() {
        use crate::metrics::ffvship::FfvshipKind;
        let dir = test_dir("ffvship");
        let cfg = CsvCfg {
            enabled: true,
            dir: dir.to_string_lossy().into_owned(),
        };
        let dist = "C:/vids/a.mkv";
        let p = write_metric_csv(
            &cfg,
            MetricKind::Ssim2,
            dist,
            &outcome(
                FrameDetail::Scores {
                    cols: FfvshipKind::Ssimulacra2.csv_cols(),
                    rows: vec![vec![81.0507], vec![79.5221]],
                },
                2,
            ),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "frame\tn\tssimulacra2\r\n0\t1\t81,0507\r\n1\t2\t79,5221\r\n"
        );
        let p = write_metric_csv(
            &cfg,
            MetricKind::But,
            dist,
            &outcome(
                FrameDetail::Scores {
                    cols: FfvshipKind::Butteraugli.csv_cols(),
                    rows: vec![vec![0.749519, 0.813082, 2.83991]],
                },
                1,
            ),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "frame\tn\tbutteraugli_2norm\tbutteraugli_3norm\tbutteraugli_infnorm\r\n\
             0\t1\t0,749519\t0,813082\t2,83991\r\n"
        );
        let p = write_metric_csv(
            &cfg,
            MetricKind::Cvvdp,
            dist,
            &outcome(
                FrameDetail::Scores {
                    cols: FfvshipKind::Cvvdp.csv_cols(),
                    rows: vec![vec![9.82829]],
                },
                1,
            ),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "frame\tn\tcvvdp\r\n0\t1\t9,82829\r\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_detail_or_bad_dir_errors() {
        let dir = test_dir("err");
        let cfg = CsvCfg {
            enabled: true,
            dir: dir.to_string_lossy().into_owned(),
        };
        let out = outcome(FrameDetail::None, 1);
        assert!(write_metric_csv(&cfg, MetricKind::Psnr, "C:/vids/a.mkv", &out).is_err());
        // A file masquerading as the target dir fails loudly.
        let blocker = dir.join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let bad = CsvCfg {
            enabled: true,
            dir: blocker.join("sub").to_string_lossy().into_owned(),
        };
        let out = outcome(FrameDetail::Psnr(vec![[1.0, 1.0, 1.0, 1.0]]), 1);
        let err = write_metric_csv(&bad, MetricKind::Psnr, "C:/vids/a.mkv", &out).unwrap_err();
        assert!(err.contains("blocker"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
