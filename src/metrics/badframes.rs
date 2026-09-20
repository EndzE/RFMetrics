//! Worst-frame PNG export (FFMetrics `BadFrames` parity + FFVship).
//!
//! Original (`FFMetrics.exe`): per file per metric take the N lowest frame
//! values (`LowestArray`, `BadFrames.Count` default 5), `offset = skip +
//! idx / ref_fps`, then one accurate-seek ffmpeg per frame for the
//! distorted file and one for the ref at the same offset:
//! `<dist>.<METRIC>.bf<NNNNNN>.png` + `<dist>.<METRIC>.bf<NNNNNN>-ref.png`.
//!
//! FFVship has no original: same offsets apply (trim-window-relative
//! indices), except Butteraugli ranks worst = max (lower-better) and VMAF
//! strides by its subsample (dense log order).

use std::path::{Path, PathBuf};

/// Options combo labels; persisted as string like `vmaf_subsample`.
pub const COUNT_LABELS: [&str; 10] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];

/// Worst-N `(value_index, value)` in worst-first order.
/// `worst_max` = true takes the largest (Butteraugli, lower-better).
/// Empty input or `n == 0` yields empty; ties break by lower index.
pub fn worst_n(values: &[f64], n: usize, worst_max: bool) -> Vec<(usize, f64)> {
    if values.is_empty() || n == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        let ord = if worst_max {
            values[b].total_cmp(&values[a])
        } else {
            values[a].total_cmp(&values[b])
        };
        ord.then_with(|| a.cmp(&b))
    });
    idx.truncate(n.min(values.len()));
    idx.into_iter().map(|i| (i, values[i])).collect()
}

/// Seconds offset of an (actual, post-stride) frame number after `skip`.
pub fn frame_offset(skip: f64, frame: usize, fps: f64) -> f64 {
    skip + frame as f64 / fps
}

/// Stride for dense value indices: VMAF subsample, else 1.
pub fn stride_for(
    kind: crate::metrics::ffmpeg::MetricKind,
    vmaf_cfg: Option<&crate::metrics::vmaf::VmafCfg>,
) -> usize {
    if kind == crate::metrics::ffmpeg::MetricKind::Vmaf {
        vmaf_cfg.map(|c| c.subsample.max(1) as usize).unwrap_or(1)
    } else {
        1
    }
}

/// Single-frame ffmpeg argv (original `BadFrames.Template` parity):
/// accurate post-input seek, per-file `-r`, `settb/setpts` normalize,
/// `bgr24` + accurate chroma flags.
pub fn ffmpeg_args(src: &str, dest: &str, offset: f64, fps: f64) -> Vec<String> {
    vec![
        "-hide_banner".to_owned(),
        "-nostdin".to_owned(),
        "-probesize".to_owned(),
        "50M".to_owned(),
        "-accurate_seek".to_owned(),
        "-r".to_owned(),
        crate::probe::format_fps(fps),
        "-i".to_owned(),
        src.to_owned(),
        "-ss".to_owned(),
        format!("{offset:.6}"),
        "-r".to_owned(),
        "1".to_owned(),
        "-frames:v".to_owned(),
        "1".to_owned(),
        "-f".to_owned(),
        "image2".to_owned(),
        "-vf".to_owned(),
        "settb=AVTB,setpts=PTS-STARTPTS".to_owned(),
        "-pix_fmt".to_owned(),
        "bgr24".to_owned(),
        "-sws_flags".to_owned(),
        "accurate_rnd+full_chroma_int+bitexact".to_owned(),
        "-update".to_owned(),
        "1".to_owned(),
        dest.to_owned(),
    ]
}

/// Union fit bounds for the viewer pair: both images centered at the
/// origin at true pixel size, so linked plots share one view.
/// Returns `(xmin, xmax, ymin, ymax)`.
pub fn viewer_fit(w1: u32, h1: u32, w2: u32, h2: u32) -> (f64, f64, f64, f64) {
    let hw = w1.max(w2) as f64 / 2.0;
    let hh = h1.max(h2) as f64 / 2.0;
    (-hw, hw, -hh, hh)
}

/// Wipe divider fraction clamped to a draggable interior band so the
/// line never collapses fully to an edge (keeps a grab target).
pub fn clamp_split(v: f32) -> f32 {
    v.clamp(0.02, 0.98)
}

/// Wipe geometry for the slider view in plot units: both frames share a
/// `w`-wide box centered at the origin (smaller frame stretches, matching
/// the stretch-to-same-rect decision). `u` is the shared texture seam:
/// left shows UV `[0, u]`, right shows `[u, 1]`, divider at `div_x`.
/// (Callers put ref left / dist right, like the side-by-side view.)
pub struct WipeLayout {
    pub div_x: f64,
    pub left_cx: f64,
    pub left_w: f64,
    pub right_cx: f64,
    pub right_w: f64,
    pub u: f32,
}

pub fn wipe_layout(w: f64, split: f32) -> WipeLayout {
    let u = clamp_split(split);
    let div_x = -w / 2.0 + w * f64::from(u);
    let left_w = (div_x + w / 2.0).max(0.0);
    let right_w = (w / 2.0 - div_x).max(0.0);
    WipeLayout {
        div_x,
        left_cx: -w / 2.0 + left_w / 2.0,
        left_w,
        right_cx: div_x + right_w / 2.0,
        right_w,
        u,
    }
}

/// Run-scoped tmp dir for viewer PNGs (bounded memory: 1080p RGBA stays
/// on disk, only the visible pair becomes textures). Per-process so two
/// instances never share it; best-effort cleanup by the caller.
pub fn tmp_dir() -> PathBuf {
    std::env::temp_dir().join(format!("rfmetrics-bf-{}", std::process::id()))
}

/// `<tmp>/<dist basename>.<METRIC>.bf<NNNNNN>.png` for viewer runs.
pub fn tmp_dest_for(tmp: &Path, dist_path: &str, kind_name: &str, frame: usize) -> PathBuf {
    let base = Path::new(dist_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dist_path.to_owned());
    tmp.join(format!("{base}.{kind_name}.bf{frame:06}.png"))
}

/// Same with `-ref` before the extension (original parity).
pub fn tmp_dest_ref_for(tmp: &Path, dist_path: &str, kind_name: &str, frame: usize) -> PathBuf {
    let base = Path::new(dist_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dist_path.to_owned());
    tmp.join(format!("{base}.{kind_name}.bf{frame:06}-ref.png"))
}

/// Run one extraction; true iff ffmpeg exited 0 and `dest` exists.
/// Failures log at warn with the repro argv (FFMetrics.log parity);
/// a partial file is removed.
pub fn extract_one(ffmpeg: &Path, src: &str, dest: &Path, offset: f64, fps: f64) -> bool {
    let args = ffmpeg_args(src, &dest.to_string_lossy(), offset, fps);
    log::info!(target: "rfmetrics::badframes", "run: \"{}\" {}", ffmpeg.display(), args.join(" "));
    let mut cmd = std::process::Command::new(ffmpeg);
    cmd.args(&args);
    match crate::cmd::output_timeout(cmd, crate::cmd::BADFRAME_TIMEOUT) {
        Ok(out) if out.status.success() && dest.is_file() => true,
        Ok(out) => {
            let _ = std::fs::remove_file(dest);
            let tail = String::from_utf8_lossy(&out.stderr);
            let last = tail
                .lines()
                .map(str::trim)
                .rfind(|l| !l.is_empty())
                .unwrap_or("");
            log::warn!(target: "rfmetrics::badframes", "extract frame {offset:.3}s from \"{src}\" failed (exit {:?}): {last}", out.status.code());
            false
        }
        Err(e) => {
            let _ = std::fs::remove_file(dest);
            log::warn!(target: "rfmetrics::badframes", "extract frame {offset:.3}s from \"{src}\" failed: {e}");
            false
        }
    }
}

#[cfg(test)]
#[path = "../tests/test_metrics_badframes.rs"]
mod tests;
