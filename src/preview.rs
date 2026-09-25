use std::path::Path;
use std::process::Command;

/// Reference thumbnail box (Python `target_size=(136, 76)`).
pub const BOX_W: u32 = 136;
pub const BOX_H: u32 = 76;

/// Python seek policy: prefer 4s on longer clips to skip fades.
pub fn seek_candidates(duration: Option<f64>) -> Vec<&'static str> {
    match duration {
        Some(d) if d >= 5.0 => vec!["4.0", "1.0", "0"],
        Some(d) if d >= 2.0 => vec!["1.0", "0"],
        Some(d) if d > 0.0 => vec!["0"],
        _ => vec!["1.0", "0"],
    }
}

/// Fit `(w, h)` into the thumbnail box, preserving aspect ratio.
fn fit_size(w: u32, h: u32) -> (u32, u32) {
    if w == 0 || h == 0 {
        return (BOX_W, BOX_H);
    }
    let ratio = (BOX_W as f32 / w as f32).min(BOX_H as f32 / h as f32);
    (
        ((w as f32 * ratio) as u32).max(1),
        ((h as f32 * ratio) as u32).max(1),
    )
}

fn decode_and_fit(png_bytes: &[u8]) -> Option<egui::ColorImage> {
    if png_bytes.len() <= 100 {
        log::debug!(target: "rfmetrics::preview", "thumbnail decode skipped: {} bytes", png_bytes.len());
        return None;
    }
    let img = match image::load_from_memory(png_bytes) {
        Ok(i) => i.to_rgba8(),
        Err(e) => {
            log::warn!(target: "rfmetrics::preview", "thumbnail PNG decode failed: {e}");
            return None;
        }
    };
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return None;
    }
    let (nw, nh) = fit_size(w, h);
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Lanczos3);
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [nw as usize, nh as usize],
        &resized.into_raw(),
    ))
}

/// One ffmpeg single-frame extract per seek candidate; first success wins.
/// `None` = empty/missing input or every seek failed (caller shows placeholder).
/// Each seek is bounded (a wedged decode must not hang the thumb worker).
pub fn extract_thumbnail(
    ffmpeg: &Path,
    path: &str,
    duration: Option<f64>,
) -> Option<egui::ColorImage> {
    if !crate::probe::path_usable(path) {
        return None;
    }
    let vf = format!(
        "scale={}:{}:force_original_aspect_ratio=decrease",
        BOX_W * 2,
        BOX_H * 2
    );
    for ss in seek_candidates(duration) {
        let start = std::time::Instant::now();
        log::debug!(target: "rfmetrics::preview", "thumbnail: -ss {ss} -i \"{path}\"");
        let mut cmd = Command::new(ffmpeg);
        cmd.args([
            "-hide_banner",
            "-nostdin",
            // FFMetrics.conf `Thumbnail.Template` parity.
            "-probesize",
            "50M",
            "-ss",
            ss,
            "-i",
            path,
            "-frames:v",
            "1",
            "-vf",
            &vf,
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "pipe:1",
        ]);
        let out = match crate::cmd::output_timeout(cmd, crate::cmd::THUMB_TIMEOUT) {
            Ok(o) => o,
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                log::warn!(target: "rfmetrics::preview", "thumbnail -ss {ss} \"{path}\" timed out after {:?} ({}ms)", crate::cmd::THUMB_TIMEOUT, start.elapsed().as_millis());
                continue;
            }
            Err(e) => {
                log::warn!(target: "rfmetrics::preview", "thumbnail ffmpeg -ss {ss} \"{path}\" spawn failed: {e}");
                return None;
            }
        };
        if let Some(img) = decode_and_fit(&out.stdout) {
            log::info!(
                target: "rfmetrics::preview",
                "thumbnail \"{path}\" -ss {ss} → {} bytes ({}ms)",
                out.stdout.len(),
                start.elapsed().as_millis(),
            );
            return Some(img);
        }
        log::debug!(
            target: "rfmetrics::preview",
            "thumbnail -ss {ss} yielded no image ({} bytes, {}ms){}",
            out.stdout.len(),
            start.elapsed().as_millis(),
            {
                let err = String::from_utf8_lossy(&out.stderr);
                let s = err.trim();
                if s.is_empty() {
                    String::new()
                } else {
                    let mut snippet: String = s.chars().take(300).collect();
                    if s.chars().count() > 300 {
                        snippet.push('…');
                    }
                    format!(" stderr: {snippet}")
                }
            },
        );
    }
    log::warn!(target: "rfmetrics::preview", "thumbnail \"{path}\" all seeks failed");
    None
}
#[cfg(test)]
#[path = "tests/test_preview.rs"]
mod tests;
