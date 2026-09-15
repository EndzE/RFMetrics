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
        return None;
    }
    let img = image::load_from_memory(png_bytes).ok()?.to_rgba8();
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
/// ponytail: no timeout on std Command; single-frame pipe:1 extract is ~ms.
pub fn extract_thumbnail(
    ffmpeg: &Path,
    path: &str,
    duration: Option<f64>,
) -> Option<egui::ColorImage> {
    if path.trim().is_empty() || !Path::new(path).is_file() {
        return None;
    }
    let vf = format!(
        "scale={}:{}:force_original_aspect_ratio=decrease",
        BOX_W * 2,
        BOX_H * 2
    );
    for ss in seek_candidates(duration) {
        let out = Command::new(ffmpeg)
            .args([
                "-hide_banner",
                "-nostdin",
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
            ])
            .output()
            .ok()?;
        if let Some(img) = decode_and_fit(&out.stdout) {
            return Some(img);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeks_match_python_policy() {
        assert_eq!(seek_candidates(Some(10.0)), vec!["4.0", "1.0", "0"]);
        assert_eq!(seek_candidates(Some(5.0)), vec!["4.0", "1.0", "0"]);
        assert_eq!(seek_candidates(Some(3.0)), vec!["1.0", "0"]);
        assert_eq!(seek_candidates(Some(1.5)), vec!["0"]);
        assert_eq!(seek_candidates(None), vec!["1.0", "0"]);
        assert_eq!(seek_candidates(Some(0.0)), vec!["1.0", "0"]);
    }

    #[test]
    fn fit_preserves_ratio_in_box() {
        // 16:9 source nearly fills the 136x76 box (float truncation → 135).
        let (w, h) = fit_size(1920, 1080);
        assert!(w <= BOX_W && h <= BOX_H && w >= 134);
        // Tall source is height-bound.
        let (w, h) = fit_size(1080, 1920);
        assert!(w < BOX_W && h == BOX_H);
        assert_eq!(fit_size(0, 0), (BOX_W, BOX_H));
    }

    #[test]
    fn rejects_tiny_or_garbage() {
        assert!(decode_and_fit(&[]).is_none());
        assert!(decode_and_fit(&[0u8; 101]).is_none());
    }

    #[test]
    fn missing_file_is_none() {
        assert!(
            extract_thumbnail(Path::new("ffmpeg"), "C:/no/such/file.mp4", Some(10.0)).is_none()
        );
        assert!(extract_thumbnail(Path::new("ffmpeg"), "   ", None).is_none());
    }
}
