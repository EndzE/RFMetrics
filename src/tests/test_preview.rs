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
    assert!(extract_thumbnail(Path::new("ffmpeg"), "C:/no/such/file.mp4", Some(10.0)).is_none());
    assert!(extract_thumbnail(Path::new("ffmpeg"), "   ", None).is_none());
}
