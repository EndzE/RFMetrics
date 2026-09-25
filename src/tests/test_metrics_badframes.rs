use super::*;
use std::path::Path;

#[test]
fn worst_min_takes_lowest_first() {
    let v = vec![90.0, 70.0, 80.0, 60.0];
    assert_eq!(worst_n(&v, 2, false), vec![(3, 60.0), (1, 70.0)]);
}

#[test]
fn worst_max_takes_highest_first() {
    let v = vec![1.0, 5.0, 3.0];
    assert_eq!(worst_n(&v, 2, true), vec![(1, 5.0), (2, 3.0)]);
}

#[test]
fn worst_empty_and_over_n() {
    assert!(worst_n(&[], 5, false).is_empty());
    assert!(worst_n(&[1.0], 0, false).is_empty());
    assert_eq!(worst_n(&[2.0, 1.0], 9, false).len(), 2);
}

#[test]
fn offset_adds_skip() {
    assert!((frame_offset(2.5, 30, 60.0) - 3.0).abs() < 1e-9);
}

#[test]
fn argv_matches_original_template() {
    let a = ffmpeg_args("in.mkv", "o.png", 1.5, 60.0);
    let j = a.join(" ");
    for token in [
        "-accurate_seek",
        "-probesize 50M",
        "-r 60",
        "-ss 1.500000",
        "settb=AVTB,setpts=PTS-STARTPTS",
        "accurate_rnd+full_chroma_int+bitexact",
    ] {
        assert!(j.contains(token), "missing {token} in {j}");
    }
}

#[test]
fn viewer_fit_centers_union() {
    assert_eq!(viewer_fit(100, 50, 80, 60), (-50.0, 50.0, -30.0, 30.0));
    assert_eq!(viewer_fit(0, 0, 0, 0), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn wipe_split_clamps() {
    assert_eq!(clamp_split(0.5), 0.5);
    assert_eq!(clamp_split(-1.0), 0.0);
    assert_eq!(clamp_split(2.0), 1.0);
}

#[test]
fn export_dest_empty_dir_lands_beside_dist() {
    let d = export_dest_for("", "C:/v/movie.mkv", "PSNR", 7, false);
    assert_eq!(d, Path::new("C:/v/movie.mkv.PSNR.bf000007.png"));
    let r = export_dest_for("   ", "C:/v/movie.mkv", "PSNR", 7, true);
    assert_eq!(r, Path::new("C:/v/movie.mkv.PSNR.bf000007-ref.png"));
}

#[test]
fn export_dest_set_dir_joins_stem() {
    let d = export_dest_for("D:/out", "C:/v/movie.mkv", "VMAF", 12, false);
    assert_eq!(d, Path::new("D:/out/movie.mkv.VMAF.bf000012.png"));
    let r = export_dest_for("D:/out", "C:/v/movie.mkv", "VMAF", 12, true);
    assert_eq!(r, Path::new("D:/out/movie.mkv.VMAF.bf000012-ref.png"));
}

#[test]
fn wipe_layout_tiles_without_gap() {
    for f in [0.0, 0.25, 0.5, 0.9, 1.0] {
        let l = wipe_layout(200.0, f);
        assert!((l.left_cx + l.left_w / 2.0 - l.div_x).abs() < 1e-9);
        assert!((l.right_cx - l.right_w / 2.0 - l.div_x).abs() < 1e-9);
        assert!(((l.left_w + l.right_w) - 200.0).abs() < 1e-9);
    }
    let l = wipe_layout(200.0, 0.5);
    assert!((l.div_x).abs() < 1e-9);
    assert!((l.left_cx + 50.0).abs() < 1e-9);
    assert!((l.right_cx - 50.0).abs() < 1e-9);
    assert_eq!(l.u, 0.5);
    // Out-of-range splits pin to the nearest edge.
    assert_eq!(wipe_layout(200.0, -5.0).u, 0.0);
    assert_eq!(wipe_layout(200.0, 5.0).u, 1.0);
}

#[test]
fn changed_overlay_maps_diff_to_purple_alpha() {
    let mut a = image::RgbaImage::new(2, 1);
    let mut b = image::RgbaImage::new(2, 1);
    a.put_pixel(0, 0, image::Rgba([10, 20, 30, 255]));
    b.put_pixel(0, 0, image::Rgba([10, 20, 30, 255]));
    a.put_pixel(1, 0, image::Rgba([0, 0, 0, 255]));
    b.put_pixel(1, 0, image::Rgba([255, 255, 255, 255]));
    let out = changed_overlay(&a, &b);
    assert_eq!(out.get_pixel(0, 0), &image::Rgba([180, 0, 255, 0]));
    assert_eq!(out.get_pixel(1, 0), &image::Rgba([180, 0, 255, 128]));
    // Noise-level change stays transparent; a small real change starts at
    // the visible floor (typical artifacts were ~1-3% opacity before).
    let mut c = image::RgbaImage::new(2, 1);
    c.put_pixel(0, 0, image::Rgba([100, 100, 100, 255]));
    c.put_pixel(1, 0, image::Rgba([100, 100, 100, 255]));
    let mut e = image::RgbaImage::new(2, 1);
    e.put_pixel(0, 0, image::Rgba([102, 102, 102, 255]));
    e.put_pixel(1, 0, image::Rgba([110, 110, 110, 255]));
    let out2 = changed_overlay(&c, &e);
    assert_eq!(out2.get_pixel(0, 0)[3], 0);
    assert!(
        out2.get_pixel(1, 0)[3] >= 48,
        "small change must be visible"
    );
    // Size mismatch resamples dist to ref dims instead of panicking.
    let big = image::RgbaImage::from_pixel(4, 4, image::Rgba([0, 0, 0, 255]));
    let small = image::RgbaImage::from_pixel(2, 2, image::Rgba([0, 0, 0, 255]));
    assert_eq!(changed_overlay(&big, &small).dimensions(), (2, 2));
}

#[test]
fn tmp_names_match_beside_file_stems() {
    let tmp = Path::new("tmp");
    let d = tmp_dest_for(tmp, "C:/v/movie.mkv", "PSNR", 7);
    assert_eq!(
        d.file_name().unwrap().to_str().unwrap(),
        "movie.mkv.PSNR.bf000007.png"
    );
    let r = tmp_dest_ref_for(tmp, "C:/v/movie.mkv", "PSNR", 7);
    assert_eq!(
        r.file_name().unwrap().to_str().unwrap(),
        "movie.mkv.PSNR.bf000007-ref.png"
    );
    assert!(tmp_dir().to_string_lossy().contains("rfmetrics-bf-"));
}
