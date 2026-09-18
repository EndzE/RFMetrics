use super::parse_ffvship_version;
use super::{
    copyright_re, ffmpeg_version_token, parse_filters_list, short_ffmpeg_version, supported_line,
};
use crate::metrics::ffmpeg::MetricKind;

#[test]
fn silent_exit_is_unknown() {
    assert_eq!(parse_ffvship_version("", ""), None);
}

#[test]
fn stderr_gpu_error_is_unknown() {
    assert_eq!(
        parse_ffvship_version("", "HIP error: no AMD device found"),
        None
    );
}

#[test]
fn version_with_backend_suffix() {
    assert_eq!(
        parse_ffvship_version("FFVship 5.1.1-a_cuda\n", ""),
        Some("5.1.1-a_cuda".to_owned())
    );
}

#[test]
fn version_plus_backend_line() {
    assert_eq!(
        parse_ffvship_version("FFVship 5.1.1\ncuda version\n", ""),
        Some("5.1.1_cuda".to_owned())
    );
}

#[test]
fn libvship_line_is_not_a_backend() {
    assert_eq!(
        parse_ffvship_version("FFVship 5.1.1\nlibvship version\n", ""),
        Some("5.1.1".to_owned())
    );
}

#[test]
fn version_on_stderr_still_counts() {
    assert_eq!(
        parse_ffvship_version("", "FFVship 5.1.1-a_cuda\n"),
        Some("5.1.1-a_cuda".to_owned())
    );
}

const FILTERS_FULL: &str = "Filters:\n \
         TSC psnr V->V : Compute the peak signal-to-noise.\n \
         TSC ssim V->V : Compute SSIM.\n \
         TSC libvmaf V->V : Apply VMAF.\n \
         TSC xpsnr V->V : Compute XPSNR.\n";

#[test]
fn filters_parse_finds_all_four() {
    assert_eq!(
        parse_filters_list(FILTERS_FULL),
        vec![
            MetricKind::Psnr,
            MetricKind::Ssim,
            MetricKind::Vmaf,
            MetricKind::Xpsnr
        ]
    );
}

#[test]
fn filters_parse_missing_libvmaf() {
    // w32threads-style build: everything but the VMAF filter.
    let text = "Filters:\n \
             TSC psnr V->V : Compute the peak signal-to-noise.\n \
             TSC ssim V->V : Compute SSIM.\n \
             TSC xpsnr V->V : Compute XPSNR.\n";
    assert_eq!(
        parse_filters_list(text),
        vec![MetricKind::Psnr, MetricKind::Ssim, MetricKind::Xpsnr]
    );
}

#[test]
fn filters_parse_ignores_config_line() {
    // `--enable-libvmaf` is the second token here but not a filter row.
    let text = "configuration: --enable-libvmaf --enable-gpl\n";
    assert!(parse_filters_list(text).is_empty());
    assert!(parse_filters_list("").is_empty());
}

#[test]
fn supported_line_shapes() {
    assert_eq!(
        supported_line(&[
            MetricKind::Psnr,
            MetricKind::Ssim,
            MetricKind::Vmaf,
            MetricKind::Xpsnr
        ]),
        "Supported: PSNR, SSIM, VMAF, XPSNR"
    );
    assert_eq!(supported_line(&[MetricKind::Psnr]), "Supported: PSNR");
    assert_eq!(supported_line(&[]), "Supported: none");
}

#[test]
fn short_version_release_and_git() {
    assert_eq!(
        short_ffmpeg_version("ffmpeg version 7.1.1"),
        "FFmpeg: 7.1.1"
    );
    assert_eq!(
        short_ffmpeg_version("ffmpeg version 9.0.1-full_build-www.gyan.dev"),
        "FFmpeg: 9.0.1-full"
    );
    assert_eq!(
        short_ffmpeg_version("ffmpeg version git-2020-08-31-4a11a6f"),
        "FFmpeg: git-2020-08-31-4a11a6f"
    );
    // Unknown shape falls back to the (already stripped) whole line.
    assert_eq!(short_ffmpeg_version("some weird build"), "some weird build");
}

#[test]
fn short_version_nightly_strips_copyright() {
    // Exact line shape from a nightly `-version` first line.
    let line = "ffmpeg version N-126626-g7070fe638e-20260917 Copyright (c) 2000-2026 the FFmpeg developers";
    let full = copyright_re().replace(line, "").trim_end().to_owned();
    assert_eq!(
        short_ffmpeg_version(&full),
        "FFmpeg: N-126626-g7070fe638e-20260917"
    );
    assert!(!short_ffmpeg_version(&full).contains("Copyright"));
}

#[test]
fn version_token_keeps_vendor_suffix() {
    // The results CSV keeps the raw token verbatim.
    assert_eq!(
        ffmpeg_version_token("ffmpeg version 9.0.1-full_build-www.gyan.dev"),
        Some("9.0.1-full_build-www.gyan.dev".to_owned())
    );
    assert_eq!(
        ffmpeg_version_token("ffmpeg version N-126626-g7070fe638e-20260917"),
        Some("N-126626-g7070fe638e-20260917".to_owned())
    );
    assert_eq!(ffmpeg_version_token("some weird build"), None);
}
