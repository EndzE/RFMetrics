use super::*;

fn ref_info() -> MediaInfo {
    MediaInfo {
        width: Some(1920),
        height: Some(1080),
        fps: Some(25.0),
        pix_fmt: Some("yuv420p".to_owned()),
        ..Default::default()
    }
}

fn cfg() -> VmafCfg {
    VmafCfg {
        model: "vmaf_v0.6.1.json".to_owned(),
        phone: false,
        scale: false,
        pooling: Pooling::Mean,
        subsample: 1,
        // Filter tests pass threads to `build_filter` directly; the
        // snapshot value only matters to `run_vmaf`.
        n_threads: 0,
    }
}

fn models_dir(names: &[&str]) -> (TempfileGuard, PathBuf) {
    // Creates <temp>/vmaf-models/<names>; guard deletes on drop.
    let base = std::env::temp_dir().join(format!(
        "rfmetrics-vmaf-test-{}-{}",
        std::process::id(),
        CTR.fetch_add(1, Ordering::SeqCst)
    ));
    let dir = base.join("vmaf-models");
    std::fs::create_dir_all(&dir).unwrap();
    for n in names {
        std::fs::write(dir.join(n), "{}").unwrap();
    }
    (TempfileGuard(base), dir)
}

struct TempfileGuard(PathBuf);
impl Drop for TempfileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

static CTR: AtomicU64 = AtomicU64::new(0);

#[test]
fn models_list_sorted_or_sentinel() {
    let (_g, dir) = models_dir(&["vmaf_4k_v0.6.1.json", "vmaf_v0.6.1.json"]);
    assert_eq!(
        list_models(&dir),
        vec![
            "vmaf_4k_v0.6.1.json".to_owned(),
            "vmaf_v0.6.1.json".to_owned()
        ]
    );
    let (_g2, empty) = models_dir(&[]);
    // Only non-JSON files: still the sentinel.
    std::fs::write(empty.join("readme.txt"), "x").unwrap();
    assert_eq!(list_models(&empty), vec!["No models found".to_owned()]);
}

#[test]
fn resolver_prefers_valid_then_fallbacks() {
    let (_g, dir) = models_dir(&["custom.json", "vmaf_v0.6.1.json"]);
    assert_eq!(
        resolve_model("custom.json", &dir),
        (
            "path=vmaf-models/custom.json".to_owned(),
            "custom.json".to_owned()
        )
    );
    // Unsafe names and missing files fall back to vmaf_v0.6.1.json.
    assert_eq!(resolve_model("../../evil.json", &dir).1, "vmaf_v0.6.1.json");
    assert_eq!(resolve_model("gone.json", &dir).1, "vmaf_v0.6.1.json");
    // Nothing on disk: builtin version fallback, empty model name.
    let (_g2, empty) = models_dir(&[]);
    assert_eq!(
        resolve_model("vmaf_v0.6.1.json", &empty),
        ("version=vmaf_v0.6.1".to_owned(), String::new())
    );
}

#[test]
fn model_resolution_picks_4k() {
    assert_eq!(model_resolution("vmaf_4k_v0.6.1.json"), (3840, 2160));
    assert_eq!(model_resolution("VMAF_4K_v0.6.1neg.json"), (3840, 2160));
    assert_eq!(model_resolution("vmaf_v0.6.1.json"), (1920, 1080));
    assert_eq!(model_resolution("vmaf_v0.6.1neg.json"), (1920, 1080));
    assert_eq!(model_resolution(""), (1920, 1080));
    // v1 names carry no `4k` marker; `*2160*` routes them instead.
    assert_eq!(model_resolution("vmaf_v1.0.16_3d0h.json"), (1920, 1080));
    assert_eq!(model_resolution("vmaf_v1.0.16_5d0h.json"), (1920, 1080));
    assert_eq!(
        model_resolution("vmaf_v1.0.16_1d5h_2160.json"),
        (3840, 2160)
    );
    assert_eq!(
        model_resolution("vmaf_v1.0.16_3d0h_2160.json"),
        (3840, 2160)
    );
    assert_eq!(model_resolution("vmaf_v1.0.16_hfr_3d0h.json"), (1920, 1080));
    assert_eq!(
        model_resolution("vmaf_v1.0.16_hfr_3d0h_2160.json"),
        (3840, 2160)
    );
}

#[test]
fn v1_detection_covers_hfr_names() {
    for m in [
        "vmaf_v1.0.16_3d0h.json",
        "vmaf_v1.0.16_5d0h.json",
        "vmaf_v1.0.16_1d5h_2160.json",
        "vmaf_v1.0.16_hfr_3d0h.json",
        "vmaf_v1.0.16_hfr_3d0h_2160.json",
    ] {
        assert!(is_v1_model(m), "{m}");
    }
    for m in [
        "vmaf_v0.6.1.json",
        "vmaf_v0.6.1neg.json",
        "vmaf_4k_v0.6.1.json",
        "",
    ] {
        assert!(!is_v1_model(m), "{m}");
    }
}

#[test]
fn filter_baseline_matches_python_shape() {
    let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
    let f = build_filter(
        &ref_info(),
        &ref_info(),
        None,
        None,
        &cfg(),
        &dir,
        "vmaf_log_1.json",
        8,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert_eq!(
        f,
        "[0:v]settb=AVTB,setpts=PTS-STARTPTS[main];\
             [1:v]settb=AVTB,setpts=PTS-STARTPTS[ref];\
             [main][ref]libvmaf=eof_action=endall:model=path=vmaf-models/vmaf_v0.6.1.json:pool=mean:n_subsample=1:n_threads=8:log_path=vmaf_log_1.json:log_fmt=json"
    );
}

#[test]
fn filter_phone_scale_pool_subsample() {
    let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
    let mut c = cfg();
    c.phone = true;
    c.scale = true;
    c.pooling = Pooling::HarmonicMean;
    c.subsample = 5;
    // Above model resolution: both legs scale down to model native.
    let ref4k = MediaInfo {
        width: Some(3840),
        height: Some(2160),
        ..ref_info()
    };
    let f = build_filter(
        &ref4k,
        &ref4k,
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    // Two backslashes before the colon (Python parity): the outer
    // filtergraph consumes one, libvmaf's model parser the other.
    // A single backslash would split `enable_transform` into an
    // unknown top-level libvmaf option ("Option not found").
    assert!(f.contains("model=path=vmaf-models/vmaf_v0.6.1.json\\\\:enable_transform=true"));
    assert!(f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic[main]"));
    assert!(f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic[ref]"));
    assert!(f.contains(":pool=harmonic_mean:"));
    assert!(f.contains(":n_subsample=5:"));
    assert!(!f.contains("n_threads"));
    // Subsample clamps to >= 1.
    c.subsample = 0;
    let f = build_filter(
        &ref_info(),
        &ref_info(),
        None,
        None,
        &c,
        &dir,
        "v.json",
        4,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(f.contains(":n_subsample=1:n_threads=4:"));
}

#[test]
fn scale_follows_model_height_threshold() {
    // Original `ScaleThreshold: 0.1`: both legs scale to model native
    // iff ref height differs from model height by >10%.
    // 1600x1080 is the live parity case (no scale filter emitted).
    let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
    let mut c = cfg();
    c.scale = true;
    let at = |w: i64, h: i64| MediaInfo {
        width: Some(w),
        height: Some(h),
        ..ref_info()
    };
    // Same height, narrower width: no scale (width alone never triggers).
    let small = at(1600, 1080);
    let f = build_filter(
        &small,
        &small,
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(!f.contains("scale="), "must not upscale: {f}");
    assert!(f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS[main]"));
    assert!(f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS[ref]"));
    // Exact model resolution is a no-op too, never a scaler pass.
    let f = build_filter(
        &ref_info(),
        &ref_info(),
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(!f.contains("scale="), "must not rescale in place: {f}");
    // 7.4% under: within threshold, no scale.
    let near = at(1920, 1000);
    let f = build_filter(
        &near,
        &near,
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(!f.contains("scale="), "within threshold: {f}");
    // 33% under: upscale both legs to model native.
    let tiny = at(1280, 720);
    let f = build_filter(
        &tiny,
        &tiny,
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(
        f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic[main]"),
        "{f}"
    );
    assert!(
        f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic[ref]"),
        "{f}"
    );
    // Per-leg aspect fit-inside, read off the original's log verbatim:
    // 1066x720 → 1599:1080, 1760x720 → 1920:785 (width 8.3% under
    // alone would not trigger; the 33% height gap does).
    let a = at(1066, 720);
    let f = build_filter(
        &a,
        &a,
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(f.contains("scale=1599:1080:flags=bicubic"), "{f}");
    let b = at(1760, 720);
    let f = build_filter(
        &b,
        &b,
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(f.contains("scale=1920:785:flags=bicubic"), "{f}");
    // Mixed aspects scale independently with no equalization (the
    // original emits this shape too; libvmaf then fails the pair).
    let wide = at(1600, 1080);
    let f = build_filter(
        &wide,
        &a,
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(
        f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1599:1080:flags=bicubic[main]"),
        "{f}"
    );
    assert!(
        f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS[ref]"),
        "{f}"
    );
}

#[test]
fn phone_guard_rejects_neg_and_4k() {
    let (_g, dir) = models_dir(&["vmaf_v0.6.1neg.json", "vmaf_4k_v0.6.1.json"]);
    let mut c = cfg();
    c.phone = true;
    c.model = "vmaf_v0.6.1neg.json".to_owned();
    assert_eq!(
        build_filter(
            &ref_info(),
            &ref_info(),
            None,
            None,
            &c,
            &dir,
            "v.json",
            0,
            ScaleMethod::Bicubic,
            crate::metrics::ffmpeg::RefPixFmt::NoConversion
        ),
        Err("Model 'vmaf_v0.6.1neg.json' has no Phone transform (use v0.6.1)".to_owned())
    );
    c.model = "vmaf_4k_v0.6.1.json".to_owned();
    assert!(
        build_filter(
            &ref_info(),
            &ref_info(),
            None,
            None,
            &c,
            &dir,
            "v.json",
            0,
            ScaleMethod::Bicubic,
            crate::metrics::ffmpeg::RefPixFmt::NoConversion
        )
        .is_err()
    );
}

#[test]
fn phone_guard_rejects_any_v1_model() {
    // Live finding: `enable_transform` parses on v1 but changes nothing
    // (identical scores), so phone+5d0h included — v1 phone means
    // picking the `5d0h` file with the box unticked.
    let (_g, dir) = models_dir(&[
        "vmaf_v1.0.16_3d0h.json",
        "vmaf_v1.0.16_5d0h.json",
        "vmaf_v1.0.16_hfr_3d0h_2160.json",
    ]);
    let mut c = cfg();
    c.phone = true;
    for m in [
        "vmaf_v1.0.16_3d0h.json",
        "vmaf_v1.0.16_5d0h.json",
        "vmaf_v1.0.16_hfr_3d0h_2160.json",
    ] {
        c.model = m.to_owned();
        assert_eq!(
            build_filter(
                &ref_info(),
                &ref_info(),
                None,
                None,
                &c,
                &dir,
                "v.json",
                0,
                ScaleMethod::Bicubic,
                crate::metrics::ffmpeg::RefPixFmt::NoConversion
            ),
            Err(format!(
                "Model '{m}' has no Phone transform (v1 uses the separate 5d0h phone file)"
            )),
        );
    }
}

#[test]
fn filter_trims_and_scales_dist_only() {
    let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
    let mut dist = ref_info();
    dist.width = Some(1280);
    dist.height = Some(720);
    let f = build_filter(
        &ref_info(),
        &dist,
        Some(5.0),
        Some(10.0),
        &cfg(),
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(f.contains(
            "[0:v]trim=start=5:end=15,settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic[main]"
        ));
    assert!(f.contains("[1:v]trim=start=5:end=15,settb=AVTB,setpts=PTS-STARTPTS[ref]"));
}

#[test]
fn filter_target_converges_and_ignores_rgb() {
    use crate::metrics::ffmpeg::RefPixFmt;
    let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
    let mut dist = ref_info();
    dist.pix_fmt = Some("yuv422p".to_owned());
    let f = build_filter(
        &ref_info(),
        &dist,
        None,
        None,
        &cfg(),
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        RefPixFmt::Yuv444p,
    )
    .unwrap();
    assert!(f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv444p[main]"));
    assert!(f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv444p[ref]"));
    // RGB targets are ignored for VMAF (requires YUV): legacy legs.
    let f = build_filter(
        &ref_info(),
        &dist,
        None,
        None,
        &cfg(),
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        RefPixFmt::Rgb24,
    )
    .unwrap();
    assert!(f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[main]"));
    assert!(f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS[ref]"));
}

#[test]
fn filter_rgb_ref_converges_on_yuv444p() {
    use crate::metrics::ffmpeg::RefPixFmt;
    let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
    let mut rgb = ref_info();
    rgb.pix_fmt = Some("rgb24".to_owned());
    // An RGB reference no longer fails libvmaf: both legs converge on
    // the map canonical format (upstream PixelFormatMap parity).
    let f = build_filter(
        &rgb,
        &rgb,
        None,
        None,
        &cfg(),
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv444p[main]"));
    assert!(f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv444p[ref]"));
}

#[test]
fn log_parses_frames_and_pooled() {
    let text = r#"{
            "frames": [
                {"metrics": {"vmaf": 90.5}},
                {"metrics": {"vmaf": "91.25"}},
                {"metrics": {"psnr": 40.0}},
                {"nope": 1}
            ],
            "pooled_metrics": {"vmaf": {"mean": 91.0, "harmonic_mean": 90.8}}
        }"#;
    let log = parse_vmaf_log(text, 100.0).unwrap();
    assert_eq!(log.values, vec![90.5, 91.25]);
    assert_eq!(log.mean, Some(91.0));
    assert_eq!(log.harmonic_mean, Some(90.8));
}

#[test]
fn log_sanitizes_and_clamps() {
    let text = r#"{"frames": [
            {"metrics": {"vmaf": "inf"}},
            {"metrics": {"vmaf": "nan"}},
            {"metrics": {"vmaf": 150.0}},
            {"metrics": {"vmaf": -5.0}}
        ]}"#;
    let log = parse_vmaf_log(text, 100.0).unwrap();
    assert_eq!(log.values, vec![100.0, 0.0, 100.0, 0.0]);
}

#[test]
fn log_clamps_to_model_range() {
    // Live v1 4K-consumer case (`3d0h_2160`, incl. `_hfr_`): real
    // scores above 100 (mean 104.355, max frame 110) must survive.
    let text = r#"{"frames": [
            {"metrics": {"vmaf": 104.355}},
            {"metrics": {"vmaf": 110.0}},
            {"metrics": {"vmaf": 150.0}},
            {"metrics": {"vmaf": -5.0}},
            {"metrics": {"vmaf": "inf"}}
        ]}"#;
    let log = parse_vmaf_log(text, 110.0).unwrap();
    assert_eq!(log.values, vec![104.355, 110.0, 110.0, 0.0, 110.0]);
    assert_eq!(model_score_max("vmaf_v1.0.16_3d0h_2160.json"), 110.0);
    assert_eq!(model_score_max("vmaf_v1.0.16_hfr_3d0h_2160.json"), 110.0);
    assert_eq!(model_score_max("vmaf_v1.0.16_3d0h.json"), 100.0);
    assert_eq!(model_score_max("vmaf_v0.6.1.json"), 100.0);
}

#[test]
fn log_pooled_sanitizes_like_series() {
    // `nan` pooled drops to `None` (frame mean wins); `inf` saturates.
    let text = r#"{"pooled_metrics": {"vmaf": {"mean": "nan", "harmonic_mean": "inf"}}}"#;
    let log = parse_vmaf_log(text, 100.0).unwrap();
    assert_eq!(log.mean, None);
    assert_eq!(log.harmonic_mean, Some(100.0));
    let text = r#"{"pooled_metrics": {"vmaf": {"mean": 150.0, "harmonic_mean": -5.0}}}"#;
    let log = parse_vmaf_log(text, 100.0).unwrap();
    assert_eq!(log.mean, Some(100.0));
    assert_eq!(log.harmonic_mean, Some(0.0));
}

#[test]
fn log_rejects_garbage() {
    assert!(parse_vmaf_log("not json", 100.0).is_none());
    assert!(parse_vmaf_log("{}", 100.0).unwrap().values.is_empty());
}

#[test]
fn setrange_tags_differing_ranges() {
    let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
    let mut dist = ref_info();
    dist.range_tag = Some("pc".to_owned());
    let mut rf = ref_info();
    rf.range_tag = Some("tv".to_owned());
    let f = build_filter(
        &rf,
        &dist,
        None,
        None,
        &cfg(),
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,setrange=range=pc[main]"));
    assert!(f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS,setrange=range=tv[ref]"));
    // Same range: no segment (FFMetrics.log parity).
    let f = build_filter(
        &ref_info(),
        &ref_info(),
        None,
        None,
        &cfg(),
        &dir,
        "v.json",
        0,
        ScaleMethod::Bicubic,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(!f.contains("setrange"));
}

#[test]
fn model_scale_legs_follow_scaler() {
    let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
    let mut c = cfg();
    c.scale = true;
    let tiny = MediaInfo {
        width: Some(1280),
        height: Some(720),
        ..ref_info()
    };
    // Non-default method lands on both model-scale legs.
    let f = build_filter(
        &tiny,
        &tiny,
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::Lanczos,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(
        f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=lanczos[main]"),
        "{f}"
    );
    assert!(
        f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=lanczos[ref]"),
        "{f}"
    );
    // FFmpeg default: flagless legs (today's exact strings).
    let f = build_filter(
        &tiny,
        &tiny,
        None,
        None,
        &c,
        &dir,
        "v.json",
        0,
        ScaleMethod::FfmpegDefault,
        crate::metrics::ffmpeg::RefPixFmt::NoConversion,
    )
    .unwrap();
    assert!(f.contains("scale=1920:1080[main]"), "{f}");
    assert!(f.contains("scale=1920:1080[ref]"), "{f}");
    assert!(!f.contains("flags="), "{f}");
}
