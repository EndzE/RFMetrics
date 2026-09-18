use super::*;
use crate::metrics::ffmpeg::FrameDetail;

fn test_dir(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("rfmetrics-csv-test-{}-{name}", std::process::id()));
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
