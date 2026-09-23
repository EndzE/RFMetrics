use super::ScaleMethod;
use super::{
    CachedStats, DropAction, METRIC_COLUMNS, ProbeMsg, QueueRow, RFMetricsApp, display_names,
    norm_key, route_drop,
};
use crate::metrics::ffmpeg::InputFpsMode;

#[test]
fn metric_columns_keep_live_cells_under_their_headers() {
    use crate::metrics::ffmpeg::MetricKind;
    // Body order must mirror the header checkboxes (PSNR SSIM VMAF
    // XPSNR SSIM2 BUTTER CVVDP); every slot is live, so no column can
    // slide under the wrong header.
    let kinds: Vec<_> = METRIC_COLUMNS.iter().map(|(k, _)| *k).collect();
    assert_eq!(
        kinds,
        [
            Some(MetricKind::Psnr),
            Some(MetricKind::Ssim),
            Some(MetricKind::Vmaf),
            Some(MetricKind::Xpsnr),
            Some(MetricKind::Ssim2),
            Some(MetricKind::But),
            Some(MetricKind::Cvvdp),
        ]
    );
    let titles: Vec<_> = METRIC_COLUMNS.iter().map(|(_, t)| *t).collect();
    assert_eq!(
        titles,
        ["PSNR", "SSIM", "VMAF", "XPSNR", "SSIM2", "BUTTER", "CVVDP"]
    );
}

#[test]
fn drop_routing() {
    use std::path::PathBuf;
    let files = || vec![PathBuf::from("C:/v/a.mp4"), PathBuf::from("C:/v/b.mp4")];
    // Empty drop: nothing, even mid-run.
    assert_eq!(route_drop(false, true, true, vec![]), DropAction::Ignore);
    assert_eq!(route_drop(true, true, true, vec![]), DropAction::Ignore);
    // Mid-run drops block with a toast instead of mutating state.
    assert_eq!(route_drop(true, true, false, files()), DropAction::Blocked);
    assert_eq!(route_drop(true, false, true, files()), DropAction::Blocked);
    // Reference takes one file; extras are reported, not lost.
    assert_eq!(
        route_drop(false, true, false, files()),
        DropAction::SetRef {
            first: PathBuf::from("C:/v/a.mp4"),
            extra: 1,
        }
    );
    // Table queues everything; outside any target is ignored.
    assert_eq!(
        route_drop(false, false, true, files()),
        DropAction::Queue(files())
    );
    assert_eq!(route_drop(false, false, false, files()), DropAction::Ignore);
}

#[cfg(windows)]
#[test]
fn same_file_keys_equal() {
    assert_eq!(norm_key("C:/Vids/a.mp4"), norm_key("c:\\vids\\A.MP4"));
    assert_ne!(norm_key("C:/Vids/a.mp4"), norm_key("C:/Vids/b.mp4"));
}

#[cfg(not(windows))]
#[test]
fn posix_keys_keep_separators_and_case() {
    // `foo\bar` is a legal literal filename on Unix, distinct from `foo/bar`.
    assert_ne!(
        norm_key("/vids/foo/bar.mp4"),
        norm_key("/vids/foo\\bar.mp4")
    );
    assert_ne!(norm_key("/vids/a.mp4"), norm_key("/vids/A.MP4"));
    assert_ne!(norm_key("/vids/a.mp4"), norm_key("/vids/b.mp4"));
}

#[test]
fn single_name_is_basename() {
    assert_eq!(
        display_names(&["C:/a/b/c.mp4".to_owned()]),
        vec!["c.mp4".to_owned()]
    );
}

#[test]
fn sibling_names_disambiguate() {
    let names = display_names(&[
        "C:/b/output tq 70.mkv".to_owned(),
        "C:/b/output tq 75.mkv".to_owned(),
    ]);
    assert_eq!(names, vec!["output tq 70.mkv", "output tq 75.mkv"]);
}

#[test]
fn same_basename_keeps_parent() {
    let names = display_names(&["C:/a/x.mp4".to_owned(), "C:/b/x.mp4".to_owned()]);
    let sep = std::path::MAIN_SEPARATOR;
    assert_eq!(names, vec![format!("a{sep}x.mp4"), format!("b{sep}x.mp4")]);
}

/// Refresh re-probes the reference and every row through the normal
/// worker channels: markers clear, rows show the placeholder, and
/// results land via drain (missing files resolve without a process).
#[test]
fn refresh_media_info_reprobes_ref_and_rows() {
    // Hermetic: no ffprobe, so workers resolve to text without spawning.
    let mut app = RFMetricsApp {
        ffprobe: None,
        ..RFMetricsApp::default()
    };
    app.rows.push(psnr_test_row("C:/no/such/a.mp4", true));
    app.rows[0].media = "old".to_owned();
    app.last_spawned_ref = "sentinel".to_owned();
    app.last_thumb_path = "sentinel".to_owned();
    app.ref_path = "C:/no/such/ref.mp4".to_owned();
    app.refresh_media_info();
    assert!(app.last_spawned_ref.is_empty());
    assert!(app.last_thumb_path.is_empty());
    assert_eq!(app.rows[0].media, "Probing…");
    for _ in 0..200 {
        app.drain_probe_results();
        if app.rows[0].media != "Probing…" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_ne!(app.rows[0].media, "Probing…");
    // Reference cheap case resolves inline on the next refresh tick.
    app.refresh_ref_info();
    assert_eq!(app.ref_info, "File not found");
}

#[test]
fn refresh_media_info_empty_queue_only_clears_markers() {
    let mut app = RFMetricsApp {
        last_spawned_ref: "sentinel".to_owned(),
        ..RFMetricsApp::default()
    };
    app.refresh_media_info();
    assert!(app.last_spawned_ref.is_empty());
    assert!(app.probe_rx.try_recv().is_err());
}

#[test]
fn ref_cheap_cases_stay_synchronous() {
    let mut app = RFMetricsApp {
        ref_path: String::new(),
        ..RFMetricsApp::default()
    };
    app.refresh_ref_info();
    assert!(app.ref_info.contains("-unknown-"));
    app.ref_path = "C:/no/such/file.mp4".to_owned();
    app.refresh_ref_info();
    assert_eq!(app.ref_info, "File not found");
    // Neither case spawns a worker: the channel stays empty.
    assert!(app.probe_rx.try_recv().is_err());
}

#[test]
fn stale_reference_result_discarded() {
    let mut app = RFMetricsApp {
        ref_info: "sentinel".to_owned(),
        ..RFMetricsApp::default()
    };
    app.probe_tx
        .send(ProbeMsg::Reference {
            generation: 999,
            text: "stale".to_owned(),
            info: None,
            timed_out: false,
        })
        .unwrap();
    app.refresh_ref_info();
    assert_eq!(app.ref_info, "sentinel");
    app.probe_tx
        .send(ProbeMsg::Reference {
            generation: app.ref_generation,
            text: "fresh".to_owned(),
            info: None,
            timed_out: false,
        })
        .unwrap();
    app.refresh_ref_info();
    assert_eq!(app.ref_info, "fresh");
}

#[test]
fn row_media_applies_by_key() {
    let mut app = RFMetricsApp::default();
    app.rows.push(QueueRow {
        path: "C:/vids/a.mp4".to_owned(),
        key: norm_key("C:/vids/a.mp4"),
        display: "a.mp4".to_owned(),
        include: true,
        color_idx: 0,
        selected: false,
        media: "Probing…".to_owned(),
        media_tip: "Probing…".to_owned(),
        info: None,
        psnr: crate::metrics::MetricCell::Idle,
        ssim: crate::metrics::MetricCell::Idle,
        vmaf: crate::metrics::MetricCell::Idle,
        xpsnr: crate::metrics::MetricCell::Idle,
        ssim2: crate::metrics::MetricCell::Idle,
        butter: crate::metrics::MetricCell::Idle,
        cvvdp: crate::metrics::MetricCell::Idle,
        psnr_cache: CachedStats::default(),
        ssim_cache: CachedStats::default(),
        vmaf_cache: CachedStats::default(),
        xpsnr_cache: CachedStats::default(),
        ssim2_cache: CachedStats::default(),
        butter_cache: CachedStats::default(),
        cvvdp_cache: CachedStats::default(),
    });
    let key = norm_key("C:/vids/a.mp4");
    app.probe_tx
        .send(ProbeMsg::RowMedia {
            key,
            media: "h264, 1080p".to_owned(),
            tip: "tip".to_owned(),
            info: None,
            timed_out: false,
        })
        .unwrap();
    app.probe_tx
        .send(ProbeMsg::RowMedia {
            key: "nope".to_owned(),
            media: "x".to_owned(),
            tip: "y".to_owned(),
            info: None,
            timed_out: false,
        })
        .unwrap();
    app.refresh_ref_info();
    assert_eq!(app.rows[0].media, "h264, 1080p");
    assert_eq!(app.rows[0].media_tip, "tip");
}

#[test]
fn plot_color_slots_survive_removal() {
    // Slots are assigned once at insert and never reused: removing a row
    // must not recolor the survivors, and a later insert takes a fresh
    // slot instead of filling the gap.
    let mut app = RFMetricsApp::default();
    app.add_queue_files(vec![
        std::path::PathBuf::from("C:/no/such/c1.mp4"),
        std::path::PathBuf::from("C:/no/such/c2.mp4"),
        std::path::PathBuf::from("C:/no/such/c3.mp4"),
    ]);
    assert_eq!(
        app.rows.iter().map(|r| r.color_idx).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    app.rows.remove(1);
    app.add_queue_files(vec![std::path::PathBuf::from("C:/no/such/c4.mp4")]);
    assert_eq!(
        app.rows.iter().map(|r| r.color_idx).collect::<Vec<_>>(),
        vec![0, 2, 3]
    );
}

#[test]
fn queue_shows_probing_placeholder() {
    let mut app = RFMetricsApp::default();
    app.add_queue_files(vec![std::path::PathBuf::from("C:/no/such/file.mp4")]);
    assert_eq!(app.rows.len(), 1);
    assert_eq!(app.rows[0].media, "Probing…");
    // Missing files resolve without spawning ffprobe; poll briefly.
    for _ in 0..200 {
        app.drain_probe_results();
        if app.rows[0].media != "Probing…" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(app.rows[0].media.contains("-unknown-"));
}

fn psnr_test_row(path: &str, include: bool) -> QueueRow {
    QueueRow {
        path: path.to_owned(),
        key: norm_key(path),
        display: "a.mp4".to_owned(),
        include,
        color_idx: 0,
        selected: false,
        media: "h264, 1080p".to_owned(),
        media_tip: "tip".to_owned(),
        info: None,
        psnr: crate::metrics::MetricCell::Idle,
        ssim: crate::metrics::MetricCell::Idle,
        vmaf: crate::metrics::MetricCell::Idle,
        xpsnr: crate::metrics::MetricCell::Idle,
        ssim2: crate::metrics::MetricCell::Idle,
        butter: crate::metrics::MetricCell::Idle,
        cvvdp: crate::metrics::MetricCell::Idle,
        psnr_cache: CachedStats::default(),
        ssim_cache: CachedStats::default(),
        vmaf_cache: CachedStats::default(),
        xpsnr_cache: CachedStats::default(),
        ssim2_cache: CachedStats::default(),
        butter_cache: CachedStats::default(),
        cvvdp_cache: CachedStats::default(),
    }
}

#[test]
fn start_psnr_gated_on_checkbox() {
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.m_psnr = false;
    app.m_vmaf = false;
    app.start_run(0.0);
    assert!(!app.measuring);
    assert!(matches!(app.rows[0].psnr, crate::metrics::MetricCell::Idle));
    assert!(app.toast.is_some());
}

#[test]
fn start_psnr_needs_included_rows() {
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        ..RFMetricsApp::default()
    };
    // Unchecked include box: the row must not be processed.
    app.rows.push(psnr_test_row("C:/vids/a.mp4", false));
    app.start_run(0.0);
    assert!(!app.measuring);
    assert!(matches!(app.rows[0].psnr, crate::metrics::MetricCell::Idle));
}

#[test]
fn start_psnr_bad_time_marks_cells() {
    let p = std::env::temp_dir().join("rfmetrics-psnr-ref.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_ssim: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        skip: "abc".to_owned(),
        ..RFMetricsApp::default()
    };
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(!app.measuring);
    assert!(matches!(
        &app.rows[0].psnr,
        crate::metrics::MetricCell::Error { msg } if msg == "bad time"
    ));
    assert!(matches!(
        &app.rows[0].ssim,
        crate::metrics::MetricCell::Error { msg } if msg == "bad time"
    ));
}

#[test]
fn psnr_progress_keeps_max_and_done_clears() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Running {
        frame: 10,
        values: Vec::new(),
    };
    app.run_generation = 1;
    app.pending = 1;
    app.measuring = true;
    let key = norm_key("C:/vids/a.mp4");
    // Stale frame ignored, fresh frame applied.
    app.metric_tx
        .send(MetricMsg::Progress {
            generation: 1,
            kind: crate::metrics::ffmpeg::MetricKind::Psnr,
            key: key.clone(),
            frame: 5,
        })
        .unwrap();
    app.metric_tx
        .send(MetricMsg::Progress {
            generation: 1,
            kind: crate::metrics::ffmpeg::MetricKind::Psnr,
            key: key.clone(),
            frame: 25,
        })
        .unwrap();
    app.drain_metric_results();
    assert!(matches!(
        app.rows[0].psnr,
        MetricCell::Running { frame: 25, .. }
    ));
    // No-summary avg falls back to the arithmetic mean; run ends.
    // Settings stamp through: the cell remembers this trim.
    app.metric_tx
        .send(MetricMsg::Done {
            generation: 1,
            kind: crate::metrics::ffmpeg::MetricKind::Psnr,
            key,
            values: vec![30.0, 32.0],
            avg: None,
            exec_s: 1.5,
            error: None,
            skip: None,
            clip_dur: Some(5.0),
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        })
        .unwrap();
    app.drain_metric_results();
    assert!(matches!(
        &app.rows[0].psnr,
        MetricCell::Done { avg, skip, clip_dur, .. }
            if (*avg - 31.0).abs() < 1e-9 && skip.is_none() && *clip_dur == Some(5.0)
    ));
    assert!(!app.measuring);
}

#[test]
fn series_appends_in_order_and_done_replaces() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Running {
        frame: 0,
        values: Vec::new(),
    };
    app.run_generation = 1;
    let key = norm_key("C:/vids/a.mp4");
    let series = |generation: u64, vals: Vec<f64>| MetricMsg::Series {
        generation,
        kind: crate::metrics::ffmpeg::MetricKind::Psnr,
        key: key.clone(),
        new_values: vals,
    };
    // Stale generation and empty batches drop silently.
    app.metric_tx.send(series(0, vec![99.0])).unwrap();
    app.metric_tx.send(series(1, vec![])).unwrap();
    app.metric_tx.send(series(1, vec![30.0, 31.0])).unwrap();
    app.metric_tx.send(series(1, vec![32.0])).unwrap();
    app.drain_metric_results();
    assert!(matches!(
        &app.rows[0].psnr,
        MetricCell::Running { values, .. } if values == &[30.0, 31.0, 32.0]
    ));
    // Plot decimates straight from values (x = 1-based frame).
    let thin = crate::plot::decimate_minmax(&[30.0, 31.0, 32.0], 512);
    let pts = thin.points();
    assert_eq!(pts.len(), 3);
    assert_eq!((pts[0].x, pts[0].y), (1.0, 30.0));
    assert_eq!((pts[2].x, pts[2].y), (3.0, 32.0));
    // Batches for a settled cell are ignored, not resurrected.
    app.rows[0].psnr = MetricCell::Idle;
    app.metric_tx.send(series(1, vec![33.0])).unwrap();
    app.drain_metric_results();
    assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
    // Done replaces the live buffer with the strict series.
    app.rows[0].psnr = MetricCell::Running {
        frame: 3,
        values: vec![30.0, 31.0, 32.0],
    };
    app.metric_tx
        .send(MetricMsg::Done {
            generation: 1,
            kind: crate::metrics::ffmpeg::MetricKind::Psnr,
            key: key.clone(),
            values: vec![29.0, 31.0],
            avg: Some(30.0),
            exec_s: 1.0,
            error: None,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        })
        .unwrap();
    app.drain_metric_results();
    assert!(matches!(
        &app.rows[0].psnr,
        MetricCell::Done { values, .. } if values == &[29.0, 31.0]
    ));
    // Done draws from the strict series (not the partials).
    let thin = crate::plot::decimate_minmax(&[29.0, 31.0], 512);
    let pts = thin.points();
    assert_eq!(pts.len(), 2);
    assert_eq!((pts[0].x, pts[0].y), (1.0, 29.0));
    assert_eq!((pts[1].x, pts[1].y), (2.0, 31.0));
}

#[test]
fn progress_and_series_track_live_kind() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    use crate::metrics::ffmpeg::MetricKind;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Running {
        frame: 0,
        values: Vec::new(),
    };
    app.run_generation = 1;
    let key = norm_key("C:/vids/a.mp4");
    assert_eq!(app.live_kind, None);
    // Stale generation touches nothing.
    app.metric_tx
        .send(MetricMsg::Progress {
            generation: 0,
            kind: MetricKind::Xpsnr,
            key: key.clone(),
            frame: 5,
        })
        .unwrap();
    app.drain_metric_results();
    assert_eq!(app.live_kind, None);
    // Live feeds record the executing job's kind.
    app.metric_tx
        .send(MetricMsg::Progress {
            generation: 1,
            kind: MetricKind::Psnr,
            key: key.clone(),
            frame: 5,
        })
        .unwrap();
    app.drain_metric_results();
    assert_eq!(app.live_kind, Some(MetricKind::Psnr));
    app.metric_tx
        .send(MetricMsg::Series {
            generation: 1,
            kind: MetricKind::Ssim,
            key,
            new_values: vec![0.9],
        })
        .unwrap();
    app.drain_metric_results();
    assert_eq!(app.live_kind, Some(MetricKind::Ssim));
}

#[test]
fn live_key_tracks_executing_job() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    use crate::metrics::ffmpeg::MetricKind;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
    for row in &mut app.rows {
        row.psnr = MetricCell::Running {
            frame: 0,
            values: Vec::new(),
        };
    }
    app.run_generation = 1;
    let key_a = norm_key("C:/vids/a.mp4");
    let key_b = norm_key("C:/vids/b.mp4");
    assert_eq!(app.live_key, None);
    // Stale generation touches nothing.
    app.metric_tx
        .send(MetricMsg::Progress {
            generation: 0,
            kind: MetricKind::Psnr,
            key: key_a.clone(),
            frame: 5,
        })
        .unwrap();
    app.drain_metric_results();
    assert_eq!(app.live_key, None);
    // First job goes live…
    app.metric_tx
        .send(MetricMsg::Progress {
            generation: 1,
            kind: MetricKind::Psnr,
            key: key_a.clone(),
            frame: 5,
        })
        .unwrap();
    app.drain_metric_results();
    assert_eq!(app.live_key, Some(key_a.clone()));
    // …then the worker moves to the next file: only that cell is live.
    app.metric_tx
        .send(MetricMsg::Progress {
            generation: 1,
            kind: MetricKind::Psnr,
            key: key_b.clone(),
            frame: 3,
        })
        .unwrap();
    app.drain_metric_results();
    assert_eq!(app.live_key, Some(key_b.clone()));
    // Its Done clears the live slot (inter-job gap shows no sweep).
    app.metric_tx
        .send(MetricMsg::Done {
            generation: 1,
            kind: MetricKind::Psnr,
            key: key_b.clone(),
            values: vec![30.0],
            avg: Some(30.0),
            exec_s: 1.0,
            error: None,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        })
        .unwrap();
    app.drain_metric_results();
    assert_eq!(app.live_key, None);
    // A Done for any other job never clears a live one (ordered
    // channel, but the guard makes it robust).
    app.metric_tx
        .send(MetricMsg::Progress {
            generation: 1,
            kind: MetricKind::Psnr,
            key: key_a.clone(),
            frame: 9,
        })
        .unwrap();
    app.drain_metric_results();
    assert_eq!(app.live_key, Some(key_a.clone()));
    app.metric_tx
        .send(MetricMsg::Done {
            generation: 1,
            kind: MetricKind::Psnr,
            key: key_b.clone(),
            values: vec![31.0],
            avg: Some(31.0),
            exec_s: 1.0,
            error: None,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        })
        .unwrap();
    app.drain_metric_results();
    assert_eq!(app.live_key, Some(key_a.clone()));
}

#[test]
fn psnr_stale_generation_dropped() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.run_generation = 2; // run 1's messages are orphans after Reset
    app.metric_tx
        .send(MetricMsg::Done {
            generation: 1,
            kind: crate::metrics::ffmpeg::MetricKind::Psnr,
            key: norm_key("C:/vids/a.mp4"),
            values: vec![30.0],
            avg: Some(30.0),
            exec_s: 1.0,
            error: None,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        })
        .unwrap();
    app.drain_metric_results();
    assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
}

#[test]
fn stop_is_noop_when_idle() {
    use std::sync::atomic::Ordering;
    let mut app = RFMetricsApp::default();
    app.stop_psnr();
    assert!(!app.abort.load(Ordering::SeqCst));
}

#[test]
fn drain_caches_stats_and_ranks_once() {
    use super::MetricMsg;
    use crate::metrics::StatRank;
    use crate::metrics::ffmpeg::MetricKind;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
    app.run_generation = 1;
    for (path, avg) in [("C:/vids/a.mp4", 30.0), ("C:/vids/b.mp4", 40.0)] {
        app.metric_tx
            .send(MetricMsg::Done {
                generation: 1,
                kind: MetricKind::Psnr,
                key: norm_key(path),
                values: vec![avg - 1.0, avg, avg + 1.0],
                avg: Some(avg),
                exec_s: 1.0,
                error: None,
                skip: None,
                clip_dur: None,
                vmaf_cfg: None,
                scaler: ScaleMethod::Bicubic,
                fps_mode: InputFpsMode::Reference,
            })
            .unwrap();
    }
    app.drain_metric_results();
    // Stats cached on arrival (no per-frame clone+sort in the render).
    assert!(app.rows[0].psnr_cache.stats.is_some());
    assert!(app.rows[1].psnr_cache.stats.is_some());
    // Ranks resolved across the scored set: avg index 0 decides the cell.
    assert_eq!(app.rows[0].psnr_cache.ranks[0], StatRank::Worst);
    assert_eq!(app.rows[1].psnr_cache.ranks[0], StatRank::Best);
    // Reset clears the cache back to Plain.
    app.reset_psnr();
    assert!(app.rows[0].psnr_cache.stats.is_none());
    assert_eq!(app.rows[0].psnr_cache.ranks, [StatRank::Plain; 10]);
}

#[test]
fn reset_metric_clears_only_that_column() {
    use crate::metrics::ffmpeg::MetricKind;
    use crate::metrics::{MetricCell, StatRank};
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    let done = |avg: f64| MetricCell::Done {
        values: vec![avg - 1.0, avg, avg + 1.0],
        avg,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].psnr = done(30.0);
    app.rows[0].ssim = done(0.95);
    app.rows[0].psnr_cache.stats = app.rows[0].psnr.done_stats();
    app.rows[0].ssim_cache.stats = app.rows[0].ssim.done_stats();
    app.rows[0].psnr_cache.text = app.rows[0].psnr.cell_text();
    app.rows[0].ssim_cache.text = app.rows[0].ssim.cell_text();
    app.live_kind = Some(MetricKind::Psnr);
    app.live_key = Some(app.rows[0].key.clone());
    app.reset_metric(MetricKind::Psnr);
    assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
    assert!(app.rows[0].psnr_cache.stats.is_none());
    assert_eq!(app.rows[0].psnr_cache.ranks, [StatRank::Plain; 10]);
    // Other columns untouched.
    assert!(matches!(app.rows[0].ssim, MetricCell::Done { .. }));
    assert!(app.rows[0].ssim_cache.stats.is_some());
    assert_eq!(app.rows[0].ssim_cache.text, "0.9500");
    // Live pointer cleared only because it pointed at the reset kind.
    assert_eq!(app.live_kind, None);
}

#[test]
fn done_stale_marking_matches_recompute_rules() {
    use super::done_is_stale;
    use crate::metrics::MetricCell;
    use crate::metrics::ffmpeg::MetricKind;
    let app = RFMetricsApp::default();
    // Default trim boxes are empty (no trim); the fixture stamps match.
    let (skip, clip) = (None, None);
    let cur_vmaf = app.current_vmaf_cfg();
    let done_psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip,
        clip_dur: clip,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    let cur = || {
        (
            skip,
            clip,
            app.current_vmaf_cfg(),
            app.scale_method,
            app.fps_mode,
        )
    };
    let (s, c, v, sc, fm) = cur();
    // Matching stamps: fresh.
    assert!(!done_is_stale(
        MetricKind::Psnr,
        &done_psnr,
        s,
        c,
        &v,
        sc,
        fm
    ));
    // Non-Done cells are never stale.
    assert!(!done_is_stale(
        MetricKind::Psnr,
        &MetricCell::Idle,
        Some(5.0),
        c,
        &v,
        sc,
        fm
    ));
    assert!(!done_is_stale(
        MetricKind::Psnr,
        &MetricCell::Error {
            msg: "x".to_owned()
        },
        Some(5.0),
        c,
        &v,
        sc,
        fm
    ));
    // Trim change stales every kind, including FFVship ones.
    assert!(done_is_stale(
        MetricKind::Psnr,
        &done_psnr,
        Some(5.0),
        c,
        &v,
        sc,
        fm
    ));
    assert!(done_is_stale(
        MetricKind::Ssim2,
        &done_psnr,
        Some(5.0),
        c,
        &v,
        sc,
        fm
    ));
    // VMAF options change stales VMAF alone.
    let vmaf_app = RFMetricsApp {
        vmaf_subsample: "5".to_owned(),
        ..RFMetricsApp::default()
    };
    let new_vmaf = vmaf_app.current_vmaf_cfg();
    assert_ne!(cur_vmaf, new_vmaf);
    assert!(done_is_stale(
        MetricKind::Vmaf,
        &done_psnr,
        s,
        c,
        &new_vmaf,
        sc,
        fm
    ));
    assert!(!done_is_stale(
        MetricKind::Psnr,
        &done_psnr,
        s,
        c,
        &new_vmaf,
        sc,
        fm
    ));
    // Scaler / fps-mode changes stale ffmpeg-backed columns only.
    let other_scaler = if sc == ScaleMethod::Bicubic {
        ScaleMethod::Lanczos
    } else {
        ScaleMethod::Bicubic
    };
    assert!(done_is_stale(
        MetricKind::Psnr,
        &done_psnr,
        s,
        c,
        &v,
        other_scaler,
        fm
    ));
    assert!(!done_is_stale(
        MetricKind::Ssim2,
        &done_psnr,
        s,
        c,
        &v,
        other_scaler,
        fm
    ));
    let other_fps = if fm == InputFpsMode::Reference {
        InputFpsMode::Off
    } else {
        InputFpsMode::Reference
    };
    assert!(done_is_stale(
        MetricKind::Ssim,
        &done_psnr,
        s,
        c,
        &v,
        sc,
        other_fps
    ));
    assert!(!done_is_stale(
        MetricKind::But,
        &done_psnr,
        s,
        c,
        &v,
        sc,
        other_fps
    ));
}

#[test]
fn refresh_ranks_single_row_stays_plain() {
    use crate::metrics::ffmpeg::MetricKind;
    use crate::metrics::{MetricCell, StatRank};
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0, 31.0],
        avg: 30.5,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    let stats = app.rows[0].psnr.done_stats();
    app.rows[0].psnr_cache.stats = stats;
    app.refresh_ranks(MetricKind::Psnr);
    assert_eq!(app.rows[0].psnr_cache.ranks, [StatRank::Plain; 10]);
}

#[test]
fn stop_keeps_done_and_settles_running_to_idle() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    use std::sync::atomic::Ordering;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
    // a finished before Stop, b was in flight.
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[1].psnr = MetricCell::Running {
        frame: 12,
        values: Vec::new(),
    };
    app.rows[1].ssim = MetricCell::Running {
        frame: 3,
        values: Vec::new(),
    };
    app.measuring = true;
    app.pending = 1;
    app.run_generation = 1;

    app.stop_psnr();
    assert!(app.abort.load(Ordering::SeqCst));

    app.metric_tx
        .send(MetricMsg::Finished {
            generation: 1,
            aborted: true,
        })
        .unwrap();
    app.drain_metric_results();
    // Processed result kept; unprocessed settled; button flips back.
    assert!(matches!(
        &app.rows[0].psnr,
        MetricCell::Done { avg, .. } if (*avg - 30.0).abs() < 1e-9
    ));
    assert!(matches!(app.rows[1].psnr, MetricCell::Idle));
    assert!(matches!(app.rows[1].ssim, MetricCell::Idle));
    assert!(!app.measuring);
    assert_eq!(app.pending, 0);
}

#[test]
fn stop_settles_killed_cell_to_idle() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    use crate::metrics::ffmpeg::MetricKind;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
    // a is the killed in-flight job, b never started.
    app.rows[0].psnr = MetricCell::Running {
        frame: 42,
        values: Vec::new(),
    };
    app.rows[1].psnr = MetricCell::Running {
        frame: 0,
        values: Vec::new(),
    };
    app.measuring = true;
    app.pending = 2;
    app.run_generation = 1;

    // The killed job reports back first: quiet settle, not an error.
    app.metric_tx
        .send(MetricMsg::Done {
            generation: 1,
            kind: MetricKind::Psnr,
            key: norm_key("C:/vids/a.mp4"),
            values: Vec::new(),
            avg: None,
            exec_s: 3.5,
            error: Some("aborted".to_owned()),
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        })
        .unwrap();
    app.drain_metric_results();
    assert!(matches!(app.rows[0].psnr, MetricCell::Idle));

    // `Finished` settles the unstarted row; the run ends.
    app.metric_tx
        .send(MetricMsg::Finished {
            generation: 1,
            aborted: true,
        })
        .unwrap();
    app.drain_metric_results();
    assert!(matches!(app.rows[1].psnr, MetricCell::Idle));
    assert!(!app.measuring);
    assert_eq!(app.pending, 0);
}

#[test]
fn clean_finish_leaves_cells_and_clears_measuring() {
    use super::MetricMsg;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.measuring = true;
    app.run_generation = 1;
    app.metric_tx
        .send(MetricMsg::Finished {
            generation: 1,
            aborted: false,
        })
        .unwrap();
    app.drain_metric_results();
    assert!(!app.measuring);
}

#[test]
fn start_psnr_bad_time_marks_all_included() {
    use crate::metrics::MetricCell;
    let p = std::env::temp_dir().join("rfmetrics-psnr-skip.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_ssim: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        skip: "abc".to_owned(), // unparseable: settings uncomparable
        ..RFMetricsApp::default()
    };
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    // Garbage settings can't be compared against the stored trim, so
    // even valid rows take the error (Python writes all targets too).
    for i in 0..2 {
        for cell in [&app.rows[i].psnr, &app.rows[i].ssim] {
            assert!(
                matches!(cell, MetricCell::Error { msg } if msg == "bad time"),
                "row {i} should be bad time, got {cell:?}",
            );
        }
    }
    assert!(!app.measuring);
}

#[test]
fn start_psnr_all_done_toasts_without_running() {
    use crate::metrics::MetricCell;
    let p = std::env::temp_dir().join("rfmetrics-psnr-skipall.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(!app.measuring);
    assert!(app.pending == 0);
    let toast = app.toast.as_ref().expect("skip toast shown");
    assert!(toast.text.contains("Skipped 1 with existing PSNR"));
}

/// Checked "Plot window at start" opens the plot viewport when a run
/// launches; unchecked (default) leaves it closed unless Plot is pressed.
#[test]
fn start_run_plot_at_start_opens_plot() {
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-plot-at-start.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        plot_at_start: true,
        ..RFMetricsApp::default()
    };
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].info = Some(MediaInfo::default());
    assert!(!app.show_plot);
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(app.show_plot);
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
}

#[test]
fn start_run_plot_stays_closed_by_default() {
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-plot-default-closed.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    assert!(!app.plot_at_start);
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(!app.show_plot);
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
}

/// Regression: the jobs loop must iterate `fresh`, never `targets`.
/// A Done row is skipped AND stays Done; only the fresh row runs.
#[test]
fn start_psnr_rerun_leaves_done_row_untouched() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-psnr-rerun.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    // Fabricate everything past the pre-flights so the run reaches the
    // jobs loop; the ffmpeg binary doesn't exist, so the worker fails
    // the spawn asynchronously and the test stays headless-safe.
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].info = Some(MediaInfo::default());
    app.rows[1].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    // Sync state right after Start: Done row untouched, fresh Running.
    assert!(
        matches!(&app.rows[0].psnr, MetricCell::Done { avg, .. } if (*avg - 30.0).abs() < 1e-9),
        "Done row must never re-enter Running, got {:?}",
        app.rows[0].psnr,
    );
    assert!(matches!(
        app.rows[1].psnr,
        MetricCell::Running { frame: 0, .. }
    ));
    assert!(app.measuring);
    // Let the doomed worker land, then settle.
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    assert!(
        matches!(&app.rows[0].psnr, MetricCell::Done { .. }),
        "Done row must survive the whole rerun, got {:?}",
        app.rows[0].psnr,
    );
    assert!(matches!(&app.rows[1].psnr, MetricCell::Error { .. }));
}

/// Guard rail: a Done value stamped with different trim settings is
/// stale — changing Duration/Skip must recompute, never skip.
#[test]
fn start_psnr_changed_trim_recomputes_done_row() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-psnr-staletrim.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        duration: "10".to_owned(), // value was computed with clip 5
        ..RFMetricsApp::default()
    };
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: Some(5.0),
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    // Stale trim: the row re-enters the run instead of skipping.
    assert!(
        matches!(app.rows[0].psnr, MetricCell::Running { .. }),
        "stale-trim Done must recompute, got {:?}",
        app.rows[0].psnr,
    );
    assert!(app.measuring);
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
}

/// Same trim stamp still skips, even with nonzero settings.
#[test]
fn start_psnr_matching_trim_still_skips() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-psnr-sametrim.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        duration: "10".to_owned(),
        ..RFMetricsApp::default()
    };
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: Some(10.0),
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(!app.measuring);
    assert!(
        matches!(&app.rows[0].psnr, MetricCell::Done { .. }),
        "matching-trim Done must skip, got {:?}",
        app.rows[0].psnr,
    );
    let toast = app.toast.as_ref().expect("skip toast shown");
    assert!(toast.text.contains("Skipped 1 with existing PSNR"));
}

/// VMAF options change invalidates VMAF alone: the stale-stamped VMAF
/// cell recomputes while a valid PSNR cell on the same row keeps
/// skipping (no Reset needed).
#[test]
fn start_run_vmaf_settings_change_recomputes_vmaf_only() {
    use crate::metrics::MetricCell;
    use crate::metrics::vmaf::{Pooling, VmafCfg};
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-vmaf-restamp.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: true,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    // Current UI settings snapshot to subsample 1; the stored VMAF
    // value was computed under subsample 5.
    app.vmaf_subsample = "1".to_owned();
    app.vmaf_threads = "4".to_owned();
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].vmaf = MetricCell::Done {
        values: vec![90.0],
        avg: 90.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: Some(VmafCfg {
            model: "vmaf_v0.6.1.json".to_owned(),
            phone: false,
            scale: false,
            pooling: Pooling::Mean,
            subsample: 5,
            n_threads: 4,
        }),
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(
        matches!(&app.rows[0].psnr, MetricCell::Done { .. }),
        "valid PSNR must keep skipping, got {:?}",
        app.rows[0].psnr,
    );
    assert!(
        matches!(&app.rows[0].vmaf, MetricCell::Running { .. }),
        "stale-stamped VMAF must recompute, got {:?}",
        app.rows[0].vmaf,
    );
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    // Spawn fails headless (bogus binary): the cell records the error
    // while PSNR still holds its skipped value.
    assert!(matches!(&app.rows[0].vmaf, MetricCell::Error { .. }));
    assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
}

/// Matching VMAF stamp still skips: unchanged options recompute nothing.
#[test]
fn start_run_vmaf_matching_settings_still_skips() {
    use crate::metrics::MetricCell;
    use crate::metrics::vmaf::{Pooling, VmafCfg};
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-vmaf-samestamp.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: true,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.vmaf_subsample = "1".to_owned();
    app.vmaf_threads = "4".to_owned();
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].vmaf = MetricCell::Done {
        values: vec![90.0],
        avg: 90.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: Some(VmafCfg {
            model: "vmaf_v0.6.1.json".to_owned(),
            phone: false,
            scale: false,
            pooling: Pooling::Mean,
            subsample: 1,
            n_threads: 4,
        }),
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(!app.measuring);
    assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
    assert!(matches!(&app.rows[0].vmaf, MetricCell::Done { .. }));
    let toast = app.toast.as_ref().expect("skip toast shown");
    assert_eq!(
        toast.text,
        "Skipped 1 with existing PSNR, VMAF (Reset to recompute)"
    );
}

/// Scaler change recomputes ffmpeg-backed columns but leaves FFVship
/// ones alone (no scale stage there): a Bicubic-stamped PSNR cell
/// recomputes under Lanczos while a Bicubic-stamped SSIM2 cell keeps
/// skipping, no Reset needed.
#[test]
fn start_run_scaler_change_recomputes_ffmpeg_only() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-scaler-restamp.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        m_ssim2: true,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.scale_method = ScaleMethod::Lanczos;
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    let done = |avg: f64| MetricCell::Done {
        values: vec![avg],
        avg,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].psnr = done(30.0);
    app.rows[0].ssim2 = done(80.0);
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(
        matches!(&app.rows[0].psnr, MetricCell::Running { .. }),
        "stale-stamped PSNR must recompute, got {:?}",
        app.rows[0].psnr,
    );
    assert!(
        matches!(&app.rows[0].ssim2, MetricCell::Done { .. }),
        "FFVship SSIM2 ignores the scaler, got {:?}",
        app.rows[0].ssim2,
    );
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    // Spawn fails headless (bogus binary): the cell records the error
    // while SSIM2 still holds its skipped value.
    assert!(matches!(&app.rows[0].psnr, MetricCell::Error { .. }));
    assert!(matches!(&app.rows[0].ssim2, MetricCell::Done { .. }));
}

/// Fps-mode change recomputes ffmpeg-backed columns but leaves FFVship
/// ones alone (no `-r` stage there): a Reference-stamped PSNR cell
/// recomputes under Off while a Reference-stamped SSIM2 cell keeps
/// skipping, no Reset needed.
#[test]
fn start_run_fps_mode_change_recomputes_ffmpeg_only() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-fpsmode-restamp.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        m_ssim2: true,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.fps_mode = InputFpsMode::Off;
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    let done = |avg: f64| MetricCell::Done {
        values: vec![avg],
        avg,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].psnr = done(30.0);
    app.rows[0].ssim2 = done(80.0);
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(
        matches!(&app.rows[0].psnr, MetricCell::Running { .. }),
        "stale-stamped PSNR must recompute, got {:?}",
        app.rows[0].psnr,
    );
    assert!(
        matches!(&app.rows[0].ssim2, MetricCell::Done { .. }),
        "FFVship SSIM2 ignores the fps mode, got {:?}",
        app.rows[0].ssim2,
    );
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    // Spawn fails headless (bogus binary): the cell records the error
    // while SSIM2 still holds its skipped value.
    assert!(matches!(&app.rows[0].psnr, MetricCell::Error { .. }));
    assert!(matches!(&app.rows[0].ssim2, MetricCell::Done { .. }));
}

/// Matching scaler stamp still skips: unchanged method recomputes nothing.
#[test]
fn start_run_matching_scaler_still_skips() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-scaler-samestamp.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].info = Some(MediaInfo::default());
    app.ref_info_data = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(!app.measuring);
    assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
    let toast = app.toast.as_ref().expect("skip toast shown");
    assert!(toast.text.contains("Skipped 1 with existing PSNR"));
}

/// All-metrics skip lists every metric: the single toast slot must name
/// PSNR, SSIM, and VMAF instead of collapsing to the last one (VMAF).
#[test]
fn start_run_skip_toast_lists_all_metrics() {
    use crate::metrics::MetricCell;
    use crate::metrics::vmaf::{Pooling, VmafCfg};
    let p = std::env::temp_dir().join("rfmetrics-skipall-metrics.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_ssim: true,
        m_vmaf: true,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    // Pin threads: the "auto" default resolves machine-dependently.
    app.vmaf_threads = "4".to_owned();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    let done = |avg: f64| MetricCell::Done {
        values: vec![avg],
        avg,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].psnr = done(30.0);
    app.rows[0].ssim = done(0.9);
    app.rows[0].vmaf = MetricCell::Done {
        values: vec![90.0],
        avg: 90.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: Some(VmafCfg {
            model: "vmaf_v0.6.1.json".to_owned(),
            phone: false,
            scale: false,
            pooling: Pooling::Mean,
            subsample: 1,
            n_threads: 4,
        }),
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(!app.measuring);
    // Shared skip set merges into one line instead of repeating it
    // per metric (previously only the VMAF line survived).
    let toast = app.toast.as_ref().expect("skip toast shown");
    assert_eq!(
        toast.text,
        "Skipped 1 with existing PSNR, SSIM, VMAF (Reset to recompute)"
    );
}

/// VMAF recompute alongside skipped metrics lists shared filenames
/// once: 2 rows skip PSNR+SSIM while VMAF alone runs on both.
#[test]
fn start_run_skip_toast_merges_shared_rows() {
    use crate::metrics::MetricCell;
    use crate::metrics::vmaf::{Pooling, VmafCfg};
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-skip-merge.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_ssim: true,
        m_vmaf: true,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.vmaf_subsample = "1".to_owned();
    app.vmaf_threads = "4".to_owned();
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
    app.rows[1].display = "b.mp4".to_owned();
    let stale_vmaf = || MetricCell::Done {
        values: vec![90.0],
        avg: 90.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: Some(VmafCfg {
            model: "vmaf_v0.6.1.json".to_owned(),
            phone: false,
            scale: false,
            pooling: Pooling::Mean,
            subsample: 5,
            n_threads: 4,
        }),
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    for i in 0..2 {
        app.rows[i].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        };
        app.rows[i].ssim = MetricCell::Done {
            values: vec![0.9],
            avg: 0.9,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        };
        app.rows[i].vmaf = stale_vmaf();
        app.rows[i].info = Some(MediaInfo::default());
    }
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    // One line, filenames once — not repeated per metric.
    let toast = app.toast.as_ref().expect("skip toast shown");
    assert_eq!(
        toast.text,
        "Skipped 2 with existing PSNR, SSIM: a.mp4, b.mp4"
    );
    assert!(matches!(&app.rows[0].vmaf, MetricCell::Running { .. }));
    assert!(matches!(&app.rows[1].vmaf, MetricCell::Running { .. }));
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
    assert!(matches!(&app.rows[1].ssim, MetricCell::Done { .. }));
}

/// SSIM-only run: only the SSIM cell enters the run, PSNR stays Idle.
#[test]
fn start_run_ssim_only_runs_ssim_cell() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-ssim-only.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_ssim: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    // Fabricate past the pre-flights; the binary doesn't exist so the
    // worker fails the spawn asynchronously (headless-safe).
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
    assert!(matches!(
        app.rows[0].ssim,
        MetricCell::Running { frame: 0, .. }
    ));
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
    assert!(matches!(&app.rows[0].ssim, MetricCell::Error { .. }));
}

/// Mixed run: valid PSNR skips while fresh SSIM on the same row runs.
#[test]
fn start_run_skips_done_psnr_but_runs_fresh_ssim() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-ssim-mixed.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_ssim: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0],
        avg: 30.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(
        matches!(&app.rows[0].psnr, MetricCell::Done { .. }),
        "valid PSNR must skip, got {:?}",
        app.rows[0].psnr,
    );
    assert!(matches!(
        app.rows[0].ssim,
        MetricCell::Running { frame: 0, .. }
    ));
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
}

/// XPSNR-only run: only the XPSNR cell enters the run, the rest stay Idle.
#[test]
fn start_run_xpsnr_only_runs_xpsnr_cell() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-xpsnr-only.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_xpsnr: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    // Fabricate past the pre-flights; the binary doesn't exist so the
    // worker fails the spawn asynchronously (headless-safe).
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
    assert!(matches!(app.rows[0].ssim, MetricCell::Idle));
    assert!(matches!(
        app.rows[0].xpsnr,
        MetricCell::Running { frame: 0, .. }
    ));
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
    assert!(matches!(&app.rows[0].xpsnr, MetricCell::Error { .. }));
}

/// Mixed run: valid XPSNR skips while fresh PSNR on the same row runs.
#[test]
fn start_run_skips_done_xpsnr_but_runs_fresh_psnr() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-xpsnr-mixed.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_xpsnr: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].xpsnr = MetricCell::Done {
        values: vec![40.0],
        avg: 40.0,
        exec_s: 1.0,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(matches!(
        app.rows[0].psnr,
        MetricCell::Running { frame: 0, .. }
    ));
    assert!(
        matches!(&app.rows[0].xpsnr, MetricCell::Done { .. }),
        "valid XPSNR must skip, got {:?}",
        app.rows[0].xpsnr,
    );
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    assert!(matches!(&app.rows[0].xpsnr, MetricCell::Done { .. }));
}

/// SSIM2-only run: only the SSIM2 cell enters the run, the rest stay
/// Idle. The binary doesn't exist so the worker fails the spawn
/// asynchronously (headless-safe).
#[test]
fn start_run_ssim2_only_runs_ssim2_cell() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-ssim2-only.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_ssim2: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.ffvship.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
    app.ffvship.usable = true;
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(app.measuring);
    assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
    assert!(matches!(
        app.rows[0].ssim2,
        MetricCell::Running { frame: 0, .. }
    ));
    for _ in 0..200 {
        app.drain_metric_results();
        if !app.measuring {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(!app.measuring);
    assert!(matches!(&app.rows[0].ssim2, MetricCell::Error { .. }));
}

/// Missing FFVship binary: FFVship cells error out without running,
/// while an ffmpeg-only selection is unaffected by the missing binary.
#[test]
fn start_run_ffvship_missing_binary_errors_cells() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-ffvship-missing.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_ssim2: true,
        m_but: true,
        m_cvvdp: true,
        m_vmaf: false,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.ffvship.path = None;
    app.ffvship.usable = false;
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].info = Some(MediaInfo::default());
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(!app.measuring);
    for cell in [&app.rows[0].ssim2, &app.rows[0].butter, &app.rows[0].cvvdp] {
        assert!(
            matches!(cell, MetricCell::Error { msg } if msg == "FFVship not found"),
            "expected FFVship gate, got {cell:?}"
        );
    }
    assert!(app.toast.is_some());
}

/// BUTTER ranks min-wins on every stat (lower is better); other
/// metrics keep max-wins.
#[test]
fn butter_rank_is_min_wins() {
    use crate::metrics::{MetricCell, StatRank};
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
    for (i, avg) in [3.0, 5.0].into_iter().enumerate() {
        app.rows[i].butter = MetricCell::Done {
            values: vec![avg, avg],
            avg,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        };
        let stats = app.rows[i].butter.done_stats();
        app.rows[i].butter_cache.stats = stats;
    }
    app.refresh_ranks(crate::metrics::ffmpeg::MetricKind::But);
    assert_eq!(app.rows[0].butter_cache.ranks[0], StatRank::Best);
    assert_eq!(app.rows[1].butter_cache.ranks[0], StatRank::Worst);
}

/// State round-trip: snapshot captures boxes, toggles, options, and
/// per-row include flags; apply restores them onto a fresh app.
#[test]
fn state_snapshot_apply_round_trip() {
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-state-rt.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        ref_path: "C:/vids/ref.mp4".to_owned(),
        skip: "5".to_owned(),
        duration: "00:10".to_owned(),
        m_psnr: true,
        m_vmaf: false,
        m_ssim2: true,
        vmaf_phone: true,
        vmaf_pooling: "Harmonic Mean".to_owned(),
        vmaf_threads: "4".to_owned(),
        scale_method: ScaleMethod::Lanczos,
        fps_mode: InputFpsMode::Off,
        plot_at_start: true,
        plot_size: crate::plot::PlotSize::S1600,
        ..RFMetricsApp::default()
    };
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows.push(psnr_test_row("C:/vids/b.mp4", false));
    app.rows[0].info = Some(MediaInfo::default());
    let snap = app.snapshot();
    std::fs::remove_file(&p).ok();

    let mut fresh = RFMetricsApp::default();
    fresh.apply_state(Some(snap));
    assert_eq!(fresh.ref_path, "C:/vids/ref.mp4");
    assert_eq!(fresh.skip, "5");
    assert_eq!(fresh.duration, "00:10");
    assert!(fresh.m_psnr && !fresh.m_vmaf && fresh.m_ssim2);
    assert!(fresh.vmaf_phone);
    assert_eq!(fresh.vmaf_pooling, "Harmonic Mean");
    assert_eq!(fresh.vmaf_threads, "4");
    assert_eq!(fresh.scale_method, ScaleMethod::Lanczos);
    assert_eq!(fresh.fps_mode, InputFpsMode::Off);
    assert!(fresh.plot_at_start);
    assert_eq!(fresh.plot_size, crate::plot::PlotSize::S1600);
    // psnr_test_row paths don't exist on disk: only pre-existing rows
    // could restore, so the queue stays empty here.
    assert!(fresh.rows.is_empty());
}

/// State apply restores live queue files with their include flags and
/// drops missing ones.
#[test]
fn state_apply_restores_files_with_include() {
    let a = std::env::temp_dir().join("rfmetrics-state-a.tmp");
    let b = std::env::temp_dir().join("rfmetrics-state-b.tmp");
    std::fs::write(&a, b"x").unwrap();
    std::fs::write(&b, b"x").unwrap();
    let state = crate::state::AppState {
        files: Some(vec![
            crate::state::FileEntry {
                path: a.to_string_lossy().into_owned(),
                include: true,
            },
            crate::state::FileEntry {
                path: b.to_string_lossy().into_owned(),
                include: false,
            },
            crate::state::FileEntry {
                path: "C:/no/such/file.mp4".to_owned(),
                include: true,
            },
        ]),
        ..Default::default()
    };
    let mut app = RFMetricsApp::default();
    app.apply_state(Some(state));
    std::fs::remove_file(&a).ok();
    std::fs::remove_file(&b).ok();
    assert_eq!(app.rows.len(), 2);
    assert!(app.rows[0].include);
    assert!(!app.rows[1].include);
    // Absent keys keep live defaults (VMAF-only).
    assert!(!app.m_psnr && app.m_vmaf);
}

/// State apply validates options: unknown models/pooling/subsamples
/// keep live values instead of poisoning the run.
#[test]
fn state_apply_validates_vmaf_options() {
    let mut app = RFMetricsApp::default();
    let state = crate::state::AppState {
        vmaf: crate::state::VmafState {
            model: Some("evil.json".to_owned()),
            pooling: Some("Median".to_owned()),
            subsample: Some("7".to_owned()),
            phone: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    app.apply_state(Some(state));
    assert_eq!(app.vmaf_model, "vmaf_v0.6.1.json");
    assert_eq!(app.vmaf_pooling, "Mean");
    assert_eq!(app.vmaf_subsample, "1");
    assert!(app.vmaf_phone);
}

/// Issue #7: restored ticks for filters this ffmpeg build lacks are
/// forced off (their header checkboxes render disabled).
#[test]
fn state_apply_unticks_unsupported_filters() {
    use crate::metrics::ffmpeg::MetricKind;
    let mut app = RFMetricsApp::default();
    // Simulate a w32threads-style build: everything but libvmaf.
    app.ffmpeg.supported_metrics = vec![MetricKind::Psnr, MetricKind::Ssim, MetricKind::Xpsnr];
    let state = crate::state::AppState {
        metrics: crate::state::MetricsState {
            psnr: Some(true),
            ssim: Some(true),
            vmaf: Some(true),
            xpsnr: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    app.apply_state(Some(state));
    assert!(app.m_psnr && app.m_ssim && app.m_xpsnr);
    assert!(!app.m_vmaf);
}

/// FFVship parity with Issue #7: restored ticks for the FFVship family are
/// forced off when the binary is absent/unusable (their header checkboxes
/// render disabled), while usable binaries keep restored ticks.
#[test]
fn state_apply_unticks_ffvship_when_unusable() {
    use crate::metrics::ffmpeg::MetricKind;
    let state = || crate::state::AppState {
        metrics: crate::state::MetricsState {
            psnr: Some(true),
            ssim2: Some(true),
            butteraugli: Some(true),
            cvvdp: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    // Unusable binary: family ticks cleared, ffmpeg ticks kept.
    let mut app = RFMetricsApp::default();
    app.ffmpeg.supported_metrics = vec![
        MetricKind::Psnr,
        MetricKind::Ssim,
        MetricKind::Vmaf,
        MetricKind::Xpsnr,
    ];
    app.ffvship.usable = false;
    app.apply_state(Some(state()));
    assert!(app.m_psnr);
    assert!(!app.m_ssim2 && !app.m_but && !app.m_cvvdp);
    // Usable binary: restored family ticks survive.
    let mut app = RFMetricsApp::default();
    app.ffmpeg.supported_metrics = vec![
        MetricKind::Psnr,
        MetricKind::Ssim,
        MetricKind::Vmaf,
        MetricKind::Xpsnr,
    ];
    app.ffvship.usable = true;
    app.apply_state(Some(state()));
    assert!(app.m_psnr);
    assert!(app.m_ssim2 && app.m_but && app.m_cvvdp);
}

/// Issue #7 backstop: a ticked-but-unsupported metric never reaches
/// the worker — with nothing runnable Start is a no-op toast.
#[test]
fn start_run_unsupported_vmaf_is_noop() {
    use crate::metrics::ffmpeg::MetricKind;
    use crate::probe::MediaInfo;
    let p = std::env::temp_dir().join("rfmetrics-unsupported-vmaf.tmp");
    std::fs::write(&p, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: false,
        m_ssim: false,
        m_vmaf: true,
        ref_path: p.to_string_lossy().into_owned(),
        ..RFMetricsApp::default()
    };
    app.ffmpeg.supported_metrics = vec![MetricKind::Psnr, MetricKind::Ssim, MetricKind::Xpsnr];
    app.ref_info_data = Some(MediaInfo::default());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.start_run(0.0);
    std::fs::remove_file(&p).ok();
    assert!(!app.measuring);
    assert!(matches!(app.rows[0].vmaf, crate::metrics::MetricCell::Idle));
}

/// CsvReport stashes the summary; the UI frame toasts it (drain has
/// no timestamp). Stale generations stay silent.
#[test]
fn csv_report_stash_and_generation() {
    use super::MetricMsg;
    let mut app = RFMetricsApp::default();
    app.metric_tx
        .send(MetricMsg::CsvReport {
            generation: app.run_generation,
            ok: 3,
            errors: Vec::new(),
        })
        .unwrap();
    assert!(app.drain_metric_results());
    assert_eq!(
        app.csv_report,
        Some((3, Vec::new())),
        "fresh report must stash"
    );
    // Stale report: dropped without touching the stash.
    app.run_generation = app.run_generation.wrapping_add(1);
    app.metric_tx
        .send(MetricMsg::CsvReport {
            generation: app.run_generation.wrapping_sub(1),
            ok: 9,
            errors: vec!["x".to_owned()],
        })
        .unwrap();
    assert!(app.drain_metric_results());
    assert_eq!(app.csv_report, Some((3, Vec::new())));
}

/// CSV options persist and restore like the other Options keys.
#[test]
fn state_apply_restores_csv_options() {
    let mut app = RFMetricsApp::default();
    let state = crate::state::AppState {
        options: crate::state::OptionsState {
            csv_export: Some(true),
            csv_dir: Some("D:/csv".to_owned()),
            ..Default::default()
        },
        ..Default::default()
    };
    app.apply_state(Some(state));
    assert!(app.csv_export);
    assert_eq!(app.csv_dir, "D:/csv");
    assert!(app.snapshot().options.csv_export == Some(true));
}

/// Bad-frames export dir persists and restores like the CSV dir.
#[test]
fn state_apply_restores_badframes_export_dir() {
    let mut app = RFMetricsApp::default();
    let state = crate::state::AppState {
        options: crate::state::OptionsState {
            badframes_export_dir: Some("D:/bf".to_owned()),
            ..Default::default()
        },
        ..Default::default()
    };
    app.apply_state(Some(state));
    assert_eq!(app.badframes_export_dir, "D:/bf");
    assert!(app.snapshot().options.badframes_export_dir == Some("D:/bf".to_owned()));
}

/// Results auto-save: Finished (clean or aborted) arms the flag when
/// opted in; consuming exports one row and disarms. Stale generations
/// stay silent.
#[test]
fn results_autosave_end_to_end() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    let dir = std::env::temp_dir().join(format!("rfmetrics-autosave-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut app = RFMetricsApp {
        results_autosave: true,
        results_path: dir
            .join("RFMetrics.Results.csv")
            .to_string_lossy()
            .into_owned(),
        ..RFMetricsApp::default()
    };
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].psnr = MetricCell::Done {
        values: vec![30.0, 31.0],
        avg: 30.5,
        exec_s: 1.0,
        skip: None,
        clip_dur: Some(5.0),
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    for aborted in [false, true] {
        app.metric_tx
            .send(MetricMsg::Finished {
                generation: app.run_generation,
                aborted,
            })
            .unwrap();
        app.drain_metric_results();
        assert!(
            app.results_autosave_pending,
            "Finished (aborted={aborted}) must arm with the option on"
        );
        app.consume_autosave(0.0);
        assert!(!app.results_autosave_pending);
    }
    let text = std::fs::read_to_string(dir.join("RFMetrics.Results.csv")).unwrap();
    let lines: Vec<&str> = text.split("\r\n").collect();
    // Header + 2 appended rows + trailing empty after final CRLF.
    assert!(lines[0].starts_with("DateTime\tPSNR-DateTime"));
    assert_eq!(
        lines.iter().filter(|l| l.contains("C:/vids/a.mp4")).count(),
        2
    );
    assert!(lines[1].contains("\t30.5\t"));
    assert_eq!(lines[3], "");
    // Opted out: Finished arms nothing.
    app.results_autosave = false;
    app.metric_tx
        .send(MetricMsg::Finished {
            generation: app.run_generation,
            aborted: false,
        })
        .unwrap();
    app.drain_metric_results();
    assert!(!app.results_autosave_pending);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Results options persist and restore like the other Options keys.
#[test]
fn state_apply_restores_results_options() {
    let mut app = RFMetricsApp::default();
    let state = crate::state::AppState {
        options: crate::state::OptionsState {
            results_autosave: Some(true),
            results_path: Some("D:/r.csv".to_owned()),
            ..Default::default()
        },
        ..Default::default()
    };
    app.apply_state(Some(state));
    assert!(app.results_autosave);
    assert_eq!(app.results_path, "D:/r.csv");
    assert!(app.snapshot().options.results_autosave == Some(true));
}

/// Wall stamp shape only (`%Y-%m-%d %H:%M:%S`): the instant is
/// host-clock dependent, the layout is not.
#[test]
fn wall_stamp_shape() {
    let s = super::wall_now_string();
    let b = s.as_bytes();
    assert_eq!(s.len(), 19);
    assert_eq!([b[4], b[7]], [b'-', b'-']);
    assert_eq!(b[10], b' ');
    assert_eq!([b[13], b[16]], [b':', b':']);
    assert!(s[..10].chars().all(|c| c.is_ascii_digit() || c == '-'));
}

/// `is_state_dirty` must agree with `snapshot() != saved_snapshot` for
/// every persisted field; a missed field silently stops saving.
#[test]
fn dirty_check_matches_snapshot_compare() {
    let mut app = RFMetricsApp::default();
    app.saved_snapshot = app.snapshot();
    assert!(!app.is_state_dirty());

    // One mutation at a time: each must read dirty under both paths,
    // and re-saving must read clean.
    let check = |app: &mut RFMetricsApp| {
        assert!(app.is_state_dirty());
        assert_ne!(app.snapshot(), app.saved_snapshot);
        app.saved_snapshot = app.snapshot();
        assert!(!app.is_state_dirty());
    };
    app.ref_path = "C:/vids/ref.mp4".to_owned();
    check(&mut app);
    app.skip = "5".to_owned();
    check(&mut app);
    app.duration = "00:10".to_owned();
    check(&mut app);
    app.m_psnr = !app.m_psnr;
    check(&mut app);
    app.m_ssim = !app.m_ssim;
    check(&mut app);
    app.m_vmaf = !app.m_vmaf;
    check(&mut app);
    app.m_xpsnr = !app.m_xpsnr;
    check(&mut app);
    app.m_ssim2 = !app.m_ssim2;
    check(&mut app);
    app.m_but = !app.m_but;
    check(&mut app);
    app.m_cvvdp = !app.m_cvvdp;
    check(&mut app);
    app.vmaf_model = "other.json".to_owned();
    check(&mut app);
    app.vmaf_phone = !app.vmaf_phone;
    check(&mut app);
    app.vmaf_scale = !app.vmaf_scale;
    check(&mut app);
    app.vmaf_pooling = "Harmonic Mean".to_owned();
    check(&mut app);
    app.vmaf_subsample = "2".to_owned();
    check(&mut app);
    app.vmaf_threads = "4".to_owned();
    check(&mut app);
    app.scale_method = ScaleMethod::Lanczos;
    check(&mut app);
    app.plot_at_start = !app.plot_at_start;
    check(&mut app);
    app.plot_size = crate::plot::PlotSize::S1600;
    check(&mut app);
    app.csv_export = !app.csv_export;
    check(&mut app);
    app.csv_dir = "D:/csv".to_owned();
    check(&mut app);
    app.results_autosave = !app.results_autosave;
    check(&mut app);
    app.results_path = "D:/r.csv".to_owned();
    check(&mut app);
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    check(&mut app);
    app.rows[0].include = false;
    check(&mut app);
    app.rows[0].path = "C:/vids/b.mp4".to_owned();
    app.rows[0].key = norm_key("C:/vids/b.mp4");
    check(&mut app);
    // Order-sensitive like the snapshot vec.
    app.rows.push(psnr_test_row("C:/vids/c.mp4", true));
    check(&mut app);
    app.rows.swap(0, 1);
    assert!(app.is_state_dirty());
    assert_ne!(app.snapshot(), app.saved_snapshot);
}

/// Before/after timing: full `snapshot()` clone+compare (old per-frame
/// path) vs `is_state_dirty` (new path) over a 200-row queue.
#[test]
fn dirty_check_perf() {
    let mut app = RFMetricsApp::default();
    for i in 0..200 {
        app.rows.push(psnr_test_row(
            &format!("C:/vids/clip_{i:04}.mp4"),
            i % 2 == 0,
        ));
    }
    app.saved_snapshot = app.snapshot();
    assert!(!app.is_state_dirty());

    let n = 2000;
    let t0 = std::time::Instant::now();
    let mut dirty_old = false;
    for _ in 0..n {
        dirty_old |= app.snapshot() != app.saved_snapshot;
    }
    let old_ms = t0.elapsed();
    let t1 = std::time::Instant::now();
    let mut dirty_new = false;
    for _ in 0..n {
        dirty_new |= app.is_state_dirty();
    }
    let new_ms = t1.elapsed();
    assert_eq!(dirty_old, dirty_new);
    eprintln!(
        "autosave dirty-check ({n} iters, 200 rows): snapshot-clone {old_ms:?} vs field-compare {new_ms:?}"
    );
}

/// Before/after timing for worker-message routing: re-normalizing every
/// row per lookup (old `Progress`-rate path) vs stored-key compare.
#[test]
fn row_routing_perf() {
    let mut app = RFMetricsApp::default();
    for i in 0..200 {
        app.rows.push(psnr_test_row(
            &format!("C:/vids/clip_{i:04}.mp4"),
            i % 2 == 0,
        ));
    }
    let key = app.rows[100].key.clone();
    let n = 5000;
    let t0 = std::time::Instant::now();
    let mut found_old = 0;
    for _ in 0..n {
        if let Some(r) = app.rows.iter().find(|r| norm_key(&r.path) == key) {
            found_old += r.path.len();
        }
    }
    let old = t0.elapsed();
    let t1 = std::time::Instant::now();
    let mut found_new = 0;
    for _ in 0..n {
        if let Some(r) = app.rows.iter().find(|r| r.key == key) {
            found_new += r.path.len();
        }
    }
    let new = t1.elapsed();
    assert_eq!(found_old, found_new);
    eprintln!("row routing ({n} lookups, 200 rows): norm_key-scan {old:?} vs stored-key {new:?}");
}

/// Done/Error arrival freezes the rendered cell text (equal to
/// `cell_text()` by construction); Reset clears it.
#[test]
fn cell_text_cache_set_and_cleared() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    use crate::metrics::ffmpeg::MetricKind;
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.run_generation = 1;
    let done = |error: Option<String>| MetricMsg::Done {
        generation: 1,
        kind: MetricKind::Psnr,
        key: norm_key("C:/vids/a.mp4"),
        values: vec![29.0, 30.0, 31.0],
        avg: Some(30.123_456),
        exec_s: 1.0,
        error,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    app.metric_tx.send(done(None)).unwrap();
    app.drain_metric_results();
    assert_eq!(app.rows[0].psnr_cache.text, "30.1235");
    assert_eq!(app.rows[0].psnr_cache.text, app.rows[0].psnr.cell_text());
    app.metric_tx.send(done(Some("boom".to_owned()))).unwrap();
    app.drain_metric_results();
    assert!(matches!(app.rows[0].psnr, MetricCell::Error { .. }));
    assert_eq!(app.rows[0].psnr_cache.text, "boom");
    app.reset_psnr();
    assert!(app.rows[0].psnr_cache.text.is_empty());
}

/// No-live-feed first Done (VMAF) on the shown tab owes one auto-fit
/// poke; a later sibling row must not disturb the first fit.
#[test]
fn vmaf_first_done_arms_follow() {
    use super::MetricMsg;
    use crate::metrics::MetricCell;
    use crate::metrics::ffmpeg::MetricKind;
    let done = |key: &str| MetricMsg::Done {
        generation: 0,
        kind: MetricKind::Vmaf,
        key: norm_key(key),
        values: vec![90.0, 91.0],
        avg: Some(90.5),
        exec_s: 1.0,
        error: None,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    let mut app = RFMetricsApp {
        show_plot: true,
        plot_tab: MetricKind::Vmaf,
        ..RFMetricsApp::default()
    };
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].vmaf = MetricCell::Running {
        frame: 0,
        values: Vec::new(),
    };
    app.metric_tx.send(done("C:/vids/a.mp4")).unwrap();
    app.drain_metric_results();
    assert!(
        app.plot_follow_pending,
        "first VMAF Done on the shown tab must arm a refit"
    );
    app.plot_follow_pending = false;
    app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
    app.metric_tx.send(done("C:/vids/b.mp4")).unwrap();
    app.drain_metric_results();
    assert!(
        !app.plot_follow_pending,
        "later rows must not disturb the first fit"
    );
}

/// Arming rules: live-feed metrics, hidden tabs, closed window, and
/// errored runs never arm the poke.
#[test]
fn follow_arm_rules() {
    use super::MetricMsg;
    use crate::metrics::ffmpeg::MetricKind;
    let done = |kind, key: &str, error: Option<String>| MetricMsg::Done {
        generation: 0,
        kind,
        key: norm_key(key),
        values: vec![30.0],
        avg: Some(30.0),
        exec_s: 1.0,
        error,
        skip: None,
        clip_dur: None,
        vmaf_cfg: None,
        scaler: ScaleMethod::Bicubic,
        fps_mode: InputFpsMode::Reference,
    };
    // Live-feed metric never arms, even first on the shown tab.
    let mut app = RFMetricsApp {
        show_plot: true,
        plot_tab: MetricKind::Psnr,
        ..RFMetricsApp::default()
    };
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.metric_tx
        .send(done(MetricKind::Psnr, "C:/vids/a.mp4", None))
        .unwrap();
    app.drain_metric_results();
    assert!(!app.plot_follow_pending);
    // First VMAF data on a hidden tab never arms.
    app.metric_tx
        .send(done(MetricKind::Vmaf, "C:/vids/a.mp4", None))
        .unwrap();
    app.drain_metric_results();
    assert!(!app.plot_follow_pending);
    // Closed window never arms.
    app.show_plot = false;
    app.plot_tab = MetricKind::Vmaf;
    app.metric_tx
        .send(done(MetricKind::Vmaf, "C:/vids/a.mp4", None))
        .unwrap();
    app.drain_metric_results();
    assert!(!app.plot_follow_pending);
    // Errored VMAF never arms.
    app.show_plot = true;
    app.metric_tx
        .send(done(
            MetricKind::Vmaf,
            "C:/vids/a.mp4",
            Some("boom".to_owned()),
        ))
        .unwrap();
    app.drain_metric_results();
    assert!(!app.plot_follow_pending);
}

/// Rerun start clears the frozen text synchronously (asserted before the
/// doomed worker's error reply is ever drained, so no race).
#[test]
fn cell_text_cache_cleared_on_rerun() {
    use crate::metrics::MetricCell;
    use crate::probe::MediaInfo;
    let dir = std::env::temp_dir();
    let ref_file = dir.join("rfmetrics-rerun-ref.tmp");
    let fake_exe = dir.join("rfmetrics-rerun-exe.tmp");
    std::fs::write(&ref_file, b"x").unwrap();
    std::fs::write(&fake_exe, b"x").unwrap();
    let mut app = RFMetricsApp {
        m_psnr: true,
        m_vmaf: false,
        ref_path: ref_file.to_string_lossy().into_owned(),
        ref_info_data: Some(MediaInfo::default()),
        ..RFMetricsApp::default()
    };
    app.ffmpeg.path = Some(fake_exe.clone());
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.rows[0].info = Some(MediaInfo::default());
    app.rows[0].psnr_cache.text = "30.1235".to_owned();
    app.start_run(0.0);
    assert!(matches!(app.rows[0].psnr, MetricCell::Running { .. }));
    assert!(app.rows[0].psnr_cache.text.is_empty());
    assert!(app.rows[0].psnr_cache.stats.is_none());
    app.reset_psnr();
    std::fs::remove_file(&ref_file).ok();
    std::fs::remove_file(&fake_exe).ok();
}

/// Before/after timing for the table cell string work: per-frame
/// `cell_text()` + `tooltip()` (old) vs borrowing statics (new).
/// Models 200 rows × 7 idle columns.
#[test]
fn cell_text_perf() {
    use crate::metrics::{MetricCell, ffmpeg::MetricKind};
    use std::hint::black_box;
    let kinds = MetricKind::ALL;
    let cells: Vec<MetricCell> = (0..200 * kinds.len()).map(|_| MetricCell::Idle).collect();
    // Borrowed text must read identically to the formatted one.
    assert_eq!(cells[0].cell_text(), "N/A");
    let n = 500;
    let t0 = std::time::Instant::now();
    let mut len_old = 0;
    for _ in 0..n {
        for (cell, kind) in cells.iter().zip(kinds.iter().cycle()) {
            let text = cell.cell_text();
            let tip = cell.tooltip(kind.name());
            len_old += text.len() + tip.len();
        }
    }
    let old = t0.elapsed();
    black_box(len_old);
    let t1 = std::time::Instant::now();
    let mut len_new = 0;
    for _ in 0..n {
        for cell in &cells {
            let text: &str = match cell {
                MetricCell::Idle => "N/A",
                _ => unreachable!("idle-only fixture"),
            };
            len_new += text.len();
        }
    }
    let new = t1.elapsed();
    black_box(len_new);
    eprintln!(
        "cell text ({n} frames, 1400 idle cells): format-per-frame {old:?} vs borrow {new:?}"
    );
}

/// Idle-heartbeat budget: our-code CPU of one heartbeat frame (dirty
/// check + borrowed cell text for 200×7 idle cells + fit/decimate of
/// 5×5000 plot series) ×2/s, vs zero frames without the heartbeat.
/// egui layout/tessellation is not included (headless can't render).
#[test]
fn idle_heartbeat_budget() {
    use crate::metrics::{MetricCell, ffmpeg::MetricKind};
    use std::hint::black_box;
    let mut app = RFMetricsApp::default();
    for i in 0..200 {
        app.rows.push(psnr_test_row(
            &format!("C:/vids/clip_{i:04}.mp4"),
            i % 2 == 0,
        ));
    }
    app.saved_snapshot = app.snapshot();
    let raw: Vec<Vec<f64>> = (0..5)
        .map(|s| {
            (0..5000)
                .map(|i| 30.0 + s as f64 + (i as f64 * 0.01).sin() * 5.0)
                .collect()
        })
        .collect();
    let borrowed: Vec<&[f64]> = raw.iter().map(Vec::as_slice).collect();
    let n = 200;
    let t0 = std::time::Instant::now();
    let mut acc = 0usize;
    for _ in 0..n {
        acc += app.is_state_dirty() as usize;
        for row in &app.rows {
            for kind in MetricKind::ALL {
                let text: &str = match row.cell(kind) {
                    MetricCell::Idle => "N/A",
                    MetricCell::Running { .. } => unreachable!("idle fixture"),
                    MetricCell::Done { .. } => &row.cached(kind).text,
                    MetricCell::Error { msg } => msg,
                };
                acc += text.len();
            }
        }
        black_box(crate::plot::fit_limits(
            &borrowed,
            crate::plot::PSNR_LO,
            crate::plot::PSNR_HI,
        ));
        for v in &raw {
            acc += crate::plot::decimate_minmax(v, 1000).points().len();
        }
    }
    let per_frame = t0.elapsed() / n;
    black_box(acc);
    let per_sec = per_frame * 2;
    eprintln!(
        "idle heartbeat: {per_frame:?}/frame our-code ×2/s = {per_sec:?}/s (~{:.3}% of a core); without heartbeat: 0 frames",
        per_sec.as_secs_f64() * 100.0,
    );
}

/// Drain activity flags: queued message → `true`; empty channel → `false`.
/// Stale messages count as traffic (one harmless extra frame).
#[test]
fn drains_report_activity() {
    use super::{MetricMsg, PngSaveMsg, ThumbMsg};
    use crate::metrics::ffmpeg::MetricKind;
    let ctx = egui::Context::default();
    let mut app = RFMetricsApp::default();
    assert!(!app.drain_probe_results());
    assert!(!app.drain_metric_results());
    assert!(!app.drain_png_results(&ctx, 0.0));
    assert!(!app.drain_thumbs(&ctx));
    // Stale metric message (wrong generation) is still traffic.
    app.metric_tx
        .send(MetricMsg::Progress {
            generation: 999,
            kind: MetricKind::Psnr,
            key: "gone".to_owned(),
            frame: 1,
        })
        .unwrap();
    assert!(app.drain_metric_results());
    assert!(!app.drain_metric_results());
    // Stale probe message likewise.
    app.probe_tx
        .send(ProbeMsg::Reference {
            generation: 999,
            text: "stale".to_owned(),
            info: None,
            timed_out: false,
        })
        .unwrap();
    assert!(app.drain_probe_results());
    assert!(!app.drain_probe_results());
    // PNG worker reply lands a toast.
    app.png_tx
        .send(PngSaveMsg::Saved {
            path: std::path::PathBuf::from("C:/vids/plot.png"),
        })
        .unwrap();
    assert!(app.drain_png_results(&ctx, 0.0));
    assert!(app.toast.is_some());
    assert!(!app.drain_png_results(&ctx, 0.0));
    // Empty-image thumbnail clears without touching the GPU path.
    app.thumb_tx
        .send(ThumbMsg {
            generation: app.thumb_generation,
            image: None,
        })
        .unwrap();
    assert!(app.drain_thumbs(&ctx));
    assert!(!app.drain_thumbs(&ctx));
    // Refresh helpers propagate their drain flags (paths unchanged, so
    // no worker spawns — pure drain reporting).
    app.probe_tx
        .send(ProbeMsg::Reference {
            generation: 999,
            text: "stale".to_owned(),
            info: None,
            timed_out: false,
        })
        .unwrap();
    assert!(app.refresh_ref_info());
    assert!(!app.refresh_ref_info());
    app.thumb_tx
        .send(ThumbMsg {
            generation: app.thumb_generation,
            image: None,
        })
        .unwrap();
    assert!(app.refresh_thumbnail(&ctx));
    assert!(!app.refresh_thumbnail(&ctx));
}

/// FPS-counter cost across regimes, one run on identical fixtures
/// (200 idle rows, 5×5000 plot series, debug build = upper bound):
/// old frame (snapshot clone+compare, per-cell format+tooltip,
/// fit/decimate, FPS format every frame) vs new frame (borrowed text,
/// dirty check, fit/decimate, FPS format), plus the counter label
/// alone (format every frame vs only when the rounded value changes).
#[test]
fn fps_counter_budget() {
    use crate::metrics::{MetricCell, ffmpeg::MetricKind};
    use std::hint::black_box;
    let mut app = RFMetricsApp::default();
    for i in 0..200 {
        app.rows.push(psnr_test_row(
            &format!("C:/vids/clip_{i:04}.mp4"),
            i % 2 == 0,
        ));
    }
    app.saved_snapshot = app.snapshot();
    let kinds = MetricKind::ALL;
    let cells: Vec<MetricCell> = (0..200 * kinds.len()).map(|_| MetricCell::Idle).collect();
    let raw: Vec<Vec<f64>> = (0..5)
        .map(|s| {
            (0..5000)
                .map(|i| 30.0 + s as f64 + (i as f64 * 0.01).sin() * 5.0)
                .collect()
        })
        .collect();
    let borrowed: Vec<&[f64]> = raw.iter().map(Vec::as_slice).collect();
    let n = 100;
    // Old frame: everything formatted per frame.
    let t0 = std::time::Instant::now();
    let mut acc = 0usize;
    for _ in 0..n {
        acc += (app.snapshot() != app.saved_snapshot) as usize;
        for (cell, kind) in cells.iter().zip(kinds.iter().cycle()) {
            let text = cell.cell_text();
            let tip = cell.tooltip(kind.name());
            acc += text.len() + tip.len();
        }
        black_box(crate::plot::fit_limits(
            &borrowed,
            crate::plot::PSNR_LO,
            crate::plot::PSNR_HI,
        ));
        for v in &raw {
            acc += crate::plot::decimate_minmax(v, 1000).points().len();
        }
        acc += format!("FPS: {:.0}", black_box(59.7)).len();
    }
    let old_frame = t0.elapsed() / n;
    // New frame: borrows + dirty check.
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        acc += app.is_state_dirty() as usize;
        for cell in &cells {
            let text: &str = match cell {
                MetricCell::Idle => "N/A",
                _ => unreachable!("idle fixture"),
            };
            acc += text.len();
        }
        black_box(crate::plot::fit_limits(
            &borrowed,
            crate::plot::PSNR_LO,
            crate::plot::PSNR_HI,
        ));
        for v in &raw {
            acc += crate::plot::decimate_minmax(v, 1000).points().len();
        }
        acc += format!("FPS: {:.0}", black_box(59.7)).len();
    }
    let new_frame = t1.elapsed() / n;
    black_box(acc);
    // Counter label alone over a jittering fps stream.
    let stream: Vec<f64> = (0..10_000)
        .map(|i| 59.5 + (i as f64 * 0.37).sin())
        .collect();
    let t2 = std::time::Instant::now();
    let mut chars_a = 0;
    for &f in &stream {
        chars_a += format!("FPS: {f:.0}").len();
    }
    let always = t2.elapsed();
    black_box(chars_a);
    let t3 = std::time::Instant::now();
    let mut chars_b = 0;
    let mut last = i64::MIN;
    for &f in &stream {
        let r = f.round() as i64;
        if r != last {
            last = r;
            chars_b += format!("FPS: {f:.0}").len();
        }
    }
    let on_change = t3.elapsed();
    black_box(chars_b);
    let pct = |per_frame: std::time::Duration, hz: f64| per_frame.as_secs_f64() * hz * 100.0;
    eprintln!(
        "fps counter: old frame {old_frame:?} | new frame {new_frame:?}\n\
             regimes (% of a core, our-code only): old-measuring 60Hz {:.2}% | old-idle 10Hz {:.2}% | \
             new-measuring 10Hz {:.2}% | new-idle 2Hz {:.3}%\n\
             counter label 10k samples: format-always {always:?} vs on-change {on_change:?}",
        pct(old_frame, 60.0),
        pct(old_frame, 10.0),
        pct(new_frame, 10.0),
        pct(new_frame, 2.0),
    );
}

/// Thumbnail duration reuse: only the completed probe's own path
/// qualifies; anything else (or no/zero duration) falls back.
#[test]
fn thumb_duration_reuse_rules() {
    use super::thumb_duration;
    assert_eq!(
        thumb_duration("C:/r.mp4", "C:/r.mp4", Some(63.0)),
        Some(63.0)
    );
    // Stale cache (other path), missing, or non-positive: re-probe.
    assert_eq!(thumb_duration("C:/r.mp4", "C:/old.mp4", Some(63.0)), None);
    assert_eq!(thumb_duration("C:/r.mp4", "", Some(63.0)), None);
    assert_eq!(thumb_duration("C:/r.mp4", "C:/r.mp4", None), None);
    assert_eq!(thumb_duration("C:/r.mp4", "C:/r.mp4", Some(0.0)), None);
    assert_eq!(thumb_duration("C:/r.mp4", "C:/r.mp4", Some(-1.0)), None);
    assert_eq!(thumb_duration("", "", Some(63.0)), None);
}

/// The reference drain records which path its info was probed from.
#[test]
fn ref_drain_tracks_info_path() {
    let mut app = RFMetricsApp {
        last_spawned_ref: "C:/vids/r.mp4".to_owned(),
        ..RFMetricsApp::default()
    };
    app.probe_tx
        .send(ProbeMsg::Reference {
            generation: app.ref_generation,
            text: "info".to_owned(),
            info: None,
            timed_out: false,
        })
        .unwrap();
    app.drain_probe_results();
    assert_eq!(app.ref_info_path, "C:/vids/r.mp4");
    // Stale generation leaves the tracked path alone.
    app.last_spawned_ref = "C:/vids/new.mp4".to_owned();
    app.ref_generation = app.ref_generation.wrapping_add(1);
    app.probe_tx
        .send(ProbeMsg::Reference {
            generation: 0,
            text: "stale".to_owned(),
            info: None,
            timed_out: false,
        })
        .unwrap();
    app.drain_probe_results();
    assert_eq!(app.ref_info_path, "C:/vids/r.mp4");
}

/// Alt+click on a checked include box isolates it; on a solo/all-off
/// column it selects all (egui_plot legend parity). The helper takes
/// POST-toggle flags — the clicked box already flipped once.
#[test]
fn alt_include_solo_and_select_all() {
    use super::apply_alt_include;
    // Others checked + Alt+click row 0 (was checked): post-toggle is
    // [F,T,T] → isolate to the clicked row.
    let mut flags = vec![false, true, true];
    apply_alt_include(&mut flags, 0);
    assert_eq!(flags, vec![true, false, false]);
    // All unchecked + Alt+click row 1: post-toggle [F,T,F] → all on.
    let mut flags = vec![false, true, false];
    apply_alt_include(&mut flags, 1);
    assert_eq!(flags, vec![true, true, true]);
    // Solo row 0 + Alt+click it (was the only checked): post-toggle
    // [F,F,F] → restore all.
    let mut flags = vec![false, false, false];
    apply_alt_include(&mut flags, 0);
    assert_eq!(flags, vec![true, true, true]);
    // Mixed + Alt+click unchecked row 2: post-toggle [T,T,T] → isolate.
    let mut flags = vec![true, true, true];
    apply_alt_include(&mut flags, 2);
    assert_eq!(flags, vec![false, false, true]);
    // Single row Alt+click stays on; out-of-range idx is a no-op.
    let mut flags = vec![false];
    apply_alt_include(&mut flags, 0);
    assert_eq!(flags, vec![true]);
    let mut flags = vec![true, false];
    apply_alt_include(&mut flags, 9);
    assert_eq!(flags, vec![true, false]);
}

/// Shift+click on an include box fills anchor..=clicked with the clicked
/// box's post-toggle value. The helper only resolves the span; stale
/// anchors (past the queue) yield `None` → plain single toggle.
#[test]
fn shift_include_range_span_and_stale_anchor() {
    use super::shift_include_range;
    assert_eq!(shift_include_range(5, 1, 3), Some((1, 3)));
    assert_eq!(shift_include_range(5, 3, 1), Some((1, 3)));
    assert_eq!(shift_include_range(5, 2, 2), Some((2, 2)));
    assert_eq!(shift_include_range(1, 0, 0), Some((0, 0)));
    assert_eq!(shift_include_range(3, 7, 1), None);
    assert_eq!(shift_include_range(3, 0, 9), None);
    // Check-direction: anchor 0, Shift+click row 3 to check it.
    let mut flags = vec![true, false, false, false];
    if let Some((lo, hi)) = shift_include_range(flags.len(), 0, 3) {
        for v in &mut flags[lo..=hi] {
            *v = true;
        }
    }
    assert_eq!(flags, vec![true, true, true, true]);
    // Uncheck-direction: anchor 0, Shift+click row 2 to uncheck it.
    let mut flags = vec![true, true, true, true];
    if let Some((lo, hi)) = shift_include_range(flags.len(), 0, 2) {
        for v in &mut flags[lo..=hi] {
            *v = false;
        }
    }
    assert_eq!(flags, vec![false, false, false, true]);
}

/// Full-row `selected` (the removal set) reuses both helpers: Alt-solo
/// mirrors the include column, Shift+click fills the span with the
/// clicked row's post-toggle state (checkbox-style).
#[test]
fn selection_alt_solo_and_shift_clicked_wins() {
    use super::{apply_alt_include, shift_include_range};
    // Alt+click row 0 with others selected → isolate to row 0.
    let mut sel = vec![false, true, true];
    apply_alt_include(&mut sel, 0);
    assert_eq!(sel, vec![true, false, false]);
    // Alt+click with nothing selected → select all.
    let mut sel = vec![false, true, false];
    apply_alt_include(&mut sel, 1);
    assert_eq!(sel, vec![true, true, true]);
    // Shift+click: anchor 3 (selected), click unselected row 1 →
    // post-toggle true selects the span.
    let mut sel = vec![false, false, false, true, false];
    if let Some((lo, hi)) = shift_include_range(sel.len(), 3, 1) {
        let v = !sel[1];
        for v2 in &mut sel[lo..=hi] {
            *v2 = v;
        }
    }
    assert_eq!(sel, vec![false, true, true, true, false]);
    // Shift+click: anchor 0 (selected), click selected row 2 →
    // post-toggle false unselects the span.
    let mut sel = vec![true, false, true];
    if let Some((lo, hi)) = shift_include_range(sel.len(), 0, 2) {
        let v = !sel[2];
        for v2 in &mut sel[lo..=hi] {
            *v2 = v;
        }
    }
    assert_eq!(sel, vec![false, false, false]);
}

/// Timed-out probes record a note for the update-level toast; stale or
/// row-less timeouts stay silent.
#[test]
fn probe_timeout_note_recorded_once() {
    let mut app = RFMetricsApp::default();
    app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
    app.probe_tx
        .send(ProbeMsg::RowMedia {
            key: norm_key("C:/vids/a.mp4"),
            media: "x".to_owned(),
            tip: "y".to_owned(),
            info: None,
            timed_out: true,
        })
        .unwrap();
    app.drain_probe_results();
    assert_eq!(app.probe_timeout_note.as_deref(), Some("a.mp4"));
    // Stale ref timeout: generation mismatch drops it, note untouched.
    app.probe_tx
        .send(ProbeMsg::Reference {
            generation: 999,
            text: "Probe timed out".to_owned(),
            info: None,
            timed_out: true,
        })
        .unwrap();
    app.drain_probe_results();
    assert_eq!(app.probe_timeout_note.as_deref(), Some("a.mp4"));
    // Timeout for a removed row stays silent.
    app.probe_tx
        .send(ProbeMsg::RowMedia {
            key: "gone".to_owned(),
            media: "x".to_owned(),
            tip: "y".to_owned(),
            info: None,
            timed_out: true,
        })
        .unwrap();
    app.drain_probe_results();
    assert_eq!(app.probe_timeout_note.as_deref(), Some("a.mp4"));
}

/// Minimal queue row for sort tests: distinct display name plus an
/// optional scored cell of one metric (everything else Idle).
fn sort_test_row(
    display: &str,
    kind: crate::metrics::ffmpeg::MetricKind,
    avg: Option<f64>,
) -> QueueRow {
    use crate::metrics::MetricCell;
    let mut r = psnr_test_row(&format!("C:/vids/{display}"), true);
    r.display = display.to_owned();
    let cell = match avg {
        Some(avg) => MetricCell::Done {
            values: vec![avg],
            avg,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        },
        None => MetricCell::Idle,
    };
    *r.cell_mut(kind) = cell;
    r
}

/// Header-click cycle: fresh click starts best-first, repeat flips,
/// third clears to insertion; switching columns restarts.
#[test]
fn sort_cycle_path_and_metrics() {
    use super::{SortColumn, SortDir, cycle_sort, initial_dir};
    use crate::metrics::CellStat;
    use crate::metrics::ffmpeg::MetricKind;
    assert_eq!(
        cycle_sort(None, SortColumn::Path, CellStat::Avg),
        Some((SortColumn::Path, SortDir::Asc))
    );
    assert_eq!(
        cycle_sort(None, SortColumn::Metric(MetricKind::Vmaf), CellStat::Avg),
        Some((SortColumn::Metric(MetricKind::Vmaf), SortDir::Desc))
    );
    // Butteraugli is lower-better: best first is ascending.
    assert_eq!(
        initial_dir(SortColumn::Metric(MetricKind::But), CellStat::Avg),
        SortDir::Asc
    );
    // StdDev is lower-better on every metric: best first is ascending.
    assert_eq!(
        initial_dir(SortColumn::Metric(MetricKind::Vmaf), CellStat::StdDev),
        SortDir::Asc
    );
    assert_eq!(
        initial_dir(SortColumn::Metric(MetricKind::Vmaf), CellStat::Max),
        SortDir::Desc
    );
    let first = cycle_sort(None, SortColumn::Metric(MetricKind::Vmaf), CellStat::Avg).unwrap();
    let second = cycle_sort(
        Some(first),
        SortColumn::Metric(MetricKind::Vmaf),
        CellStat::Avg,
    )
    .unwrap();
    assert_eq!(second.1, SortDir::Asc);
    assert_eq!(
        cycle_sort(
            Some(second),
            SortColumn::Metric(MetricKind::Vmaf),
            CellStat::Avg
        ),
        None
    );
    assert_eq!(
        cycle_sort(Some(second), SortColumn::Path, CellStat::Avg),
        Some((SortColumn::Path, SortDir::Asc))
    );
}

/// Display order: identity unsorted, A-Z paths, best-first scores with
/// unscored rows always last in both directions, stable ties.
#[test]
fn sort_view_orders_and_restores() {
    use super::{SortColumn, SortDir, sort_view};
    use crate::metrics::CellStat;
    use crate::metrics::ffmpeg::MetricKind;
    let rows = vec![
        sort_test_row("c.mp4", MetricKind::Vmaf, Some(30.0)),
        sort_test_row("a.mp4", MetricKind::Vmaf, None),
        sort_test_row("b.mp4", MetricKind::Vmaf, Some(40.0)),
        sort_test_row("d.mp4", MetricKind::Vmaf, Some(40.0)),
    ];
    // Insertion identity when unsorted.
    assert_eq!(sort_view(&rows, None, CellStat::Avg), vec![0, 1, 2, 3]);
    // Path A-Z.
    assert_eq!(
        sort_view(&rows, Some((SortColumn::Path, SortDir::Asc)), CellStat::Avg),
        vec![1, 2, 0, 3]
    );
    // VMAF best first: highest scored first, unscored last, ties stable.
    assert_eq!(
        sort_view(
            &rows,
            Some((SortColumn::Metric(MetricKind::Vmaf), SortDir::Desc)),
            CellStat::Avg
        ),
        vec![2, 3, 0, 1]
    );
    // Reversed: lowest scored first, unscored still last.
    assert_eq!(
        sort_view(
            &rows,
            Some((SortColumn::Metric(MetricKind::Vmaf), SortDir::Asc)),
            CellStat::Avg
        ),
        vec![0, 2, 3, 1]
    );
    // Butteraugli best first is ascending (lower wins).
    let rows = vec![
        sort_test_row("a.mp4", MetricKind::But, Some(2.0)),
        sort_test_row("b.mp4", MetricKind::But, Some(1.0)),
    ];
    assert_eq!(
        sort_view(
            &rows,
            Some((SortColumn::Metric(MetricKind::But), SortDir::Asc)),
            CellStat::Avg
        ),
        vec![1, 0]
    );
}

/// Sort follows the cell-value selector: same rows order differently
/// under Avg vs Min. Multi-value cells exercise both the cached-stats
/// path (populated) and the uncached fallback (cleared).
#[test]
fn sort_view_follows_cell_stat_selector() {
    use super::{SortColumn, SortDir, sort_view};
    use crate::metrics::CellStat;
    use crate::metrics::ffmpeg::MetricKind;
    fn scored(display: &str, values: Vec<f64>, avg: f64, cache: bool) -> QueueRow {
        use crate::metrics::MetricCell;
        let mut r = psnr_test_row(&format!("C:/vids/{display}"), true);
        r.display = display.to_owned();
        r.psnr = MetricCell::Done {
            values,
            avg,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
            fps_mode: InputFpsMode::Reference,
        };
        if cache {
            let stats = r.psnr.done_stats();
            r.psnr_cache.stats = stats;
        }
        r
    }
    // a: avg 15, min 10 — b: avg 14, min 14.
    let spec = Some((SortColumn::Metric(MetricKind::Psnr), SortDir::Desc));
    let rows = vec![
        scored("a.mp4", vec![10.0, 20.0], 15.0, true),
        scored("b.mp4", vec![14.0, 14.0], 14.0, true),
    ];
    assert_eq!(sort_view(&rows, spec, CellStat::Avg), vec![0, 1]);
    assert_eq!(sort_view(&rows, spec, CellStat::Min), vec![1, 0]);
    // Uncached fallback agrees (caches never populated, cells Done).
    let bare = vec![
        scored("a.mp4", vec![10.0, 20.0], 15.0, false),
        scored("b.mp4", vec![14.0, 14.0], 14.0, false),
    ];
    assert_eq!(sort_view(&bare, spec, CellStat::Avg), vec![0, 1]);
    assert_eq!(sort_view(&bare, spec, CellStat::Min), vec![1, 0]);
}

/// Cell-value selector persists and restores; unknown labels keep Avg.
#[test]
fn state_apply_restores_cell_stat() {
    use crate::metrics::CellStat;
    let mut app = RFMetricsApp::default();
    assert_eq!(app.cell_stat, CellStat::Avg);
    let state = crate::state::AppState {
        options: crate::state::OptionsState {
            cell_stat: Some("Max".to_owned()),
            ..Default::default()
        },
        ..Default::default()
    };
    app.apply_state(Some(state));
    assert_eq!(app.cell_stat, CellStat::Max);
    assert!(app.snapshot().options.cell_stat == Some("Max".to_owned()));
    let state = crate::state::AppState {
        options: crate::state::OptionsState {
            cell_stat: Some("Frames".to_owned()),
            ..Default::default()
        },
        ..Default::default()
    };
    app.apply_state(Some(state));
    assert_eq!(app.cell_stat, CellStat::Max);
}

#[test]
fn running_sweep_ping_pongs() {
    use super::running_sweep_pos;
    // 0.7s per leg: 0 → 1 → 0 → 1 …
    assert!((running_sweep_pos(0.0) - 0.0).abs() < 1e-6);
    assert!((running_sweep_pos(0.35) - 0.5).abs() < 1e-6);
    assert!((running_sweep_pos(0.7) - 1.0).abs() < 1e-6);
    assert!((running_sweep_pos(1.05) - 0.5).abs() < 1e-6);
    assert!((running_sweep_pos(1.4) - 0.0).abs() < 1e-6);
    for t in [0.1, 0.5, 0.9, 1.3, 2.0, 10.0] {
        let p = running_sweep_pos(t);
        assert!((0.0..=1.0).contains(&p), "pos {p} out of range at t={t}");
    }
}
