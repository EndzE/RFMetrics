//! FFVship metrics: SSIMULACRA2, Butteraugli, CVVDP (Python
//! `_compute_ffvship_series` parity). Rides the shared worker plumbing
//! (`RunInputs`, abort slot, progress channel); only the argv, the
//! `--live-score-output` protocol, and the strict whole-output parser are
//! FFVship-specific. Scores are never clamped (Python parity: finite-check
//! only) — unlike the PSNR/SSIM frame clamps.

use crate::metrics::ffmpeg::{FrameDetail, RunInputs, RunOutcome, no_data_outcome, pump_process};
use crate::probe::MediaInfo;

/// Which FFVship metric a run computes: CLI name, per-line score count,
/// and pooling rule (CVVDP pools the last frame, the others the mean).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfvshipKind {
    Ssimulacra2,
    Butteraugli,
    Cvvdp,
}

impl FfvshipKind {
    /// `-m` CLI value (Python `_compute_*_series` parity).
    pub fn metric_arg(self) -> &'static str {
        match self {
            Self::Ssimulacra2 => "SSIMULACRA2",
            Self::Butteraugli => "Butteraugli",
            Self::Cvvdp => "CVVDP",
        }
    }
    /// Scores per live-output line (Butteraugli emits 3; the first is used).
    fn n_scores(self) -> usize {
        match self {
            Self::Butteraugli => 3,
            Self::Ssimulacra2 | Self::Cvvdp => 1,
        }
    }

    /// CVVDP alone pools the last frame value; the rest use the mean.
    fn pool_last(self) -> bool {
        matches!(self, Self::Cvvdp)
    }

    /// CSV column headers (the writer prepends `frame`/`n`): measured
    /// score names in live-output order. Butteraugli's first column is
    /// the default `--qnorm 2` norm — verified live (`--qnorm 3` moves
    /// it onto the second column, which is always the 3Norm).
    pub fn csv_cols(self) -> Vec<String> {
        match self {
            Self::Ssimulacra2 => vec!["ssimulacra2".to_owned()],
            Self::Butteraugli => [
                "butteraugli_2norm",
                "butteraugli_3norm",
                "butteraugli_infnorm",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            Self::Cvvdp => vec!["cvvdp".to_owned()],
        }
    }
}

/// CVVDP display-model registry (`cvvdp-displays.json`, normalized from
/// upstream `display_models.json`; every entry verified live against
/// FFVship 5.1.1). File-driven, not an enum: 26 entries would duplicate
/// the file key-for-key. The default is the binary's own default, so it
/// is omitted from argv (byte-identical runs to before).
#[derive(Debug, Clone)]
pub struct DisplayModel {
    /// `--displayModel` value and state-file key.
    pub key: String,
    /// Combo text (VideoMetricsLab's short names for its 8, same style
    /// for the rest).
    pub name: String,
    /// In the VideoMetricsLab friendly list (the "More models" checkbox
    /// gates the rest).
    pub vmlab: bool,
    /// Normalized display object, written verbatim into the per-run
    /// `--displayConfig` file.
    pub display: serde_json::Map<String, serde_json::Value>,
}

/// The default display: the binary's own when no flags are passed.
pub const DEFAULT_DISPLAY_KEY: &str = "standard_fhd";

/// Display fields FFVship requires (`Display Missing …` errors); entries
/// lacking any are skipped at load so one bad entry can't poison the file.
const REQUIRED_DISPLAY_FIELDS: &[&str] = &[
    "colorspace",
    "contrast",
    "diagonal_size_inches",
    "E_ambient",
    "max_luminance",
    "resolution",
    "viewing_distance_meters",
];

fn load_registry() -> Vec<DisplayModel> {
    let raw: serde_json::Value = serde_json::from_str(include_str!("cvvdp-displays.json"))
        .unwrap_or(serde_json::Value::Null);
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for entry in raw
        .get("models")
        .and_then(|m| m.as_array())
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        let Some(key) = entry.get("key").and_then(|k| k.as_str()) else {
            log::warn!(target: "rfmetrics::metric", "skipping CVVDP display entry without a key");
            continue;
        };
        let Some(name) = entry.get("name").and_then(|n| n.as_str()) else {
            log::warn!(target: "rfmetrics::metric", "skipping CVVDP display {key:?} without a name");
            continue;
        };
        let group = entry
            .get("group")
            .and_then(|g| g.as_str())
            .unwrap_or("extended");
        let Some(display) = entry.get("display").and_then(|d| d.as_object()) else {
            log::warn!(target: "rfmetrics::metric", "skipping CVVDP display {key:?} without a display object");
            continue;
        };
        if !seen.insert(key.to_owned())
            || REQUIRED_DISPLAY_FIELDS
                .iter()
                .any(|f| display.get(*f).is_none())
        {
            log::warn!(target: "rfmetrics::metric", "skipping CVVDP display {key:?}: duplicate or incomplete");
            continue;
        }
        out.push(DisplayModel {
            key: key.to_owned(),
            name: name.to_owned(),
            vmlab: group == "vmlab",
            display: display.clone(),
        });
    }
    out
}

/// Parsed-once display registry (the file is embedded, so this only ever
/// fails closed: empty list, everything falls back to the binary default).
pub fn display_registry() -> &'static [DisplayModel] {
    static REGISTRY: std::sync::OnceLock<Vec<DisplayModel>> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(load_registry)
}

/// Registry lookup by `--displayModel` key (also the state-file value).
pub fn display_named(key: &str) -> Option<&'static DisplayModel> {
    display_registry().iter().find(|m| m.key == key)
}

impl DisplayModel {
    /// One-line summary for the options panel (VideoMetricsLab `describe`
    /// parity): `30" 3840x2160 SDR, 200 nits, 250 lux, 0.75 m (2.0 x
    /// screen height)`. Falls back to the key on degenerate geometry
    /// (unreachable through the validated registry).
    pub fn describe(&self) -> String {
        let num = |f: &str| self.display.get(f).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let (w, h) = match self.display.get("resolution").and_then(|r| r.as_array()) {
            Some(r) => (
                r.first().and_then(|v| v.as_f64()).unwrap_or(0.0),
                r.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0),
            ),
            None => (0.0, 0.0),
        };
        if w <= 0.0 || h <= 0.0 {
            return self.key.clone();
        }
        let ar = w / h;
        let height_m = num("diagonal_size_inches") * 0.0254 / (1.0 + ar * ar).sqrt();
        if height_m <= 0.0 {
            return self.key.clone();
        }
        let sdr = self.display.get("colorspace").and_then(|c| c.as_str()) == Some("SDR");
        format!(
            "{}\" {}x{} {}, {} nits, {} lux, {:.2} m ({:.1} x screen height)",
            num("diagonal_size_inches"),
            w,
            h,
            if sdr { "SDR" } else { "HDR" },
            num("max_luminance"),
            num("E_ambient"),
            num("viewing_distance_meters"),
            num("viewing_distance_meters") / height_m,
        )
    }
}

/// Per-run `--displayConfig` file (`{key: display}` — the shape FFVship
/// parses; a bare object crashes its parser). Unique per call so
/// concurrent runs never share; the caller deletes it (see
/// [`DisplayConfigGuard`]).
fn write_display_config(model: &DisplayModel) -> std::io::Result<std::path::PathBuf> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!(
        "rfmetrics-cvvdp-{}-{n}-{}.json",
        std::process::id(),
        model.key
    ));
    let mut obj = serde_json::Map::new();
    obj.insert(
        model.key.clone(),
        serde_json::Value::Object(model.display.clone()),
    );
    std::fs::write(&path, serde_json::to_string(&obj).unwrap_or_default())?;
    Ok(path)
}

/// Owns a per-run `--displayConfig` file; deleted on drop so aborts and
/// early returns can't leak it (hard-kill residue is KBs in tempdir).
pub struct DisplayConfig {
    key: String,
    path: std::path::PathBuf,
}

impl DisplayConfig {
    /// `Ok(None)` = default display: no file, no flags (binary default).
    /// Unknown keys also fall back to the default (logged): a run must
    /// never fail on a display lookup.
    pub fn new(key: &str) -> std::io::Result<Option<Self>> {
        if key == DEFAULT_DISPLAY_KEY {
            return Ok(None);
        }
        let Some(model) = display_named(key) else {
            log::warn!(target: "rfmetrics::metric", "unknown CVVDP display {key:?}, using binary default");
            return Ok(None);
        };
        Ok(Some(Self {
            key: model.key.clone(),
            path: write_display_config(model)?,
        }))
    }

    /// `(key, path)` argv pair.
    pub fn argv(&self) -> (&str, &std::path::Path) {
        (&self.key, &self.path)
    }
}

impl Drop for DisplayConfig {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// `--start/--end` frame window + expected frame count (Python
/// `_compute_ffvship_series` parity): skip/clip seconds become frames via
/// fps, and the end clamps to the probed duration. `Ok((empty, None))` =
/// full video. `Err` when a trim is set but no fps is known.
pub fn trim_window_frames(
    ref_info: &MediaInfo,
    dist_info: &MediaInfo,
    skip: Option<f64>,
    clip_dur: Option<f64>,
) -> Result<(Vec<String>, Option<i64>), String> {
    if !skip.is_some_and(|v| v != 0.0) && !clip_dur.is_some_and(|v| v != 0.0) {
        return Ok((Vec::new(), None));
    }
    let Some(fps) = ref_info.fps.or(dist_info.fps).filter(|f| *f > 0.0) else {
        return Err("unknown fps for skip/duration".to_owned());
    };
    let start = (skip.unwrap_or(0.0) * fps) as i64;
    if let Some(d) = clip_dur.filter(|&d| d != 0.0) {
        let mut end = start + (d * fps).round() as i64;
        if let Some(total) = ref_info.duration.or(dist_info.duration) {
            end = end.min((total * fps) as i64);
        }
        let expected = (end - start).max(0);
        Ok((
            vec![
                "--start".to_owned(),
                start.to_string(),
                "--end".to_owned(),
                end.to_string(),
            ],
            Some(expected),
        ))
    } else {
        Ok((vec!["--start".to_owned(), start.to_string()], None))
    }
}

/// Strict whole-output parse of `--live-score-output` text (Python
/// `_parse_ffvship_live` parity): the first non-empty line is the total
/// frame count (checked against `expected_n` when the trim window sets
/// one), then one `<idx> <scores…>` line per frame — exact arity, all
/// finite, no duplicate indices, covering `0..n`.
fn parse_live_rows(text: &str, expected_n: Option<i64>, n_scores: usize) -> Option<Vec<Vec<f64>>> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let n: i64 = lines.first()?.parse().ok()?;
    if n <= 0 || expected_n.is_some_and(|e| e != n) {
        return None;
    }
    let mut scores: std::collections::HashMap<usize, Vec<f64>> = std::collections::HashMap::new();
    for line in &lines[1..] {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 1 + n_scores {
            return None;
        }
        let idx: usize = parts[0].parse().ok()?;
        let mut vals = Vec::with_capacity(n_scores);
        for p in &parts[1..] {
            let v: f64 = p.parse().ok()?;
            if !v.is_finite() {
                return None;
            }
            vals.push(v);
        }
        if scores.insert(idx, vals).is_some() {
            return None;
        }
    }
    (0..n as usize)
        .map(|i| scores.remove(&i))
        .collect::<Option<Vec<_>>>()
}

/// Best-effort per-line curve value for live plots: the first score of a
/// well-formed `<idx> <scores…>` line — exactly what the runner collects
/// per row. Anything else is skipped live; the strict end-parse
/// (arity, finiteness, duplicates, coverage) still decides `Done`, which
/// replaces the live buffer.
pub fn live_value(line: &str, n_scores: usize) -> Option<f64> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 1 + n_scores {
        return None;
    }
    let _: usize = parts[0].parse().ok()?;
    let mut scores = parts[1..].iter();
    let first: f64 = scores.next()?.parse().ok()?;
    if !first.is_finite() {
        return None;
    }
    for p in scores {
        if !p.parse::<f64>().map(|x| x.is_finite()).unwrap_or(false) {
            return None;
        }
    }
    Some(first)
}

/// Pooled average: last frame for CVVDP, arithmetic mean otherwise.
fn pooled_avg(values: &[f64], kind: FfvshipKind) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(if kind.pool_last() {
        values[values.len() - 1]
    } else {
        crate::metrics::mean(values)
    })
}

/// Full FFVship argv (minus the exe) for one metric run. `display` only
/// affects CVVDP: `Some((key, config path))` appends `--displayConfig`
/// and `--displayModel`; `None` (the default display) omits both, keeping
/// default runs byte-identical to before.
pub fn build_args(
    kind: FfvshipKind,
    ref_path: &str,
    dist_path: &str,
    window: &[String],
    display: Option<(&str, &std::path::Path)>,
) -> Vec<String> {
    let mut args = vec![
        "-s".to_owned(),
        ref_path.to_owned(),
        "-e".to_owned(),
        dist_path.to_owned(),
        "-m".to_owned(),
        kind.metric_arg().to_owned(),
        "--live-score-output".to_owned(),
    ];
    args.extend(window.iter().cloned());
    if kind == FfvshipKind::Cvvdp
        && let Some((key, path)) = display
    {
        args.push("--displayConfig".to_owned());
        args.push(path.display().to_string());
        args.push("--displayModel".to_owned());
        args.push(key.to_owned());
    }
    args
}

/// Blocking FFVship run; call off the UI thread. Progress counts scored
/// stdout lines (Python parity); the strict parse runs at the end.
/// ponytail: no wait-timeout (worker-only pattern, like `run_metric`).
pub fn run_ffvship(
    job: &RunInputs,
    kind: FfvshipKind,
    display_key: &str,
    on_progress: &(dyn Fn(u64) + Sync),
    on_series: &(dyn Fn(&[f64]) + Sync),
) -> RunOutcome {
    let RunInputs {
        exe,
        ref_path,
        dist_path,
        ref_info,
        dist_info,
        skip,
        clip_dur,
        abort,
        child_slot,
        ..
    } = *job;
    let name = kind.metric_arg();
    // FFVship kinds are out of CSV scope: no frame detail is retained.
    let fail = |msg: String| RunOutcome::error(msg, 0.0);
    let (window, expected_n) = match trim_window_frames(ref_info, dist_info, skip, clip_dur) {
        Ok(w) => w,
        Err(e) => {
            log::warn!(target: "rfmetrics::metric", "{name} {e}");
            return fail(e);
        }
    };
    // Per-run display file (guard deletes it on every exit path,
    // including abort). Temp-dir write failure fails the run loudly
    // instead of silently scoring the wrong display.
    let display_cfg = match DisplayConfig::new(display_key) {
        Ok(cfg) => cfg,
        Err(e) => {
            log::warn!(target: "rfmetrics::metric", "{name} display file: {e}");
            return fail(format!("display file: {e}"));
        }
    };
    let display_argv = display_cfg.as_ref().map(|c| c.argv());
    let args = build_args(kind, ref_path, dist_path, &window, display_argv);
    // `info`: the exact repro command is the core artifact of an issue
    // report (FFMetrics.log parity) — one line per metric job.
    log::info!(target: "rfmetrics::metric", "run: \"{}\" {}", exe.display(), args.join(" "));
    let mut out_lines: Vec<String> = Vec::new();
    let mut frames = 0u64;
    // Live-curve tap: first score per well-formed line, mirroring what
    // the strict end-parse takes (`r[0]`).
    // Malformed lines are skipped live; `Done` replaces the buffer with
    // the strict result either way. Same throttle as `run_metric`.
    let n_scores = kind.n_scores();
    let mut series = crate::metrics::ffmpeg::SeriesEmitter::new(on_series);
    let pumped = match pump_process(
        exe,
        &args,
        None,
        abort,
        child_slot,
        |line| {
            out_lines.push(line.to_owned());
            // Python progress: lines with 2+ tokens count as a frame
            // (the leading count line has one and is skipped).
            if line.split_whitespace().count() >= 2 {
                frames += 1;
                on_progress(frames);
            }
            if let Some(v) = live_value(line, n_scores) {
                series.push(v);
            }
        },
        on_progress,
    ) {
        Ok(p) => p,
        Err(e) => {
            log::warn!(target: "rfmetrics::metric", "{name} {e}");
            return fail(e);
        }
    };
    if pumped.aborted {
        log::info!(target: "rfmetrics::metric", "{name} \"{dist_path}\" aborted after {:.1}s", pumped.exec_s);
        return RunOutcome::error("aborted".to_owned(), pumped.exec_s);
    }
    // Python ignores the exit code and trusts the strict parse instead.
    // Full ordered rows feed both the pooled series (first score, the
    // runner's live projection) and the CSV detail.
    let text = out_lines.join("\n");
    match parse_live_rows(&text, expected_n, n_scores).filter(|r| !r.is_empty()) {
        Some(rows) => {
            let values: Vec<f64> = rows.iter().map(|r| r[0]).collect();
            let avg = pooled_avg(&values, kind);
            let show = avg.unwrap_or_else(|| crate::metrics::mean(&values));
            log::info!(
                target: "rfmetrics::metric",
                "{name} \"{dist_path}\" → {show:.4} ({} frames, exit {:?}, {:.1}s)",
                values.len(),
                pumped.code,
                pumped.exec_s,
            );
            RunOutcome {
                values,
                avg,
                exec_s: pumped.exec_s,
                error: None,
                detail: FrameDetail::Scores {
                    cols: kind.csv_cols(),
                    rows,
                },
            }
        }
        None => no_data_outcome(
            name,
            dist_path,
            pumped.code,
            pumped.exec_s,
            &pumped.stderr,
            &format!("no {name} data"),
        ),
    }
}
#[cfg(test)]
#[path = "../tests/test_metrics_ffvship.rs"]
mod tests;
