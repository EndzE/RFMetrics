//! FFVship metrics: SSIMULACRA2, Butteraugli, CVVDP (Python
//! `_compute_ffvship_series` parity). Rides the shared worker plumbing
//! (`RunInputs`, abort slot, progress channel); only the argv, the
//! `--live-score-output` protocol, and the strict whole-output parser are
//! FFVship-specific. Scores are never clamped (Python parity: finite-check
//! only) — unlike the PSNR/SSIM frame clamps.

use crate::metrics::ffmpeg::{
    FrameDetail, RunInputs, RunOutcome, STDERR_TAIL_LINES, pump_process, stderr_tail,
};
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

/// Full FFVship argv (minus the exe) for one metric run.
pub fn build_args(
    kind: FfvshipKind,
    ref_path: &str,
    dist_path: &str,
    window: &[String],
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
    args
}

/// Blocking FFVship run; call off the UI thread. Progress counts scored
/// stdout lines (Python parity); the strict parse runs at the end.
/// ponytail: no wait-timeout (worker-only pattern, like `run_metric`).
pub fn run_ffvship(
    job: &RunInputs,
    kind: FfvshipKind,
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
    let fail = |msg: String| RunOutcome {
        values: Vec::new(),
        avg: None,
        exec_s: 0.0,
        error: Some(msg),
        detail: crate::metrics::ffmpeg::FrameDetail::None,
    };
    let (window, expected_n) = match trim_window_frames(ref_info, dist_info, skip, clip_dur) {
        Ok(w) => w,
        Err(e) => {
            log::warn!(target: "rfmetrics::metric", "{name} {e}");
            return fail(e);
        }
    };
    let args = build_args(kind, ref_path, dist_path, &window);
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
    let mut pending: Vec<f64> = Vec::new();
    let mut last_emit = std::time::Instant::now();
    let emit = |pending: &mut Vec<f64>, last_emit: &mut std::time::Instant| {
        if !pending.is_empty()
            && (pending.len() >= crate::metrics::ffmpeg::SERIES_BATCH
                || last_emit.elapsed() >= crate::metrics::ffmpeg::SERIES_THROTTLE)
        {
            on_series(pending);
            pending.clear();
            *last_emit = std::time::Instant::now();
        }
    };
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
                pending.push(v);
                emit(&mut pending, &mut last_emit);
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
        return RunOutcome {
            values: Vec::new(),
            avg: None,
            exec_s: pumped.exec_s,
            error: Some("aborted".to_owned()),
            detail: crate::metrics::ffmpeg::FrameDetail::None,
        };
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
        None => {
            let msg = pumped
                .stderr
                .lines()
                .map(str::trim)
                .rfind(|l| !l.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("no {name} data"));
            let dump = stderr_tail(&pumped.stderr, STDERR_TAIL_LINES);
            log::warn!(target: "rfmetrics::metric", "{name} no data for \"{dist_path}\" (exit {:?}, {:.1}s): {msg}\n{dump}", pumped.code, pumped.exec_s);
            RunOutcome {
                values: Vec::new(),
                avg: None,
                exec_s: pumped.exec_s,
                error: Some(msg),
                detail: crate::metrics::ffmpeg::FrameDetail::None,
            }
        }
    }
}
#[cfg(test)]
#[path = "../tests/test_metrics_ffvship.rs"]
mod tests;
