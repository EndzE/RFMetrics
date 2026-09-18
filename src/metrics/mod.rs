pub mod csv;
pub mod ffmpeg;
pub mod ffvship;
pub mod results;
pub mod vmaf;

/// Per-cell lifecycle of one metric on one queue row (Python `results` dict
/// entry + label text rolled into one).
#[derive(Debug, Clone, Default)]
pub enum MetricCell {
    #[default]
    Idle,
    Running {
        frame: u64,
        /// Per-frame values streamed so far (live plot curves); replaced
        /// by the strict full series on `Done`, dropped on abort/Reset
        /// with the cell itself.
        values: Vec<f64>,
    },
    Done {
        values: Vec<f64>,
        avg: f64,
        exec_s: f64,
        /// Trim settings the run used; a rerun under different skip/clip
        /// must recompute instead of trusting this value.
        skip: Option<f64>,
        clip_dur: Option<f64>,
        /// VMAF settings the run used (`None` for other metrics); a rerun
        /// under different VMAF options recomputes just that column.
        vmaf_cfg: Option<crate::metrics::vmaf::VmafCfg>,
        /// Scaling method the run used; a rerun under a different method
        /// recomputes every ffmpeg-backed column (FFVship has no scale
        /// stage and ignores it).
        scaler: crate::metrics::ffmpeg::ScaleMethod,
    },
    Error {
        msg: String,
    },
}

impl MetricCell {
    /// Short cell text (Python `_done_metric` / progress handler parity).
    pub fn cell_text(&self) -> String {
        match self {
            Self::Idle => "N/A".to_owned(),
            Self::Running { frame, .. } => format!("Frame: {frame}"),
            Self::Done { avg, .. } => format!("{avg:.4}"),
            Self::Error { msg } => msg.clone(),
        }
    }

    /// Hover tooltip; `title` is the metric name (e.g. "PSNR").
    pub fn tooltip(&self, title: &str) -> String {
        match self {
            Self::Idle => "No data yet".to_owned(),
            Self::Running { .. } => "Computing…".to_owned(),
            Self::Done {
                values,
                avg,
                exec_s,
                ..
            } => {
                format!("{title}\n{}", stats_text(Some(*avg), *exec_s, values))
            }
            Self::Error { msg } => format!("Error: {msg}"),
        }
    }

    /// Comparable statistics for a finished run; `None` unless `Done`.
    pub fn done_stats(&self) -> Option<DoneStats> {
        match self {
            Self::Done {
                values,
                avg,
                exec_s,
                ..
            } => DoneStats::new(values, *avg, *exec_s),
            _ => None,
        }
    }
}

/// Cross-row rank of one stat value (screenshot green/red/yellow rules).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatRank {
    /// Holds the column extreme on the winning side: green.
    Best,
    /// Holds the column extreme on the losing side: red.
    Worst,
    /// Single row or all equal: dim yellow (neither won nor lost).
    Tie,
    /// Mid-pack (3+ rows) or unrankable: no highlight.
    #[default]
    Plain,
}

/// Max-wins rank of `v` against the column range `[lo, hi]`.
pub fn rank(v: f64, lo: f64, hi: f64) -> StatRank {
    if lo == hi {
        StatRank::Tie
    } else if v == hi {
        StatRank::Best
    } else if v == lo {
        StatRank::Worst
    } else {
        StatRank::Plain
    }
}

/// Min-wins rank (lower = better): StdDev only. A tighter spread means a
/// more consistent encode, so FFMetrics-original colors the lower StdDev
/// green (e.g. 1.117 beats 1.326).
pub fn rank_low(v: f64, lo: f64, hi: f64) -> StatRank {
    if lo == hi {
        StatRank::Tie
    } else if v == lo {
        StatRank::Best
    } else if v == hi {
        StatRank::Worst
    } else {
        StatRank::Plain
    }
}

/// The comparable per-run statistics of one finished cell, in tooltip order.
#[derive(Debug, Clone)]
pub struct DoneStats {
    pub avg: f64,
    pub mean: f64,
    pub harm: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
    pub p1: f64,
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub exec_s: f64,
    pub frames: usize,
}

impl DoneStats {
    pub fn new(values: &[f64], avg: f64, exec_s: f64) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mut s = values.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        Some(Self {
            avg,
            mean: mean(values),
            harm: harm_mean(values),
            min: s[0],
            max: s[s.len() - 1],
            stddev: pstdev(values),
            p1: percentile(&s, 1.0),
            p5: percentile(&s, 5.0),
            p10: percentile(&s, 10.0),
            p25: percentile(&s, 25.0),
            exec_s,
            frames: values.len(),
        })
    }

    /// Comparable (label, value, lower_better) triples in tooltip order.
    /// Only StdDev is lower-better; everything else is max-wins.
    pub fn comparable(&self) -> [(&'static str, f64, bool); 10] {
        [
            ("Avg:", self.avg, false),
            ("Mean:", self.mean, false),
            ("Mean (harm):", self.harm, false),
            ("Min:", self.min, false),
            ("Max:", self.max, false),
            ("StdDev (pop):", self.stddev, true),
            ("Percentile 1:", self.p1, false),
            ("Percentile 5:", self.p5, false),
            ("Percentile 10:", self.p10, false),
            ("Percentile 25:", self.p25, false),
        ]
    }
}

/// `MM:SS.ss` rendering of an execution time (shared by text/grid tooltips).
pub fn format_exec(exec_s: f64) -> String {
    format!(
        "{:02}:{:05.2}",
        (exec_s / 60.0).floor() as i64,
        exec_s % 60.0
    )
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Harmonic mean of positive values only; 0.0 when there are none
/// (Python `statistics.harmonic_mean(positive) if positive else 0.0`).
pub fn harm_mean(values: &[f64]) -> f64 {
    let positive: Vec<f64> = values.iter().copied().filter(|v| *v > 0.0).collect();
    if positive.is_empty() {
        return 0.0;
    }
    positive.len() as f64 / positive.iter().map(|v| 1.0 / v).sum::<f64>()
}

/// Population standard deviation (Python `statistics.pstdev`).
pub fn pstdev(values: &[f64]) -> f64 {
    let m = mean(values);
    (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

/// Nearest-rank percentile over ascending-sorted values
/// (Python `sorted[ceil(pct / 100 * n) - 1]`).
pub fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    sorted[(pct / 100.0 * n as f64).ceil() as usize - 1]
}

/// Python `_stats_text`: `avg` is ffmpeg's summary when present, else the
/// arithmetic mean. `exec_s` renders as `MM:SS.ss`.
pub fn stats_text(avg: Option<f64>, exec_s: f64, values: &[f64]) -> String {
    let n = values.len();
    if n == 0 {
        return "No data".to_owned();
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let show = avg.unwrap_or_else(|| mean(values));
    let exec = format_exec(exec_s);
    let mut lines = vec![
        format!("Avg: {show:.6}"),
        format!("Exec time: {exec}"),
        format!("Frames count: {n}"),
        String::new(),
        "Frames statistics".to_owned(),
        format!("Mean: {:.6}", mean(values)),
        format!("Mean (harm): {:.6}", harm_mean(values)),
        format!("Min: {:.6}", s[0]),
        format!("Max: {:.6}", s[n - 1]),
        format!("StdDev (pop): {:.6}", pstdev(values)),
        String::new(),
    ];
    for p in [1.0, 5.0, 10.0, 25.0] {
        lines.push(format!("Percentile {}: {:.6}", p as i64, percentile(&s, p)));
    }
    lines.join("\n")
}

/// Python `_parse_time_spec` verbatim port: plain seconds (`"12.5"`) or
/// `mm:ss[.xxx]` / `hh:mm:ss[.xxx]`. Empty → `None` (no trim); anything
/// else unparseable → `None` (caller reports "bad time"). Quirks kept:
/// `-` anywhere rejects (`"1e-3"` invalid), heads must be integers
/// (`"1.5:02"` invalid), seconds may exceed 59 (`"1:75"` is 135s).
pub fn parse_time_spec(raw: &str) -> Option<f64> {
    let s = raw.trim();
    if s.is_empty() || s.contains('-') {
        return None;
    }
    if !s.contains(':') {
        return s.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0);
    }
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() > 3 || parts.iter().any(|p| p.trim().is_empty()) {
        return None;
    }
    let mut total = 0.0;
    for p in &parts[..parts.len() - 1] {
        let v: i64 = p.trim().parse().ok()?;
        if v < 0 {
            return None;
        }
        total = total * 60.0 + v as f64;
    }
    let last: f64 = parts[parts.len() - 1].trim().parse().ok()?;
    if !last.is_finite() || last < 0.0 {
        return None;
    }
    let total = total * 60.0 + last;
    total.is_finite().then_some(total)
}
#[cfg(test)]
#[path = "../tests/test_metrics_mod.rs"]
mod tests;
