pub mod ffmpeg;

/// Per-cell lifecycle of one metric on one queue row (Python `results` dict
/// entry + label text rolled into one).
#[derive(Debug, Clone, Default)]
pub enum MetricCell {
    #[default]
    Idle,
    Running {
        frame: u64,
    },
    Done {
        values: Vec<f64>,
        avg: f64,
        exec_s: f64,
        /// Trim settings the run used; a rerun under different skip/clip
        /// must recompute instead of trusting this value.
        skip: Option<f64>,
        clip_dur: Option<f64>,
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
            Self::Running { frame } => format!("Frame: {frame}"),
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
mod tests {
    use super::*;

    #[test]
    fn stats_match_python_shape() {
        let v = vec![28.1, 33.97, 45.2, 30.2, 32.4];
        let t = stats_text(Some(33.977_123), 12.34, &v);
        assert!(t.starts_with("Avg: 33.977123\nExec time: 00:12.34\nFrames count: 5\n"));
        assert!(t.contains("\nFrames statistics\nMean: "));
        assert!(t.contains("Min: 28.100000\nMax: 45.200000\n"));
        assert!(t.ends_with("Percentile 25: 30.200000"));
    }

    #[test]
    fn stats_empty_and_fallback_avg() {
        assert_eq!(stats_text(None, 0.0, &[]), "No data");
        let t = stats_text(None, 61.5, &[2.0, 4.0]);
        assert!(t.starts_with("Avg: 3.000000\nExec time: 01:01.50\n"));
        assert!(t.contains("Mean (harm): 2.666667"));
    }

    #[test]
    fn harm_ignores_non_positive() {
        assert_eq!(harm_mean(&[]), 0.0);
        assert_eq!(harm_mean(&[0.0, -1.0]), 0.0);
        assert!((harm_mean(&[1.0, 2.0, 4.0]) - 12.0 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_is_ceil_rank() {
        let s: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        assert_eq!(percentile(&s, 1.0), 1.0);
        assert_eq!(percentile(&s, 25.0), 25.0);
        let s5: Vec<f64> = vec![28.1, 30.2, 32.4, 33.97, 45.2];
        assert_eq!(percentile(&s5, 1.0), 28.1);
        assert_eq!(percentile(&s5, 25.0), 30.2);
    }

    #[test]
    fn time_specs() {
        assert_eq!(parse_time_spec(""), None);
        assert_eq!(parse_time_spec("12.5"), Some(12.5));
        assert_eq!(parse_time_spec(" 90 "), Some(90.0));
        assert_eq!(parse_time_spec("01:02.5"), Some(62.5));
        assert_eq!(parse_time_spec("1:02:03"), Some(3723.0));
        assert_eq!(parse_time_spec("00:00.000"), Some(0.0));
        assert_eq!(parse_time_spec("-5"), None);
        assert_eq!(parse_time_spec("abc"), None);
        assert_eq!(parse_time_spec("1:2:3:4"), None);
        assert_eq!(parse_time_spec("inf"), None);
        assert_eq!(parse_time_spec("1:"), None);
        // Python-parity quirks:
        assert_eq!(parse_time_spec("1e-3"), None); // `-` anywhere rejects
        assert_eq!(parse_time_spec("1.5:02"), None); // heads must be integers
        assert_eq!(parse_time_spec("1:75"), Some(135.0)); // seconds may exceed 59
        assert_eq!(parse_time_spec("1: 02"), Some(62.0)); // inner space tolerated
        assert_eq!(parse_time_spec("0:75"), Some(75.0));
    }

    #[test]
    fn rank_rules() {
        use super::StatRank;
        assert_eq!(rank(48.0, 46.0, 48.0), StatRank::Best);
        assert_eq!(rank(46.0, 46.0, 48.0), StatRank::Worst);
        assert_eq!(rank(47.0, 46.0, 48.0), StatRank::Plain);
        // Single row / all equal: tie, never best-vs-worst.
        assert_eq!(rank(30.0, 30.0, 30.0), StatRank::Tie);
        // NaN compares equal to nothing: no highlight.
        assert_eq!(rank(f64::NAN, 1.0, 2.0), StatRank::Plain);
    }

    #[test]
    fn rank_low_inverts_best_worst() {
        use super::StatRank;
        // FFMetrics-original: StdDev 1.117 beats 1.326 (lower = greener).
        assert_eq!(rank_low(1.117, 1.117, 1.326), StatRank::Best);
        assert_eq!(rank_low(1.326, 1.117, 1.326), StatRank::Worst);
        assert_eq!(rank_low(1.2, 1.117, 1.326), StatRank::Plain);
        assert_eq!(rank_low(1.2, 1.2, 1.2), StatRank::Tie);
    }

    #[test]
    fn done_stats_shape() {
        let s = DoneStats::new(&[28.0, 30.0, 32.0], 30.5, 61.5).unwrap();
        assert_eq!(s.frames, 3);
        assert_eq!(s.avg, 30.5);
        assert_eq!(s.comparable().len(), 10);
        assert_eq!(s.comparable()[0], ("Avg:", 30.5, false));
        assert!(s.comparable()[5].2); // only StdDev is lower-better
        assert_eq!(format_exec(s.exec_s), "01:01.50");
        assert!(DoneStats::new(&[], 0.0, 0.0).is_none());
    }

    #[test]
    fn cell_text_and_tooltip() {
        assert_eq!(MetricCell::Idle.cell_text(), "N/A");
        assert_eq!(MetricCell::Running { frame: 7 }.cell_text(), "Frame: 7");
        assert_eq!(
            MetricCell::Done {
                values: vec![30.0],
                avg: 30.123_456,
                exec_s: 1.0,
                skip: None,
                clip_dur: None,
            }
            .cell_text(),
            "30.1235"
        );
        assert!(
            MetricCell::Done {
                values: vec![30.0],
                avg: 30.0,
                exec_s: 1.0,
                skip: None,
                clip_dur: None,
            }
            .tooltip("PSNR")
            .starts_with("PSNR\nAvg: 30.000000")
        );
        assert_eq!(
            MetricCell::Error {
                msg: "probe failed".to_owned()
            }
            .tooltip("PSNR"),
            "Error: probe failed"
        );
    }
}
