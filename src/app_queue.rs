//! Queue table domain: rows, cached stats, sorting, drop routing.
//!
//! Extracted from `app.rs` (High 1 split, step 1). Channel-free and
//! UI-free except for tiny `egui` sort-mark helpers that stay in `app.rs`;
//! everything here is pure over `QueueRow`/slices so unit tests cover it
//! without a GUI harness.

use crate::metrics::ffmpeg::MetricKind;
use crate::metrics::ffmpeg::ScaleMethod;
use std::path::Path;

#[derive(Debug)]
pub(crate) struct QueueRow {
    pub(crate) path: String,
    /// `norm_key(path)` computed once at insert; `path` is never mutated
    /// after push, so worker-message routing compares this instead of
    /// re-normalizing (and re-hitting `current_dir()`) per row per message.
    pub(crate) key: String,
    pub(crate) display: String,
    pub(crate) include: bool,
    /// Permanent plot-color slot, assigned from `next_color_idx` at insert
    /// and never reused: hiding or removing one row never recolors the
    /// survivors. Session-only (reassigned 0..n in file order on load).
    pub(crate) color_idx: usize,
    /// Probe token, assigned from `next_probe_seq` at insert and never
    /// reused: `RowMedia` applies only on match, so removing a row and
    /// re-adding the same path can't let the old probe paint the new row.
    pub(crate) probe_gen: u64,
    pub(crate) selected: bool,
    pub(crate) media: String,
    pub(crate) media_tip: String,
    pub(crate) info: Option<crate::probe::MediaInfo>,
    pub(crate) psnr: crate::metrics::MetricCell,
    pub(crate) ssim: crate::metrics::MetricCell,
    pub(crate) vmaf: crate::metrics::MetricCell,
    pub(crate) xpsnr: crate::metrics::MetricCell,
    pub(crate) ssim2: crate::metrics::MetricCell,
    pub(crate) butter: crate::metrics::MetricCell,
    pub(crate) cvvdp: crate::metrics::MetricCell,
    pub(crate) psnr_cache: CachedStats,
    pub(crate) ssim_cache: CachedStats,
    pub(crate) vmaf_cache: CachedStats,
    pub(crate) xpsnr_cache: CachedStats,
    pub(crate) ssim2_cache: CachedStats,
    pub(crate) butter_cache: CachedStats,
    pub(crate) cvvdp_cache: CachedStats,
}

/// Cached per-row stats + cross-row ranks for one metric column (H1: the
/// values-vec clone+sort in `DoneStats::new` and the rank scan run on
/// result arrival, not per frame; the render loop only reads).
/// Plot lines decimate directly from the cell `values` (`x = i+1.0`) at
/// draw time, capped at ~8192 points — no full-res `PlotPoint` cache.
#[derive(Debug, Clone, Default)]
pub(crate) struct CachedStats {
    pub(crate) stats: Option<crate::metrics::DoneStats>,
    pub(crate) ranks: [crate::metrics::StatRank; 10],
    /// Rendered Done text (Avg at the Options Precision), frozen at Done arrival
    /// so the table loop never formats per frame. Cleared wherever `stats`
    /// is cleared (rerun start, Reset via wholesale `default()`).
    pub(crate) text: String,
    /// Wall-clock completion stamp (`%Y-%m-%d %H:%M:%S` local) for the
    /// results CSV `*-DateTime` columns; frozen with the rest, cleared
    /// with it.
    pub(crate) finished: Option<String>,
}

impl QueueRow {
    pub(crate) fn cell(&self, kind: MetricKind) -> &crate::metrics::MetricCell {
        match kind {
            MetricKind::Psnr => &self.psnr,
            MetricKind::Ssim => &self.ssim,
            MetricKind::Vmaf => &self.vmaf,
            MetricKind::Xpsnr => &self.xpsnr,
            MetricKind::Ssim2 => &self.ssim2,
            MetricKind::But => &self.butter,
            MetricKind::Cvvdp => &self.cvvdp,
        }
    }

    pub(crate) fn cell_mut(&mut self, kind: MetricKind) -> &mut crate::metrics::MetricCell {
        match kind {
            MetricKind::Psnr => &mut self.psnr,
            MetricKind::Ssim => &mut self.ssim,
            MetricKind::Vmaf => &mut self.vmaf,
            MetricKind::Xpsnr => &mut self.xpsnr,
            MetricKind::Ssim2 => &mut self.ssim2,
            MetricKind::But => &mut self.butter,
            MetricKind::Cvvdp => &mut self.cvvdp,
        }
    }

    pub(crate) fn cached(&self, kind: MetricKind) -> &CachedStats {
        match kind {
            MetricKind::Psnr => &self.psnr_cache,
            MetricKind::Ssim => &self.ssim_cache,
            MetricKind::Vmaf => &self.vmaf_cache,
            MetricKind::Xpsnr => &self.xpsnr_cache,
            MetricKind::Ssim2 => &self.ssim2_cache,
            MetricKind::But => &self.butter_cache,
            MetricKind::Cvvdp => &self.cvvdp_cache,
        }
    }

    pub(crate) fn cached_mut(&mut self, kind: MetricKind) -> &mut CachedStats {
        match kind {
            MetricKind::Psnr => &mut self.psnr_cache,
            MetricKind::Ssim => &mut self.ssim_cache,
            MetricKind::Vmaf => &mut self.vmaf_cache,
            MetricKind::Xpsnr => &mut self.xpsnr_cache,
            MetricKind::Ssim2 => &mut self.ssim2_cache,
            MetricKind::But => &mut self.butter_cache,
            MetricKind::Cvvdp => &mut self.cvvdp_cache,
        }
    }

    /// Drop a stale result cache when its cell stops being `Done`
    /// (pre-flight `Error`, Reset): rank/sort/tooltip readers trust the
    /// cache, so leaving it behind resurrects the old result.
    pub(crate) fn clear_cached(&mut self, kind: MetricKind) {
        *self.cached_mut(kind) = CachedStats::default();
    }
}

/// File picker extensions (FFMetrics.conf `VideoFilesList` parity).
pub(crate) const VIDEO_EXTS: &[&str] = &[
    "264", "avi", "avs", "h264", "hevc", "m2ts", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "mts",
    "mxf", "ts", "webm",
];

/// Purely lexical `.`/`..`/duplicate-separator resolution (the `normpath`
/// half of Python `abspath`): no filesystem access, no symlink resolution,
/// so pending-drop paths work and spellings stay stable. `pop` on an empty
/// or root path is a no-op, which clamps `..` at the filesystem root exactly
/// like `normpath` does for absolute inputs.
pub(crate) fn lexical_normalize(p: &Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut out = std::path::PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            _ => out.push(c.as_os_str()),
        }
    }
    out
}

/// Python `normcase(abspath)` equivalent for the same-file guard rail.
/// Windows: lowercase + `/` -> `\`. POSIX `normcase` is the identity, so
/// Unix keeps separators and case as-is (`foo/bar` vs `foo\bar` are distinct).
pub(crate) fn norm_key(p: &str) -> String {
    let path = Path::new(p);
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(path),
            Err(e) => {
                log::warn!(target: "rfmetrics::app", "current_dir failed ({e}); norm_key falling back to relative path for \"{p}\"");
                std::path::PathBuf::new().join(path)
            }
        }
    };
    let abs = lexical_normalize(&abs);
    #[cfg(windows)]
    {
        abs.to_string_lossy().replace('/', "\\").to_lowercase()
    }
    #[cfg(not(windows))]
    {
        abs.to_string_lossy().into_owned()
    }
}

/// Reveal a queued file in the OS file manager without blocking the UI.
/// Windows selects the file (`explorer /select,`); other platforms open the
/// containing folder (select-on-open has no portable equivalent).
/// Spawn-only: never waits on the child, so a slow Explorer can't freeze a frame.
pub(crate) fn reveal_in_explorer(path: &str) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::process::Command::new("explorer")
            .args(["/select,", path])
            .spawn()
            .map(|_| ())
    }
    #[cfg(not(windows))]
    {
        let target = Path::new(path)
            .parent()
            .map(|p| p.as_os_str().to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| path.to_owned());
        open::that(&target)
    }
}

/// Thumbnail seek duration: reuse the completed reference probe's
/// duration when it belongs to the current path; otherwise `None` and the
/// worker falls back to a dedicated probe (`media_duration`).
pub(crate) fn thumb_duration(
    ref_path: &str,
    ref_info_path: &str,
    probed: Option<f64>,
) -> Option<f64> {
    if !ref_path.is_empty() && ref_path == ref_info_path {
        probed.filter(|&d| d > 0.0)
    } else {
        None
    }
}

/// Shortest unique trailing-path suffix per entry (Python `_display_names`).
pub(crate) fn display_names(paths: &[String]) -> Vec<String> {
    let parts: Vec<Vec<String>> = paths
        .iter()
        .map(|p| {
            Path::new(p)
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect()
        })
        .collect();
    let sep = std::path::MAIN_SEPARATOR.to_string();
    parts
        .iter()
        .enumerate()
        .map(|(i, part)| {
            for n in 1..=part.len() {
                let cand = &part[part.len() - n..];
                let unique = parts.iter().enumerate().all(|(j, q)| {
                    j == i || {
                        let tail = if q.len() >= n {
                            &q[q.len() - n..]
                        } else {
                            &q[..]
                        };
                        tail != cand
                    }
                });
                if unique {
                    return cand.join(&sep);
                }
            }
            part.join(&sep)
        })
        .collect()
}

/// Alt+click solo/select-all for the first-column include checkboxes
/// (egui_plot legend parity): operates on the POST-toggle flags — the
/// single checkbox already flipped before this runs, and the other rows
/// are untouched, so "any other checked" is identical pre/post.
/// Others checked → isolate (only `idx` stays on); no others checked
/// (was all-off, or was solo on `idx`) → select all. Out-of-range `idx`
/// is a no-op. Runs only on discrete Alt+clicks, never per frame.
pub(crate) fn apply_alt_include(includes: &mut [bool], idx: usize) {
    if idx >= includes.len() {
        return;
    }
    if includes.iter().enumerate().any(|(j, &v)| j != idx && v) {
        for (j, v) in includes.iter_mut().enumerate() {
            *v = j == idx;
        }
    } else {
        for v in includes.iter_mut() {
            *v = true;
        }
    }
}

/// Shift+click range for the first-column include checkboxes
/// (Gmail-style): the closed `(lo, hi)` span between the anchor row and
/// the clicked row. The caller fills the span with the clicked box's
/// post-toggle value. `None` when the anchor or `idx` points past the
/// queue (stale anchor after row removal). Runs only on discrete
/// Shift+clicks, never per frame.
pub(crate) fn shift_include_range(len: usize, anchor: usize, idx: usize) -> Option<(usize, usize)> {
    if anchor >= len || idx >= len {
        return None;
    }
    Some((anchor.min(idx), anchor.max(idx)))
}

/// Sortable table columns: queue path + the 7 metric columns (checkbox,
/// play, and Media info columns stay unsorted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SortColumn {
    Path,
    Metric(MetricKind),
}

/// Rendered direction: first click lands the initial direction (best
/// first — ascending names, descending scores, ascending Butteraugli),
/// second click flips it, third click clears back to insertion order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SortDir {
    Asc,
    Desc,
}

impl SortDir {
    pub(crate) fn flipped(self) -> SortDir {
        match self {
            SortDir::Asc => SortDir::Desc,
            SortDir::Desc => SortDir::Asc,
        }
    }
}

/// First-click direction per column (best first).
pub(crate) fn initial_dir(col: SortColumn, stat: crate::metrics::CellStat) -> SortDir {
    use crate::metrics::CellStat;
    match col {
        SortColumn::Path => SortDir::Asc,
        // Butteraugli is lower-better on every stat, StdDev on every
        // metric (mirrors the rank logic).
        SortColumn::Metric(MetricKind::But) => SortDir::Asc,
        SortColumn::Metric(_) if stat == CellStat::StdDev => SortDir::Asc,
        SortColumn::Metric(_) => SortDir::Desc,
    }
}

/// Header-click cycle: new column starts at its initial direction, a
/// repeat click flips, a third click clears to insertion order.
pub(crate) fn cycle_sort(
    current: Option<(SortColumn, SortDir)>,
    col: SortColumn,
    stat: crate::metrics::CellStat,
) -> Option<(SortColumn, SortDir)> {
    match current {
        None => Some((col, initial_dir(col, stat))),
        Some((c, _)) if c != col => Some((col, initial_dir(col, stat))),
        Some((_, dir)) if dir == initial_dir(col, stat) => Some((col, dir.flipped())),
        Some(_) => None,
    }
}

/// Scored selector value for sorting; unscored cells (Idle/Running/Error)
/// sort after every scored row in both directions. Reads the
/// arrival-cached stats (O(1)); uncached `Done` cells only exist in
/// tests and compute from the values instead.
pub(crate) fn sort_stat(
    row: &QueueRow,
    kind: MetricKind,
    stat: crate::metrics::CellStat,
) -> Option<f64> {
    use crate::metrics::CellStat;
    if let Some(s) = &row.cached(kind).stats {
        return Some(s.value(stat));
    }
    match row.cell(kind) {
        crate::metrics::MetricCell::Done { values, avg, .. } if !values.is_empty() => {
            Some(match stat {
                CellStat::Avg => *avg,
                CellStat::Mean => crate::metrics::mean(values),
                CellStat::Harm => crate::metrics::harm_mean(values),
                CellStat::Min => values
                    .iter()
                    .copied()
                    .max_by(|a, b| a.total_cmp(b).reverse())?,
                CellStat::Max => values.iter().copied().max_by(|a, b| a.total_cmp(b))?,
                CellStat::StdDev => crate::metrics::pstdev(values),
                CellStat::P1 | CellStat::P5 | CellStat::P10 | CellStat::P25 => {
                    let mut s = values.to_vec();
                    s.sort_by(|a, b| a.total_cmp(b));
                    let pct = match stat {
                        CellStat::P1 => 1.0,
                        CellStat::P5 => 5.0,
                        CellStat::P10 => 10.0,
                        CellStat::P25 => 25.0,
                        _ => 1.0,
                    };
                    crate::metrics::percentile(&s, pct)
                }
            })
        }
        _ => None,
    }
}

pub(crate) fn cmp_rows(
    col: SortColumn,
    dir: SortDir,
    a: &QueueRow,
    b: &QueueRow,
    stat: crate::metrics::CellStat,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match col {
        SortColumn::Path => match dir {
            SortDir::Asc => a.display.cmp(&b.display),
            SortDir::Desc => b.display.cmp(&a.display),
        },
        SortColumn::Metric(kind) => match (sort_stat(a, kind, stat), sort_stat(b, kind, stat)) {
            (Some(x), Some(y)) => {
                let ord = x.total_cmp(&y);
                match dir {
                    SortDir::Asc => ord,
                    SortDir::Desc => ord.reverse(),
                }
            }
            // Scored rows always precede unscored ones, either direction.
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        },
    }
}

/// Display order as underlying row indices (identity when unsorted).
/// Stable sort, so ties keep insertion order. Rebuilt per frame while a
/// sort is active — trivial at queue sizes, and `None` skips it entirely.
/// Values come from the arrival-cached stats, so sorting stays O(1) per
/// comparison no matter which selector is active.
pub(crate) fn sort_view(
    rows: &[QueueRow],
    spec: Option<(SortColumn, SortDir)>,
    stat: crate::metrics::CellStat,
) -> Vec<usize> {
    let mut view: Vec<usize> = (0..rows.len()).collect();
    if let Some((col, dir)) = spec {
        view.sort_by(|&a, &b| cmp_rows(col, dir, &rows[a], &rows[b], stat));
    }
    view
}

/// Table metric-column layout, left to right — MUST match the header
/// checkbox order.
pub(crate) const METRIC_COLUMNS: [(Option<MetricKind>, &str); 7] = [
    (Some(MetricKind::Psnr), "PSNR"),
    (Some(MetricKind::Ssim), "SSIM"),
    (Some(MetricKind::Vmaf), "VMAF"),
    (Some(MetricKind::Xpsnr), "XPSNR"),
    (Some(MetricKind::Ssim2), "SSIM2"),
    (Some(MetricKind::But), "BUTTER"),
    (Some(MetricKind::Cvvdp), "CVVDP"),
];

/// Whether a `Done` cell's stamped options no longer match the current
/// settings — the same comparison `start_run` uses to decide recompute
/// vs. skip. Pure so the badge and the partition can never disagree.
/// Non-`Done` cells are never stale.
#[allow(clippy::too_many_arguments)]
pub(crate) fn done_is_stale(
    kind: MetricKind,
    cell: &crate::metrics::MetricCell,
    skip: Option<f64>,
    clip_dur: Option<f64>,
    vmaf_cfg: &crate::metrics::vmaf::VmafCfg,
    scaler: ScaleMethod,
    fps_mode: crate::metrics::ffmpeg::InputFpsMode,
    ref_pixfmt: crate::metrics::ffmpeg::RefPixFmt,
) -> bool {
    if let crate::metrics::MetricCell::Done {
        skip: s,
        clip_dur: c,
        vmaf_cfg: v,
        scaler: sc,
        fps_mode: fm,
        ref_pixfmt: pf,
        ..
    } = cell
    {
        !(*s == skip
            && *c == clip_dur
            && (kind != MetricKind::Vmaf || v.as_ref() == Some(vmaf_cfg))
            && (kind.is_ffvship() || *sc == scaler)
            && (kind.is_ffvship() || *fm == fps_mode)
            && (kind.is_ffvship() || *pf == ref_pixfmt))
    } else {
        false
    }
}

/// Drop routing decision: pure so the guard rails stay unit-tested.
/// Mid-run drops are `Blocked` (toast) — the worker snapshotted its jobs
/// at Start, so ref/queue changes must wait for Stop.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DropAction {
    Ignore,
    Blocked,
    SetRef {
        first: std::path::PathBuf,
        extra: usize,
    },
    Queue(Vec<std::path::PathBuf>),
}

pub(crate) fn route_drop(
    measuring: bool,
    is_over_ref: bool,
    is_over_table: bool,
    dropped: Vec<std::path::PathBuf>,
) -> DropAction {
    if dropped.is_empty() {
        return DropAction::Ignore;
    }
    if measuring {
        return DropAction::Blocked;
    }
    if is_over_ref {
        let mut iter = dropped.into_iter();
        // `dropped` is non-empty (checked above), so `first` exists.
        let first = iter.next().unwrap_or_default();
        let extra = iter.len();
        DropAction::SetRef { first, extra }
    } else if is_over_table {
        DropAction::Queue(dropped)
    } else {
        DropAction::Ignore
    }
}

impl crate::app::RFMetricsApp {
    pub(crate) fn refresh_ranks(&mut self, kind: MetricKind) {
        let mut stat_lo = [f64::INFINITY; 10];
        let mut stat_hi = [f64::NEG_INFINITY; 10];
        let mut scored = 0usize;
        for row in &self.rows {
            if let Some(s) = &row.cached(kind).stats {
                scored += 1;
                for (k, (_, v, _)) in s.comparable().iter().enumerate() {
                    stat_lo[k] = stat_lo[k].min(*v);
                    stat_hi[k] = stat_hi[k].max(*v);
                }
            }
        }
        for row in &mut self.rows {
            let cached = row.cached_mut(kind);
            let mut ranks = [crate::metrics::StatRank::Plain; 10];
            if scored >= 2
                && let Some(s) = &cached.stats
            {
                let comp = s.comparable();
                for k in 0..10 {
                    let (_, v, lower_better) = comp[k];
                    // BUTTERAUGLI is lower-is-better on every stat (Python
                    // "lower is better, min 0"); StdDev already is.
                    ranks[k] = if lower_better || kind == MetricKind::But {
                        crate::metrics::rank_low(v, stat_lo[k], stat_hi[k])
                    } else {
                        crate::metrics::rank(v, stat_lo[k], stat_hi[k])
                    };
                }
            }
            cached.ranks = ranks;
        }
    }
}
