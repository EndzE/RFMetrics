use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use regex::Regex;

use crate::metrics::ffmpeg::MetricKind;

/// Compiled once (startup/version probes), not per call — same `OnceLock`
/// pattern as the metric parsers in `metrics/ffmpeg.rs`. Literals below
/// are proven-valid (they compile on every current call), hence `unwrap`.
fn ffmpeg_ver_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // First whitespace-delimited token after `ffmpeg version`: dotted
    // releases (`7.1.1`), git snapshots (`git-2020-08-31-4a11a6f`) and
    // nightly builds (`N-126626-g7070fe638e-20260917`).
    RE.get_or_init(|| Regex::new(r"(?i)ffmpeg version (\S+)").unwrap())
}

fn copyright_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\s*copyright.*$").unwrap())
}

/// Bottom-bar label from a copyright-stripped first `-version` line:
/// `FFmpeg: <first version token>`. Unknown shapes fall back to the whole
/// line (already stripped, so the copyright tail can never leak in).
fn short_ffmpeg_version(first_line: &str) -> String {
    ffmpeg_ver_re()
        .captures(first_line)
        .map(|c| {
            // Vendor build-domain suffix (`9.0.1-full_build-www.gyan.dev`):
            // drop the domain, keep the variant (`9.0.1-full`).
            let raw = &c[1];
            let ver = raw.split_once("_build-www.").map_or(raw, |(head, _)| head);
            format!("FFmpeg: {ver}")
        })
        .unwrap_or_else(|| first_line.to_owned())
}

fn ffvship_ver_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)FFVship\s+([^\s\r\n]+)").unwrap())
}

fn ffvship_back_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^([a-zA-Z0-9]+)\s+version\s*$").unwrap())
}

#[derive(Debug, Clone)]
pub struct BinaryInfo {
    pub path: Option<PathBuf>,
    #[allow(dead_code)]
    pub origin: &'static str,
    pub short: String,
    pub detail: String,
    /// False when the binary is absent or its `--version` gave nothing
    /// parseable (e.g. wrong-GPU FFVship build). Measure step gates on this.
    #[allow(dead_code)]
    pub usable: bool,
    /// ffmpeg filter-backed metrics this build can run (`-filters` probe).
    /// Meaningless for FFVship (always empty); fail-open to all four when
    /// the probe itself won't run (see `ffmpeg_supported_filters`).
    pub supported_metrics: Vec<MetricKind>,
    /// Raw `ffmpeg version` token (`9.0.1-full_build-www.gyan.dev`):
    /// the results CSV keeps the vendor suffix the bottom-bar `short`
    /// strips. `None` when ffmpeg is missing/unusable (FFVship: always).
    pub ffmpeg_version: Option<String>,
}

/// Directory holding the running executable (Rust analog of Python's script dir).
pub(crate) fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

/// Writable app dir with Python `app_dir` parity: exe dir, then cwd,
/// then temp dir when the exe dir is unresolvable.
pub(crate) fn app_dir() -> PathBuf {
    exe_dir()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(std::env::temp_dir)
}

fn nearby(name: &str) -> Option<PathBuf> {
    let cand = exe_dir()?.join(name);
    if cand.is_file() { Some(cand) } else { None }
}

/// Bundled-suite layout (`<exe_dir>/ffmpeg/ffmpeg.exe`, …): keeps a full
/// extracted ffmpeg release next to the app without cluttering its folder
/// (mirrors the `FFVship/` subfolder convention below).
fn nearby_in(dir: &str, name: &str) -> Option<PathBuf> {
    let cand = exe_dir()?.join(dir).join(name);
    if cand.is_file() { Some(cand) } else { None }
}

fn find_ffmpeg() -> (Option<PathBuf>, &'static str) {
    // ponytail: near-exe preferred over PATH per user requirement
    if let Some(p) = nearby("ffmpeg.exe") {
        return (Some(p), "next to app");
    }
    if let Some(p) = nearby_in("ffmpeg", "ffmpeg.exe") {
        return (Some(p), "in ffmpeg folder");
    }
    if let Ok(p) = which::which("ffmpeg") {
        return (Some(p), "in PATH");
    }
    (None, "")
}

fn find_ffvship() -> (Option<PathBuf>, &'static str) {
    if let Some(dir) = exe_dir() {
        let sub = dir.join("FFVship").join("FFVship.exe");
        if sub.is_file() {
            return (Some(sub), "in FFVship folder");
        }
    }
    if let Some(p) = nearby("FFVship.exe") {
        return (Some(p), "next to app");
    }
    if let Ok(p) = which::which("ffvship") {
        return (Some(p), "in PATH");
    }
    (None, "")
}

fn find_ffprobe(ffmpeg_path: Option<&Path>) -> (Option<PathBuf>, &'static str) {
    if let Some(p) = nearby("ffprobe.exe") {
        return (Some(p), "next to app");
    }
    if let Some(p) = nearby_in("ffmpeg", "ffprobe.exe") {
        return (Some(p), "in ffmpeg folder");
    }
    if let Some(dir) = ffmpeg_path.and_then(|p| p.parent()) {
        let cand = dir.join("ffprobe.exe");
        if cand.is_file() {
            return (Some(cand), "next to ffmpeg");
        }
    }
    if let Ok(p) = which::which("ffprobe") {
        return (Some(p), "in PATH");
    }
    (None, "")
}

/// Captured `--version` probe. stderr matters: a wrong-GPU FFVship build
/// may exit silently on stdout while reporting the backend error on stderr.
struct VersionOutput {
    stdout: String,
    stderr: String,
    code: Option<i32>,
}

// Bounded version probe: a hung `--version` (wedged exe, AV stall) must
// never freeze startup — 5 s, then `None` like a spawn failure.
fn run_version(exe: &Path, arg: &str) -> Option<VersionOutput> {
    let start = std::time::Instant::now();
    log::debug!(target: "rfmetrics::binaries", "run: \"{}\" {arg}", exe.display());
    let mut cmd = Command::new(exe);
    cmd.arg(arg);
    let out = match crate::cmd::output_timeout(cmd, crate::cmd::VERSION_TIMEOUT) {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
            log::warn!(target: "rfmetrics::binaries", "run: \"{}\" {arg} timed out after {:?} — treating as missing", exe.display(), crate::cmd::VERSION_TIMEOUT);
            return None;
        }
        Err(e) => {
            log::warn!(target: "rfmetrics::binaries", "run: \"{}\" {arg} spawn failed: {e}", exe.display());
            return None;
        }
    };
    let v = VersionOutput {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        code: out.status.code(),
    };
    log::info!(
        target: "rfmetrics::binaries",
        "run: \"{}\" {arg} → exit {} ({}ms){}",
        exe.display(),
        exit_str(v.code),
        start.elapsed().as_millis(),
        stderr_snippet(&v.stderr),
    );
    Some(v)
}

fn exit_str(code: Option<i32>) -> String {
    code.map_or_else(|| "unknown".to_owned(), |c| c.to_string())
}

/// First non-empty trimmed line, preferring stdout (Python parity).
fn first_line(stdout: &str, stderr: &str) -> String {
    stdout
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .or_else(|| stderr.lines().map(str::trim).find(|l| !l.is_empty()))
        .unwrap_or("")
        .to_owned()
}

/// Stderr snippet for hover details (capped so a chatty binary can't flood it).
fn stderr_snippet(stderr: &str) -> String {
    let s = stderr.trim();
    if s.is_empty() {
        return String::new();
    }
    let mut snippet: String = s.chars().take(300).collect();
    if s.chars().count() > 300 {
        snippet.push('…');
    }
    format!("\nstderr: {snippet}")
}

/// ffmpeg-backed metric kinds, in `MetricKind::ALL` order (FFVship kinds
/// have no ffmpeg filter and never appear here).
fn ffmpeg_kinds() -> impl Iterator<Item = MetricKind> {
    MetricKind::ALL.into_iter().filter(|k| !k.is_ffvship())
}

/// Pure `ffmpeg -filters` listing parser: a filter counts only as the exact
/// second whitespace token (` TSC libvmaf …`), so `--enable-libvmaf`
/// configure lines never false-positive.
fn parse_filters_list(text: &str) -> Vec<MetricKind> {
    use std::collections::HashSet;
    let present: HashSet<&str> = text
        .lines()
        .filter_map(|l| l.split_whitespace().nth(1))
        .collect();
    ffmpeg_kinds()
        .filter(|k| present.contains(k.filter()))
        .collect()
}

/// Bottom-bar hover line for the ffmpeg tooltip (`Supported: …`).
fn supported_line(kinds: &[MetricKind]) -> String {
    if kinds.is_empty() {
        return "Supported: none".to_owned();
    }
    let names: Vec<&str> = kinds.iter().map(|k| k.name()).collect();
    format!("Supported: {}", names.join(", "))
}

/// Startup capability probe: which metric filters this ffmpeg build has
/// (issue #7: w32threads builds report `--enable-libvmaf` yet ship no
/// `libvmaf` filter). Fail-open to all four + warn when the probe itself
/// won't run, so a transient spawn failure can't brick the checkboxes;
/// genuine absences list honestly.
pub fn ffmpeg_supported_filters(exe: &Path) -> Vec<MetricKind> {
    let all: Vec<MetricKind> = ffmpeg_kinds().collect();
    // Hermetic tests: host ffmpeg capability must not leak into
    // assertions (same rationale as the state-load skip in `Default`).
    // Gating tests pin `supported_metrics` explicitly instead.
    if cfg!(test) {
        return all;
    }
    let Some(v) = run_version(exe, "-filters") else {
        log::warn!(target: "rfmetrics::binaries", "filter probe failed, assuming all filters present");
        return all;
    };
    parse_filters_list(&v.stdout)
}

pub fn ffmpeg_info() -> BinaryInfo {
    let (path, origin) = find_ffmpeg();
    let Some(exe) = path.clone() else {
        log::warn!(target: "rfmetrics::binaries", "ffmpeg not found (checked next to app and PATH)");
        return BinaryInfo {
            path: None,
            origin: "",
            short: "ffmpeg not found in PATH".to_owned(),
            detail: "Checked next to app and in PATH, none found".to_owned(),
            usable: false,
            supported_metrics: Vec::new(),
            ffmpeg_version: None,
        };
    };
    let missing = |msg: String| BinaryInfo {
        path: Some(exe.clone()),
        origin,
        short: "ffmpeg not found in PATH".to_owned(),
        detail: msg,
        usable: false,
        supported_metrics: Vec::new(),
        ffmpeg_version: None,
    };
    let Some(v) = run_version(&exe, "-version") else {
        log::warn!(target: "rfmetrics::binaries", "ffmpeg at {} ({origin}) failed to run", exe.display());
        return missing(format!(
            "Found at {} ({origin}) but failed to run",
            exe.display()
        ));
    };
    let full = v
        .stdout
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    if full.is_empty() {
        log::warn!(target: "rfmetrics::binaries", "ffmpeg at {} ({origin}) gave no version output (exit {})", exe.display(), exit_str(v.code));
        return missing(format!(
            "Found at {} ({origin}) but got no version output (exit {}){}",
            exe.display(),
            exit_str(v.code),
            stderr_snippet(&v.stderr),
        ));
    }
    // `ffmpeg -version` puts "Copyright (c) ..." on the same first line;
    // strip it BEFORE the short label: nightly `N-…` tokens previously
    // missed the release-only regex and the fallback leaked the whole
    // copyright tail into the bottom bar.
    let full = copyright_re().replace(full, "").trim_end().to_owned();
    let short = short_ffmpeg_version(&full);
    let version = ffmpeg_version_token(&full);
    log::info!(target: "rfmetrics::binaries", "ffmpeg: {short} at {} ({origin})", exe.display());
    let supported = ffmpeg_supported_filters(&exe);
    let line = supported_line(&supported);
    log::info!(target: "rfmetrics::binaries", "ffmpeg filters: {line}");
    BinaryInfo {
        path: Some(exe.clone()),
        origin,
        short,
        detail: format!("{full}\n{} ({origin})\n{line}", exe.display()),
        usable: true,
        supported_metrics: supported,
        ffmpeg_version: version,
    }
}

/// Raw `ffmpeg version` token (`9.0.1-full_build-www.gyan.dev`,
/// `N-126626-g7070fe638e-20260917`, …) from a copyright-stripped first
/// `-version` line. The results CSV keeps it verbatim (vendor suffix
/// included); the bottom bar shows the stripped `short` form instead.
fn ffmpeg_version_token(first_line: &str) -> Option<String> {
    ffmpeg_ver_re()
        .captures(first_line)
        .map(|c| c[1].to_owned())
}

/// Pure version-token parser over stdout+stderr. `Some` = usable version
/// string (`ver` or `ver_backend`); `None` = binary gave nothing parseable
/// (e.g. wrong-GPU build exiting silently).
fn parse_ffvship_version(stdout: &str, stderr: &str) -> Option<String> {
    let mut ver: Option<String> = None;
    let mut backend: Option<String> = None;
    for line in format!("{stdout}\n{stderr}").lines() {
        let s = line.trim();
        if ver.is_none()
            && let Some(c) = ffvship_ver_re().captures(s)
        {
            ver = Some(c[1].to_owned());
        } else if backend.is_none()
            && let Some(c) = ffvship_back_re().captures(s)
        {
            let word = c[1].to_lowercase();
            if word != "libvship" && word != "ffvship" {
                backend = Some(word);
            }
        }
    }
    match (ver, backend) {
        (Some(v), Some(b)) => Some(format!("{v}_{b}")),
        (Some(v), None) => Some(v),
        (None, _) => None,
    }
}

pub fn ffvship_info() -> BinaryInfo {
    let (path, origin) = find_ffvship();
    let Some(exe) = path.clone() else {
        log::warn!(target: "rfmetrics::binaries", "FFVship not found (checked next to app, FFVship folder, PATH)");
        return BinaryInfo {
            path: None,
            origin: "",
            short: "FFVship: not found".to_owned(),
            detail: "Checked next to app, FFVship folder, and in PATH, none found".to_owned(),
            usable: false,
            // Filter support is an ffmpeg concept; always empty here.
            supported_metrics: Vec::new(),
            ffmpeg_version: None,
        };
    };
    let missing = |msg: String| BinaryInfo {
        path: Some(exe.clone()),
        origin,
        short: "FFVship: not found".to_owned(),
        detail: msg,
        usable: false,
        supported_metrics: Vec::new(),
        ffmpeg_version: None,
    };
    let Some(v) = run_version(&exe, "--version") else {
        log::warn!(target: "rfmetrics::binaries", "FFVship at {} ({origin}) failed to run", exe.display());
        return missing(format!(
            "Found at {} ({origin}) but failed to run",
            exe.display()
        ));
    };
    if let Some(vs) = parse_ffvship_version(&v.stdout, &v.stderr) {
        let raw = first_line(&v.stdout, &v.stderr);
        log::info!(target: "rfmetrics::binaries", "FFVship: {vs} at {} ({origin})", exe.display());
        return BinaryInfo {
            path: Some(exe.clone()),
            origin,
            short: format!("FFVship: {vs}"),
            detail: format!("{raw}\n{} ({origin})", exe.display()),
            usable: true,
            supported_metrics: Vec::new(),
            ffmpeg_version: None,
        };
    }
    // Guard rail: binary persists but reports no version (e.g. AMD build on
    // an NVIDIA system). Warn instead of claiming "not found"; keep the path
    // so the failure is diagnosable. Measure step must gate on `usable`.
    let spoke = first_line(&v.stdout, &v.stderr);
    let mut detail = format!(
        "Found at {} ({origin}) but --version gave no version (exit {}); \
         possible GPU-build mismatch (e.g. AMD build on NVIDIA system)",
        exe.display(),
        exit_str(v.code),
    );
    if spoke.is_empty() {
        detail.push_str(&stderr_snippet(&v.stderr));
    } else {
        detail.push_str(&format!("\noutput: {spoke}"));
        detail.push_str(&stderr_snippet(&v.stderr));
    }
    log::warn!(target: "rfmetrics::binaries", "FFVship at {} ({origin}) gave no version (exit {})", exe.display(), exit_str(v.code));
    BinaryInfo {
        path: Some(exe.clone()),
        origin,
        short: "FFVship: found (version unknown)".to_owned(),
        detail,
        usable: false,
        supported_metrics: Vec::new(),
        ffmpeg_version: None,
    }
}

/// ffprobe has no bottom-bar label; path is held for the later probe step.
pub fn ffprobe_path(ffmpeg_path: Option<&Path>) -> Option<PathBuf> {
    let (path, origin) = find_ffprobe(ffmpeg_path);
    match &path {
        Some(p) => {
            log::info!(target: "rfmetrics::binaries", "ffprobe at {} ({origin})", p.display())
        }
        None => log::warn!(target: "rfmetrics::binaries", "ffprobe not found"),
    }
    path
}
#[cfg(test)]
#[path = "tests/test_binaries.rs"]
mod tests;
