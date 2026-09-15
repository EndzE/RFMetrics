use std::path::{Path, PathBuf};
use std::process::Command;

use regex::Regex;

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
}

/// Directory holding the running executable (Rust analog of Python's script dir).
pub(crate) fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

fn nearby(name: &str) -> Option<PathBuf> {
    let cand = exe_dir()?.join(name);
    if cand.is_file() { Some(cand) } else { None }
}

fn find_ffmpeg() -> (Option<PathBuf>, &'static str) {
    // ponytail: near-exe preferred over PATH per user requirement
    if let Some(p) = nearby("ffmpeg.exe") {
        return (Some(p), "next to app");
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

// ponytail: no timeout on std Command; one local spawn at startup is ~ms
fn run_version(exe: &Path, arg: &str) -> Option<VersionOutput> {
    let start = std::time::Instant::now();
    log::debug!(target: "rfmetrics::binaries", "run: \"{}\" {arg}", exe.display());
    let out = match Command::new(exe).arg(arg).output() {
        Ok(o) => o,
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
        };
    };
    let missing = |msg: String| BinaryInfo {
        path: Some(exe.clone()),
        origin,
        short: "ffmpeg not found in PATH".to_owned(),
        detail: msg,
        usable: false,
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
    let short = Regex::new(r"(?i)ffmpeg version (\d+(?:\.\d+)*)")
        .ok()
        .and_then(|re| re.captures(full).map(|c| format!("FFmpeg: {}", &c[1])))
        .unwrap_or_else(|| full.to_owned());
    // `ffmpeg -version` puts "Copyright (c) ..." on the same first line;
    // strip it so the hover tooltip stays to version + path.
    let full = Regex::new(r"(?i)\s*copyright.*$")
        .map(|re| re.replace(full, "").trim_end().to_owned())
        .unwrap_or_else(|_| full.to_owned());
    log::info!(target: "rfmetrics::binaries", "ffmpeg: {short} at {} ({origin})", exe.display());
    BinaryInfo {
        path: Some(exe.clone()),
        origin,
        short,
        detail: format!("{full}\n{} ({origin})", exe.display()),
        usable: true,
    }
}

/// Pure version-token parser over stdout+stderr. `Some` = usable version
/// string (`ver` or `ver_backend`); `None` = binary gave nothing parseable
/// (e.g. wrong-GPU build exiting silently).
fn parse_ffvship_version(stdout: &str, stderr: &str) -> Option<String> {
    let ver_re = Regex::new(r"(?i)FFVship\s+([^\s\r\n]+)").ok();
    let back_re = Regex::new(r"(?i)^([a-zA-Z0-9]+)\s+version\s*$").ok();
    let mut ver: Option<String> = None;
    let mut backend: Option<String> = None;
    for line in format!("{stdout}\n{stderr}").lines() {
        let s = line.trim();
        if ver.is_none()
            && let Some(re) = &ver_re
            && let Some(c) = re.captures(s)
        {
            ver = Some(c[1].to_owned());
        } else if backend.is_none()
            && let Some(re) = &back_re
            && let Some(c) = re.captures(s)
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
        };
    };
    let missing = |msg: String| BinaryInfo {
        path: Some(exe.clone()),
        origin,
        short: "FFVship: not found".to_owned(),
        detail: msg,
        usable: false,
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
mod tests {
    use super::parse_ffvship_version;

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
}
