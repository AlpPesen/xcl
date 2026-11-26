//! Where the title's threads spend their time in XCL: a `profile (2 s)` line beside the
//! `activity (2 s)` one (`XCL_ACTIVITY`). A console title's calls into the graphics driver take
//! next to no time; here a call can wait for a pipeline still being built, for XCL's uploads and
//! tile mappings before a submission, or for the GPU. Each kind of wait is timed where it happens
//! (`span`), summed over all the threads, so a frame rate near 0 shows what holds it: the line
//! gives each kind's time in the 2 s (over 2 000 ms: several threads at once), its calls and the
//! longest one. A wait counts when it ends; one that has not ended yet (a draw 36 s into waiting
//! for its pipeline) is given as still going, with its time so far.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub enum Kind {
    /// a draw waiting for its graphics pipeline (`pso_async::resolve_wait_in`)
    GraphicsPipelineWait,
    /// a dispatch or a submission waiting for a compute pipeline, its build on the title's
    /// thread included (`pso_async`)
    ComputePipelineWait,
    /// `ExecuteCommandLists` / `KickoffX`: XCL's uploads and tile mappings, then the driver's
    Submit,
    /// `CreatePlacedResourceX`: the PC's reserved resource and its tile mappings
    Placement,
    /// `WaitFrameEventX`: the title waiting for its frame pacing (the GPU, a flip)
    FrameEvent,
    /// `PresentX`
    Present,
    /// the driver compiling a pipeline, on any thread (workers too)
    DriverCompile,
    /// in a submit: the write-watch look for the pages the CPU wrote (`title_textures::refresh_once`)
    SubmitWatch,
    /// in a submit: those pages copied to the sparse windows' video memory (`sparse::before_execute`)
    SubmitSparse,
    /// in a submit: textures in title memory uploaded again (`title_textures::upload_all`)
    SubmitTextures,
    /// in a submit: the title's lists expanded into their PC segments (put-off dispatches recorded)
    SubmitRecord,
    /// in a submit: the PC `ExecuteCommandLists` (the driver)
    SubmitDriver,
    /// in a submit: after the driver's call (scratch copies, Write64 signals, readbacks, CPU writes)
    SubmitAfter,
}

const KINDS: usize = 13;
const NAMES: [&str; KINDS] = [
    "graphics pipeline waits",
    "compute pipeline waits",
    "submits",
    "placements",
    "frame-event waits",
    "presents",
    "driver compiles (all threads)",
    "[submit: write-watch",
    "sparse copies",
    "texture uploads",
    "list segments",
    "driver",
    "after]",
];

static NS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static CALLS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
/// the longest single one since the last line
static LONGEST: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];

/// Adds `d` of `kind`.
pub fn add(kind: Kind, d: Duration) {
    let k = kind as usize;
    let ns = d.as_nanos().min(u64::MAX as u128) as u64;
    NS[k].fetch_add(ns, Ordering::Relaxed);
    CALLS[k].fetch_add(1, Ordering::Relaxed);
    LONGEST[k].fetch_max(ns, Ordering::Relaxed);
}

/// spans of each kind not dropped yet, and the sum of their starts (`now_ns`, wrapping)
static OPEN: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static OPEN_STARTS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
/// spans still going are in the line once they took this long in all (a frame's own short
/// waits, caught open as the line is written, are not)
const OPEN_SHOWN_NS: u64 = 500_000_000;

/// Nanoseconds since the first span.
fn now_ns() -> u64 {
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos().min(u64::MAX as u128) as u64
}

/// Times from now to its drop as `kind`.
pub struct Span(Kind, u64);

pub fn span(kind: Kind) -> Span {
    let at = now_ns();
    let k = kind as usize;
    OPEN_STARTS[k].fetch_add(at, Ordering::Relaxed);
    OPEN[k].fetch_add(1, Ordering::Relaxed);
    Span(kind, at)
}

impl Span {
    /// The time since it started.
    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(now_ns().saturating_sub(self.1))
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let k = self.0 as usize;
        OPEN[k].fetch_sub(1, Ordering::Relaxed);
        OPEN_STARTS[k].fetch_sub(self.1, Ordering::Relaxed);
        add(self.0, self.elapsed());
    }
}

/// The spans of kind `k` still going: how many, and their time so far in all. The two counters
/// are read one after the other while other threads change them: a pair that cannot be (more
/// time than every span running since the first) is given as none.
fn still_going(k: usize) -> (u64, u64) {
    let (n, starts, now) = (OPEN[k].load(Ordering::Relaxed), OPEN_STARTS[k].load(Ordering::Relaxed), now_ns());
    let ns = n.wrapping_mul(now).wrapping_sub(starts);
    if n == 0 || n > 1 << 20 || ns > n.saturating_mul(now) { (0, 0) } else { (n, ns) }
}

/// ` ; N still going, M ms so far` for a kind's part of the line (nothing below `OPEN_SHOWN_NS`).
fn still_going_note((n, ns): (u64, u64)) -> String {
    if n == 0 || ns < OPEN_SHOWN_NS { String::new() } else { format!("; {n} still going, {} ms so far", ns / 1_000_000) }
}

/// Totals so far: (ns, calls) of each kind.
#[derive(Clone, Copy)]
pub struct Totals([(u64, u64); KINDS]);

pub fn totals() -> Totals {
    Totals(std::array::from_fn(|k| (NS[k].load(Ordering::Relaxed), CALLS[k].load(Ordering::Relaxed))))
}

/// The line for what was added between `before` and `now` (the longest ones start again).
pub fn line(before: Totals, now: Totals, presents: u64, secs: f64) -> String {
    let mut parts = Vec::new();
    for k in 0..KINDS {
        let ns = now.0[k].0.saturating_sub(before.0[k].0);
        let calls = now.0[k].1.saturating_sub(before.0[k].1);
        let longest = LONGEST[k].swap(0, Ordering::Relaxed);
        let going = still_going_note(still_going(k));
        if calls == 0 && going.is_empty() {
            continue;
        }
        parts.push(format!("{} {} ms ({calls}, longest {} ms{going})", NAMES[k], ns / 1_000_000, longest / 1_000_000));
    }
    let frame = if presents > 0 { format!("{:.0} ms a frame", secs * 1000.0 / presents as f64) } else { "no frame".to_string() };
    let (_, failed, queued) = super::pso_async::counts();
    let (running, done) = crate::gpu::recomp::dxc::compile_counts();
    format!(
        "profile ({secs:.0} s): {frame}; {}; pipelines queued {queued}, failed {failed}; DXC {running} running, {done} done",
        if parts.is_empty() { "no waits".to_string() } else { parts.join(", ") }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span adds its time and a call to its kind; the line gives the difference since the
    /// totals before, and leaves out kinds with no call.
    #[test]
    fn spans_add_up_by_kind_in_the_line() {
        let before = totals();
        add(Kind::Placement, Duration::from_millis(30));
        add(Kind::Placement, Duration::from_millis(20));
        {
            let _s = span(Kind::FrameEvent);
        }
        let now = totals();
        let k = Kind::Placement as usize;
        assert!(now.0[k].0 - before.0[k].0 >= 50_000_000);
        assert!(now.0[k].1 - before.0[k].1 >= 2);
        let l = line(before, now, 4, 2.0);
        assert!(l.contains("500 ms a frame"), "{l}");
        assert!(l.contains("placements "), "{l}");
        assert!(l.contains("frame-event waits "), "{l}");
    }

    /// A span that has not ended is counted as still going with its time so far, and no longer
    /// once it is dropped; the line names it only when it is long.
    #[test]
    fn a_span_not_ended_yet_is_still_going() {
        let k = Kind::SubmitAfter as usize;
        let s = span(Kind::SubmitAfter);
        std::thread::sleep(Duration::from_millis(5));
        let (n, ns) = still_going(k);
        assert!(n >= 1 && ns >= 5_000_000, "{n} spans, {ns} ns");
        assert!(s.elapsed() >= Duration::from_millis(5));
        let calls = CALLS[k].load(Ordering::Relaxed);
        drop(s);
        assert!(CALLS[k].load(Ordering::Relaxed) > calls, "a dropped span is a call");
        assert_eq!(still_going_note((1, 36_000_000_000)), "; 1 still going, 36000 ms so far");
        assert_eq!(still_going_note((2, OPEN_SHOWN_NS - 1)), "");
        assert_eq!(still_going_note((0, 0)), "");
    }
}
