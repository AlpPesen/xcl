//! `XCL_PLACE_TIMES=1`: where the time of the title's texture placements goes
//! (`CreatePlacedResourceX` → `vram::place_pages`), logged every 4096 of them. Off: one load each.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Step {
    /// the whole call, for a texture
    Total,
    /// the PC description, the alias note, the console's texture layout
    Desc,
    /// the CPU view of a GPU-view address (`canonical_va`)
    Va,
    /// waiting for the page table's lock, twice a placement
    Lock,
    /// `CreateReservedResource`
    Reserved,
    /// `GetResourceTiling`
    Tiling,
    /// pages gone idle, under the lock
    Sweep,
    /// `CreateHeap` of a pool heap, under the lock
    Heap,
    /// `UpdateTileMappings`, under the lock
    Map,
    /// the fence signal after them, under the lock
    Signal,
    /// `title_textures::register`
    Register,
}

const STEPS: [Step; 11] = [
    Step::Total,
    Step::Desc,
    Step::Va,
    Step::Lock,
    Step::Reserved,
    Step::Tiling,
    Step::Sweep,
    Step::Heap,
    Step::Map,
    Step::Signal,
    Step::Register,
];
const EVERY: u64 = 4096;

struct Sum {
    calls: AtomicU64,
    ns: AtomicU64,
    max: AtomicU64,
}

static SUMS: [Sum; STEPS.len()] = [const { Sum { calls: AtomicU64::new(0), ns: AtomicU64::new(0), max: AtomicU64::new(0) } }; STEPS.len()];

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PLACE_TIMES").is_ok_and(|v| v.trim() == "1"))
}

/// Times a step until dropped.
pub(super) struct Timer(Option<(Step, Instant)>);

pub(super) fn step(step: Step) -> Timer {
    Timer(on().then(|| (step, Instant::now())))
}

impl Drop for Timer {
    fn drop(&mut self) {
        let Some((step, since)) = self.0.take() else { return };
        let calls = add(step, since.elapsed().as_nanos() as u64);
        if step == Step::Total && calls % EVERY == 0 {
            crate::xlog_info!("D3D12.X: texture placements: {}", text());
        }
    }
}

/// How many times `step` has been counted, this one included.
fn add(step: Step, ns: u64) -> u64 {
    let s = &SUMS[step as usize];
    s.ns.fetch_add(ns, Relaxed);
    s.max.fetch_max(ns, Relaxed);
    s.calls.fetch_add(1, Relaxed) + 1
}

fn text() -> String {
    STEPS
        .iter()
        .map(|&s| {
            let v = &SUMS[s as usize];
            let (calls, ns, max) = (v.calls.load(Relaxed), v.ns.load(Relaxed), v.max.load(Relaxed));
            format!("{s:?} {calls}x {} ms (max {:.1})", ns / 1_000_000, max as f64 / 1e6)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_names_every_step_with_its_count_sum_and_longest() {
        for (i, s) in STEPS.iter().enumerate() {
            assert_eq!(*s as usize, i, "{s:?} has its own sum");
        }
        assert_eq!(add(Step::Heap, 2_000_000), 1);
        assert_eq!(add(Step::Heap, 5_500_000), 2);
        let t = text();
        assert!(t.contains("Heap 2x 7 ms (max 5.5)"), "{t}");
        assert!(t.starts_with("Total 0x 0 ms"), "{t}");
    }
}
