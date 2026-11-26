//! The driver's shader compiler crashing on one pipeline: an access violation inside
//! `Create*PipelineState` ends the process, and the next run ends at the same pipeline (seen in
//! AMD's compiler on a 331 KiB mesh pipeline, every run, a minute into a retail world).
//!
//! The pipeline a thread hands to the driver is noted while the driver has it (`building`). The
//! unhandled-exception filter, on the thread that crashed, writes that pipeline's key to
//! `driver-crashes.txt` beside the DXIL cache and its shaders to the log folder (`on_crash`). A
//! later run does not hand a pipeline with a key of that file to the driver (`blocked`): it fails
//! as one the driver refused. The key is of the translated shaders and the state, so a translation
//! that changes makes a new key and is tried again; deleting the file tries them all again.
//!
//! A preparation worker (`shaders::in_workers`) has no such filter, and a fast-fail inside the
//! driver calls none anyway: it says which pipeline it builds in a file of its own beside the
//! list (`driver-building-<pid>-<thread>.txt`), gone when the pipeline is built. One left by a
//! process that is gone names a pipeline the driver did not come back from: `blocked` moves it
//! into the list, so the worker started again for that part (`in_workers`) goes on after it.

use std::cell::Cell;
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// A pipeline in the driver: its key and its shaders (file name ending, address, bytes).
struct Building {
    key: String,
    shaders: Vec<(String, usize, usize)>,
}

thread_local! {
    /// the pipeline this thread's driver call is building (null: none)
    static BUILDING: Cell<*const Building> = const { Cell::new(std::ptr::null()) };
}

/// Until dropped, this thread's driver call builds the pipeline it was made for.
pub(crate) struct Guard {
    _now: Box<Building>,
    before: *const Building,
    /// a worker's file that names the pipeline while the driver has it
    marker: Option<PathBuf>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = BUILDING.try_with(|b| b.set(self.before));
        if let Some(m) = &self.marker {
            let _ = std::fs::remove_file(m);
        }
    }
}

/// This process is a preparation worker (`shaders::worker_command`).
fn worker() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var_os("XCL_PRECOMPILE_PARENT").is_some())
}

fn marker_name(pid: u32, thread: u32) -> String {
    format!("driver-building-{pid}-{thread}.txt")
}

/// The driver call this thread makes next builds the pipeline `key` from `shaders` (each with the
/// ending of the file it is dumped to after a crash); they outlive the returned guard.
pub(crate) fn building(key: &str, shaders: &[(String, &[u8])]) -> Guard {
    let now = Box::new(Building { key: key.to_string(), shaders: shaders.iter().map(|(n, c)| (n.clone(), c.as_ptr() as usize, c.len())).collect() });
    let before = BUILDING.with(|b| b.replace(&*now));
    let marker = worker().then(|| mark(&path(), key)).flatten();
    Guard { _now: now, before, marker }
}

/// A file beside `list` that names `key` as in the driver for this thread of this process.
fn mark(list: &Path, key: &str) -> Option<PathBuf> {
    let dir = list.parent()?;
    std::fs::create_dir_all(dir).ok()?;
    let m = dir.join(marker_name(std::process::id(), unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }));
    std::fs::write(&m, key).ok()?;
    Some(m)
}

/// The process `pid` is gone (or cannot be asked about, which a live worker of this user can).
fn gone(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};
    unsafe {
        let h = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
        if h.is_null() {
            return true;
        }
        let ended = WaitForSingleObject(h, 0) == 0;
        CloseHandle(h);
        ended
    }
}

/// The markers beside `list` of processes that are gone (`is_gone`): their keys go into the list
/// and the markers are removed. The keys.
fn collect_markers(list: &Path, is_gone: impl Fn(u32) -> bool) -> Vec<String> {
    let Some(dir) = list.parent() else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut found = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(pid) = name.strip_prefix("driver-building-").and_then(|r| r.split('-').next()).and_then(|p| p.parse::<u32>().ok()) else { continue };
        if !is_gone(pid) {
            continue;
        }
        if let Some(key) = std::fs::read_to_string(e.path()).ok().and_then(|t| t.split_whitespace().next().map(str::to_string)) {
            if note(list, &key).is_ok() {
                found.push(key);
            }
        }
        let _ = std::fs::remove_file(e.path());
    }
    found
}

/// The key of the compute pipelines of a console program (`record`: the hash of its container),
/// whole or in pieces: what a preparation worker and the title both know it by.
pub(crate) fn compute_key(record: u64) -> String {
    format!("cs-{record:016x}")
}

/// The keys of the pipelines the driver crashed on, one a line.
fn path() -> PathBuf {
    crate::gpu::recomp::dxc::cache_dir().join("driver-crashes.txt")
}

fn keys(text: &str) -> HashSet<String> {
    text.lines().filter_map(|l| l.split_whitespace().next()).map(str::to_string).collect()
}

fn note(list: &Path, key: &str) -> std::io::Result<()> {
    if let Some(dir) = list.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(list)?;
    writeln!(f, "{key}")?;
    f.sync_all()
}

static BLOCKED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// The driver crashed building the pipeline `key` in an earlier run, or a process that is gone
/// was still building it: it is not built.
pub(crate) fn blocked(key: &str) -> bool {
    let mut g = BLOCKED.lock().unwrap_or_else(|e| e.into_inner());
    // tests start with none, not with what the machine's runs left
    g.get_or_insert_with(|| if cfg!(test) { HashSet::new() } else { load(&path()) }).contains(key)
}

/// The keys of `list`, with those of the markers that processes now gone left beside it.
fn load(list: &Path) -> HashSet<String> {
    for key in collect_markers(list, gone) {
        crate::xlog_warn!("D3D12.X: a process that ended was building pipeline {key} in the driver – noted in {} and not built again", list.display());
    }
    std::fs::read_to_string(list).map(|t| keys(&t)).unwrap_or_default()
}

#[cfg(test)]
pub(super) fn block(key: &str) {
    BLOCKED.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::new).insert(key.to_string());
}

/// The pipeline this thread's driver call was building, if any: its key goes to `list` and its
/// shaders to `dir`; what to log.
unsafe fn crashed(list: &Path, dir: Option<&Path>) -> Option<String> {
    let p = BUILDING.try_with(|b| b.replace(std::ptr::null())).ok()?;
    let b = p.as_ref()?;
    let noted = match note(list, &b.key) {
        Ok(()) => format!("noted in {} and not built again", list.display()),
        Err(e) => format!("not noted in {} ({e})", list.display()),
    };
    let mut shaders = Vec::new();
    for (name, at, len) in &b.shaders {
        if !crate::gpu::shader::readable(*at, *len) {
            continue;
        }
        let code = std::slice::from_raw_parts(*at as *const u8, *len);
        let file = format!("driver-crash-{}-{name}", b.key);
        let written = dir.is_some_and(|d| std::fs::create_dir_all(d).and_then(|()| std::fs::write(d.join(&file), code)).is_ok());
        shaders.push(format!("{file} ({:016x}, {len} B{})", crate::gpu::shader::fnv64(code), if written { "" } else { ", not written" }));
    }
    let dir = dir.map_or_else(String::new, |d| format!(" in {}", d.display()));
    Some(format!("the driver crashed building pipeline {}: {noted}; its shaders{dir}: {}", b.key, shaders.join(", ")))
}

/// From the unhandled-exception filter, on the thread that crashed.
pub fn on_crash() {
    // (a process without a title, a preparation worker: beside the list)
    let dir = crate::state::boot().map_or_else(|| path().with_file_name("driver-crashes"), |b| b.logs_dir());
    if let Some(msg) = unsafe { crashed(&path(), Some(&dir)) } {
        crate::xlog_error!("D3D12.X: {msg}");
        crate::log::flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crash_notes_the_pipeline_being_built_and_dumps_its_shaders() {
        let dir = std::env::temp_dir().join(format!("xcl-driver-crash-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let list = dir.join("cache").join("driver-crashes.txt");
        let logs = dir.join("logs");
        unsafe {
            // no driver call on this thread: nothing to note
            assert!(crashed(&list, Some(&logs)).is_none());
            assert!(!list.exists());

            let (ms, ps) = (vec![1u8, 2, 3], vec![4u8; 5]);
            let outer = building("s-outer-10", &[("ms.dxil".to_string(), &ms[..])]);
            {
                let _inner = building("s-inner-8", &[("ms.dxil".to_string(), &ms[..]), ("ps.dxil".to_string(), &ps[..])]);
                let msg = crashed(&list, Some(&logs)).expect("the inner pipeline");
                assert!(msg.contains("s-inner-8") && msg.contains("driver-crash-s-inner-8-ps.dxil"), "{msg}");
                // once: the filter may run again for the same crash
                assert!(crashed(&list, Some(&logs)).is_none());
            }
            // the inner call returned: the outer pipeline is the one being built again
            assert!(crashed(&list, Some(&logs)).expect("the outer pipeline").contains("s-outer-10"));
            drop(outer);
            assert!(crashed(&list, Some(&logs)).is_none());
        }
        assert_eq!(std::fs::read(logs.join("driver-crash-s-inner-8-ms.dxil")).unwrap(), [1, 2, 3]);
        assert_eq!(std::fs::read(logs.join("driver-crash-s-inner-8-ps.dxil")).unwrap(), [4; 5]);
        let noted = keys(&std::fs::read_to_string(&list).unwrap());
        assert_eq!(noted, HashSet::from(["s-inner-8".to_string(), "s-outer-10".to_string()]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A worker's marker names the pipeline while the driver has it and is gone after; one left
    /// by a process that is gone puts its key into the list, one of a live process stays.
    #[test]
    fn a_marker_left_by_a_process_that_is_gone_blocks_its_pipeline() {
        let dir = std::env::temp_dir().join(format!("xcl-driver-marker-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let list = dir.join("driver-crashes.txt");
        let m = mark(&list, "cs-00000000000000aa").expect("a marker");
        assert_eq!(std::fs::read_to_string(&m).unwrap(), "cs-00000000000000aa");
        // built: the guard removes it (as `Guard::drop` does)
        std::fs::remove_file(&m).unwrap();
        assert!(collect_markers(&list, |_| true).is_empty());

        std::fs::write(dir.join(marker_name(111, 1)), "cs-0000000000000001").unwrap();
        std::fs::write(dir.join(marker_name(222, 7)), "cs-0000000000000002").unwrap();
        assert_eq!(collect_markers(&list, |pid| pid == 111), ["cs-0000000000000001"]);
        assert!(!dir.join(marker_name(111, 1)).exists() && dir.join(marker_name(222, 7)).exists());
        assert_eq!(keys(&std::fs::read_to_string(&list).unwrap()), HashSet::from(["cs-0000000000000001".to_string()]));
        assert!(!gone(std::process::id()), "this process is not gone");
        assert_eq!(compute_key(0xab), "cs-00000000000000ab");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn keys_are_the_first_word_of_each_line() {
        assert_eq!(keys("s-1-a\r\n\n  s-2-b  later text\n"), HashSet::from(["s-1-a".to_string(), "s-2-b".to_string()]));
    }
}
