//! End-to-end integration test for cleanroom-staging.
//!
//! Pipeline: `write multiple files → manifest snapshot → commit → target
//! source tree reflects the writes; abort leaves the target untouched`.

use std::fs;
use std::path::{Path, PathBuf};

use cleanroom_staging::{StagingWorkspace, TempDirBackend};
use tempfile::tempdir;

#[test]
fn write_multiple_files_and_commit() {
    let mut ws = TempDirBackend::new("int-1").expect("init");
    let target = tempdir().expect("target");

    // Stage 3 files in different sub-directories.
    ws.write(
        Path::new("src/main.rs"),
        "fn main() { println!(\"hi\"); }\n",
    )
    .expect("write main");
    ws.write(Path::new("src/lib.rs"), "pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
        .expect("write lib");
    ws.write(
        Path::new("tests/integration.rs"),
        "#[test] fn t() { assert_eq!(2+2, 4); }\n",
    )
    .expect("write test");

    // Manifest should have 3 entries.
    assert_eq!(ws.manifest().len(), 3);

    // Commit to target.
    let report = ws.commit(target.path()).expect("commit");
    assert_eq!(report.files_written.len(), 3);
    assert!(report.elapsed_ms < 5_000);

    // Target tree should have all 3 files.
    let main_content = fs::read_to_string(target.path().join("src/main.rs")).expect("read main");
    assert!(main_content.contains("println"));
    let lib_content = fs::read_to_string(target.path().join("src/lib.rs")).expect("read lib");
    assert!(lib_content.contains("add"));
    let test_content =
        fs::read_to_string(target.path().join("tests/integration.rs")).expect("read test");
    assert!(test_content.contains("assert_eq"));
}

#[test]
fn edit_then_commit() {
    let mut ws = TempDirBackend::new("int-2").expect("init");
    let target = tempdir().expect("target");

    ws.write(Path::new("a.txt"), "hello world\n").expect("write");
    ws.edit(Path::new("a.txt"), "world", "rust").expect("edit");
    ws.commit(target.path()).expect("commit");

    let s = fs::read_to_string(target.path().join("a.txt")).expect("read");
    assert_eq!(s, "hello rust\n");
}

#[test]
fn abort_does_not_touch_target() {
    let mut ws = TempDirBackend::new("int-3").expect("init");
    let target = tempdir().expect("target");

    // Pre-existing file in target — must not be deleted by abort.
    fs::write(target.path().join("existing.txt"), "untouched").expect("seed");

    ws.write(Path::new("new.txt"), "staged").expect("write");
    ws.abort().expect("abort");

    assert!(target.path().join("existing.txt").exists());
    assert!(!target.path().join("new.txt").exists());
}

#[test]
fn write_then_delete_collapses_to_noop() {
    let mut ws = TempDirBackend::new("int-4").expect("init");
    let target = tempdir().expect("target");

    // Staging a write+delete of the same path should result in NO change
    // at commit time (write is shadowed by the subsequent delete).
    ws.write(Path::new("a.txt"), "tmp").expect("write");
    ws.delete(Path::new("a.txt")).expect("delete");
    let report = ws.commit(target.path()).expect("commit");
    assert!(report.is_empty(), "write+delete should cancel out");
    assert!(!target.path().join("a.txt").exists());
}

#[test]
fn diff_shows_changes() {
    let mut ws = TempDirBackend::new("int-5").expect("init");
    let target = tempdir().expect("target");
    fs::write(target.path().join("a.txt"), "old\n").expect("seed");

    ws.write(Path::new("a.txt"), "new\n").expect("write");
    let diff = ws.diff(target.path()).expect("diff");
    assert!(diff.contains("a.txt"));
    assert!(diff.contains("new") || diff.contains("+"));
}

#[test]
#[ignore = "requires git; run with `cargo test -- --ignored`"]
fn git_worktree_backend_open_write_commit() {
    use cleanroom_staging::GitWorktreeBackend;
    use cleanroom_staging::StagingWorkspace;

    let repo = tempdir().expect("repo");
    // Initialize a git repo with a baseline commit.
    std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(repo.path())
        .output()
        .expect("git init");
    std::process::Command::new("git")
        .args(["config", "user.email", "test@localhost"])
        .current_dir(repo.path())
        .output()
        .expect("git config email");
    std::process::Command::new("git")
        .args(["config", "user.name", "test"])
        .current_dir(repo.path())
        .output()
        .expect("git config name");
    std::fs::write(repo.path().join("README.md"), "# test\n").expect("write");
    std::process::Command::new("git")
        .args(["add", "-A"])
        .current_dir(repo.path())
        .output()
        .expect("git add");
    std::process::Command::new("git")
        .args(["commit", "-q", "-m", "init"])
        .current_dir(repo.path())
        .output()
        .expect("git commit");

    let mut ws = GitWorktreeBackend::open("gw-1", repo.path()).expect("open worktree");
    ws.write(Path::new("src/lib.rs"), "pub fn hello() {}\n").expect("write");
    let report = ws.commit(repo.path()).expect("commit");
    assert_eq!(report.files_written, vec![PathBuf::from("src/lib.rs")]);
    let s = std::fs::read_to_string(repo.path().join("src/lib.rs")).expect("read");
    assert!(s.contains("hello"));
}
