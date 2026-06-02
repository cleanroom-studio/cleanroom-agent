//! Staging backend abstraction and backends.
//!
//! See `PLAN2.md` §5 for the design.

use std::path::{Path, PathBuf};

use crate::error::{StagingError, StagingResult};
use crate::manifest::StagingEntry;

/// Staging backend strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StagingMode {
    /// Stage to a `tempfile::tempdir()`. MVP backend, no git required.
    TempDir,
    /// Stage to a `git worktree`. Future Phase (Phase D.4).
    GitWorktree,
}

impl StagingMode {
    pub fn from_str(s: &str) -> StagingResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "tempdir" | "tmp" | "tmpdir" => Ok(StagingMode::TempDir),
            "git-worktree" | "git_worktree" | "worktree" => Ok(StagingMode::GitWorktree),
            other => Err(StagingError::InvalidMode(other.to_string())),
        }
    }
}

/// The trait every backend implements.
pub trait StagingWorkspace: Send {
    /// Write a new file (or overwrite) inside the staging area.
    fn write(&mut self, path: &Path, content: &str) -> StagingResult<StagingEntry>;

    /// Apply a textual edit. If `old_text` is not found, returns
    /// `StagingError::FileNotFound` (caller decides whether to fall back to
    /// `write`).
    fn edit(&mut self, path: &Path, old_text: &str, new_text: &str) -> StagingResult<StagingEntry>;

    /// Delete a file from the staging area (does not affect the source tree
    /// until `commit` is called).
    fn delete(&mut self, path: &Path) -> StagingResult<StagingEntry>;

    /// Read a file from the staging area.
    fn read(&self, path: &Path) -> StagingResult<String>;

    /// Diff between staging and `target_dir`. Returns a unified diff string
    /// (empty when no changes).
    fn diff(&self, target_dir: &Path) -> StagingResult<String>;

    /// Apply all staged changes to `target_dir` (atomic move). After
    /// `commit`, the staging area is dropped.
    fn commit(&mut self, target_dir: &Path) -> StagingResult<CommitReport>;

    /// Drop the staging area without applying changes.
    fn abort(&mut self) -> StagingResult<()>;

    /// The task_id this workspace belongs to.
    fn task_id(&self) -> &str;

    /// The staging root path (where files physically live during staging).
    fn root(&self) -> &Path;

    /// Manifest entries (snapshot).
    fn manifest(&self) -> Vec<StagingEntry>;
}

/// What `commit` returns.
#[derive(Debug, Clone, Default)]
pub struct CommitReport {
    pub files_written: Vec<PathBuf>,
    pub files_deleted: Vec<PathBuf>,
    pub elapsed_ms: u64,
}

impl CommitReport {
    pub fn is_empty(&self) -> bool {
        self.files_written.is_empty() && self.files_deleted.is_empty()
    }
    pub fn total_changes(&self) -> usize {
        self.files_written.len() + self.files_deleted.len()
    }
}
