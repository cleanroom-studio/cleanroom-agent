//! `git-worktree` backend — wraps `git worktree add` to stage files
//! inside a real git worktree, then `git add` + `git commit` + `git merge`
//! on commit. This is the "production" staging backend (better crash
//! recovery than `tempdir` because the worktree is on the same filesystem
//! and the orchestrator can resume from the staged files directly).
//!
//! Used when the skill's `x-cleanroom.staging.mode` is `git-worktree`.
//! See [`crate::workspace::StagingMode::GitWorktree`].
//!
//! # Requirements
//!
//! - The host system must have `git` on `$PATH`.
//! - The target directory must be inside (or equal to) a git working tree.
//! - The current `git` user must have a valid `user.email` / `user.name`.
//!
//! # Implementation note
//!
//! We shell out to `git` via [`std::process::Command`] rather than linking
//! a git library — keeps the dep graph small and the behavior easy to
//! reason about for crash recovery.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{StagingError, StagingResult};
use crate::manifest::{StagingEntry, StagingOp};
use crate::workspace::{CommitReport, StagingWorkspace};

/// Backend that stages inside a `git worktree add`-ed branch.
pub struct GitWorktreeBackend {
    task_id: String,
    /// The original git working tree the worktree is attached to.
    repo_root: PathBuf,
    /// Path to the staging worktree (e.g. `/path/to/repo.cleanroom-tasks/<task_id>`).
    worktree: Option<PathBuf>,
    /// Branch name (per-task; never pushed).
    branch: String,
    manifest: Vec<StagingEntry>,
    /// Logical path → staged content (for `read`).
    files: HashMap<PathBuf, String>,
}

impl GitWorktreeBackend {
    /// Open a new worktree attached to the git repo at `repo_root`.
    /// Creates a branch named `cleanroom-task-<task_id>` from HEAD.
    ///
    /// `git worktree add <path> -b <branch>` is used; the staging worktree
    /// lives at `<repo_root>/.cleanroom/staging/<task_id>` by default.
    pub fn open(task_id: impl Into<String>, repo_root: impl AsRef<Path>) -> StagingResult<Self> {
        let task_id = task_id.into();
        let repo_root = repo_root.as_ref().to_path_buf();
        let branch = format!("cleanroom-task-{}", sanitize_branch(&task_id));
        let worktree_path = repo_root
            .join(".cleanroom")
            .join("staging")
            .join(&task_id);

        // Make sure parent dirs exist
        if let Some(parent) = worktree_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // `git worktree add <path> -b <branch>`
        let status = Command::new("git")
            .current_dir(&repo_root)
            .args([
                "worktree",
                "add",
                "-b",
                &branch,
                worktree_path.to_str().ok_or_else(|| StagingError::Other(
                    "worktree path is not valid UTF-8".to_string(),
                ))?,
            ])
            .output();
        match status {
            Ok(out) if out.status.success() => {}
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                return Err(StagingError::Git(format!(
                    "git worktree add failed: {stderr}"
                )));
            }
            Err(e) => {
                return Err(StagingError::Git(format!(
                    "failed to invoke `git worktree add`: {e}"
                )));
            }
        }

        Ok(Self {
            task_id,
            repo_root,
            worktree: Some(worktree_path),
            branch,
            manifest: Vec::new(),
            files: HashMap::new(),
        })
    }

    /// Resolve a logical path under the staging worktree. Rejects
    /// absolute paths and `..` traversal.
    fn resolve(&self, path: &Path) -> StagingResult<PathBuf> {
        if path.is_absolute() {
            return Err(StagingError::Other(format!(
                "absolute paths not allowed: {}",
                path.display()
            )));
        }
        for c in path.components() {
            if matches!(c, std::path::Component::ParentDir) {
                return Err(StagingError::Other(format!(
                    "parent-dir traversal not allowed: {}",
                    path.display()
                )));
            }
        }
        let worktree = self
            .worktree
            .as_ref()
            .ok_or_else(|| StagingError::NotInitialized(self.task_id.clone()))?;
        Ok(worktree.join(path))
    }

    fn run_git(&self, args: &[&str]) -> StagingResult<()> {
        let output = Command::new("git")
            .current_dir(self.worktree.as_ref().unwrap_or(&self.repo_root))
            .args(args)
            .output()
            .map_err(|e| StagingError::Git(format!("failed to spawn git: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(StagingError::Git(format!(
                "git {} failed: {}",
                args.join(" "),
                stderr
            )));
        }
        Ok(())
    }
}

impl StagingWorkspace for GitWorktreeBackend {
    fn write(&mut self, path: &Path, content: &str) -> StagingResult<StagingEntry> {
        let resolved = self.resolve(path)?;
        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&resolved, content)?;
        let entry = StagingEntry::new(&self.task_id, path.to_path_buf(), StagingOp::Write, content);
        self.files.insert(path.to_path_buf(), content.to_string());
        self.manifest.push(entry.clone());
        Ok(entry)
    }

    fn edit(&mut self, path: &Path, old_text: &str, new_text: &str) -> StagingResult<StagingEntry> {
        let resolved = self.resolve(path)?;
        let existing = std::fs::read_to_string(&resolved)
            .map_err(|_| StagingError::FileNotFound(path.to_path_buf()))?;
        let new_content = if let Some(idx) = existing.find(old_text) {
            let mut s = String::with_capacity(existing.len() + new_text.len());
            s.push_str(&existing[..idx]);
            s.push_str(new_text);
            s.push_str(&existing[idx + old_text.len()..]);
            s
        } else {
            return Err(StagingError::FileNotFound(path.to_path_buf()));
        };
        std::fs::write(&resolved, &new_content)?;
        let entry = StagingEntry::new(&self.task_id, path.to_path_buf(), StagingOp::Edit, &new_content);
        self.files.insert(path.to_path_buf(), new_content);
        self.manifest.push(entry.clone());
        Ok(entry)
    }

    fn delete(&mut self, path: &Path) -> StagingResult<StagingEntry> {
        let resolved = self.resolve(path)?;
        std::fs::remove_file(&resolved)
            .map_err(|_| StagingError::FileNotFound(path.to_path_buf()))?;
        let entry = StagingEntry::new(&self.task_id, path.to_path_buf(), StagingOp::Delete, "");
        self.files.remove(&path.to_path_buf());
        self.manifest.push(entry.clone());
        Ok(entry)
    }

    fn read(&self, path: &Path) -> StagingResult<String> {
        if let Some(c) = self.files.get(&path.to_path_buf()) {
            return Ok(c.clone());
        }
        let resolved = self.resolve(path)?;
        Ok(std::fs::read_to_string(&resolved)?)
    }

    fn diff(&self, target_dir: &Path) -> StagingResult<String> {
        let worktree = self
            .worktree
            .as_ref()
            .ok_or_else(|| StagingError::NotInitialized(self.task_id.clone()))?;
        // `git diff` from inside the worktree against HEAD (the branch's
        // parent at worktree-add time).
        let output = Command::new("git")
            .current_dir(worktree)
            .args(["diff", "--no-color"])
            .output()
            .map_err(|e| StagingError::Git(format!("git diff: {e}")))?;
        if !output.status.success() {
            return Err(StagingError::Git(format!(
                "git diff failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        let _ = target_dir; // unused — git diff runs against the worktree
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn commit(&mut self, target_dir: &Path) -> StagingResult<CommitReport> {
        let started = std::time::Instant::now();
        let mut report = CommitReport::default();

        let worktree = self
            .worktree
            .take()
            .ok_or_else(|| StagingError::NotInitialized(self.task_id.clone()))?;

        // Collapse to latest op per file.
        let mut latest: HashMap<PathBuf, StagingOp> = HashMap::new();
        for entry in &self.manifest {
            latest.insert(entry.file_path.clone(), entry.op.clone());
        }

        // First, copy each file from the worktree into the target dir.
        for (path, op) in &latest {
            let src = worktree.join(path);
            let dest = target_dir.join(path);
            match op {
                StagingOp::Write | StagingOp::Edit => {
                    if !src.exists() {
                        return Err(StagingError::FileNotFound(path.clone()));
                    }
                    let content = std::fs::read_to_string(&src)?;
                    if let Some(parent) = dest.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    let already_matches = dest.exists()
                        && std::fs::read_to_string(&dest).map(|c| c == content).unwrap_or(false);
                    std::fs::write(&dest, content)?;
                    if !already_matches {
                        report.files_written.push(path.clone());
                    }
                }
                StagingOp::Delete => {
                    if dest.exists() {
                        std::fs::remove_file(&dest)?;
                        report.files_deleted.push(path.clone());
                    }
                }
            }
        }

        // Then, commit inside the worktree so the branch carries the
        // history (useful for `git log` / `git diff` audit).
        // We do this only if there's something to commit.
        if !latest.is_empty() {
            let _ = self.run_git(&["add", "-A"]); // best effort
            let _ = self.run_git(&[
                "-c",
                "user.email=cleanroom-agent@localhost",
                "-c",
                "user.name=cleanroom-agent",
                "commit",
                "-m",
                &format!("cleanroom task {}", self.task_id),
            ]);
        }

        // Remove the worktree.
        if let Err(e) = Command::new("git")
            .current_dir(&self.repo_root)
            .args([
                "worktree",
                "remove",
                "--force",
                worktree.to_str().unwrap_or(""),
            ])
            .output()
        {
            tracing::warn!(error = %e, "failed to spawn git worktree remove");
        }

        report.elapsed_ms = started.elapsed().as_millis() as u64;
        Ok(report)
    }

    fn abort(&mut self) -> StagingResult<()> {
        self.manifest.clear();
        self.files.clear();
        if let Some(wt) = self.worktree.take() {
            // `git worktree remove --force` to clean up; ignore errors.
            let _ = Command::new("git")
                .current_dir(&self.repo_root)
                .args([
                    "worktree",
                    "remove",
                    "--force",
                    wt.to_str().unwrap_or(""),
                ])
                .output();
            // Also delete the throwaway branch.
            let _ = Command::new("git")
                .current_dir(&self.repo_root)
                .args(["branch", "-D", &self.branch])
                .output();
        }
        Ok(())
    }

    fn task_id(&self) -> &str {
        &self.task_id
    }

    fn root(&self) -> &Path {
        self.worktree.as_deref().unwrap_or(&self.repo_root)
    }

    fn manifest(&self) -> Vec<StagingEntry> {
        self.manifest.clone()
    }
}

/// Make `task_id` safe for use as a git branch name (lowercase, replace
/// non-alphanumeric with `-`).
fn sanitize_branch(task_id: &str) -> String {
    let mut s = String::with_capacity(task_id.len());
    for c in task_id.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            s.push(c.to_ascii_lowercase());
        } else {
            s.push('-');
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_branch_handles_special_chars() {
        assert_eq!(sanitize_branch("task-001"), "task-001");
        // "TASK 001 / foo" → uppercases lowered, spaces and `/` become
        // `-` (no consecutive-collapse). Three separators between 1 and foo
        // (one for space, one for /, one for space).
        assert_eq!(sanitize_branch("TASK 001 / foo"), "task-001---foo");
        assert_eq!(sanitize_branch("a.b@c"), "a-b-c");
    }

    #[test]
    fn resolve_rejects_unsafe_paths() {
        // We need a constructed backend just to exercise `resolve`.
        let backend = GitWorktreeBackend {
            task_id: "t".into(),
            repo_root: PathBuf::from("/tmp/x"),
            worktree: Some(PathBuf::from("/tmp/x/.cleanroom/staging/t")),
            branch: "cleanroom-task-t".into(),
            manifest: vec![],
            files: HashMap::new(),
        };
        assert!(backend.resolve(Path::new("../escape")).is_err());
        assert!(backend.resolve(Path::new("/abs")).is_err());
        assert!(backend.resolve(Path::new("ok/path.txt")).is_ok());
    }
}
