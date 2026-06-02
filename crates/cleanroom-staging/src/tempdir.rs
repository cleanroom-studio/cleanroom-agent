//! `tempdir` backend — MVP. Stages files into a `tempfile::tempdir()` and
//! moves them atomically to the target source tree on `commit`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{StagingError, StagingResult};
use crate::manifest::{StagingEntry, StagingOp};
use crate::workspace::{CommitReport, StagingWorkspace};

pub struct TempDirBackend {
    task_id: String,
    /// Wrapped in Option so `commit` / `abort` can move it out.
    root: Option<tempfile::TempDir>,
    /// Logical path → relative path under root.
    /// (We keep absolute paths in manifest, but root is the temp dir.)
    manifest: Vec<StagingEntry>,
    /// In-memory staging area, mirrors the file system. Used for `read`.
    files: HashMap<PathBuf, String>,
}

impl TempDirBackend {
    pub fn new(task_id: impl Into<String>) -> StagingResult<Self> {
        let task_id = task_id.into();
        let root = tempfile::Builder::new()
            .prefix(&format!("cleanroom-staging-{task_id}-"))
            .tempdir()?;
        Ok(Self {
            task_id,
            root: Some(root),
            manifest: Vec::new(),
            files: HashMap::new(),
        })
    }

    /// Resolve a logical path under the staging root. Rejects `..` traversal
    /// and absolute paths.
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
        let root = self
            .root
            .as_ref()
            .ok_or_else(|| StagingError::NotInitialized(self.task_id.clone()))?;
        Ok(root.path().join(path))
    }
}

impl StagingWorkspace for TempDirBackend {
    fn write(&mut self, path: &Path, content: &str) -> StagingResult<StagingEntry> {
        let resolved = self.resolve(path)?;
        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&resolved, content)?;
        let entry = StagingEntry::new(&self.task_id, path.to_path_buf(), StagingOp::Write, content);
        self.files.insert(path.to_path_buf(), content.to_string());
        self.manifest.push(entry.clone());
        Ok(entry)
    }

    fn edit(&mut self, path: &Path, old_text: &str, new_text: &str) -> StagingResult<StagingEntry> {
        let resolved = self.resolve(path)?;
        let existing = fs::read_to_string(&resolved)
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
        fs::write(&resolved, &new_content)?;
        let entry = StagingEntry::new(&self.task_id, path.to_path_buf(), StagingOp::Edit, &new_content);
        self.files.insert(path.to_path_buf(), new_content);
        self.manifest.push(entry.clone());
        Ok(entry)
    }

    fn delete(&mut self, path: &Path) -> StagingResult<StagingEntry> {
        let resolved = self.resolve(path)?;
        fs::remove_file(&resolved).map_err(|_| StagingError::FileNotFound(path.to_path_buf()))?;
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
        Ok(fs::read_to_string(&resolved)?)
    }

    fn diff(&self, target_dir: &Path) -> StagingResult<String> {
        let mut out = String::new();
        for entry in &self.manifest {
            let target_path = target_dir.join(&entry.file_path);
            match entry.op {
                StagingOp::Write | StagingOp::Edit => {
                    let new_content = self.read(&entry.file_path)?;
                    let old_content = fs::read_to_string(&target_path).unwrap_or_default();
                    if old_content != new_content {
                        out.push_str(&format!(
                            "--- {}\n+++ {}\n",
                            target_path.display(),
                            target_path.display()
                        ));
                        out.push_str(&simple_diff(&old_content, &new_content));
                    }
                }
                StagingOp::Delete => {
                    if target_path.exists() {
                        out.push_str(&format!("--- {}\n", target_path.display()));
                    }
                }
            }
        }
        Ok(out)
    }

    fn commit(&mut self, target_dir: &Path) -> StagingResult<CommitReport> {
        let started = std::time::Instant::now();
        let mut report = CommitReport::default();

        // Collapse to the latest op per file (write+delete → delete only).
        let mut latest: HashMap<PathBuf, StagingOp> = HashMap::new();
        for entry in &self.manifest {
            latest.insert(entry.file_path.clone(), entry.op.clone());
        }

        for (path, op) in latest {
            let dest = target_dir.join(&path);
            match op {
                StagingOp::Write | StagingOp::Edit => {
                    let content = self.read(&path)?;
                    if let Some(parent) = dest.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::write(&dest, content)?;
                    report.files_written.push(path);
                }
                StagingOp::Delete => {
                    if dest.exists() {
                        fs::remove_file(&dest)?;
                    }
                    report.files_deleted.push(path);
                }
            }
        }
        report.elapsed_ms = started.elapsed().as_millis() as u64;

        // Drop the temp dir.
        if let Some(root) = self.root.take() {
            root.close().map_err(StagingError::Io)?;
        }
        Ok(report)
    }

    fn abort(&mut self) -> StagingResult<()> {
        self.manifest.clear();
        self.files.clear();
        if let Some(root) = self.root.take() {
            root.close().map_err(StagingError::Io)?;
        }
        Ok(())
    }

    fn task_id(&self) -> &str {
        &self.task_id
    }

    fn root(&self) -> &Path {
        self.root
            .as_ref()
            .map(|r| r.path())
            .unwrap_or_else(|| Path::new(""))
    }

    fn manifest(&self) -> Vec<StagingEntry> {
        self.manifest.clone()
    }
}

/// Very small unified-diff-like output. Real `git diff` would be nicer but
/// keeps the MVP dependency-free.
fn simple_diff(old: &str, new: &str) -> String {
    let mut out = String::new();
    out.push_str(format!("-{old}\n").as_str());
    out.push_str(format!("+{new}\n").as_str());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_and_commit_atomic() {
        let mut ws = TempDirBackend::new("t1").expect("init");
        let target = tempfile::tempdir().unwrap();
        ws.write(Path::new("src/main.rs"), "fn main() {}\n").unwrap();
        let r = ws.commit(target.path()).unwrap();
        assert_eq!(r.files_written, vec![PathBuf::from("src/main.rs")]);
        let written = std::fs::read_to_string(target.path().join("src/main.rs")).unwrap();
        assert_eq!(written, "fn main() {}\n");
    }

    #[test]
    fn edit_finds_and_replaces() {
        let mut ws = TempDirBackend::new("t2").expect("init");
        ws.write(Path::new("a.txt"), "hello world\n").unwrap();
        ws.edit(Path::new("a.txt"), "world", "rust").unwrap();
        let target = tempfile::tempdir().unwrap();
        ws.commit(target.path()).unwrap();
        let s = std::fs::read_to_string(target.path().join("a.txt")).unwrap();
        assert_eq!(s, "hello rust\n");
    }

    #[test]
    fn edit_missing_old_returns_error() {
        let mut ws = TempDirBackend::new("t3").expect("init");
        ws.write(Path::new("a.txt"), "abc\n").unwrap();
        let r = ws.edit(Path::new("a.txt"), "zzz", "qqq");
        assert!(matches!(r, Err(StagingError::FileNotFound(_))));
    }

    #[test]
    fn delete_removes_file() {
        let mut ws = TempDirBackend::new("t4").expect("init");
        ws.write(Path::new("a.txt"), "x").unwrap();
        let target = tempfile::tempdir().unwrap();
        std::fs::write(target.path().join("a.txt"), "old").unwrap();
        ws.delete(Path::new("a.txt")).unwrap();
        let r = ws.commit(target.path()).unwrap();
        assert_eq!(r.files_deleted, vec![PathBuf::from("a.txt")]);
        assert!(!target.path().join("a.txt").exists());
    }

    #[test]
    fn abort_does_not_touch_target() {
        let mut ws = TempDirBackend::new("t5").expect("init");
        let target = tempfile::tempdir().unwrap();
        ws.write(Path::new("a.txt"), "staged\n").unwrap();
        ws.abort().unwrap();
        assert!(!target.path().join("a.txt").exists());
    }

    #[test]
    fn read_returns_staged_content() {
        let mut ws = TempDirBackend::new("t6").expect("init");
        ws.write(Path::new("a.txt"), "stage").unwrap();
        assert_eq!(ws.read(Path::new("a.txt")).unwrap(), "stage");
    }

    #[test]
    fn parent_dir_traversal_rejected() {
        let mut ws = TempDirBackend::new("t7").expect("init");
        let r = ws.write(Path::new("../escape.txt"), "pwned");
        assert!(r.is_err());
    }
}
