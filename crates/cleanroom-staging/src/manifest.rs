//! Manifest of staged files (per task).
//!
//! Persisted to SQLite via `migrations/010_staging_manifest.sql` so an
//! orchestrator can resume after a crash.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StagingOp {
    Write,
    Edit,
    Delete,
}

/// One row in `staging_manifest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagingEntry {
    pub task_id: String,
    pub file_path: PathBuf,
    pub content_hash: String,
    pub op: StagingOp,
    pub created_at: i64,
}

impl StagingEntry {
    pub fn new(task_id: impl Into<String>, file_path: PathBuf, op: StagingOp, content: &str) -> Self {
        Self {
            task_id: task_id.into(),
            file_path,
            content_hash: sha256_hex(content.as_bytes()),
            op,
            created_at: chrono::Utc::now().timestamp(),
        }
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_computes_hash() {
        let e = StagingEntry::new("t1", PathBuf::from("/x"), StagingOp::Write, "hello");
        assert_eq!(e.content_hash.len(), 64);
        assert_eq!(e.op, StagingOp::Write);
    }
}
