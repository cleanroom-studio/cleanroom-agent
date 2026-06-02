//! Error types for cleanroom-staging.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StagingError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("staging not initialized for task `{0}`")]
    NotInitialized(String),

    #[error("file not found in staging: {0}")]
    FileNotFound(PathBuf),

    #[error("conflict on path `{0}` — already exists with different content")]
    Conflict(PathBuf),

    #[error("git error: {0}")]
    Git(String),

    #[error("invalid backend mode: {0} (expected `git-worktree` or `tempdir`)")]
    InvalidMode(String),

    #[error("commit failed: {0}")]
    CommitFailed(String),

    #[error("other: {0}")]
    Other(String),
}

pub type StagingResult<T> = Result<T, StagingError>;
