//! Staging bridge — a thin shim so `SkillError` can carry staging errors
//! without depending on the `cleanroom-staging` crate (which would be a
//! circular dep since staging depends on skill for AuthorizationToken types).

use thiserror::Error;

/// Errors that can occur when bridging to the staging workspace.
#[derive(Debug, Error)]
pub enum StagingBridgeError {
    #[error("staging not initialized for task {0}")]
    NotInitialized(String),

    #[error("staging io error: {0}")]
    Io(String),

    #[error("staging conflict on path {0}")]
    Conflict(String),

    #[error("staging backend error: {0}")]
    Backend(String),
}
