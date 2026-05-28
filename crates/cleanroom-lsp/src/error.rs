//! LSP error types.

use thiserror::Error;

/// LSP operation errors.
#[derive(Debug, Error)]
pub enum LspError {
    #[error("Failed to start LSP server: {0}")]
    ServerStartFailed(String),

    #[error("Server communication error: {0}")]
    CommunicationError(String),

    #[error("Server timeout: {0}")]
    Timeout(String),

    #[error("Language not supported: {0}")]
    UnsupportedLanguage(String),

    #[error("File analysis failed: {0}")]
    AnalysisFailed(String),

    #[error("Server not available for language: {0}")]
    ServerNotAvailable(String),
}

/// Result type alias for LSP operations.
pub type LspResult<T> = Result<T, LspError>;