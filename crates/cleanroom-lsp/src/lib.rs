//! cleanroom-lsp — LSP client for Cleanroom Agent.

#![warn(missing_docs)]

pub mod error;
pub mod server_pool;
pub mod file_analysis;
pub mod language_detection;

pub use error::{LspError, LspResult};
pub use server_pool::{LspConfig, LspServerPool, LspServerHandle};
pub use file_analysis::{FileAnalysis, DocumentSymbol, Diagnostic, DiagnosticSeverity};
pub use language_detection::{detect_language, supported_languages, is_language_supported};
pub use lsp_types::{SymbolKind as SymbolKind, Location};