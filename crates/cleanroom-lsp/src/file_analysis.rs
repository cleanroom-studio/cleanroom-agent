//! File analysis types and results.

use serde::{Deserialize, Serialize};
use lsp_types::{Location, SymbolKind};

/// Document symbol with additional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSymbol {
    /// Symbol name.
    pub name: String,
    
    /// Symbol kind (Class, Function, etc.).
    pub kind: SymbolKind,
    
    /// Source location range.
    pub range: Option<(u32, u32, u32, u32)>, // (start_line, start_col, end_line, end_col)
    
    /// Child symbols.
    pub children: Vec<DocumentSymbol>,
    
    /// Additional detail information.
    pub detail: Option<String>,
}

/// File analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAnalysis {
    /// File path.
    pub file_path: String,
    
    /// Language of the file.
    pub language: String,
    
    /// All document symbols (top-level and nested).
    pub symbols: Vec<DocumentSymbol>,
    
    /// Import statements.
    pub imports: Vec<String>,
    
    /// Export statements.
    pub exports: Vec<String>,
    
    /// References to other symbols.
    pub references: Vec<Location>,
    
    /// Diagnostics (errors/warnings).
    pub diagnostics: Vec<Diagnostic>,
}

/// A diagnostic (error or warning).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Severity level.
    pub severity: DiagnosticSeverity,
    
    /// Message.
    pub message: String,
    
    /// Source location.
    pub range: Option<(u32, u32, u32, u32)>,
    
    /// Error code (if applicable).
    pub code: Option<String>,
}

/// Diagnostic severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Information,
    Hint,
}

impl From<lsp_types::DiagnosticSeverity> for DiagnosticSeverity {
    fn from(severity: lsp_types::DiagnosticSeverity) -> Self {
        match severity {
            lsp_types::DiagnosticSeverity::ERROR => DiagnosticSeverity::Error,
            lsp_types::DiagnosticSeverity::WARNING => DiagnosticSeverity::Warning,
            lsp_types::DiagnosticSeverity::INFORMATION => DiagnosticSeverity::Information,
            lsp_types::DiagnosticSeverity::HINT => DiagnosticSeverity::Hint,
            _ => DiagnosticSeverity::Information,
        }
    }
}