//! LSP integration MCP tool parameters and result types.
//!
//! These tools expose Language Server Protocol (LSP) analysis capabilities to the
//! LLM agent via the MCP interface. The LSP provides rich code understanding
//! including symbol navigation, type info, references, and diagnostics.
//!
//! # LSP Server Pool
//!
//! The MCP server maintains a pool of LSP servers (one per language) via
//! [`LspServerPool`]. Servers are initialized lazily on first use and
//! reused for subsequent requests.
//!
//! # Supported Languages
//!
//! - TypeScript/JavaScript (via `typescript-language-server`)
//! - Rust (via `rust-analyzer`)
//! - Python (via `pylsp`)
//! - Go (via `gopls`)
//! - And other LSP-compatible servers
//!
//! # Tools
//!
//! | Tool | Description |
//! |------|-------------|
//! | [`LspInitParams`] | Initialize LSP server for a language |
//! | [`LspDocumentSymbolsParams`] | Get all symbols in a document |
//! | [`LspTypeInfoParams`] | Get type info at cursor position |
//! | [`LspFindReferencesParams`] | Find all references to a symbol |
//! | [`LspDiagnosticsParams`] | Get errors/warnings for a file |
//! | [`LspHierarchyParams`] | Get type hierarchy (supertypes/subtypes) |
//!
//! # Example
//!
//! ```rust,ignore
//! // Initialize LSP for TypeScript
//! let init = call("lsp_initialize", json!({ "language": "typescript" }));
//!
//! // Get symbols in a file
//! let symbols = call("lsp_get_document_symbols", json!({
//!     "file_path": "/project/src/user.ts",
//!     "language": "typescript"
//! }));
//! ```

use rmcp::schemars;
use serde::Deserialize;

/// Initialize LSP server for a language.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LspInitParams {
    /// Language identifier (e.g., "typescript", "rust", "python", "go").
    pub language: String,
    /// Maximum time to wait for initialization in seconds (default: 30).
    #[serde(default = "default_init_timeout")]
    pub timeout_secs: u64,
}

fn default_init_timeout() -> u64 { 30 }

/// Get document symbols from a file.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LspDocumentSymbolsParams {
    /// Absolute path to the file.
    pub file_path: String,
    /// Language identifier.
    pub language: String,
}

/// Get type information at a position.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LspTypeInfoParams {
    /// Absolute path to the file.
    pub file_path: String,
    /// Language identifier.
    pub language: String,
    /// Zero-based line number.
    pub line: u32,
    /// Zero-based character offset.
    pub character: u32,
}

/// Find references of a symbol.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LspFindReferencesParams {
    /// Absolute path to the file.
    pub file_path: String,
    /// Language identifier.
    pub language: String,
    /// Zero-based line number.
    pub line: u32,
    /// Zero-based character offset.
    pub character: u32,
    /// Include the declaration itself (default: false).
    #[serde(default)]
    pub include_declaration: bool,
}

/// Get diagnostics (errors/warnings) for a file.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LspDiagnosticsParams {
    /// Absolute path to the file.
    pub file_path: String,
    /// Language identifier.
    pub language: String,
}

/// Get type hierarchy (parents and children of a symbol).
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LspHierarchyParams {
    /// Absolute path to the file.
    pub file_path: String,
    /// Language identifier.
    pub language: String,
    /// Zero-based line number.
    pub line: u32,
    /// Zero-based character offset.
    pub character: u32,
    /// Direction: "both" (default), "parents", or "children".
    #[serde(default = "default_hierarchy_direction")]
    pub direction: String,
}

fn default_hierarchy_direction() -> String { "both".to_string() }

/// Result of LSP initialization.
#[derive(Debug, serde::Serialize)]
pub struct LspInitResult {
    pub initialized: bool,
    pub language: String,
    pub server_info: String,
}

/// Document symbol result for MCP output.
#[derive(Debug, serde::Serialize)]
pub struct DocumentSymbolResult {
    pub name: String,
    pub kind: String,
    pub range: Option<String>,
    pub detail: Option<String>,
    pub children: Vec<DocumentSymbolResult>,
}

impl From<cleanroom_lsp::DocumentSymbol> for DocumentSymbolResult {
    fn from(s: cleanroom_lsp::DocumentSymbol) -> Self {
        Self {
            name: s.name,
            kind: format!("{:?}", s.kind),
            range: s.range.map(|(sl, sc, el, ec)| format!("{}:{}-{}:{}", sl, sc, el, ec)),
            detail: s.detail,
            children: s.children.into_iter().map(Into::into).collect(),
        }
    }
}
