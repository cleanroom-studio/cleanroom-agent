//! Naming service MCP tool parameters and result types.
//!
//! The naming service resolves S.DEF URIs to language-specific concrete names.
//! It supports both auto-generated names (via [`NameResolutionService`]) and
//! user-defined custom names registered via [`RegisterCustomNameParams`].
//!
//! # Name Resolution Process
//!
//! 1. Check if a custom name exists in the symbol registry
//! 2. If not, use [`NameResolutionService`] to auto-generate a name based on language conventions
//! 3. Cache the result for subsequent lookups
//!
//! # Language Conventions
//!
//! - `rust` — snake_case for variables/functions, PascalCase for types
//! - `typescript` — camelCase for variables, PascalCase for types
//! - `python` — snake_case for all identifiers
//! - `go` — PascalCase for exports, camelCase for locals
//!
//! # Tools
//!
//! - [`ResolveNameParams`] — Resolve a single S.DEF URI to a concrete name
//! - [`BatchResolveParams`] — Resolve multiple URIs in one call
//! - [`ListSymbolsParams`] — List all registered symbols for a document
//! - [`RegisterCustomNameParams`] — Register a user-defined name override

use rmcp::schemars;
use serde::{Deserialize, Serialize};

/// Resolve name parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ResolveNameParams {
    pub document_name: String,
    pub sdef_uri: String,
    pub language: String,
    /// Symbol type: "class", "interface", "function", "variable", "constant", "enum", "type"
    pub symbol_type: String,
}

/// Batch resolve parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BatchResolveParams {
    pub document_name: String,
    pub uris: Vec<String>,
    pub language: String,
    /// Optional symbol type to apply to all URIs. Defaults to "variable".
    pub symbol_type: Option<String>,
}

/// List symbols parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListSymbolsParams {
    pub document_name: String,
    pub language: String,
    pub symbol_type: Option<String>,
}

/// Register custom name parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RegisterCustomNameParams {
    pub document_name: String,
    pub sdef_uri: String,
    pub language: String,
    pub symbol_type: String,
    pub concrete_name: String,
}

/// Symbol result.
#[derive(Debug, Clone, Serialize)]
pub struct SymbolResult {
    pub sdef_uri: String,
    pub concrete_name: String,
    pub is_user_defined: bool,
    pub symbol_type: String,
    pub language: String,
    pub created_at: Option<String>,
}

/// Resolution result.
#[derive(Debug, Clone, Serialize)]
pub struct ResolutionResult {
    pub sdef_uri: String,
    pub concrete_name: String,
    pub is_user_defined: bool,
}
