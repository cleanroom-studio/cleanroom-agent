//! Naming service MCP tool parameters.

use rmcp::schemars;
use serde::{Deserialize, Serialize};

/// Resolve name parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ResolveNameParams {
    pub document_name: String,
    pub sdef_uri: String,
    pub language: String,
}

/// Batch resolve parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BatchResolveParams {
    pub document_name: String,
    pub uris: Vec<String>,
    pub language: String,
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
