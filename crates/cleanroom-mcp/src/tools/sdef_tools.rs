//! S.DEF query MCP tool parameters and result types.
//!
//! Defines parameters for querying Software Definition entities from the
//! Cleanroom Agent database. These are read-only tools that retrieve
//! structured data about data models, contracts, functions, and UI screens.
//!
//! # Tools
//!
//! - [`GetDataModelParams`] — Retrieve a data model with its attributes
//! - [`GetContractParams`] — Retrieve an interface/class contract
//! - [`GetFunctionSpecParams`] — Retrieve a function specification
//! - [`GetUiScreenParams`] — Retrieve a UI screen definition
//! - [`ListDocumentsParams`] — List all S.DEF documents
//! - [`SearchSdefParams`] — Full-text search across S.DEF entities
//! - [`ListShardsParams`] — List document shards
//!
//! # Result Types
//!
//! - [`DataModelResult`] — Data model with attributes
//! - [`ContractResult`] — Contract with type and status
//! - [`FunctionSpecResult`] — Function with logic and complexity
//! - [`DocumentResult`] — Document metadata
//! - [`ShardResult`] — Shard with content hash

use rmcp::schemars;
use serde::{Deserialize, Serialize};

/// Get data model parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetDataModelParams {
    pub document_name: String,
    pub entity: String,
}

/// Get contract parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetContractParams {
    pub document_name: String,
    pub name: String,
}

/// Get function spec parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetFunctionSpecParams {
    pub document_name: String,
    pub name: String,
}

/// Search S.DEF parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchSdefParams {
    pub query: String,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

fn default_search_limit() -> usize { 20 }

/// List documents parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListDocumentsParams {}

/// Get UI screen parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetUiScreenParams {
    pub document_name: String,
    pub screen_id: String,
}

/// List shards parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListShardsParams {
    pub document_name: String,
    pub section_type: Option<String>,
    pub status: Option<String>,
}

/// Data model result.
#[derive(Debug, Clone, Serialize)]
pub struct DataModelResult {
    pub entity: String,
    pub status: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub logical_model: Option<String>,
    pub attributes: Vec<DataAttributeResult>,
}

/// Data attribute result.
#[derive(Debug, Clone, Serialize)]
pub struct DataAttributeResult {
    pub name: String,
    pub attr_type: String,
    pub format: Option<String>,
    pub description: Option<String>,
    pub required: bool,
    pub identity: bool,
    pub generated: bool,
    pub unique_flag: bool,
    pub internal: bool,
    pub deprecated: bool,
    pub default_value: Option<String>,
}

/// Contract result.
#[derive(Debug, Clone, Serialize)]
pub struct ContractResult {
    pub name: String,
    pub contract_type: String,
    pub status: String,
    pub version: Option<String>,
    pub is_abstract: bool,
    pub description: Option<String>,
    pub http_method: Option<String>,
    pub api_path: Option<String>,
    pub auth: Option<String>,
}

/// Function spec result.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionSpecResult {
    pub name: String,
    pub description: Option<String>,
    pub logic: Option<String>,
    pub complexity: Option<String>,
    pub pure_function: bool,
}

/// Shard result.
#[derive(Debug, Clone, Serialize)]
pub struct ShardResult {
    pub shard_id: String,
    pub document_name: String,
    pub sdef_uri: String,
    pub section_type: String,
    pub status: String,
    pub content_hash: Option<String>,
}

/// Document result.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentResult {
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}
