//! Import/Export MCP tool parameters.

use rmcp::schemars;
use serde::Deserialize;

/// Export S.DEF parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExportSdefParams {
    pub document_name: String,
    #[serde(default = "default_format")]
    pub format: String,
}

fn default_format() -> String { "json".to_string() }

/// Import S.DEF parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ImportSdefParams {
    pub sdef_json: String,
}

/// Export shard parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExportShardParams {
    pub sdef_uri: String,
}

/// Import shard parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ImportShardParams {
    pub shard_id: String,
    pub document_name: String,
    pub sdef_uri: String,
    pub section_type: String,
    pub content_json: String,
}

/// Checkpoint parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CheckpointParams {
    pub document_name: String,
    pub description: Option<String>,
}

/// Checkpoint ID parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CheckpointIdParams {
    pub checkpoint_id: String,
}

/// Consistency check parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConsistencyCheckParams {
    pub document_name: String,
    #[serde(default = "default_check_type")]
    pub check_type: String,
}

fn default_check_type() -> String { "fast".to_string() }

/// Consistency fix parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConsistencyFixParams {
    pub document_name: String,
    pub entity_uri: String,
    pub strategy: String,
}

/// Fingerprint compute parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FingerprintParams {
    pub document_name: String,
}

/// Compatibility mode parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SetCompatModeParams {
    pub document_name: String,
    pub mode: String,
}

/// List compat layers parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListCompatLayersParams {
    pub document_name: String,
}
