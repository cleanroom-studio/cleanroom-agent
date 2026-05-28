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
