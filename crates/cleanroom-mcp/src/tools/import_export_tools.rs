//! Import/Export MCP tool parameters and transaction types.
//!
//! Handles serialization of S.DEF documents to/from JSON and shard-level
//! operations. Also provides checkpoint functionality for workflow snapshots
//! and transaction management for atomic operations.
//!
//! # S.DEF Import/Export
//!
//! - [`ExportSdefParams`] — Export document to JSON/YAML string
//! - [`ExportSdefDiskParams`] — Export document to directory structure
//! - [`ImportSdefParams`] — Import S.DEF from JSON string
//! - [`ExportShardParams`] — Export a single shard by URI
//! - [`ImportShardParams`] — Import a shard with content
//!
//! # Checkpoint System
//!
//! Checkpoints capture a point-in-time snapshot of:
//! - All tasks (status, input, progress)
//! - All shards (content hash, file path)
//!
//! This enables workflow resumption after crashes.
//!
//! # Transaction Management
//!
//! Tools like `begin_transaction`, `commit_transaction`, `rollback_transaction`
//! support atomic operations via a prepared transaction table.

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

/// Export S.DEF to disk parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExportSdefDiskParams {
    /// Document name to export.
    pub document_name: String,
    /// Output directory (will create {document_name}/ subdir).
    pub output_dir: String,
}

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

/// Transaction ID parameters (for commit/rollback).
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TransactionIdParams {
    pub transaction_id: String,
}
