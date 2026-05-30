//! Consistency check MCP tool parameters and result types.
//!
//! These tools manage fingerprint-based consistency verification between
//! S.DEF documents, the database, and generated code. They enable
//! detection and repair of drift between layers.
//!
//! # Three-Way Consistency Model
//!
//! Each tracked entity maintains three hashes:
//!
//! - **sdef_hash** — Hash of the S.DEF document definition
//! - **db_hash** — Hash of the database record
//! - **code_hash** — Hash of the generated code
//!
//! An entity is **consistent** when all three hashes match.
//! An entity is **inconsistent** when any hash differs.
//!
//! # Fix Strategies
//!
//! | Strategy | Description |
//! |----------|-------------|
//! | `sync_code_to_sdef` | Update code to match S.DEF |
//! | `regenerate_code` | Regenerate code from scratch |
//! | `sync_db_to_sdef` | Update DB to match S.DEF |
//! | `sync_sdef_to_db` | Update S.DEF to match DB |
//! | `accept_external` | Mark external modification as intentional |
//!
//! # Tools
//!
//! - [`ConsistencyCheckParams`] — Run consistency check, return inconsistent entities
//! - [`FingerprintParams`] — Compute/refresh fingerprints for a document
//! - [`ResolveInconsistencyParams`] — Apply a fix strategy to resolve drift
//! - [`InconsistencyReportParams`] — Get detailed report with suggested strategies

use rmcp::schemars;
use serde::{Deserialize, Serialize};

/// Run a consistency check.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConsistencyCheckParams {
    /// Document name to check.
    pub document_name: String,
    /// Check type: "fast" (default), "full", or "deep".
    #[serde(default = "default_check_type")]
    pub check_type: String,
}

fn default_check_type() -> String { "fast".to_string() }

/// Compute fingerprints parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FingerprintParams {
    /// Document name.
    pub document_name: String,
}

/// Resolve an inconsistency.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ResolveInconsistencyParams {
    /// Document name.
    pub document_name: String,
    /// Entity URI to fix.
    pub entity_uri: String,
    /// Fix strategy: "sync_code_to_sdef", "regenerate_code", "sync_db_to_sdef",
    /// "sync_sdef_to_db", or "accept_external".
    pub strategy: String,
}

/// Get inconsistency report.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct InconsistencyReportParams {
    /// Document name.
    pub document_name: String,
    /// Optional filter by entity type.
    pub entity_type: Option<String>,
}

/// Inconsistency item for report output.
#[derive(Debug, Clone, Serialize)]
pub struct InconsistencyItem {
    pub entity_uri: String,
    pub entity_type: String,
    pub code_path: Option<String>,
    pub sdef_hash: Option<String>,
    pub db_hash: Option<String>,
    pub code_hash: Option<String>,
    pub last_consistent_at: Option<String>,
    pub suggested_strategies: Vec<String>,
}

/// Inconsistency report output.
#[derive(Debug, Clone, Serialize)]
pub struct InconsistencyReport {
    pub document_name: String,
    pub total_inconsistencies: usize,
    pub items: Vec<InconsistencyItem>,
}
