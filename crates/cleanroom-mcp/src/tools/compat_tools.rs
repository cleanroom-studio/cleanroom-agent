//! Compatibility mode MCP tool parameters and result types.
//!
//! These tools manage compatibility layers — deprecated and legacy entities
//! that provide backward compatibility across API versions. They enable
//! filtering, inspection, and resolution of compatibility concerns.
//!
//! # Compatibility Modes
//!
//! | Mode | Description |
//! |------|-------------|
//! | `full` | Include all legacy elements, 100% compatibility |
//! | `mixed` | Include compat layers but mark deprecated |
//! | `clean` | Current version only, strip legacy code |
//! | `custom` | User-defined filtering rules |
//!
//! # Compatibility Layers
//!
//! A compatibility layer represents an entity (contract/interface) that
//! provides backward compatibility. Layers can be:
//!
//! - **Deprecated** — Marked for removal, still functional
//! - **Legacy** — Old implementations kept for migration
//! - **Ignored** — User manually resolved (marked as active)
//!
//! # Tools
//!
//! - [`SetCompatModeParams`] — Set the compatibility mode for a document
//! - [`ListCompatLayersParams`] — List all compat layers with status
//! - [`GetCompatLayerParams`] — Get detailed info for a specific layer
//! - [`IgnoreCompatLayerParams`] — Mark a layer as resolved (ignore deprecation)

use rmcp::schemars;
use serde::{Deserialize, Serialize};

/// Set compatibility mode for a document.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SetCompatModeParams {
    /// Document name.
    pub document_name: String,
    /// Compatibility mode: "full", "mixed", "clean", or "custom".
    pub mode: String,
}

/// List compatibility layers for a document.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListCompatLayersParams {
    /// Document name.
    pub document_name: String,
}

/// Get compatibility layer detail.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetCompatLayerParams {
    /// Document name.
    pub document_name: String,
    /// Layer identifier.
    pub layer_id: String,
}

/// Ignore a compatibility layer (mark as resolved).
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct IgnoreCompatLayerParams {
    /// Document name.
    pub document_name: String,
    /// Layer identifier to ignore.
    pub layer_id: String,
}

/// Compatibility layer info.
#[derive(Debug, Clone, Serialize)]
pub struct CompatLayerInfo {
    pub layer_id: String,
    pub source_interface: Option<String>,
    pub target_interface: Option<String>,
    pub transform_type: String,
    pub bidirectional: bool,
    pub is_ignored: bool,
    pub priority: i32,
}

/// Compat mode info.
#[derive(Debug, Clone, Serialize)]
pub struct CompatModeInfo {
    pub document_name: String,
    pub current_mode: String,
    pub layers: Vec<CompatLayerInfo>,
    pub incompatibilities_remaining: usize,
}
