//! Figma-specific UI extensions for S.DEF.
//!
//! These types back the `S.DEF/proposals/0000-figma-ui-import.md` proposal.
//! They are extensions to the S.DEF UI section that allow round-trip
//! representation of Figma-specific concepts: component variants, variables
//! and modes, layout grids, and design-tool import provenance.
//!
//! All types in this module are `Option`-flavored at the use site (i.e. they
//! are nested inside the base UI types as optional fields) so existing
//! S.DEF documents that do not use Figma remain valid unchanged.

use serde::{Deserialize, Serialize};

use super::ui::UIComponentType;

// ============================================================================
// Component variants
// ============================================================================

/// A Figma component set variant (e.g. `Size=Small, State=Default`).
///
/// References a shared `base_component_ref` and lists the property values
/// that distinguish this variant from its siblings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIComponentVariant {
    /// Variant identifier. For Figma: the variant node id.
    pub id: String,

    /// Variant name. For Figma: e.g. `"Size=Small, State=Default"`.
    pub name: String,

    /// Property values that distinguish this variant.
    /// Key = property name, value = property value.
    pub property_values: std::collections::BTreeMap<String, String>,

    /// Reference to the base component's reusable UI element.
    pub base_component_ref: String,

    /// Property overrides that this variant applies on top of the base.
    /// Keys are property names from [`UIComponentProperty`] on the base;
    /// values are the variant-specific settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub property_overrides: Option<serde_json::Value>,
}

/// A property descriptor on a component set.
///
/// In Figma, this is `componentPropertyDefinitions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIComponentProperty {
    /// Property name. For Figma: e.g. `"Size"`, `"State"`, `"Icon"`.
    pub name: String,

    /// Property type — controls the consumer's behavior when rendering
    /// a variant instance.
    pub property_type: UIComponentPropertyType,

    /// Allowed values for `"text"` and `"variant"` type properties.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_values: Option<Vec<String>>,

    /// Default value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_value: Option<UIPropertyDefault>,
}

/// Type discriminator for a component property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UIComponentPropertyType {
    /// Boolean toggle (e.g. `HasIcon`).
    Boolean,
    /// Text input (e.g. `Label = "Submit"`).
    Text,
    /// Swap-in for another component (e.g. `Icon` → pick an icon component).
    InstanceSwap,
    /// Nested variant selection.
    Variant,
}

/// Default value for a component property. Either a string or a boolean.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UIPropertyDefault {
    String(String),
    Bool(bool),
}

// ============================================================================
// Layout grids
// ============================================================================

/// A Figma-style layout grid (columns / rows / grid / stretched).
///
/// Mirrors the `layoutGrids` field on a Figma frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UILayoutGrid {
    /// Grid pattern.
    pub pattern: UILayoutGridPattern,

    /// Cell size in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_size: Option<f64>,

    /// Number of columns / rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,

    /// Gutter between cells, in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gutter: Option<f64>,

    /// Margin from the frame edges, in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,

    /// Tint color for the grid overlay (e.g. `"rgba(255,0,0,0.1)"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tint: Option<String>,

    /// Whether the grid is visible by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UILayoutGridPattern {
    Columns,
    Rows,
    Grid,
    Stretched,
}

// ============================================================================
// Variable modes
// ============================================================================

/// A Figma variable mode (e.g. `"Light"`, `"Dark"`, `"Brand A"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIVariableMode {
    /// Mode identifier. For Figma: the mode's `modeId`.
    pub id: String,

    /// Display name.
    pub name: String,

    /// When `true`, this is the default mode for the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_default: Option<bool>,
}

// ============================================================================
// UI import provenance
// ============================================================================

/// Per-document provenance for the UI.
///
/// Records which source design tool, file, page, and frame each UI element
/// came from. The `node_map` enables round-trip integrity checks:
///
/// ```text
/// node_map["figma-node-id"] -> "sdef-element-id"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIImportProvenance {
    /// Source tool.
    pub source: UIImportSource,

    /// Free-form source identifier (Figma file key, Sketch Cloud ID, etc.).
    pub source_id: String,

    /// Source-file version / revision / branch
    /// (Figma's `lastModified`, Git SHA, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_version: Option<String>,

    /// ISO-8601 timestamp of the import.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub imported_at: Option<String>,

    /// SHA-256 of the source file content, for round-trip integrity checks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,

    /// Per-page mapping: source page id → S.DEF UI shard id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_map: Option<std::collections::BTreeMap<String, String>>,

    /// Per-element mapping: source node id → S.DEF element id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_map: Option<std::collections::BTreeMap<String, String>>,
}

/// The source design tool of a UI import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UIImportSource {
    Figma,
    Sketch,
    Xd,
    Pen,
    /// Anything else. The `source_id` is free-form.
    Other,
}

// ============================================================================
// `UIComponentType` extension helpers
// ============================================================================

impl UIComponentType {
    /// Whether this component is a Figma-style component set (has variants).
    pub fn is_component_set(&self) -> bool {
        self.variants.as_ref().is_some_and(|v| !v.is_empty())
    }
}
