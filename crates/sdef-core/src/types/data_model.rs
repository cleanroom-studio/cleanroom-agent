//! Data model types — entities, attributes, relationships, and physical design.
//!
//! The data model section describes the structural data aspects of the software:
//! entities (like "User", "Order"), their attributes, relationships between them,
//! and physical design hints for database implementation.
//!
//! # Data Model Lifecycle
//!
//! | Status | Meaning |
//! |--------|---------|
//! | `active` | Currently in use, fully supported |
//! | `deprecated` | Still functional but marked for removal |
//! | `legacy` | Old implementation, kept for backward compatibility |
//!
//! # Attribute Types
//!
//! Common logical types: `UUID`, `string`, `boolean`, `integer`, `decimal`,
//! `timestamp`, `json`, `email`, `uri`, `url`
//!
//! # Relationships
//!
//! | Kind | Meaning |
//! |------|---------|
//! | `belongs_to` | Foreign key in this entity points to target |
//! | `has_many` | Target has foreign key pointing to this entity |
//! | `many_to_many` | Uses join table |

use serde::{Deserialize, Serialize};

use super::reconstruction_policy::ElementOrigin;
use super::versioning::{CompatibilityMapping, DeprecationInfo};

/// A Data Model describes a data entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataModel {
    /// Entity name (e.g. "User", "Order").
    pub entity: String,

    /// Lifecycle status: "active" | "deprecated" | "legacy".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<DeprecationInfo>,

    /// What this entity represents in the domain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The logical model in natural language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_model: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Vec<DataAttribute>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub relationships: Option<Vec<DataRelationship>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_rules: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_design: Option<PhysicalDesign>,

    /// Reconstruction provenance (PTDL) — see [`ElementOrigin`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<ElementOrigin>,
}

/// An attribute of a data entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAttribute {
    pub name: String,

    /// Logical type (e.g. "UUID", "string", "boolean", "timestamp", "Decimal").
    #[serde(rename = "type")]
    pub attr_type: String,

    /// A format hint (e.g. "email", "uri", "uuid").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default)]
    pub required: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,

    /// Whether this is the identity / primary key.
    #[serde(default)]
    pub identity: bool,

    /// Whether this value is auto-generated.
    #[serde(default)]
    pub generated: bool,

    #[serde(default)]
    pub unique: bool,

    #[serde(default)]
    pub internal: bool,

    #[serde(default)]
    pub deprecated: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<CompatibilityMapping>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub constraints: Option<Vec<String>>,

    /// Reconstruction provenance (PTDL) — see [`ElementOrigin`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<ElementOrigin>,
}

/// A relationship between data entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRelationship {
    /// Relationship kind (e.g. "belongs_to", "has_many", "many_to_many").
    pub kind: String,

    pub target: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub foreign_key: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub join_table: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub on_delete: Option<String>,
}

/// Physical design hints.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhysicalDesign {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_key: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexes: Option<Vec<IndexDefinition>>,
}

/// Index definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub fields: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
}
