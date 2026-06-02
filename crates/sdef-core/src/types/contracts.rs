//! Contract types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::reconstruction_policy::ElementOrigin;
use super::versioning::{CompatibilityMapping, DeprecationInfo};

/// Contracts — interfaces, classes, enums, and API endpoints.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Contracts {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interfaces: Option<Vec<InterfaceContract>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub classes: Option<Vec<ClassContract>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub enums: Option<Vec<EnumContract>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub apis: Option<Vec<ApiContract>>,

    /// Compatibility modules.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compatibility_modules: Option<Vec<CompatibilityModule>>,

    /// Data migration specifications.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_migrations: Option<Vec<DataMigrationStub>>,
}

/// A compatibility module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityModule {
    pub id: String,
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub targets_versions: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub interfaces: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<String>>,
}

/// An abstract interface contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceContract {
    pub name: String,

    #[serde(default)]
    pub is_abstract: bool,

    /// Lifecycle status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<DeprecationInfo>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub methods: Option<Vec<ContractMethod>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub invariants: Option<Vec<String>>,

    /// Reconstruction provenance (PTDL) — see [`ElementOrigin`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<ElementOrigin>,
}

/// A concrete class contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassContract {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<DeprecationInfo>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub implements: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub methods: Option<Vec<ContractMethod>>,

    /// Reconstruction provenance (PTDL) — see [`ElementOrigin`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<ElementOrigin>,
}

/// A method on an interface or class contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractMethod {
    /// Method signature (e.g. "create_task(input: CreateTaskInput) -> Task").
    pub signature: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<DeprecationInfo>,

    /// Behavioral description in natural language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub behavior: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub preconditions: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub postconditions: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub errors: Option<Vec<String>>,

    /// Reconstruction provenance (PTDL) — see [`ElementOrigin`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<ElementOrigin>,
}

/// An enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumContract {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<DeprecationInfo>,

    pub values: Vec<String>,
}

/// An API endpoint contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiContract {
    /// HTTP method or protocol operation.
    pub method: String,

    /// API path.
    pub path: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<DeprecationInfo>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<CompatibilityMapping>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<ApiRequest>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<HashMap<String, ApiResponse>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub side_effects: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<String>,

    /// Reconstruction provenance (PTDL) — see [`ElementOrigin`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<ElementOrigin>,
}

/// API request specification.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApiRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_rules: Option<Vec<String>>,
}

/// API response specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Stub for data migration referenced in Contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMigrationStub {
    pub id: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    pub from_entity: String,
    pub to_entity: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_version: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_version: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
}
