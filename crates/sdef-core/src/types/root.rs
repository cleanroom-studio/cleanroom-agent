//! Root document type — the top-level [`SoftwareDefinition`].
//!
//! This is the entry point for any S.DEF document. It aggregates all
//! other type modules as optional fields, allowing partial documents
//! while ensuring the schema version and name are always present.
//!
//! # Document Structure
//!
//! ```text
//! SoftwareDefinition
//! ├── sdef_version          (required)
//! ├── name                  (required)
//! ├── description
//! ├── version
//! ├── metadata
//! ├── system_boundary
//! ├── design_decisions
//! ├── version_history
//! ├── domain
//! ├── architecture
//! ├── data_models
//! ├── contracts
//! ├── behavior
//! ├── ui
//! ├── tests
//! ├── reconstruction_rules         (time dimension: v1 vs v2 compat)
//! ├── reconstruction_policy        (PTDL: language dimension: C vs Rust)
//! ├── dependencies
//! ├── deployment
//! └── resources
//! ```
//!
//! # URI References
//!
//! Entities within the document are referenced using `sdef://` URIs:
//!
//! | Entity | URI Pattern |
//! |--------|-------------|
//! | Data Model | `sdef://{name}/data-models/{entity}` |
//! | Interface | `sdef://{name}/contracts/interfaces/{name}` |
//! | Function | `sdef://{name}/behavior/functions/{name}` |
//!
//! # Example
//!
//! ```rust
//! use sdef_core::SoftwareDefinition;
//!
//! let sdef = SoftwareDefinition {
//!     sdef_version: "2026-05-27".to_string(),
//!     name: "com.example.myapp".to_string(),
//!     description: Some("Example application".to_string()),
//!     ..Default::default()
//! };
//! ```

use serde::{Deserialize, Serialize};

use super::{metadata, system_boundary, design_decisions, versioning, domain, architecture, data_model, contracts, behavior, ui, tests, reconstruction, reconstruction_policy, deployment, dependencies};

/// The root object of any S.DEF document.
/// Contains all layers of software description.
///
/// # Mandatory Fields
///
/// - `sdef_version` — Schema version (date-based, e.g. "2026-05-27")
/// - `name` — Software identifier (e.g. "com.example.todoapp")
///
/// # All Optional Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `description` | `String` | Human-readable description |
/// | `version` | `String` | Software version being described |
/// | `metadata` | `SoftwareMetadata` | Authors, license, repository |
/// | `system_boundary` | `SystemBoundary` | Inclusions/exclusions |
/// | `design_decisions` | `Vec<DesignDecision>` | Architectural decisions |
/// | `version_history` | `Vec<VersionRecord>` | Version changelog |
/// | `domain` | `Domain` | Domain model |
/// | `architecture` | `Architecture` | Structural layers |
/// | `data_models` | `Vec<DataModel>` | Entity definitions |
/// | `contracts` | `Contracts` | Interfaces, classes, APIs |
/// | `behavior` | `Behavior` | Functions, flows, state machines |
/// | `ui` | `UserInterface` | Screens and components |
/// | `tests` | `TestContract` | Test specifications |
/// | `reconstruction_rules` | `ReconstructionRules` | Generation directives (time dimension) |
/// | `reconstruction_policy` | `ReconstructionPolicy` | PTDL — cross-language/paradigm directives |
/// | `dependencies` | `Vec<Dependency>` | External dependencies |
/// | `deployment` | `Deployment` | Runtime requirements |
/// | `resources` | `Vec<Resource>` | Provided/consumed resources |
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SoftwareDefinition {
    /// Schema version (date-based, e.g. "2026-05-27").
    pub sdef_version: String,

    /// The software identifier (e.g. "com.example.todoapp").
    pub name: String,

    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Version of the software being described.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<metadata::SoftwareMetadata>,

    /// System boundary — what the software does and does NOT do.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_boundary: Option<system_boundary::SystemBoundary>,

    /// Design decisions with rationale.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub design_decisions: Option<Vec<design_decisions::DesignDecision>>,

    /// Version history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version_history: Option<Vec<versioning::VersionRecord>>,

    /// Domain model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<domain::Domain>,

    /// Architecture — structural layers, modules, and communication patterns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<architecture::Architecture>,

    /// Data model — entities, attributes, relationships, and validation rules.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_models: Option<Vec<data_model::DataModel>>,

    /// Contracts — interfaces, classes, enums, and API endpoints.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contracts: Option<contracts::Contracts>,

    /// Behavior — functions, flows, and state machines.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub behavior: Option<behavior::Behavior>,

    /// User interface — screens, components, interactions, and navigation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ui: Option<ui::UserInterface>,

    /// Test contracts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tests: Option<tests::TestContract>,

    /// Reconstruction rules — fidelity target, technology constraints, and directives
    /// (time dimension: v1 vs v2 compatibility).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reconstruction_rules: Option<reconstruction::ReconstructionRules>,

    /// Reconstruction policy (PTDL) — cross-language/paradigm directives
    /// (language dimension: C vs Rust, OCaml vs Java). See [`reconstruction_policy::ReconstructionPolicy`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reconstruction_policy: Option<reconstruction_policy::ReconstructionPolicy>,

    /// External software dependencies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<Vec<dependencies::Dependency>>,

    /// Deployment and runtime requirements.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deployment: Option<deployment::Deployment>,

    /// Resources the software provides or consumes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<Vec<dependencies::Resource>>,
}
