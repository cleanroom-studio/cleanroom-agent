//! S.DEF type modules.
//!
//! This module organizes all type definitions that make up the S.DEF schema.
//! Each submodule corresponds to a major section of the Software Definition.
//!
//! # Module Structure
//!
//! | Module | Root Field | Description |
//! |--------|------------|-------------|
//! | [`root`] | — | [`SoftwareDefinition`] — the root document type |
//! | [`metadata`] | `metadata` | [`SoftwareMetadata`], [`Author`] |
//! | [`system_boundary`] | `system_boundary` | Scope inclusions/exclusions |
//! | [`design_decisions`] | `design_decisions` | Architectural decisions |
//! | [`versioning`] | `version_history` | [`VersionRecord`], deprecation, migrations |
//! | [`domain`] | `domain` | Domain model entities |
//! | [`architecture`] | `architecture` | Layers, modules, communication |
//! | [`data_model`] | `data_models` | [`DataModel`], [`DataAttribute`], relationships |
//! | [`contracts`] | `contracts` | Interfaces, classes, enums, APIs |
//! | [`behavior`] | `behavior` | [`FunctionSpec`], flows, state machines |
//! | [`ui`] | `ui` | Screens, components, navigation |
//! | [`tests`] | `tests` | Test specifications |
//! | [`reconstruction`] | `reconstruction_rules` | Code generation directives |
//! | [`reconstruction_policy`] | `reconstruction_policy` | PTDL — cross-language/paradigm directives |
//! | [`deployment`] | `deployment` | Runtime requirements |
//! | [`dependencies`] | `dependencies` | External dependencies, resources |
//! | [`shard`] | — | [`ShardMetadata`] — internal runtime tracking (not exchanged) |
//!
//! # Type Annotations
//!
//! All types use `#[derive(Serialize, Deserialize)]` for JSON/YAML support.
//! Optional fields use `#[serde(skip_serializing_if = "Option::is_none")]` for concise output.
//!
//! # Naming Conventions
//!
//! - `sdef_version` — Date-based schema version (YYYY-MM-DD)
//! - Entity names — PascalCase (e.g., `User`, `OrderItem`)
//! - Attribute names — camelCase (e.g., `createdAt`, `userId`)
//! - S.DEF URIs — `sdef://{document}/{section}/{entity}`

pub mod root;       // SoftwareDefinition
pub mod metadata;   // SoftwareMetadata, Author
pub mod system_boundary;
pub mod design_decisions;
pub mod versioning;    // VersionRecord, DeprecationInfo, CompatibilityMapping, DataMigration
pub mod domain;
pub mod architecture;
pub mod data_model;
pub mod contracts;
pub mod behavior;
pub mod ui;
pub mod tests;
pub mod reconstruction;
pub mod reconstruction_policy;   // PTDL — ElementOrigin, ParadigmMetadata, LibrarySubstitution, TransformationHint
pub mod deployment;
pub mod dependencies;
pub mod shard;       // ShardMetadata, ShardType, ShardStatus
