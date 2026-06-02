//! sdef-core — Rust type definitions for S.DEF (Software Definition Exchange Format).
//!
//! S.DEF is a structured format for describing software systems comprehensively.
//! It captures everything from data models and contracts to behavior, architecture,
//! UI design, and deployment requirements.
//!
//! # Canonical Source
//!
//! The canonical TypeScript schema lives at: `S.DEF/schema/draft/schema.ts`
//! This Rust crate is a direct port of that schema with Rust-specific optimizations.
//!
//! # S.DEF Document Structure
//!
//! A complete S.DEF document contains these major sections:
//!
//! | Section | Type | Description |
//! |---------|------|-------------|
//! | Metadata | [`SoftwareMetadata`] | Project info, authors, license |
//! | System Boundary | [`SystemBoundary`] | What the software does/doesn't do |
//! | Design Decisions | [`Vec<DesignDecision>`] | Architectural choices with rationale |
//! | Domain | [`Domain`] | Domain model with entities and relationships |
//! | Architecture | [`Architecture`] | Layers, modules, communication patterns |
//! | Data Models | [`Vec<DataModel>`] | Entity definitions with attributes |
//! | Contracts | [`Contracts`] | Interfaces, classes, enums, APIs |
//! | Behavior | [`Behavior`] | Functions, flows, state machines |
//! | UI | [`UserInterface`] | Screens, components, navigation |
//! | Tests | [`TestContract`] | Test specifications |
//! | Reconstruction Rules | [`ReconstructionRules`] | Time-dimension generation directives (v1 vs v2) |
//! | Reconstruction Policy (PTDL) | [`ReconstructionPolicy`] | Language-dimension directives (C vs Rust) |
//! | Versioning | [`Vec<VersionRecord>`] | Version history and migrations |
//! | Dependencies | [`Vec<Dependency>`] | External dependencies |
//! | Deployment | [`Deployment`] | Runtime and deployment requirements |
//!
//! # Serialization
//!
//! All types implement [`serde::Serialize`] and [`serde::Deserialize`] for
//! JSON/YAML export. The schema version is date-based (e.g., `"2026-05-27"`).
//!
//! # Example
//!
//! ```rust
//! use sdef_core::{SoftwareDefinition, SoftwareMetadata, DataModel, DataAttribute};
//!
//! let mut sdef = SoftwareDefinition::default();
//! sdef.sdef_version = "2026-05-27".to_string();
//! sdef.name = "com.example.todoapp".to_string();
//! sdef.metadata = Some(SoftwareMetadata {
//!     authors: None,
//!     license: Some("MIT".to_string()),
//!     homepage: None,
//!     repository: None,
//!     category: Some("web_application".to_string()),
//!     tags: None,
//!     target_platforms: None,
//!     compatibility_policy: None,
//!     annotations: None,
//! });
//!
//! // Serialize to JSON
//! let json = serde_json::to_string_pretty(&sdef).unwrap();
//! ```
//!
//! # Validation
//!
//! Use [`validator::validate()`] to check an S.DEF document for:
//! - Required fields (`sdef_version`, `name`)
//! - URI reference integrity (sdef:// URIs point to existing entities)
//! - Naming conventions
//! - Cross-reference validation (e.g., class implements existing interfaces)

pub mod types;
pub mod version;
pub mod validator;

// Re-export types from submodules
pub use types::root::SoftwareDefinition;
pub use types::metadata::{Author, SoftwareMetadata};
pub use types::data_model::{DataAttribute, DataModel, DataRelationship, IndexDefinition, PhysicalDesign};
pub use types::behavior::{Behavior, EdgeCase, FlowParticipant, FlowSpec, FlowStep, FunctionParam, FunctionSpec, StateMachine};
pub use types::contracts::{ApiContract, ClassContract, CompatibilityModule, ContractMethod, Contracts, DataMigrationStub, EnumContract, InterfaceContract};
pub use types::design_decisions::DesignDecision;
pub use types::architecture::{Architecture, ArchitectureLayer, ArchitectureModule};
pub use types::versioning::{CompatibilityMapping, DataMigration, DeprecationInfo, VersionRecord};
pub use version::CURRENT_SCHEMA_VERSION;
pub use types::shard::{ShardMetadata, ShardStatus, ShardType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_software_definition_default() {
        let def = SoftwareDefinition::default();
        assert_eq!(def.sdef_version, "");
        assert_eq!(def.name, "");
        assert!(def.description.is_none());
    }

    #[test]
    fn test_json_roundtrip() {
        let mut def = SoftwareDefinition::default();
        def.sdef_version = "2026-05-27".to_string();
        def.name = "com.example.test".to_string();
        def.description = Some("A test project".to_string());
        def.metadata = Some(SoftwareMetadata {
            authors: None,
            license: Some("MIT".to_string()),
            homepage: None,
            repository: None,
            category: None,
            tags: None,
            target_platforms: None,
            compatibility_policy: None,
            annotations: None,
        });

        let json = serde_json::to_string_pretty(&def).unwrap();
        let deserialized: SoftwareDefinition = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.sdef_version, "2026-05-27");
        assert_eq!(deserialized.name, "com.example.test");
        assert_eq!(deserialized.description.unwrap(), "A test project");
        assert_eq!(deserialized.metadata.unwrap().license.unwrap(), "MIT");
    }

    #[test]
    fn test_yaml_roundtrip() {
        let mut def = SoftwareDefinition::default();
        def.sdef_version = "2026-05-27".to_string();
        def.name = "com.example.yaml-test".to_string();

        let yaml = serde_yaml::to_string(&def).unwrap();
        let deserialized: SoftwareDefinition = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(deserialized.sdef_version, "2026-05-27");
        assert_eq!(deserialized.name, "com.example.yaml-test");
    }

    #[test]
    fn test_data_model_roundtrip() {
        let model = DataModel {
            entity: "User".to_string(),
            status: None,
            version: Some("1.0".to_string()),
            deprecated: None,
            description: Some("A system user".to_string()),
            logical_model: None,
            attributes: Some(vec![
                DataAttribute {
                    name: "id".to_string(),
                    attr_type: "UUID".to_string(),
                    format: None,
                    description: Some("Primary key".to_string()),
                    required: true,
                    default: None,
                    identity: true,
                    generated: true,
                    unique: true,
                    internal: false,
                    deprecated: false,
                    compatibility: None,
                    constraints: None,
                    origin: None,
                },
                DataAttribute {
                    name: "email".to_string(),
                    attr_type: "string".to_string(),
                    format: Some("email".to_string()),
                    description: Some("Email address".to_string()),
                    required: true,
                    default: None,
                    identity: false,
                    generated: false,
                    unique: true,
                    internal: false,
                    deprecated: false,
                    compatibility: None,
                    constraints: None,
                    origin: None,
                },
            ]),
            relationships: None,
            validation_rules: None,
            physical_design: None,
            origin: None,
        };

        let json = serde_json::to_string_pretty(&model).unwrap();
        let deserialized: DataModel = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.entity, "User");
        assert_eq!(deserialized.version.clone().unwrap(), "1.0");
        let attrs = deserialized.attributes.unwrap();
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].name, "id");
        assert_eq!(attrs[1].name, "email");
        assert_eq!(attrs[1].format.clone().unwrap(), "email");
    }

    #[test]
    fn test_design_decision_roundtrip() {
        let decision = DesignDecision {
            id: "dec-001".to_string(),
            topic: "Database selection".to_string(),
            decision: "Use PostgreSQL".to_string(),
            rationale: "Need ACID compliance".to_string(),
            context: Some("Initial architecture planning".to_string()),
            alternatives: Some(vec!["MySQL".to_string(), "SQLite".to_string()]),
            consequences: Some(vec!["Requires dedicated server".to_string()]),
            constraints: Some(vec!["Must be open source".to_string()]),
        };

        let json = serde_json::to_string(&decision).unwrap();
        let deserialized: DesignDecision = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "dec-001");
        assert_eq!(deserialized.topic, "Database selection");
        assert_eq!(deserialized.alternatives.unwrap().len(), 2);
    }

    #[test]
    fn test_version_record_roundtrip() {
        let record = VersionRecord {
            version: "2.0.0".to_string(),
            release_date: Some("2024-01-15".to_string()),
            deprecated: false,
            eol_date: Some("2025-01-15".to_string()),
            breaking_changes: Some(vec!["API v1 removed".to_string()]),
            compatibility_notes: Some("Use migration guide".to_string()),
        };

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: VersionRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.version, "2.0.0");
        assert_eq!(deserialized.breaking_changes.unwrap()[0], "API v1 removed");
    }

    #[test]
    fn test_function_spec_roundtrip() {
        let func = types::behavior::FunctionSpec {
            name: "createUser".to_string(),
            description: Some("Creates a new user".to_string()),
            inputs: Some(vec![
                types::behavior::FunctionParam {
                    name: "email".to_string(),
                    param_type: "string".to_string(),
                    description: Some("User email".to_string()),
                },
            ]),
            outputs: Some(vec![
                types::behavior::FunctionParam {
                    name: "userId".to_string(),
                    param_type: "UUID".to_string(),
                    description: Some("Created user ID".to_string()),
                },
            ]),
            logic: Some("INSERT INTO users (email) VALUES (email)".to_string()),
            complexity: Some("O(1)".to_string()),
            pure_function: false,
            edge_cases: Some(vec![types::behavior::EdgeCase {
                condition: "Duplicate email".to_string(),
                expected_behavior: "Return error".to_string(),
            }]),
            origin: None,
        };

        let json = serde_json::to_string(&func).unwrap();
        let deserialized: types::behavior::FunctionSpec = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "createUser");
        assert!(deserialized.inputs.is_some());
        assert!(!deserialized.pure_function);
    }

    #[test]
    fn test_author_serialization() {
        let author = Author {
            name: "Test Author".to_string(),
            email: Some("test@example.com".to_string()),
            url: Some("https://example.com".to_string()),
        };

        let json = serde_json::to_string(&author).unwrap();
        let deserialized: Author = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "Test Author");
        assert_eq!(deserialized.email.unwrap(), "test@example.com");
    }

    #[test]
    fn test_shard_types_serialization() {
        let shard = ShardMetadata {
            shard_id: "shard-001".to_string(),
            sdef_uri: "sdef://test/entity".to_string(),
            section_type: ShardType::Metadata,
            file_path: None,
            size_bytes: None,
            token_estimate: Some(1500),
            content_hash: Some("abc123".to_string()),
            dependencies: None,
            status: ShardStatus::Generated,
        };

        let json = serde_json::to_string(&shard).unwrap();
        let deserialized: ShardMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.shard_id, "shard-001");
        assert_eq!(deserialized.token_estimate.unwrap(), 1500);
        assert_eq!(deserialized.status, ShardStatus::Generated);
    }
}