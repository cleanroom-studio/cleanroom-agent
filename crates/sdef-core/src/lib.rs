//! sdef-core — Rust types for S.DEF (Software Definition Exchange Format)
//!
//! Canonical source: S.DEF/schema/draft/schema.ts
//!
//! All types are serde-serializable for JSON/YAML export.

pub mod types;
pub mod version;

// Re-export types from submodules
pub use types::root::SoftwareDefinition;
pub use types::metadata::{Author, SoftwareMetadata};
pub use types::data_model::{DataAttribute, DataModel, DataRelationship, IndexDefinition, PhysicalDesign};
pub use types::behavior::{Behavior, EdgeCase, FlowParticipant, FlowSpec, FlowStep, FunctionParam, FunctionSpec, StateMachine};
pub use types::contracts::{ApiContract, ClassContract, CompatibilityModule, ContractMethod, Contracts, DataMigrationStub, EnumContract, InterfaceContract};
pub use types::design_decisions::DesignDecision;
pub use types::versioning::{CompatibilityMapping, DataMigration, DeprecationInfo, VersionRecord};
pub use version::CURRENT_SCHEMA_VERSION;
pub use types::shard::{ShardMetadata, ShardStatus, ShardType};