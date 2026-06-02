//! MCP tool parameter definitions.
//!
//! Each submodule contains [`schemars::JsonSchema`] annotated structs that define
//! the parameters for a group of related MCP tools. These structs are used for:
//!
//! 1. JSON Schema generation — Enables LLMs to understand tool inputs
//! 2. Deserialization — Parses JSON arguments from MCP protocol
//! 3. Documentation — Each field has doc comments describing purpose
//!
//! # Module Organization
//!
//! | Module | Tools | Description |
//! |--------|-------|-------------|
//! | [`task_tools`] | Task lifecycle | Create, claim, complete, fail, list tasks |
//! | [`sdef_tools`] | S.DEF queries | Read data models, contracts, functions |
//! | [`naming_tools`] | Name resolution | Resolve/Register S.DEF URI → concrete names |
//! | [`import_export_tools`] | Serialization | Import/Export S.DEF to/from JSON |
//! | [`lsp_tools`] | Code analysis | LSP-based symbol navigation, diagnostics |
//! | [`consistency_tools`] | Fingerprint checks | Consistency verification and repair |
//! | [`compat_tools`] | Compatibility layers | Deprecation and legacy entity management |
pub mod task_tools;
pub mod sdef_tools;
pub mod naming_tools;
pub mod import_export_tools;
pub mod lsp_tools;
pub mod consistency_tools;
pub mod compat_tools;
pub mod eval_tools;
pub mod task_queue_tools;
pub mod bridge_tools;
pub mod skill_tools;
