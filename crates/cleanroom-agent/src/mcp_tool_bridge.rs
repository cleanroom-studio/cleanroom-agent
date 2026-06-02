//! `mcp_tool_bridge` — wrap MCP server tools as `cleanroom_meta_core::tool::MetaToolT`
//! so the LLM agent loop can call them through ReAct / tool-calling.
//!
//! # Motivation
//!
//! The Producer / Consumer in Phase 0.5+ are driven by `llm_loop::run_loop_via_basic_agent`
//! (and, in later phases, a ReAct executor). The agent abstraction hands the
//! LLM a list of `MetaToolT` impls; when the LLM emits a tool call the runtime
//! invokes `MetaToolT::execute(args)` and feeds the result back into the
//! conversation.
//!
//! `cleanroom-mcp` already exposes 56 tools (DB queries, S.DEF I/O, naming,
//! consistency checks, LSP dispatch, etc.) through a single
//! `dispatch_tool_call` entry point. Instead of reimplementing each one as a
//! `MetaToolT`, this module provides a thin bridge. The bundled
//! [`mcp_tool_catalog`] is a curated **40-entry subset** of those 56 tools —
//! the ones that are useful for code analysis / generation (LSP lookups,
//! task queue, S.DEF I/O, naming, consistency). Mutating-only / debug-only
//! tools (e.g. `set_compatibility_mode`, raw DB transactions) are
//! intentionally excluded from the LLM-facing catalog.
//!
//! ```text
//! LLM tool call (json args)
//!     |
//!     v
//! McpToolBridge::execute(args)        <-- MetaToolT impl (this file)
//!     |
//!     v
//! Arc<dyn Fn(Value) -> Result<Value, String>>   <-- user-supplied closure
//!     |
//!     v
//! cleanroom_mcp::CleanroomMcpServer::dispatch_tool_call(...)
//! ```
//!
//! # Why a closure, not a concrete `Arc<CleanroomMcpServer>` field
//!
//! - `cleanroom-agent` does not (and should not) depend on `cleanroom-mcp`
//!   directly; the wire boundary between the two is the binary CLI
//!   (`cleanroom-cli`). A closure keeps the bridge crate-agnostic.
//! - Tests can inject a mock dispatch closure (a captured `Vec<Value>` or a
//!   `Mutex<HashMap<String, Value>>`) without spinning up an MCP server.
//! - Crates that DO have the MCP server (e.g. `cleanroom-cli`, integration
//!   tests) wire the closure in two lines: `|args| server.dispatch_tool_call(...)`.
//!
//! # Wiring example (in `cleanroom-cli` or any integration code)
//!
//! ```ignore
//! use std::sync::Arc;
//! use cleanroom_agent::mcp_tool_bridge::{McpToolBridge, bridge_all_mcp_tools};
//! use cleanroom_meta_core::tool::to_llm_tool;
//! use cleanroom_mcp::CleanroomMcpServer;
//!
//! let server = Arc::new(CleanroomMcpServer::new(db.clone()));
//! // bridge_all_mcp_tools() is a hand-curated list of 37 tools -- name +
//! // description + JSON schema. Each McpToolBridge holds a closure that
//! // routes to `server.dispatch_tool_call(...)`.
//! let bridges: Vec<McpToolBridge> = bridge_all_mcp_tools(server.clone());
//! let boxed: Vec<Box<dyn MetaToolT>> = bridges.into_iter().map(|b| Box::new(b) as _).collect();
//! // Convert to LLM `Tool` definitions (name + description + parameters) for
//! // the chat() call:
//! let llm_tools: Vec<Tool> = boxed.iter().map(|b| to_llm_tool(b)).collect();
//! ```

use std::fmt;
use std::sync::Arc;

use cleanroom_meta_core::tool::{MetaToolT, ToolCallError, ToolRuntime};
use cleanroom_meta_llm::chat::MetaFunctionTool;
use serde_json::{json, Value};

/// Type alias for the user-supplied dispatch closure.
///
/// Takes a JSON `args` object, returns a JSON result or a `String` error
/// (which the bridge wraps as `ToolCallError::RuntimeError`).
pub type McpDispatchFn = Arc<dyn Fn(Value) -> Result<Value, String> + Send + Sync>;

/// A `MetaToolT` adapter that delegates `execute()` to a user-supplied closure.
///
/// Use [`McpToolBridge::new`] for fully custom tools or
/// [`McpToolBridge::from_dispatch_fn`] when you already have a closure lying
/// around. For the common case of "wrap a single MCP tool", pass
/// `|args| server.dispatch_tool_call(CallToolRequestParams { name, arguments })`
/// in.
pub struct McpToolBridge {
    name: String,
    description: String,
    args_schema: Value,
    dispatch: McpDispatchFn,
}

impl McpToolBridge {
    /// Build a new bridge with the given metadata + dispatch closure.
    ///
    /// `args_schema` MUST be a valid JSON Schema object (the LLM uses it to
    /// decide when to call this tool). If you don't have a real schema,
    /// pass `serde_json::json!({"type": "object"})` as a permissive default.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        args_schema: Value,
        dispatch: McpDispatchFn,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            args_schema,
            dispatch,
        }
    }

    /// Build a bridge where `name` and `args` are both captured in the
    /// closure (typical "wrap one MCP tool" usage).
    pub fn from_dispatch_fn(
        name: impl Into<String>,
        description: impl Into<String>,
        args_schema: Value,
        dispatch: McpDispatchFn,
    ) -> Self {
        // Functionally identical to `new` -- alias kept for call-site
        // readability at the call sites that read like
        //   "give me a bridge whose dispatch is this closure".
        Self::new(name, description, args_schema, dispatch)
    }

    /// Borrow the bridge's name (handy for logging / debug).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the bridge's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Borrow the bridge's JSON schema for arguments.
    pub fn args_schema(&self) -> &Value {
        &self.args_schema
    }

    /// Convert this bridge into an LLM `MetaFunctionTool` description (the
    /// name + description + parameters the LLM sees). Equivalent to
    /// `cleanroom_meta_core::tool::to_llm_tool` for boxed tools but works on
    /// the unboxed bridge directly.
    pub fn to_function_tool(&self) -> MetaFunctionTool {
        MetaFunctionTool {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.args_schema.clone(),
        }
    }
}

impl fmt::Debug for McpToolBridge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpToolBridge")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("args_schema", &self.args_schema)
            // Don't print the closure (Fn isn't Debug)
            .field("dispatch", &"<fn>")
            .finish()
    }
}

impl MetaToolT for McpToolBridge {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args_schema(&self) -> Value {
        self.args_schema.clone()
    }
}

#[cleanroom_meta::async_trait]
impl ToolRuntime for McpToolBridge {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        // The dispatch closure returns `Result<Value, String>`; wrap the
        // String error into the trait's expected `ToolCallError`.
        (self.dispatch)(args).map_err(|e| ToolCallError::RuntimeError(e.into()))
    }
}

// ============================================================================
// MCP tool catalog
// ============================================================================

/// One entry in [`McpToolCatalog`]: tool name + human description + JSON
/// schema for the arguments. Hand-curated to match the 37 tools registered
/// in `cleanroom-mcp/src/tools/*.rs` (see `bridge_tools.rs`,
/// `compat_tools.rs`, `consistency_tools.rs`, `eval_tools.rs`,
/// `import_export_tools.rs`, `lsp_tools.rs`, `naming_tools.rs`,
/// `sdef_tools.rs`, `task_queue_tools.rs`, `task_tools.rs`).
///
/// Schemas are kept permissive (`additionalProperties: true`) on purpose --
/// the MCP server's own `dispatch_tool_call` does the strict validation, and
/// over-specifying here would duplicate logic. Tighten the schemas when
/// individual tools are wired up to the LLM in later phases.
#[derive(Debug, Clone)]
pub struct McpToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub args_schema: Value,
}

impl McpToolSpec {
    pub const fn new(name: &'static str, description: &'static str, args_schema: Value) -> Self {
        Self {
            name,
            description,
            args_schema,
        }
    }
}

/// The full catalog of MCP tools we expose to the LLM. Order is roughly
/// grouped by source file in `cleanroom-mcp/src/tools/` for easy diffing.
///
/// `args_schema` for each tool is `{"type": "object", "additionalProperties": true}`
/// (permissive) -- the real validation happens in the MCP dispatch path.
pub fn mcp_tool_catalog() -> Vec<McpToolSpec> {
    let any_args = || {
        json!({
            "type": "object",
            "additionalProperties": true,
        })
    };

    vec![
        // --- sdef_tools.rs ---
        McpToolSpec::new("upsert_data_model", "Create or update a data model entity in the S.DEF document.", any_args()),
        McpToolSpec::new("get_data_model", "Fetch a single data model by entity name.", any_args()),
        McpToolSpec::new("list_data_models", "List all data models in the current document.", any_args()),
        McpToolSpec::new("upsert_contract", "Create or update an interface contract.", any_args()),
        McpToolSpec::new("upsert_function", "Create or update a pure function spec.", any_args()),
        McpToolSpec::new("upsert_module", "Create or update a module (architecture) record.", any_args()),
        McpToolSpec::new("upsert_design_decision", "Record a design decision.", any_args()),
        McpToolSpec::new("list_shards", "List the S.DEF shards (16K-token fragments) for the current document.", any_args()),
        // --- naming_tools.rs ---
        McpToolSpec::new("resolve_name", "Resolve a URI to a language-specific deterministic name.", any_args()),
        McpToolSpec::new("register_symbol", "Reserve a name in the symbol registry.", any_args()),
        // --- task_tools.rs / task_queue_tools.rs ---
        McpToolSpec::new("claim_task", "Atomically claim the next runnable task for this agent.", any_args()),
        McpToolSpec::new("complete_task", "Mark a claimed task as completed with output JSON.", any_args()),
        McpToolSpec::new("fail_task", "Mark a claimed task as failed (transient -- allows retry).", any_args()),
        McpToolSpec::new("list_tasks", "List tasks (filter by status / type).", any_args()),
        McpToolSpec::new("heartbeat", "Send a heartbeat for a long-running task (prevents zombie reassignment).", any_args()),
        // --- import_export_tools.rs ---
        McpToolSpec::new("import_sdef", "Import an S.DEF document (JSON or YAML) into the DB.", any_args()),
        McpToolSpec::new("export_sdef", "Export the current document as S.DEF (JSON or YAML).", any_args()),
        // --- consistency_tools.rs ---
        McpToolSpec::new("consistency_check", "Run a 3-way consistency check (S.DEF <-> DB <-> code).", any_args()),
        McpToolSpec::new("fingerprint_check", "Compute SHA-256 fingerprints for S.DEF, DB, and source code.", any_args()),
        // --- compat_tools.rs ---
        McpToolSpec::new("detect_compat", "Detect compatibility shims / legacy patterns in source code.", any_args()),
        McpToolSpec::new("resolve_compat", "Resolve a compatibility decision for a specific pattern.", any_args()),
        // --- lsp_tools.rs ---
        McpToolSpec::new("lsp_lookup_type", "Resolve a symbol to its fully-qualified type via the LSP server pool.", any_args()),
        McpToolSpec::new("lsp_analyze_file", "Run an LSP-powered semantic analysis on a single file.", any_args()),
        // --- eval_tools.rs ---
        McpToolSpec::new("evaluate_project", "Run a benchmark evaluation against a benchmark project.", any_args()),
        McpToolSpec::new("coverage_report", "Compute coverage metrics for the current document.", any_args()),
        // --- bridge_tools.rs (cross-cutting) ---
        McpToolSpec::new("checkpoint", "Create a workflow checkpoint (for resume).", any_args()),
        McpToolSpec::new("restore_checkpoint", "Restore workflow state from a checkpoint.", any_args()),
        McpToolSpec::new("list_documents", "List all S.DEF documents in the database.", any_args()),
        McpToolSpec::new("delete_document", "Delete an S.DEF document and all its entities.", any_args()),
        // --- 11 more tools (capped at 37 spec slots) ---
        McpToolSpec::new("get_progress", "Get workflow progress (total/in_progress/completed/failed).", any_args()),
        McpToolSpec::new("get_shard", "Fetch a single S.DEF shard by id.", any_args()),
        McpToolSpec::new("get_task", "Fetch a single task by id.", any_args()),
        McpToolSpec::new("update_task", "Update task priority / input / assigned_to.", any_args()),
        McpToolSpec::new("reprioritize_task", "Change a task's priority.", any_args()),
        McpToolSpec::new("search_sdef", "Full-text search across all S.DEF documents.", any_args()),
        McpToolSpec::new("naming_stats", "Get statistics about the symbol registry (registered count, collisions).", any_args()),
        McpToolSpec::new("completeness_check", "Validate the completeness of the current S.DEF document.", any_args()),
        McpToolSpec::new("two_phase_prepare", "Begin a two-phase commit (locks + snapshot).", any_args()),
        McpToolSpec::new("two_phase_commit", "Finalize a two-phase commit (apply changes).", any_args()),
        McpToolSpec::new("two_phase_rollback", "Roll back a two-phase commit (release locks).", any_args()),
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn echo_dispatch(args: Value) -> Result<Value, String> {
        Ok(json!({ "echo": args }))
    }

    fn failing_dispatch(_args: Value) -> Result<Value, String> {
        Err("simulated MCP failure".to_string())
    }

    #[test]
    fn test_meta_tool_t_contract() {
        let dispatch: McpDispatchFn = Arc::new(echo_dispatch);
        let bridge = McpToolBridge::new("echo", "echoes back its args", json!({"type":"object"}), dispatch);

        // MetaToolT accessors
        assert_eq!(bridge.name(), "echo");
        assert_eq!(bridge.description(), "echoes back its args");
        assert_eq!(bridge.args_schema()["type"], "object");
    }

    #[tokio::test]
    async fn test_execute_passes_args_through_dispatch() {
        let dispatch: McpDispatchFn = Arc::new(echo_dispatch);
        let bridge = McpToolBridge::new("echo", "d", json!({}), dispatch);

        let result = bridge.execute(json!({"x": 1})).await.expect("ok");
        assert_eq!(result["echo"]["x"], 1);
    }

    #[tokio::test]
    async fn test_execute_wraps_dispatch_error_as_runtime_error() {
        let dispatch: McpDispatchFn = Arc::new(failing_dispatch);
        let bridge = McpToolBridge::new("fail", "d", json!({}), dispatch);

        let err = bridge.execute(json!({})).await.expect_err("should fail");
        match err {
            ToolCallError::RuntimeError(e) => {
                assert_eq!(e.to_string(), "simulated MCP failure");
            }
            other => panic!("expected RuntimeError, got {other:?}"),
        }
    }

    #[test]
    fn test_to_function_tool_roundtrip() {
        let dispatch: McpDispatchFn = Arc::new(echo_dispatch);
        let bridge = McpToolBridge::new(
            "my_tool",
            "useful",
            json!({"type":"object","properties":{"q":{"type":"string"}}}),
            dispatch,
        );
        let ft = bridge.to_function_tool();
        assert_eq!(ft.name, "my_tool");
        assert_eq!(ft.description, "useful");
        assert_eq!(ft.parameters["properties"]["q"]["type"], "string");
    }

    #[test]
    fn test_catalog_has_expected_entry_count() {
        // The MCP server has 56 tools total; this catalog exposes a curated
        // subset that are useful for the LLM-driven Producer / Consumer.
        // 40 today; expect that number to grow as more phases wire up more
        // tools. We assert "at least 30" so adding a new tool doesn't break
        // the build, but cap at 50 to catch accidental bulk additions.
        let catalog = mcp_tool_catalog();
        assert!(
            (30..=50).contains(&catalog.len()),
            "catalog should have 30-50 entries (got {})",
            catalog.len(),
        );
        // Names must be unique -- collisions would silently break routing.
        let mut names: Vec<&str> = catalog.iter().map(|s| s.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), catalog.len(), "tool names must be unique");
    }

    #[test]
    fn test_catalog_tool_schemas_are_objects() {
        // Per the JSON Schema spec, top-level `parameters` for an OpenAI-style
        // tool call must be an object. Catch typos that pass `null` or `[]`.
        for spec in mcp_tool_catalog() {
            assert_eq!(
                spec.args_schema["type"], "object",
                "tool {} must declare `type: object`",
                spec.name,
            );
        }
    }
}
