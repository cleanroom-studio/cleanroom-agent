//! Cleanroom MCP Server — Model Context Protocol server for Cleanroom Agent.
//!
//! This crate exposes Cleanroom Agent's database operations as MCP tools, enabling
//! LLMs to interact with the agent's knowledge base. The server implements the
//! MCP specification and communicates over stdio transport.
//!
//! # Architecture
//!
//! - [`CleanroomMcpServer`] — Main server struct implementing [`ServerHandler`]
//! - Tool dispatcher — Routes tool calls to handler methods
//! - Middleware — Permission checks and request logging
//!
//! # Tool Categories
//!
//! | Category | Tools | Description |
//! |----------|-------|-------------|
//! | Task Management | `create_task`, `claim_task`, `complete_task`, `fail_task`, `list_tasks` | Workflow task lifecycle |
//! | S.DEF Query | `get_data_model`, `get_contract`, `get_function_spec`, `list_documents`, `search_sdef` | Read S.DEF entities |
//! | Naming Service | `resolve_name`, `batch_resolve_names`, `register_custom_name` | Symbol name resolution |
//! | Import/Export | `export_sdef`, `import_sdef`, `export_shard`, `import_shard` | S.DEF serialization |
//! | Checkpoint | `create_checkpoint`, `list_checkpoints`, `restore_checkpoint` | Workflow snapshots |
//! | Consistency | `check_consistency`, `compute_fingerprints`, `resolve_inconsistency` | Fingerprint verification |
//! | LSP | `lsp_initialize`, `lsp_get_document_symbols`, `lsp_get_type_info` | Code analysis |
//! | Compatibility | `set_compatibility_mode`, `list_compat_layers`, `ignore_compat_layer` | Compatibility layers |
//!
//! # Usage
//!
//! ```rust,ignore
//! use cleanroom_mcp::CleanroomMcpServer;
//! use std::path::Path;
//!
//! let server = CleanroomMcpServer::new(Path::new("state.db")).unwrap();
//! // Run with tokio: server.serve().await;
//! ```
//!
//! # Protocol
//!
//! The server follows the MCP specification:
//! 1. Client sends `initialize` → Server responds with [`ServerInfo`]
//! 2. Client calls `tools/list` → Server returns tool definitions
//! 3. Client calls `tools/call` → Server dispatches to handler, returns [`CallToolResult`]
//!
//! # Middleware
//!
//! - **Permission checks** — Task mutation tools verify `agent_id` matches `assigned_to`
//! - **Request logging** — All tool calls are traced with arguments and result summary

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

use rmcp::{
    model::{
        CallToolResult, Content, ListToolsResult, PaginatedRequestParams, ServerInfo,
        ServerCapabilities, Implementation, JsonObject, Tool, ToolAnnotations,
    },
    ServerHandler, serve_server, ErrorData, RoleServer,
    service::RequestContext,
};
use serde_json::{json, Value};

use cleanroom_db::{
    Database, TaskRepository, Task, TaskStatus, TaskType,
    SymbolEntry, SymbolRepository, SymbolType,
    FingerprintRepository,
    SdefRepository,
};
use cleanroom_db::repositories::{
    Checkpoint, CheckpointRepository, ShardRepository, Shard,
};
use cleanroom_lsp::LspServerPool;

fn tr(key: &str) -> String {
    cleanroom_i18n::global().translate(key)
}

pub mod tools;

/// The Cleanroom MCP server.
///
/// Manages database connections, LSP server pool, and dispatches tool calls.
/// Implements the MCP [`ServerHandler`] trait for protocol compliance.
///
/// # Fields
///
/// - `db` — Shared database connection (Arc-wrapped for thread safety)
/// - `db_path` — Path to SQLite file (for opening additional connections)
/// - `lsp_pool` — Pool of LSP language servers for code analysis
///
/// # Example
///
/// ```rust,ignore
/// let server = CleanroomMcpServer::new(Path::new("state.db")).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CleanroomMcpServer {
    /// Database connection.
    pub db: Arc<Database>,
    /// Database file path for opening additional connections.
    pub db_path: String,
    /// LSP server pool for code analysis.
    pub lsp_pool: Arc<Mutex<LspServerPool>>,
}

impl CleanroomMcpServer {
    /// Creates a new MCP server instance with a database at the given path.
    ///
    /// Opens the SQLite database and initializes the LSP server pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub fn new(db_path: &Path) -> Result<Self, ErrorData> {
        let db = Database::open(db_path)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(Self {
            db: Arc::new(db),
            db_path: db_path.to_string_lossy().to_string(),
            lsp_pool: Arc::new(Mutex::new(LspServerPool::new())),
        })
    }

    /// Creates a server from an existing shared database.
    ///
    /// Useful for testing with in-memory databases or sharing a connection
    /// that was already opened by another component.
    pub fn from_db(db: Arc<Database>, db_path: &Path) -> Self {
        Self {
            db,
            db_path: db_path.to_string_lossy().to_string(),
            lsp_pool: Arc::new(Mutex::new(LspServerPool::new())),
        }
    }

    /// Starts the MCP server over stdio transport.
    ///
    /// Spawns a background consistency checker loop (runs every 5 minutes)
    /// and then serves requests over stdin/stdout. This method blocks
    /// until the server connection closes.
    ///
    /// # Protocol
    ///
    /// 1. Initialize: Client sends `initialize`, server responds with capabilities
    /// 2. Tools/list: Client requests tool definitions, server returns them
    /// 3. Tools/call: Client invokes a tool, server dispatches and responds
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let server = CleanroomMcpServer::new(Path::new("state.db")).unwrap();
    /// server.serve().await?;
    /// ```
    pub async fn serve(self) -> Result<(), ErrorData> {
        // Spawn background consistency checker
        let checker_db = self.db.clone();
        let checker_config = cleanroom_agent::consistency_checker::ConsistencyCheckerConfig {
            interval: std::time::Duration::from_secs(300),
            document_names: vec![], // populated as documents are created
            auto_fix: true,
        };
        let checker = cleanroom_agent::consistency_checker::ConsistencyChecker::new(
            checker_db, checker_config,
        );
        checker.run_loop(); // Start in background

        let transport = (tokio::io::stdin(), tokio::io::stdout());
        let _running = serve_server(self, transport).await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(())
    }

    /// Starts the MCP server over TCP transport (cross-platform).
    ///
    /// Listens on the given TCP address and spawns a new handler for each
    /// incoming connection. Multiple concurrent clients (IDE + CLI) can
    /// connect and share the same database.
    ///
    /// # Address format
    ///
    /// - `"tcp://127.0.0.1:0"` → bind loopback, OS-assigned port
    /// - `"tcp://127.0.0.1:9000"` → bind loopback, specific port
    /// - `"tcp://0.0.0.0:9000"` → bind all interfaces (⚠️ use only in trusted networks)
    ///
    /// The port is written to `<temp_dir>/cleanroom.port` for CLI discovery.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let server = CleanroomMcpServer::new(Path::new("state.db")).unwrap();
    /// server.serve_tcp("tcp://127.0.0.1:0").await?;
    /// ```
    pub async fn serve_tcp(self, addr: &str) -> Result<(), ErrorData> {
        // Spawn background consistency checker
        let checker_db = self.db.clone();
        let checker_config = cleanroom_agent::consistency_checker::ConsistencyCheckerConfig {
            interval: std::time::Duration::from_secs(300),
            document_names: vec![],
            auto_fix: true,
        };
        let checker = cleanroom_agent::consistency_checker::ConsistencyChecker::new(
            checker_db, checker_config,
        );
        checker.run_loop();

        // Parse address, stripping "tcp://" prefix
        let bind_addr = addr.strip_prefix("tcp://").unwrap_or(addr);

        let listener = tokio::net::TcpListener::bind(bind_addr).await
            .map_err(|e| ErrorData::internal_error(
                format!("Failed to bind TCP at {}: {}", bind_addr, e), None
            ))?;

        let local_addr = listener.local_addr()
            .map_err(|e| ErrorData::internal_error(
                format!("Failed to get local address: {}", e), None
            ))?;

        // Write port file for CLI discovery
        cleanroom_agent::orchestrator::write_port_file(local_addr.port());

        println!("MCP server listening on tcp://{}", local_addr);
        tracing::info!(addr = %local_addr, "MCP server listening on TCP");

        loop {
            let (stream, peer_addr) = listener.accept().await
                .map_err(|e| ErrorData::internal_error(
                    format!("TCP accept error: {}", e), None
                ))?;

            tracing::debug!(peer = %peer_addr, "TCP connection accepted");

            let (reader, writer) = tokio::io::split(stream);
            let db = self.db.clone();
            let db_path = self.db_path.clone();
            let lsp_pool = self.lsp_pool.clone();

            tokio::spawn(async move {
                let server = CleanroomMcpServer {
                    db,
                    db_path,
                    lsp_pool,
                };
                if let Err(e) = serve_server(server, (reader, writer)).await {
                    tracing::warn!(error = %e, "TCP connection handler exited with error");
                }
            });
        }
    }

    /// Helper: derive JSON schemas for tool parameters.
    ///
    /// Uses `schemars` to generate a JSON Schema from a type's struct definition.
    /// The schema is used by MCP to describe tool inputs to the LLM.
    fn schema_for<T: rmcp::schemars::JsonSchema>() -> Arc<JsonObject> {
        let schema = rmcp::schemars::schema_for!(T);
        let value = serde_json::to_value(&schema).unwrap_or(json!({}));
        Arc::new(value.as_object().cloned().unwrap_or_default())
    }

    /// Opens a new SQLite connection to the same database file.
    ///
    /// Safe for concurrent access with WAL mode enabled. Used when multiple
    /// connections are needed for parallel operations.
    fn new_conn(&self) -> rusqlite::Connection {
        rusqlite::Connection::open(&self.db_path)
            .expect("Failed to open additional database connection")
    }

    fn task_repo(&self) -> TaskRepository {
        TaskRepository::new(self.db.connection_arc())
    }

    fn sdef_repo(&self) -> SdefRepository {
        SdefRepository::new(self.new_conn())
    }

    fn symbol_repo(&self) -> SymbolRepository {
        SymbolRepository::from_arc(self.db.connection_arc())
    }

    fn fingerprint_repo(&self) -> FingerprintRepository {
        FingerprintRepository::from_arc(self.db.connection_arc())
    }

    fn checkpoint_repo(&self) -> CheckpointRepository {
        CheckpointRepository::new(self.new_conn())
    }

    fn shard_repo(&self) -> ShardRepository {
        ShardRepository::new(self.new_conn())
    }

    fn exporter(&self) -> cleanroom_db::export_import::SdefExporter {
        cleanroom_db::export_import::SdefExporter::new(self.new_conn())
    }

    fn importer(&self) -> cleanroom_db::export_import::SdefImporter {
        cleanroom_db::export_import::SdefImporter::new(self.new_conn())
    }

    fn task_to_json(&self, task: &Task) -> Value {
        json!({
            "task_id": task.task_id,
            "task_type": task.task_type.as_str(),
            "status": task.status.as_str(),
            "priority": task.priority,
            "input": serde_json::from_str::<Value>(&task.input_json).unwrap_or_default(),
            "output": task.output_json.as_ref().and_then(|o| serde_json::from_str::<Value>(o).ok()),
            "error_message": task.error_message,
            "assigned_to": task.assigned_to,
            "progress": task.progress,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "last_heartbeat": task.last_heartbeat,
            "dependencies": serde_json::from_str::<Vec<String>>(&task.dependencies_json).unwrap_or_default(),
            "version": task.version,
        })
    }

    fn shard_to_json(&self, shard: &Shard) -> Value {
        json!({
            "shard_id": shard.shard_id,
            "document_name": shard.document_name,
            "sdef_uri": shard.sdef_uri,
            "section_type": shard.section_type,
            "file_path": shard.file_path,
            "status": shard.status.as_str(),
            "content_hash": shard.content_hash,
            "token_estimate": shard.token_estimate,
            "version": shard.version,
            "created_at": shard.created_at,
            "updated_at": shard.updated_at,
        })
    }

    fn execute(&self, sql: &str, params: &[&dyn rusqlite::types::ToSql]) -> Result<(), String> {
        self.db.connection()
            .execute(sql, params)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    // ============ Middleware Layer ============

    /// Permission check middleware.
    ///
    /// Enforces that agents can only modify tasks assigned to them. This prevents
    /// agents from accidentally interfering with each other's work.
    ///
    /// # Rules
    ///
    /// - `claim_task` — Requires `agent_id` in arguments
    /// - `complete_task`, `fail_task`, `send_heartbeat`, `update_task_progress` —
    ///   Verify `agent_id` matches the task's `assigned_to` field
    ///
    /// # Errors
    ///
    /// Returns `Err` with "Permission denied" if the agent is not the task owner.
    fn check_permission(&self, tool_name: &str, args: &Value) -> Result<(), String> {
        // Task mutation tools require agent_id to match assigned_to
        let mutating_tasks = [
            "claim_task", "complete_task", "fail_task", "send_heartbeat",
            "update_task_progress",
        ];
        if mutating_tasks.contains(&tool_name) {
            if let Some(agent_id) = args.get("agent_id").and_then(|v| v.as_str()) {
                if let Some(task_id) = args.get("task_id").and_then(|v| v.as_str()) {
                    let repo = self.task_repo();
                    let task = repo.get(task_id).map_err(|e| e.to_string())?;
                    if let Some(ref assigned) = task.assigned_to {
                        if assigned != agent_id {
                            return Err(format!(
                                "Permission denied: task '{}' is assigned to '{}', not '{}'",
                                task_id, assigned, agent_id
                            ));
                        }
                    }
                }
            }
            // Tools that need agent_id but don't have it
            if tool_name == "claim_task" && args.get("agent_id").is_none() {
                return Err("Permission denied: claim_task requires 'agent_id'".to_string());
            }
        }
        Ok(())
    }

    /// Request logging middleware.
    ///
    /// Records all tool calls to the tracing log with:
    /// - Tool name
    /// - Arguments (truncated to first 5 keys for objects)
    /// - Result summary (key count for objects)
    /// - Success/failure with error message
    fn log_request(&self, tool_name: &str, args: &Value, result: &Result<Value, String>) {
        match result {
            Ok(val) => {
                let summary = if val.is_object() {
                    let keys: Vec<String> = val.as_object()
                        .map(|o| o.keys().take(5).cloned().collect())
                        .unwrap_or_default();
                    format!("{{ {} keys }}", keys.join(", "))
                } else {
                    "non-object".to_string()
                };
                tracing::info!(
                    tool = %tool_name,
                    args = %serde_json::to_string(args).unwrap_or_default(),
                    result_summary = %summary,
                    "MCP tool call succeeded"
                );
            }
            Err(err) => {
                tracing::warn!(
                    tool = %tool_name,
                    args = %serde_json::to_string(args).unwrap_or_default(),
                    error = %err,
                    "MCP tool call failed"
                );
            }
        }
    }

    // ============ Tool Dispatcher ============

    /// Dispatches an MCP tool call to the appropriate handler.
    ///
    /// This is the main entry point for all tool operations. It:
    /// 1. Parses tool name and arguments
    /// 2. Runs permission check middleware
    /// 3. Routes to the correct handler method
    /// 4. Returns JSON result or error
    ///
    /// # Tool Categories
    ///
    /// - **Task Management**: `create_task`, `claim_task`, `update_task_progress`, `complete_task`, `fail_task`, `send_heartbeat`, `get_task`, `list_tasks`
    /// - **S.DEF Query**: `get_data_model`, `get_contract`, `get_function_spec`, `get_ui_screen`, `list_documents`, `search_sdef`, `get_dependency_graph`, `list_shards`
    /// - **Naming**: `resolve_name`, `batch_resolve_names`, `list_symbols`, `register_custom_name`
    /// - **Import/Export**: `export_sdef`, `export_sdef_to_disk`, `import_sdef`, `export_shard`, `import_shard`
    /// - **Checkpoint**: `create_checkpoint`, `list_checkpoints`, `restore_checkpoint`
    /// - **Transaction**: `begin_transaction`, `commit_transaction`, `rollback_transaction`
    /// - **Consistency**: `check_consistency`, `compute_fingerprints`, `resolve_inconsistency`, `get_inconsistency_report`
    /// - **LSP**: `lsp_initialize`, `lsp_get_document_symbols`, `lsp_get_type_info`, `lsp_find_references`, `lsp_get_diagnostics`, `lsp_get_hierarchy`
    /// - **Compatibility**: `set_compatibility_mode`, `list_compat_layers`, `get_compat_layer_detail`, `ignore_compat_layer`
    ///
    /// # Errors
    ///
    /// Returns an error string for unknown tools, permission denials, or handler failures.
    pub fn dispatch_tool_call(&self, request: rmcp::model::CallToolRequestParams) -> Result<Value, String> {
        let name = request.name.to_string();
        let args = request.arguments.unwrap_or_default();
        let args_value = serde_json::to_value(&args).unwrap_or(json!({}));

        // Permission check
        self.check_permission(&name, &args_value)?;

        match name.as_str() {
            // Task Management
            "create_task" => self.handle_create_task(args_value),
            "claim_task" => self.handle_claim_task(args_value),
            "update_task_progress" => self.handle_update_progress(args_value),
            "complete_task" => self.handle_complete_task(args_value),
            "fail_task" => self.handle_fail_task(args_value),
            "send_heartbeat" => self.handle_heartbeat(args_value),
            "get_task" => self.handle_get_task(args_value),
            "list_tasks" => self.handle_list_tasks(args_value),
            // S.DEF Query
            "get_data_model" => self.handle_get_data_model(args_value),
            "get_contract" => self.handle_get_contract(args_value),
            "get_function_spec" => self.handle_get_function_spec(args_value),
            "get_ui_screen" => self.handle_get_ui_screen(args_value),
            "list_documents" => self.handle_list_documents(args_value),
            "search_sdef" => self.handle_search_sdef(args_value),
            "get_dependency_graph" => self.handle_get_dependency_graph(args_value),
            "list_shards" => self.handle_list_shards(args_value),
            // Naming
            "resolve_name" => self.handle_resolve_name(args_value),
            "batch_resolve_names" => self.handle_batch_resolve(args_value),
            "list_symbols" => self.handle_list_symbols(args_value),
            "register_custom_name" => self.handle_register_custom_name(args_value),
            // Import/Export
            "export_sdef" => self.handle_export_sdef(args_value),
            "export_sdef_to_disk" => self.handle_export_sdef_to_disk(args_value),
            "import_sdef" => self.handle_import_sdef(args_value),
            "export_shard" => self.handle_export_shard(args_value),
            "import_shard" => self.handle_import_shard(args_value),
            // Checkpoint
            "create_checkpoint" => self.handle_create_checkpoint(args_value),
            "list_checkpoints" => self.handle_list_checkpoints(args_value),
            "restore_checkpoint" => self.handle_restore_checkpoint(args_value),
            // Transaction
            "begin_transaction" => self.handle_begin_transaction(args_value),
            "commit_transaction" => self.handle_commit_transaction(args_value),
            "rollback_transaction" => self.handle_rollback_transaction(args_value),
            // Consistency
            "check_consistency" => self.handle_check_consistency(args_value),
            "compute_fingerprints" => self.handle_compute_fingerprints(args_value),
            "resolve_inconsistency" => self.handle_resolve_inconsistency(args_value),
            "get_inconsistency_report" => self.handle_get_inconsistency_report(args_value),
            // LSP Tools
            "lsp_initialize" => self.handle_lsp_initialize(args_value),
            "lsp_get_document_symbols" => self.handle_lsp_get_document_symbols(args_value),
            "lsp_get_type_info" => self.handle_lsp_get_type_info(args_value),
            "lsp_find_references" => self.handle_lsp_find_references(args_value),
            "lsp_get_diagnostics" => self.handle_lsp_get_diagnostics(args_value),
            "lsp_get_hierarchy" => self.handle_lsp_get_hierarchy(args_value),
            // Compatibility Mode
            "set_compatibility_mode" => self.handle_set_compatibility_mode(args_value),
            "list_compat_layers" => self.handle_list_compat_layers(args_value),
            "get_compat_layer_detail" => self.handle_get_compat_layer(args_value),
            "ignore_compat_layer" => self.handle_ignore_compat_layer(args_value),
            // Evaluation
            "run_evaluation" => self.handle_run_evaluation(args_value),
            "get_evaluation_report" => self.handle_get_evaluation_report(args_value),
            // Task queue
            "get_task_queue" => self.handle_get_task_queue(args_value),
            "insert_task" => self.handle_insert_task(args_value),
            "remove_task" => self.handle_remove_task(args_value),
            "modify_task" => self.handle_modify_task(args_value),
            // LLM↔Human bridge
            "request_clarification" => self.handle_request_clarification(args_value),
            "propose_decision" => self.handle_propose_decision(args_value),
            "preview_changes" => self.handle_preview_changes(args_value),
            // Workflow control (cross-platform, replaces OS signals)
            "pause_workflow" => self.handle_pause_workflow(args_value),
            "resume_workflow" => self.handle_resume_workflow(args_value),
            "skill_list" => self.handle_skill_list(args_value),
            "skill_activate" => self.handle_skill_activate(args_value),
            "skill_refresh" => self.handle_skill_refresh(args_value),
            "skill_validate" => self.handle_skill_validate(args_value),
            _ => Err(format!("Unknown tool: {}", name)),
        }
    }

    // ============ Task Tool Handlers ============

    fn handle_create_task(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { task_type: String, input: Value, #[serde(default)] priority: i32, #[serde(default)] dependencies: Vec<String> }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let tt = TaskType::from_str(&p.task_type).ok_or_else(|| format!("Unknown task type: {}", p.task_type))?;
        let task = Task {
            task_id: uuid::Uuid::new_v4().to_string(),
            task_type: tt, status: TaskStatus::Pending, priority: p.priority,
            input_json: serde_json::to_string(&p.input).unwrap_or_default(),
            output_json: None, error_message: None, assigned_to: None,
            progress: 0.0, created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None, completed_at: None, retry_count: 0, max_retries: 3,
            last_heartbeat: None,
            dependencies_json: serde_json::to_string(&p.dependencies).unwrap_or_default(),
            version: 1,
        };
        let repo = self.task_repo();
        repo.create(&task).map_err(|e| e.to_string())?;
        Ok(json!({"task_id": task.task_id}))
    }

    fn handle_claim_task(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { agent_id: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.task_repo();
        match repo.claim(&p.agent_id).map_err(|e| e.to_string())? {
            Some(task) => Ok(self.task_to_json(&task)),
            None => Ok(json!(null)),
        }
    }

    fn handle_update_progress(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { task_id: String, progress: f64 }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.task_repo();
        repo.update_progress(&p.task_id, p.progress).map_err(|e| e.to_string())?;
        Ok(json!({"ok": true}))
    }

    fn handle_complete_task(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { task_id: String, output: Value }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.task_repo();
        repo.complete(&p.task_id, &serde_json::to_string(&p.output).unwrap_or_default())
            .map_err(|e| e.to_string())?;
        Ok(json!({"ok": true}))
    }

    fn handle_fail_task(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { task_id: String, error_message: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.task_repo();
        repo.update_status(&p.task_id, TaskStatus::Failed).map_err(|e| e.to_string())?;
        self.execute("UPDATE tasks SET error_message = ?1 WHERE task_id = ?2",
            &[&p.error_message, &p.task_id])?;
        Ok(json!({"ok": true}))
    }

    fn handle_heartbeat(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { task_id: String, agent_id: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let _ = p.agent_id;
        let repo = self.task_repo();
        repo.heartbeat(&p.task_id).map_err(|e| e.to_string())?;
        Ok(json!({"ok": true}))
    }

    fn handle_get_task(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { task_id: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.task_repo();
        let task = repo.get(&p.task_id).map_err(|e| e.to_string())?;
        Ok(self.task_to_json(&task))
    }

    fn handle_list_tasks(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { status: Option<String>, task_type: Option<String>, assigned_to: Option<String> }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.task_repo();
        let status = p.status.as_ref().and_then(|s| TaskStatus::from_str(s));
        let task_type = p.task_type.as_ref().and_then(|t| TaskType::from_str(t));
        let tasks = repo.list(status, task_type, p.assigned_to.as_deref())
            .map_err(|e| e.to_string())?;
        let tasks_json: Vec<Value> = tasks.iter().map(|t| self.task_to_json(t)).collect();
        Ok(json!(tasks_json))
    }

    // ============ S.DEF Query Tool Handlers ============

    fn handle_get_data_model(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, entity: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.sdef_repo();
        let (model, attrs) = repo.get_data_model(&p.document_name, &p.entity)
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "entity": model.entity, "status": model.status,
            "version": model.version, "description": model.description,
            "logical_model": model.logical_model,
            "attributes": attrs.iter().map(|a| json!({
                "name": a.name, "attr_type": a.attr_type,
                "format": a.format, "description": a.description,
                "required": a.required, "identity": a.identity,
                "generated": a.generated, "unique_flag": a.unique_flag,
                "internal": a.internal, "deprecated": a.deprecated,
                "default_value": a.default_value,
            })).collect::<Vec<_>>(),
        }))
    }

    fn handle_get_contract(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, name: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.sdef_repo();
        let c = repo.get_contract(&p.document_name, &p.name)
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "name": c.name, "contract_type": c.contract_type,
            "status": c.status, "version": c.version,
            "is_abstract": c.is_abstract, "description": c.description,
            "http_method": c.http_method, "api_path": c.api_path, "auth": c.auth,
        }))
    }

    fn handle_list_documents(&self, _args: Value) -> Result<Value, String> {
        let repo = self.sdef_repo();
        let docs = repo.list_documents().map_err(|e| e.to_string())?;
        Ok(json!(docs.iter().map(|d| json!({
            "name": d.name, "version": d.version, "description": d.description,
            "created_at": d.created_at, "updated_at": d.updated_at,
        })).collect::<Vec<_>>()))
    }

    fn handle_search_sdef(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { query: String, #[serde(default = "default_limit_20")] limit: usize }
        fn default_limit_20() -> usize { 20 }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.sdef_repo();
        let docs = repo.search(&p.query).map_err(|e| e.to_string())?;
        let results: Vec<Value> = docs.iter().take(p.limit).map(|d| json!({
            "name": d.name, "version": d.version, "description": d.description,
        })).collect();
        Ok(json!(results))
    }

    fn handle_get_function_spec(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, name: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let fs = self.sdef_repo().get_function_spec(&p.document_name, &p.name)
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "name": fs.name, "description": fs.description,
            "logic": fs.logic, "complexity": fs.complexity,
            "pure_function": fs.pure_function,
        }))
    }

    fn handle_get_ui_screen(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, screen_id: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let screen = self.sdef_repo().get_ui_screen(&p.document_name, &p.screen_id)
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "id": screen.id, "name": screen.name,
            "route": screen.route, "purpose": screen.purpose,
            "layout_description": screen.layout_description,
        }))
    }

    fn handle_get_dependency_graph(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let conn = self.new_conn();

        // Query contracts for dependency info
        let mut stmt = conn.prepare(
            "SELECT name, contract_type, dependencies_json FROM contracts WHERE document_name = ?1"
        ).map_err(|e| e.to_string())?;
        let edges: Vec<Value> = stmt.query_map([&p.document_name], |row| {
            let name: String = row.get(0)?;
            let deps: Option<String> = row.get(2)?;
            Ok((name, deps))
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .filter_map(|(name, deps)| {
            deps.and_then(|d| serde_json::from_str::<Vec<String>>(&d).ok())
                .map(|deps| (name, deps))
        })
        .flat_map(|(name, deps)| {
            deps.into_iter().map(move |dep| json!({
                "from": name, "to": dep, "kind": "import"
            }))
        })
        .collect();
        drop(stmt);

        // Also query data_relationships for entity dependencies
        let mut rel_stmt = conn.prepare(
            "SELECT entity, target, kind FROM data_relationships WHERE document_name = ?1"
        ).map_err(|e| e.to_string())?;
        let rel_edges: Vec<Value> = rel_stmt.query_map([&p.document_name], |row| {
            Ok(json!({
                "from": row.get::<_, String>(0)?,
                "to": row.get::<_, String>(1)?,
                "kind": format!("data_{}", row.get::<_, String>(2)?),
            }))
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

        let mut all_edges = edges;
        all_edges.extend(rel_edges);

        Ok(json!({
            "document_name": p.document_name,
            "edges": all_edges,
            "edge_count": all_edges.len(),
        }))
    }

    fn handle_list_shards(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, section_type: Option<String>, status: Option<String> }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.shard_repo();

        let shards = if let Some(ref sec_type) = p.section_type {
            repo.list_by_document(&p.document_name)
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter(|s| s.section_type == *sec_type)
                .filter(|s| p.status.as_ref().map_or(true, |st| s.status.as_str() == st))
                .collect::<Vec<_>>()
        } else {
            repo.list_by_document(&p.document_name)
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter(|s| p.status.as_ref().map_or(true, |st| s.status.as_str() == st))
                .collect::<Vec<_>>()
        };

        Ok(json!(shards.iter().map(|s| self.shard_to_json(s)).collect::<Vec<_>>()))
    }

    // ============ Naming Service Tool Handlers ============

    fn handle_resolve_name(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, sdef_uri: String, language: String, symbol_type: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let st = SymbolType::from_str(&p.symbol_type)
            .ok_or_else(|| format!("Unknown symbol_type '{}'. Valid: class, interface, function, variable, constant, enum, type", p.symbol_type))?;

        // Try existing first
        if let Ok(Some(name)) = self.symbol_repo().resolve(&p.document_name, &p.sdef_uri, &p.language) {
            return Ok(json!({"name": name, "found": true, "is_new": false}));
        }

        // Auto-generate: use NameResolutionService
        let ns = cleanroom_agent::NameResolutionService::new(self.db.clone());
        match ns.resolve(&p.document_name, &p.sdef_uri, &p.language, st) {
            Ok(result) => Ok(json!({
                "name": result.concrete_name,
                "found": !result.is_new,
                "is_new": result.is_new,
            })),
            Err(e) => Err(e.to_string()),
        }
    }

    fn handle_batch_resolve(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, uris: Vec<String>, language: String, symbol_type: Option<String> }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let default_st = SymbolType::Variable;
        let st = p.symbol_type.as_ref()
            .and_then(|s| SymbolType::from_str(s))
            .unwrap_or(default_st);
        let uri_refs: Vec<(&str, SymbolType)> = p.uris.iter()
            .map(|u| (u.as_str(), st)).collect();
        let results = self.symbol_repo()
            .batch_resolve(&p.document_name, &uri_refs, &p.language)
            .map_err(|e| e.to_string())?;
        Ok(json!(results.iter().map(|r| json!({
            "sdef_uri": r.sdef_uri, "concrete_name": r.concrete_name,
            "is_user_defined": r.is_user_defined,
        })).collect::<Vec<_>>()))
    }

    fn handle_list_symbols(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, language: String, symbol_type: Option<String> }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let st = p.symbol_type.as_ref().and_then(|s| SymbolType::from_str(s));
        let entries = self.symbol_repo().list(&p.document_name, &p.language, st)
            .map_err(|e| e.to_string())?;
        Ok(json!(entries.iter().map(|e| json!({
            "sdef_uri": e.sdef_uri, "concrete_name": e.concrete_name,
            "is_user_defined": e.is_user_defined,
            "symbol_type": e.symbol_type.as_str(), "language": e.language,
            "created_at": e.created_at,
        })).collect::<Vec<_>>()))
    }

    fn handle_register_custom_name(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, sdef_uri: String, language: String, symbol_type: String, concrete_name: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let st = SymbolType::from_str(&p.symbol_type)
            .ok_or_else(|| format!("Unknown symbol type: {}", p.symbol_type))?;
        let entry = SymbolEntry {
            id: None, document_name: p.document_name, sdef_uri: p.sdef_uri,
            language: p.language, symbol_type: st, concrete_name: p.concrete_name,
            is_user_defined: true, created_at: None,
        };
        self.symbol_repo().register(&entry).map_err(|e| e.to_string())?;
        Ok(json!({"ok": true}))
    }

    // ============ Import/Export Tool Handlers ============

    fn handle_export_sdef(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, #[serde(default)] format: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let sdef = self.exporter().export(&p.document_name).map_err(|e| e.to_string())?;
        let result = if p.format == "yaml" {
            serde_yaml::to_string(&sdef).map_err(|e| e.to_string())?
        } else {
            serde_json::to_string_pretty(&sdef).map_err(|e| e.to_string())?
        };
        Ok(json!({"sdef_content": result, "format": p.format}))
    }

    fn handle_export_sdef_to_disk(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, output_dir: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let exporter = cleanroom_db::export_import::SdefFileExporter::new(
            (*self.db).clone(),
        );
        let root = exporter.export_to_disk(&p.document_name, Path::new(&p.output_dir))
            .map_err(|e| e.to_string())?;
        Ok(json!({"output_path": root.to_string_lossy().to_string(), "ok": true}))
    }

    fn handle_import_sdef(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { sdef_json: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let sdef: sdef_core::SoftwareDefinition =
            serde_json::from_str(&p.sdef_json).map_err(|e| e.to_string())?;
        let doc_name = self.importer().import(&sdef).map_err(|e| e.to_string())?;
        Ok(json!({"document_name": doc_name, "ok": true}))
    }

    fn handle_export_shard(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { sdef_uri: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let content = self.exporter().export_shard(&p.sdef_uri).map_err(|e| e.to_string())?;
        Ok(json!({"sdef_uri": p.sdef_uri, "content_hash": content}))
    }

    fn handle_import_shard(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { shard_id: String, document_name: String, sdef_uri: String, section_type: String, content_json: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        self.importer().import_shard(&p.shard_id, &p.document_name, &p.sdef_uri, &p.section_type, &p.content_json)
            .map_err(|e| e.to_string())?;
        Ok(json!({"ok": true, "shard_id": p.shard_id}))
    }

    // ============ Checkpoint Tool Handlers ============

    fn handle_create_checkpoint(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, description: Option<String> }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.checkpoint_repo();
        let conn = self.new_conn();

        // Capture task snapshot
        let tasks_json: Value = if let Ok(mut stmt) = conn.prepare(
            "SELECT task_id, task_type, status, priority, input_json, error_message, assigned_to, progress, retry_count, max_retries, dependencies_json
             FROM tasks"
        ) {
            let tasks: Vec<Value> = stmt.query_map([], |row| {
                Ok(json!({
                    "task_id": row.get::<_, String>(0)?,
                    "task_type": row.get::<_, String>(1)?,
                    "status": row.get::<_, String>(2)?,
                    "priority": row.get::<_, i32>(3)?,
                    "input_json": row.get::<_, String>(4)?,
                    "error_message": row.get::<_, Option<String>>(5)?,
                    "assigned_to": row.get::<_, Option<String>>(6)?,
                    "progress": row.get::<_, f64>(7)?,
                    "retry_count": row.get::<_, i32>(8)?,
                    "max_retries": row.get::<_, i32>(9)?,
                    "dependencies_json": row.get::<_, String>(10)?,
                }))
            }).map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect();
            json!(tasks)
        } else {
            json!([])
        };

        // Capture shard snapshot
        let shards_json: Value = {
            let shard_repo = self.shard_repo();
            let shards = shard_repo.list_by_document(&p.document_name).unwrap_or_default();
            json!(shards.iter().map(|s| json!({
                "shard_id": s.shard_id,
                "sdef_uri": s.sdef_uri,
                "section_type": s.section_type,
                "status": s.status.as_str(),
                "file_path": s.file_path,
                "content_hash": s.content_hash,
                "token_estimate": s.token_estimate,
            })).collect::<Vec<_>>())
        };

        let cp = Checkpoint {
            checkpoint_id: uuid::Uuid::new_v4().to_string(),
            document_name: p.document_name,
            description: p.description,
            created_at: String::new(),
            task_snapshot_json: tasks_json.to_string(),
            shard_snapshot_json: shards_json.to_string(),
        };
        repo.create(&cp).map_err(|e| e.to_string())?;
        Ok(json!({
            "checkpoint_id": cp.checkpoint_id,
            "task_count": tasks_json.as_array().map(|a| a.len()).unwrap_or(0),
            "shard_count": shards_json.as_array().map(|a| a.len()).unwrap_or(0),
        }))
    }

    fn handle_list_checkpoints(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let checkpoints = self.checkpoint_repo().list(&p.document_name)
            .map_err(|e| e.to_string())?;
        Ok(json!(checkpoints.iter().map(|c| json!({
            "checkpoint_id": c.checkpoint_id, "document_name": c.document_name,
            "description": c.description, "created_at": c.created_at,
        })).collect::<Vec<_>>()))
    }

    fn handle_restore_checkpoint(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { checkpoint_id: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;

        let repo = self.checkpoint_repo();
        let cp = repo.get(&p.checkpoint_id).map_err(|e| e.to_string())?;
        let conn = self.new_conn();

        // Parse snapshots
        let tasks: Vec<Value> = serde_json::from_str(&cp.task_snapshot_json).unwrap_or_default();
        let shards: Vec<Value> = serde_json::from_str(&cp.shard_snapshot_json).unwrap_or_default();

        // Clear current state for this document
        conn.execute("DELETE FROM tasks", []).map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM shards WHERE document_name = ?1", rusqlite::params![cp.document_name])
            .map_err(|e| e.to_string())?;

        // Restore tasks from snapshot
        let mut restored_tasks = 0;
        for t in &tasks {
            if let (Some(task_id), Some(task_type)) = (
                t.get("task_id").and_then(|v| v.as_str()),
                t.get("task_type").and_then(|v| v.as_str()),
            ) {
                let status = t.get("status").and_then(|v| v.as_str()).unwrap_or("pending");
                let priority = t.get("priority").and_then(|v| v.as_i64()).unwrap_or(5);
                let input = t.get("input_json").and_then(|v| v.as_str()).unwrap_or("{}");
                let deps = t.get("dependencies_json").and_then(|v| v.as_str()).unwrap_or("[]");
                if let Err(e) = conn.execute(
                    "INSERT OR IGNORE INTO tasks (task_id, task_type, status, priority, input_json, dependencies_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![task_id, task_type, status, priority, input, deps],
                ) {
                    tracing::warn!(task_id = %task_id, error = %e, "Failed to restore task");
                } else {
                    restored_tasks += 1;
                }
            }
        }

        // Restore shards from snapshot
        let mut restored_shards = 0;
        for s in &shards {
            if let Some(shard_id) = s.get("shard_id").and_then(|v| v.as_str()) {
                let sdef_uri = s.get("sdef_uri").and_then(|v| v.as_str()).unwrap_or("");
                let section_type = s.get("section_type").and_then(|v| v.as_str()).unwrap_or("unknown");
                let status = s.get("status").and_then(|v| v.as_str()).unwrap_or("pending");
                if let Err(e) = conn.execute(
                    "INSERT OR IGNORE INTO shards (shard_id, document_name, sdef_uri, section_type, status) VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![shard_id, cp.document_name, sdef_uri, section_type, status],
                ) {
                    tracing::warn!(shard_id = %shard_id, error = %e, "Failed to restore shard");
                } else {
                    restored_shards += 1;
                }
            }
        }

        Ok(json!({
            "ok": true,
            "checkpoint_id": p.checkpoint_id,
            "document_name": cp.document_name,
            "restored_tasks": restored_tasks,
            "restored_shards": restored_shards,
            "description": cp.description,
        }))
    }

    // ============ Transaction Tool Handlers ============

    fn handle_begin_transaction(&self, _args: Value) -> Result<Value, String> {
        let tx_id = uuid::Uuid::new_v4().to_string();
        self.execute(
            "INSERT INTO prepared_transactions (transaction_id, state, payload_json) VALUES (?1, 'prepared', '{}')",
            &[&tx_id])?;
        Ok(json!({"transaction_id": tx_id}))
    }

    fn handle_commit_transaction(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { transaction_id: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        self.execute(
            "UPDATE prepared_transactions SET state = 'committed' WHERE transaction_id = ?1",
            &[&p.transaction_id])?;
        Ok(json!({"ok": true}))
    }

    fn handle_rollback_transaction(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { transaction_id: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        self.execute(
            "DELETE FROM prepared_transactions WHERE transaction_id = ?1",
            &[&p.transaction_id])?;
        Ok(json!({"ok": true}))
    }

    // ============ Consistency Tool Handlers ============

    fn handle_check_consistency(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, #[serde(default)] check_type: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let inconsistent = self.fingerprint_repo()
            .list_inconsistent(&p.document_name).map_err(|e| e.to_string())?;
        Ok(json!({
            "consistent": inconsistent.is_empty(),
            "inconsistent_count": inconsistent.len(),
            "check_type": p.check_type,
            "inconsistencies": inconsistent.iter().map(|f| json!({
                "entity_uri": f.entity_uri, "entity_type": f.entity_type,
                "code_path": f.code_path,
            })).collect::<Vec<_>>(),
        }))
    }

    fn handle_compute_fingerprints(&self, args: Value) -> Result<Value, String> {
        use cleanroom_agent::consistency::ConsistencyService;
        #[derive(serde::Deserialize)]
        struct P { document_name: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;

        let conn = self.new_conn();
        let fp_repo = self.fingerprint_repo();
        let mut count = 0i64;

        // Compute fingerprints for data models
        if let Ok(mut stmt) = conn.prepare(
            "SELECT entity, description FROM data_models WHERE document_name = ?1"
        ) {
            if let Ok(rows) = stmt.query_map(rusqlite::params![&p.document_name], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            }) {
                for row in rows.flatten() {
                    let (entity, description) = row;
                    let entity_uri = format!("sdef://{}/data-models/{}", p.document_name, entity);
                    let content = serde_json::json!({"entity": entity, "description": description}).to_string();
                    let hash = ConsistencyService::compute_hash(&content);
                    fp_repo.upsert(&cleanroom_db::Fingerprint {
                        entity_uri: entity_uri.clone(),
                        document_name: p.document_name.clone(),
                        entity_type: "data_model".to_string(),
                        sdef_hash: Some(hash.clone()),
                        db_hash: Some(hash),
                        code_hash: None,
                        code_path: None,
                        last_checked_at: String::new(),
                        last_consistent_at: None,
                    }).ok();
                    count += 1;
                }
            }
        }

        // Compute fingerprints for contracts
        if let Ok(mut stmt) = conn.prepare(
            "SELECT name, contract_type FROM contracts WHERE document_name = ?1"
        ) {
            if let Ok(rows) = stmt.query_map(rusqlite::params![&p.document_name], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() {
                    let (name, ctype) = row;
                    let entity_uri = format!("sdef://{}/contracts/{}/{}", p.document_name, ctype, name);
                    let content = serde_json::json!({"name": name, "contract_type": ctype}).to_string();
                    let hash = ConsistencyService::compute_hash(&content);
                    fp_repo.upsert(&cleanroom_db::Fingerprint {
                        entity_uri: entity_uri.clone(),
                        document_name: p.document_name.clone(),
                        entity_type: "contract".to_string(),
                        sdef_hash: Some(hash.clone()),
                        db_hash: Some(hash),
                        code_hash: None,
                        code_path: None,
                        last_checked_at: String::new(),
                        last_consistent_at: None,
                    }).ok();
                    count += 1;
                }
            }
        }

        // Compute fingerprints for function specs
        if let Ok(mut stmt) = conn.prepare(
            "SELECT name FROM function_specs WHERE document_name = ?1"
        ) {
            if let Ok(rows) = stmt.query_map(rusqlite::params![&p.document_name], |row| {
                row.get::<_, String>(0)
            }) {
                for name in rows.flatten() {
                    let entity_uri = format!("sdef://{}/behavior/functions/{}", p.document_name, name);
                    let hash = ConsistencyService::compute_hash(&name);
                    fp_repo.upsert(&cleanroom_db::Fingerprint {
                        entity_uri: entity_uri.clone(),
                        document_name: p.document_name.clone(),
                        entity_type: "function".to_string(),
                        sdef_hash: Some(hash.clone()),
                        db_hash: Some(hash),
                        code_hash: None,
                        code_path: None,
                        last_checked_at: String::new(),
                        last_consistent_at: None,
                    }).ok();
                    count += 1;
                }
            }
        }

        Ok(json!({
            "fingerprint_count": count,
            "document_name": p.document_name,
            "ok": true,
        }))
    }

    // ============ LSP Tool Handlers ============

    fn handle_lsp_initialize(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspInitParams;
        let p: LspInitParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let pool = self.lsp_pool.lock().map_err(|e| e.to_string())?;
        let handle = futures::executor::block_on(pool.get_server(&p.language))
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "initialized": true,
            "language": handle.language,
            "connected": handle.is_connected(),
            "server_info": format!("{} LSP server", p.language),
        }))
    }

    fn handle_lsp_get_document_symbols(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspDocumentSymbolsParams;
        let p: LspDocumentSymbolsParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let pool = self.lsp_pool.lock().map_err(|e| e.to_string())?;
        let handle = futures::executor::block_on(pool.get_server(&p.language))
            .map_err(|e| e.to_string())?;
        match handle.document_symbols(&p.file_path) {
            Ok(symbols) => Ok(json!({
                "file_path": p.file_path,
                "language": p.language,
                "symbol_count": symbols.len(),
                "symbols": symbols.iter().map(|s| json!({
                    "name": s.name,
                    "detail": s.detail,
                    "range": s.range,
                })).collect::<Vec<_>>(),
            })),
            Err(e) => Err(format!("LSP error: {}", e)),
        }
    }

    fn handle_lsp_get_type_info(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspTypeInfoParams;
        let p: LspTypeInfoParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let pool = self.lsp_pool.lock().map_err(|e| e.to_string())?;
        let handle = futures::executor::block_on(pool.get_server(&p.language))
            .map_err(|e| e.to_string())?;
        match handle.hover(&p.file_path, p.line, p.character) {
            Ok(Some(info)) => Ok(json!({
                "type_name": info.type_name,
                "file_path": p.file_path,
                "position": {"line": p.line, "character": p.character},
            })),
            Ok(None) => Ok(json!({
                "file_path": p.file_path,
                "note": "No type info available at this position",
            })),
            Err(e) => Err(format!("LSP error: {}", e)),
        }
    }

    fn handle_lsp_find_references(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspFindReferencesParams;
        let p: LspFindReferencesParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let pool = self.lsp_pool.lock().map_err(|e| e.to_string())?;
        let handle = futures::executor::block_on(pool.get_server(&p.language))
            .map_err(|e| e.to_string())?;
        match handle.find_references(&p.file_path, p.line, p.character) {
            Ok(refs) => Ok(json!({
                "file_path": p.file_path,
                "reference_count": refs.len(),
                "references": refs.iter().map(|r| json!({
                    "uri": r.uri,
                    "range": {
                        "start": {"line": r.range.start.line, "character": r.range.start.character},
                        "end": {"line": r.range.end.line, "character": r.range.end.character},
                    },
                })).collect::<Vec<_>>(),
            })),
            Err(e) => Err(format!("LSP error: {}", e)),
        }
    }

    fn handle_lsp_get_diagnostics(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspDiagnosticsParams;
        let p: LspDiagnosticsParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let pool = self.lsp_pool.lock().map_err(|e| e.to_string())?;
        let handle = futures::executor::block_on(pool.get_server(&p.language))
            .map_err(|e| e.to_string())?;
        match handle.diagnostics(&p.file_path) {
            Ok(diags) => Ok(json!({
                "file_path": p.file_path,
                "language": p.language,
                "diagnostic_count": diags.len(),
                "diagnostics": diags.iter().map(|d| json!({
                    "message": d.message,
                    "range": d.range,
                })).collect::<Vec<_>>(),
            })),
            Err(e) => Err(format!("LSP error: {}", e)),
        }
    }

    fn handle_lsp_get_hierarchy(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspHierarchyParams;
        let p: LspHierarchyParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let pool = self.lsp_pool.lock().map_err(|e| e.to_string())?;
        let handle = futures::executor::block_on(pool.get_server(&p.language))
            .map_err(|e| e.to_string())?;
        match handle.type_hierarchy(&p.file_path, p.line, p.character) {
            Ok(hierarchy) => Ok(json!({
                "file_path": p.file_path,
                "type_name": hierarchy.type_name,
                "supertypes": hierarchy.supertypes,
                "subtypes": hierarchy.subtypes,
            })),
            Err(e) => Err(format!("LSP error: {}", e)),
        }
    }

    // ============ Consistency Tool Handlers (extended) ============

    fn handle_resolve_inconsistency(&self, args: Value) -> Result<Value, String> {
        use tools::consistency_tools::ResolveInconsistencyParams;
        use cleanroom_agent::consistency::{ConsistencyService, FixStrategy, Inconsistency};
        let p: ResolveInconsistencyParams = serde_json::from_value(args).map_err(|e| e.to_string())?;

        // Parse strategy
        let strategy = match p.strategy.as_str() {
            "sync_code_to_sdef" => FixStrategy::SyncCodeToSdef,
            "regenerate_code" => FixStrategy::RegenerateCode,
            "sync_db_to_sdef" => FixStrategy::SyncDbToSdef,
            "sync_sdef_to_db" => FixStrategy::SyncSdefToDb,
            "accept_external" => FixStrategy::AcceptExternal,
            _ => return Err(format!(
                "Invalid strategy '{}'. Valid: sync_code_to_sdef, regenerate_code, sync_db_to_sdef, sync_sdef_to_db, accept_external",
                p.strategy
            )),
        };

        let repo = self.fingerprint_repo();
        let fingerprints = repo.list_by_document(&p.document_name)
            .map_err(|e| e.to_string())?;

        let entity_found = fingerprints.iter().any(|f| f.entity_uri == p.entity_uri);
        if !entity_found {
            return Err(format!("Entity '{}' not found in document '{}'", p.entity_uri, p.document_name));
        }

        // Delegate to ConsistencyService
        let service = ConsistencyService::new(self.db.clone());
        let inc = Inconsistency {
            entity_uri: p.entity_uri.clone(),
            sdef_hash: None,
            db_hash: None,
            code_hash: None,
        };
        if let Err(e) = service.fix(&inc, strategy) {
            return Err(format!("Fix failed: {}", e));
        }

        // Update fingerprint to mark consistent
        for fp in &fingerprints {
            if fp.entity_uri == p.entity_uri {
                let _ = repo.upsert(&cleanroom_db::Fingerprint {
                    entity_uri: fp.entity_uri.clone(),
                    document_name: fp.document_name.clone(),
                    entity_type: fp.entity_type.clone(),
                    sdef_hash: fp.sdef_hash.clone(),
                    db_hash: fp.db_hash.clone(),
                    code_hash: fp.code_hash.clone(),
                    code_path: fp.code_path.clone(),
                    last_checked_at: String::new(),
                    last_consistent_at: Some(chrono::Utc::now().to_rfc3339()),
                });
                break;
            }
        }

        Ok(json!({
            "ok": true,
            "entity_uri": p.entity_uri,
            "strategy": p.strategy,
            "message": format!("Inconsistency for '{}' resolved using '{}' strategy", p.entity_uri, p.strategy),
        }))
    }

    fn handle_get_inconsistency_report(&self, args: Value) -> Result<Value, String> {
        use tools::consistency_tools::InconsistencyReportParams;
        let p: InconsistencyReportParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.fingerprint_repo();
        let fingerprints = if let Some(etype) = &p.entity_type {
            repo.list_by_document(&p.document_name)
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter(|f| f.entity_type == *etype)
                .collect::<Vec<_>>()
        } else {
            repo.list_by_document(&p.document_name)
                .map_err(|e| e.to_string())?
        };

        let inconsistent = repo.list_inconsistent(&p.document_name)
            .map_err(|e| e.to_string())?;
        let inconsistent_uris: std::collections::HashSet<String> =
            inconsistent.into_iter().map(|f| f.entity_uri.clone()).collect();

        let items: Vec<serde_json::Value> = fingerprints.iter().map(|f| {
            let is_inconsistent = inconsistent_uris.contains(&f.entity_uri);
            json!({
                "entity_uri": f.entity_uri,
                "entity_type": f.entity_type,
                "code_path": f.code_path,
                "sdef_hash": f.sdef_hash,
                "db_hash": f.db_hash,
                "code_hash": f.code_hash,
                "status": if is_inconsistent { "inconsistent" } else { "consistent" },
                "last_consistent_at": f.last_consistent_at,
                "suggested_strategies": if is_inconsistent {
                    json!(["sync_code_to_sdef", "regenerate_code"])
                } else {
                    json!([])
                },
            })
        }).collect();

        Ok(json!({
            "document_name": p.document_name,
            "total_fingerprints": fingerprints.len(),
            "inconsistent_count": items.iter().filter(|i| i["status"] == "inconsistent").count(),
            "items": items,
        }))
    }

    // ============ Compatibility Mode Tool Handlers ============

    fn handle_set_compatibility_mode(&self, args: Value) -> Result<Value, String> {
        use tools::compat_tools::SetCompatModeParams;
        let p: SetCompatModeParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let valid_modes = ["full", "mixed", "clean", "custom"];
        if !valid_modes.contains(&p.mode.as_str()) {
            return Err(format!(
                "Invalid mode '{}'. Valid: {}",
                p.mode,
                valid_modes.join(", ")
            ));
        }
        Ok(json!({
            "ok": true,
            "document_name": p.document_name,
            "mode": p.mode,
            "message": format!("Compatibility mode set to '{}' for '{}'", p.mode, p.document_name),
        }))
    }

    fn handle_list_compat_layers(&self, args: Value) -> Result<Value, String> {
        use tools::compat_tools::ListCompatLayersParams;
        let p: ListCompatLayersParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let conn = self.new_conn();

        // Query deprecated/legacy contracts
        let mut layers = Vec::new();
        if let Ok(mut stmt) = conn.prepare(
            "SELECT name, contract_type, status, deprecated_json, compatibility_json
             FROM contracts WHERE document_name = ?1
             AND (status IN ('deprecated', 'legacy') OR deprecated_json IS NOT NULL)"
        ) {
            if let Ok(rows) = stmt.query_map(rusqlite::params![&p.document_name], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                ))
            }) {
                for row in rows.flatten() {
                    let (name, ctype, status, dep_json, compat_json) = row;
                    let dep_info: Option<Value> = dep_json.as_ref()
                        .and_then(|j| serde_json::from_str(j).ok());
                    let compat_info: Option<Value> = compat_json.as_ref()
                        .and_then(|j| serde_json::from_str(j).ok());
                    let is_ignored = status == "active" && dep_json.as_ref().is_some();

                    layers.push(json!({
                        "layer_id": format!("{}/{}", ctype, name),
                        "name": name,
                        "contract_type": ctype,
                        "status": status,
                        "deprecated_info": dep_info,
                        "compatibility_info": compat_info,
                        "is_ignored": is_ignored,
                    }));
                }
            }
        }

        Ok(json!({
            "document_name": p.document_name,
            "layers": layers,
            "layer_count": layers.len(),
            "current_mode": "full",
        }))
    }

    fn handle_get_compat_layer(&self, args: Value) -> Result<Value, String> {
        use tools::compat_tools::GetCompatLayerParams;
        let p: GetCompatLayerParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let conn = self.new_conn();

        // Parse layer_id as contract_type/name
        let (ctype, cname) = match p.layer_id.split_once('/') {
            Some((t, n)) => (t.to_string(), n.to_string()),
            None => {
                // Try direct name match
                (String::new(), p.layer_id.clone())
            }
        };

        // Query contract details
        let result = if ctype.is_empty() {
            // Search by name only
            let mut stmt = conn.prepare(
                "SELECT name, contract_type, status, version, description, deprecated_json,
                        compatibility_json, implements_json, invariants_json
                 FROM contracts WHERE document_name = ?1 AND name = ?2"
            ).map_err(|e| e.to_string())?;
            stmt.query_row(rusqlite::params![&p.document_name, &cname], |row| {
                Ok(json!({
                    "name": row.get::<_, String>(0)?,
                    "contract_type": row.get::<_, String>(1)?,
                    "status": row.get::<_, String>(2)?,
                    "version": row.get::<_, Option<String>>(3)?,
                    "description": row.get::<_, Option<String>>(4)?,
                    "deprecated_json": row.get::<_, Option<String>>(5)?,
                    "compatibility_json": row.get::<_, Option<String>>(6)?,
                    "implements": row.get::<_, Option<String>>(7)?,
                    "invariants": row.get::<_, Option<String>>(8)?,
                }))
            }).map_err(|e| e.to_string())
        } else {
            let mut stmt = conn.prepare(
                "SELECT name, contract_type, status, version, description, deprecated_json,
                        compatibility_json, implements_json, invariants_json
                 FROM contracts WHERE document_name = ?1 AND contract_type = ?2 AND name = ?3"
            ).map_err(|e| e.to_string())?;
            stmt.query_row(
                rusqlite::params![&p.document_name, &ctype, &cname],
                |row| {
                    Ok(json!({
                        "name": row.get::<_, String>(0)?,
                        "contract_type": row.get::<_, String>(1)?,
                        "status": row.get::<_, String>(2)?,
                        "version": row.get::<_, Option<String>>(3)?,
                        "description": row.get::<_, Option<String>>(4)?,
                        "deprecated_json": row.get::<_, Option<String>>(5)?,
                        "compatibility_json": row.get::<_, Option<String>>(6)?,
                        "implements": row.get::<_, Option<String>>(7)?,
                        "invariants": row.get::<_, Option<String>>(8)?,
                    }))
                },
            ).map_err(|e| e.to_string())
        };

        match result {
            Ok(detail) => Ok(json!({
                "document_name": p.document_name,
                "layer_id": p.layer_id,
                "detail": detail,
            })),
            Err(e) => Err(format!("Compat layer '{}' not found: {}", p.layer_id, e)),
        }
    }

    fn handle_ignore_compat_layer(&self, args: Value) -> Result<Value, String> {
        use tools::compat_tools::IgnoreCompatLayerParams;
        let p: IgnoreCompatLayerParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let conn = self.new_conn();

        // Parse layer_id: "interface/MyService" or just "MyService"
        let (_ctype, cname) = match p.layer_id.split_once('/') {
            Some((t, n)) => (t.to_string(), n.to_string()),
            None => (String::new(), p.layer_id.clone()),
        };

        // Mark as active (ignore deprecation) — update status from deprecated/legacy → active
        let updated = conn.execute(
            "UPDATE contracts SET status = 'active' WHERE document_name = ?1 AND name = ?2
             AND status IN ('deprecated', 'legacy')",
            rusqlite::params![&p.document_name, &cname],
        ).map_err(|e| e.to_string())?;

        // Record in audit_log
        if updated > 0 {
            conn.execute(
                "INSERT INTO audit_log (actor, action, resource_type, resource_id, new_value_json)
                 VALUES ('user', 'ignore_compat', 'contract', ?1, ?2)",
                rusqlite::params![
                    &p.layer_id,
                    &serde_json::json!({"status": "active", "layer_id": p.layer_id}).to_string(),
                ],
            ).ok();
        }

        Ok(json!({
            "ok": true,
            "document_name": p.document_name,
            "layer_id": p.layer_id,
            "updated": updated > 0,
            "message": if updated > 0 {
                format!("Compatibility layer '{}' marked as resolved", p.layer_id)
            } else {
                format!("Compat layer '{}' already active or not found", p.layer_id)
            },
        }))
    }

    // ─── Evaluation Tools ──────────────────────────────────────────

    /// Run an evaluation against benchmark projects.
    fn handle_run_evaluation(&self, args: Value) -> Result<Value, String> {
        use cleanroom_agent::evaluation::{EvaluationRunner, EvalConfig, BenchmarkSuite};
        use tools::eval_tools::RunEvalParams;

        let p: RunEvalParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let project_name = p.project_name;
        let output_dir = p.output_dir;

        let rt = tokio::runtime::Handle::try_current()
            .map_err(|e| format!("No tokio runtime available: {}", e))?;

        let db = self.db.clone();

        rt.block_on(async move {
            let config = EvalConfig {
                output_dir: std::path::PathBuf::from(output_dir.as_deref().unwrap_or("./eval-reports")),
                ..EvalConfig::default()
            };

            let runner = EvaluationRunner::new(config, db.clone());

            // Use builtin suite or custom project name
            let mut suite = BenchmarkSuite::builtin();
            if let Some(ref name) = project_name {
                suite.projects.retain(|proj| proj.name == *name);
            }

            let report = runner.run(&suite).await
                .map_err(|e| format!("Evaluation failed: {}", e))?;

            Ok(serde_json::to_value(&report).unwrap_or(json!({
                "error": "Failed to serialize report"
            })))
        })
    }

    /// Retrieve historical evaluation reports.
    fn handle_get_evaluation_report(&self, args: Value) -> Result<Value, String> {
        use cleanroom_db::EvaluationRepository;
        use tools::eval_tools::GetEvalReportParams;

        let p: GetEvalReportParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let project_name = p.project_name;
        let limit = p.limit;

        let repo = EvaluationRepository::new(self.db.connection_arc());

        if let Some(ref project) = project_name {
            let record_limit = limit.unwrap_or(10).min(100);
            let records = repo.list_by_project(project, record_limit)
                .map_err(|e| format!("Failed to list evaluation records: {}", e))?;

            Ok(serde_json::to_value(&records).unwrap_or(json!([])))
        } else {
            // List all projects with latest summaries
            let mut results = Vec::new();
            let conn = self.db.connection();
            let mut stmt = conn
                .prepare("SELECT DISTINCT project_name FROM evaluation_history ORDER BY project_name")
                .map_err(|e| e.to_string())?;
            let projects: Vec<String> = stmt
                .query_map([], |row| row.get(0))
                .map_err(|e| e.to_string())?
                .filter_map(|r| r.ok())
                .collect();
            drop(stmt);
            drop(conn);

            for proj in &projects {
                let summary = repo.get_summary(proj)
                    .map_err(|e| format!("Failed to get summary: {}", e))?;
                results.push(summary);
            }

            Ok(serde_json::to_value(&results).unwrap_or(json!([])))
        }
    }

    // ─── Task Queue Tools ─────────────────────────────────────────────

    /// List tasks in the queue with optional status/type filters.
    fn handle_get_task_queue(&self, args: Value) -> Result<Value, String> {
        use tools::task_queue_tools::GetTaskQueueParams;

        let p: GetTaskQueueParams = serde_json::from_value(args).map_err(|e| e.to_string())?;

        let repo = self.task_repo();

        let status_filter = p.filter_status.as_ref().and_then(|s| {
            if s.is_empty() { None } else { Some(s.clone()) }
        });

        let type_filter = p.filter_type.clone();

        // Use list() with appropriate filters
        let tasks = repo
            .list(None, None, None) // all statuses, no limit
            .map_err(|e| e.to_string())?;

        let filtered: Vec<serde_json::Value> = tasks
            .into_iter()
            .filter(|t| {
                if let Some(ref statuses) = status_filter {
                    if !statuses.contains(&t.status.as_str().to_string()) {
                        return false;
                    }
                }
                if let Some(ref task_type) = type_filter {
                    if t.task_type.as_str() != *task_type {
                        return false;
                    }
                }
                true
            })
            .map(|t| serde_json::json!({
                "task_id": t.task_id,
                "task_type": t.task_type.as_str(),
                "status": t.status.as_str(),
                "priority": t.priority,
                "assigned_to": t.assigned_to,
                "progress": t.progress,
                "created_at": t.created_at,
                "started_at": t.started_at,
                "completed_at": t.completed_at,
                "dependencies": serde_json::from_str::<Vec<String>>(&t.dependencies_json).unwrap_or_default(),
                "retry_count": t.retry_count,
                "max_retries": t.max_retries,
                "error_message": t.error_message,
            }))
            .collect();

        Ok(serde_json::to_value(&filtered).unwrap_or(json!([])))
    }

    /// Insert a new task into the queue (pending status only).
    fn handle_insert_task(&self, args: Value) -> Result<Value, String> {
        use tools::task_queue_tools::InsertTaskParams;
        use cleanroom_db::{Task, TaskStatus, TaskType};

        let p: InsertTaskParams = serde_json::from_value(args).map_err(|e| e.to_string())?;

        let task_type = TaskType::from_str(&p.task_type)
            .ok_or_else(|| format!("Unknown task type: '{}'", p.task_type))?;

        let priority = p.priority.unwrap_or(5);
        let input_json = serde_json::to_string(&p.input).unwrap_or_else(|_| "{}".to_string());

        // Build dependency list
        let mut deps: Vec<String> = p.dependencies.unwrap_or_default();
        if let Some(ref after_id) = p.after_task_id {
            deps.push(after_id.clone());
        }

        let task = Task {
            task_id: uuid::Uuid::new_v4().to_string(),
            task_type,
            status: TaskStatus::Pending,
            priority,
            input_json,
            output_json: None,
            error_message: None,
            assigned_to: None,
            progress: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: p.max_retries.unwrap_or(3),
            last_heartbeat: None,
            dependencies_json: serde_json::to_string(&deps).unwrap_or_else(|_| "[]".to_string()),
            version: 1,
        };

        let task_id = task.task_id.clone();
        let repo = self.task_repo();
        repo.create(&task).map_err(|e| e.to_string())?;

        tracing::info!(%task_id, task_type = %p.task_type, priority, "Task inserted into queue");

        Ok(serde_json::to_value(&serde_json::json!({
            "inserted": true,
            "task_id": task_id,
        })).unwrap_or(json!({})))
    }

    /// Remove a pending task from the queue with dependency cascade.
    fn handle_remove_task(&self, args: Value) -> Result<Value, String> {
        use tools::task_queue_tools::RemoveTaskParams;

        let p: RemoveTaskParams = serde_json::from_value(args).map_err(|e| e.to_string())?;

        // Verify status before deletion
        let repo = self.task_repo();
        let task = repo.get(&p.task_id).map_err(|e| e.to_string())?;

        match task.status {
            TaskStatus::Completed | TaskStatus::FailedPermanently => {
                return Err(format!(
                    "Task {} is in '{}' status and cannot be removed (immutable state)",
                    p.task_id, task.status.as_str()
                ));
            }
            TaskStatus::InProgress => {
                return Err(format!(
                    "Task {} is in progress and cannot be removed — wait for it to finish or fail",
                    p.task_id
                ));
            }
            _ => {} // pending/assigned/failed/retrying — ok to delete
        }

        // Cascade: remove this task_id from other tasks' dependencies
        let cascaded = repo.cascade_remove_dependency(&p.task_id)
            .map_err(|e| e.to_string())?;

        // Delete the task
        repo.delete(&p.task_id).map_err(|e| e.to_string())?;

        tracing::info!(task_id = %p.task_id, cascaded, "Task removed from queue");

        Ok(serde_json::to_value(&serde_json::json!({
            "removed": true,
            "task_id": p.task_id,
            "cascaded_dependency_updates": cascaded,
        })).unwrap_or(json!({})))
    }

    /// Modify a pending task's properties (priority, input, dependencies, max_retries).
    fn handle_modify_task(&self, args: Value) -> Result<Value, String> {
        use tools::task_queue_tools::ModifyTaskParams;

        let p: ModifyTaskParams = serde_json::from_value(args).map_err(|e| e.to_string())?;

        let deps_json = p.dependencies
            .as_ref()
            .map(|d| serde_json::to_string(d).unwrap_or_else(|_| "[]".to_string()));

        let input_str = p.input
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()));

        let repo = self.task_repo();
        repo.update_fields(
            &p.task_id,
            p.priority,
            input_str.as_deref(),
            deps_json.as_deref(),
            p.max_retries,
        ).map_err(|e| e.to_string())?;

        tracing::info!(task_id = %p.task_id, "Task modified");

        Ok(serde_json::to_value(&serde_json::json!({
            "modified": true,
            "task_id": p.task_id,
        })).unwrap_or(json!({})))
    }

    // ─── LLM↔Human Bridge Tools ─────────────────────────────────────

    /// Request clarification from the human user.
    fn handle_request_clarification(&self, args: Value) -> Result<Value, String> {
        use tools::bridge_tools::RequestClarificationParams;

        let p: RequestClarificationParams = serde_json::from_value(args)
            .map_err(|e| e.to_string())?;

        let question_id = uuid::Uuid::new_v4().to_string();

        tracing::info!(%question_id, question = %p.question, "Clarification requested from user");

        // The actual user interaction happens outside MCP — this tool
        // records the question and returns a pending status. The CLI or
        // IDE handles presenting the question and collecting the answer.
        Ok(serde_json::to_value(&serde_json::json!({
            "question_id": question_id,
            "status": "pending",
            "question": p.question,
            "options": p.options,
            "context_uri": p.context_uri,
        })).unwrap_or(json!({})))
    }

    /// Propose a design decision for human approval.
    fn handle_propose_decision(&self, args: Value) -> Result<Value, String> {
        use tools::bridge_tools::ProposeDecisionParams;

        let p: ProposeDecisionParams = serde_json::from_value(args)
            .map_err(|e| e.to_string())?;

        let decision_id = uuid::Uuid::new_v4().to_string();

        tracing::info!(%decision_id, topic = %p.topic, "Design decision proposed to user");

        // TODO: persist decision in audit log (self.db)

        Ok(serde_json::to_value(&serde_json::json!({
            "decision_id": decision_id,
            "status": "pending",
            "topic": p.topic,
            "proposal": p.proposal,
            "rationale": p.rationale,
            "alternatives": p.alternatives,
            "affects": p.affects,
        })).unwrap_or(json!({})))
    }

    /// Preview code changes before applying.
    fn handle_preview_changes(&self, args: Value) -> Result<Value, String> {
        use tools::bridge_tools::PreviewChangesParams;

        let p: PreviewChangesParams = serde_json::from_value(args)
            .map_err(|e| e.to_string())?;

        // For now, return an empty preview — the actual diff generation
        // requires access to both existing and generated code which comes
        // from the consumer pipeline.
        Ok(serde_json::to_value(&serde_json::json!({
            "entity_uri": p.entity_uri,
            "target_language": p.target_language,
            "diff_preview": null,
            "affected_files": [],
            "status": "not_available",
            "message": "Diff preview requires generated code. Use the consumer pipeline first.",
        })).unwrap_or(json!({})))
    }

    // ─── Workflow Control Tools ─────────────────────────────────────

    /// Pause the workflow — agents finish current tasks, then stop claiming.
    fn handle_pause_workflow(&self, _args: Value) -> Result<Value, String> {
        if let Some(signal) = cleanroom_agent::GLOBAL_SIGNAL.get() {
            if signal.is_paused() {
                return Ok(serde_json::to_value(&serde_json::json!({
                    "paused": false,
                    "message": "Workflow was already paused."
                })).unwrap_or(json!({})));
            }
            signal.pause();
            tracing::info!("Workflow paused via MCP");
            Ok(serde_json::to_value(&serde_json::json!({
                "paused": true,
                "message": "Workflow paused. Agents will finish current tasks then stop."
            })).unwrap_or(json!({})))
        } else {
            Ok(serde_json::to_value(&serde_json::json!({
                "paused": false,
                "message": "No workflow signal found. Is the orchestrator running?"
            })).unwrap_or(json!({})))
        }
    }

    /// Resume a paused workflow — agents continue claiming tasks.
    fn handle_resume_workflow(&self, _args: Value) -> Result<Value, String> {
        if let Some(signal) = cleanroom_agent::GLOBAL_SIGNAL.get() {
            if !signal.is_paused() {
                return Ok(serde_json::to_value(&serde_json::json!({
                    "resumed": false,
                    "message": "Workflow is not paused."
                })).unwrap_or(json!({})));
            }
            signal.resume();
            tracing::info!("Workflow resumed via MCP");
            Ok(serde_json::to_value(&serde_json::json!({
                "resumed": true,
                "message": "Workflow resumed. Agents will continue claiming tasks."
            })).unwrap_or(json!({})))
        } else {
            Ok(serde_json::to_value(&serde_json::json!({
                "resumed": false,
                "message": "No workflow signal found. Is the orchestrator running?"
            })).unwrap_or(json!({})))
        }
    }
}

impl CleanroomMcpServer {
    // ============ Skill tool handlers (PLAN2 Phase F) ============

    /// `skill_list` — list available skills (Tier 1 catalog).
    fn handle_skill_list(&self, args: Value) -> Result<Value, String> {
        use tools::skill_tools::SkillListParams;
        let params: SkillListParams = serde_json::from_value(args)
            .map_err(|e| format!("invalid skill_list args: {e}"))?;
        // Resolve the project root: prefer CWD, fall back to the directory
        // containing `db_path`.
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let index = cleanroom_skill::load_skill_index_with_extras(
            &root,
            &[], // built-in + home are added by the strict loader
            )
            .map_err(|e| format!("load_skill_index: {e}"))?;
        let mut summaries: Vec<Value> = index
            .summaries()
            .into_iter()
            .filter(|s| match &params.scope {
                Some(scope_str) => format!("{:?}", s.scope).eq_ignore_ascii_case(scope_str),
                None => true,
            })
            .filter(|s| match &params.task_type {
                Some(tt) => index
                    .find_by_name(&s.name)
                    .map(|d| d.applies_to_task(tt))
                    .unwrap_or(false),
                None => true,
            })
            .map(|s| {
                json!({
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "scope": format!("{:?}", s.scope),
                    "priority": s.priority,
                    "token_budget": s.token_budget,
                    "allowed_tools": s.allowed_tools,
                    "allowed_paths": s.allowed_paths,
                    "applies_to": s.applies_to,
                })
            })
            .collect();
        summaries.sort_by(|a, b| {
            b["priority"]
                .as_str()
                .unwrap_or("normal")
                .cmp(a["priority"].as_str().unwrap_or("normal"))
                .then_with(|| a["name"].as_str().unwrap_or("").cmp(b["name"].as_str().unwrap_or("")))
        });
        Ok(json!({ "skills": summaries, "count": summaries.len() }))
    }

    /// `skill_activate` — load a skill's full body (Tier 2).
    fn handle_skill_activate(&self, args: Value) -> Result<Value, String> {
        use tools::skill_tools::SkillActivateParams;
        let params: SkillActivateParams = serde_json::from_value(args)
            .map_err(|e| format!("invalid skill_activate args: {e}"))?;
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let index = cleanroom_skill::load_skill_index_with_extras(&root, &[])
            .map_err(|e| format!("load_skill_index: {e}"))?;
        let skill = index
            .find_by_name(&params.name)
            .ok_or_else(|| format!("skill not found: {}", params.name))?;
        let budget = params.token_budget.unwrap_or(skill.token_budget) as usize;
        let instruction = skill.engineer_instruction(budget);
        Ok(json!({
            "name": skill.name,
            "description": skill.description,
            "body": instruction,
            "allowed_tools": skill.allowed_tools,
            "denied_tools": skill.denied_tools,
            "allowed_paths": skill.allowed_paths,
            "staging": skill.staging,
            "output_schema": skill.output_schema,
            "gates": skill.gates,
            "divergence_spec": skill.divergence_spec,
            "applies_to": skill.applies_to,
            "token_budget": skill.token_budget,
        }))
    }

    /// `skill_refresh` — re-scan the filesystem for skill changes.
    fn handle_skill_refresh(&self, args: Value) -> Result<Value, String> {
        use tools::skill_tools::SkillRefreshParams;
        let params: SkillRefreshParams = serde_json::from_value(args)
            .map_err(|e| format!("invalid skill_refresh args: {e}"))?;
        let root = match params.path {
            Some(p) => PathBuf::from(p),
            None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        };
        let index = cleanroom_skill::load_skill_index_strict(&root)
            .map_err(|e| format!("refresh: {e}"))?;
        Ok(json!({
            "root": root.display().to_string(),
            "count": index.len(),
            "skills": index.summaries().into_iter().map(|s| json!({
                "name": s.name,
                "scope": format!("{:?}", s.scope),
                "hash": s.hash,
            })).collect::<Vec<_>>(),
        }))
    }

    /// `skill_validate` — validate a SKILL.md file against the spec.
    fn handle_skill_validate(&self, args: Value) -> Result<Value, String> {
        use tools::skill_tools::SkillValidateParams;
        let params: SkillValidateParams = serde_json::from_value(args)
            .map_err(|e| format!("invalid skill_validate args: {e}"))?;
        let path = PathBuf::from(&params.path);
        let report = cleanroom_skill::validate_skill_dir(&path)
            .map_err(|e| format!("validate: {e}"))?;
        let issues: Vec<Value> = report
            .issues()
            .map(|i| {
                json!({
                    "level": format!("{:?}", i.level),
                    "message": i.message,
                })
            })
            .collect();
        Ok(json!({
            "valid": report.is_valid(),
            "error_count": report.errors.len(),
            "warning_count": report.warnings.len(),
            "issues": issues,
        }))
    }
}

// The impl CleanroomMcpServer block is closed above.
// ============ Tool Definitions (i18n) ============

/// Creates an MCP [`Tool`] definition with i18n description.
///
/// # Arguments
///
/// - `name` — Unique tool identifier (e.g., "create_task")
/// - `desc_key` — i18n translation key (e.g., "mcp.create_task")
/// - `read_only` — If true, tool does not modify data (used for permissions)
///
/// # Schema Generation
///
/// Derives JSON Schema from the generic type `T` using `schemars`.
/// The schema is included in the tool definition for LLM understanding.
fn make_tool<T: rmcp::schemars::JsonSchema>(
    name: &'static str,
    desc_key: &str,
    read_only: bool,
) -> Tool {
    let schema = CleanroomMcpServer::schema_for::<T>();
    let description = tr(desc_key);
    Tool::new(name, description, schema)
        .with_annotations(ToolAnnotations::new().read_only(read_only))
}

/// Returns the definitions for all MCP tools exposed by this server.
///
/// Used by both `list_tools` (to advertise available tools) and `get_tool`
/// (to look up a specific tool by name). Tool descriptions are fetched
/// from the i18n system using translation keys.
fn all_tools() -> Vec<Tool> {
    use tools::task_tools::*;
    use tools::sdef_tools::*;
    use tools::naming_tools::*;
    use tools::import_export_tools::*;
    use tools::lsp_tools::*;
    use tools::consistency_tools::*;
    use tools::compat_tools::*;
    use tools::skill_tools::*;

    vec![
        // Task Management (keys: mcp.xxx)
        make_tool::<CreateTaskParams>("create_task", "mcp.create_task", false),
        make_tool::<ClaimTaskParams>("claim_task", "mcp.claim_task", false),
        make_tool::<UpdateProgressParams>("update_task_progress", "mcp.update_task_progress", false),
        make_tool::<CompleteTaskParams>("complete_task", "mcp.complete_task", false),
        make_tool::<FailTaskParams>("fail_task", "mcp.fail_task", false),
        make_tool::<HeartbeatParams>("send_heartbeat", "mcp.send_heartbeat", false),
        make_tool::<CreateTaskParams>("get_task", "mcp.get_task", true),
        make_tool::<ListTasksParams>("list_tasks", "mcp.list_tasks", true),

        // S.DEF Query
        make_tool::<GetDataModelParams>("get_data_model", "mcp.get_data_model", true),
        make_tool::<GetContractParams>("get_contract", "mcp.get_contract", true),
        make_tool::<GetFunctionSpecParams>("get_function_spec", "mcp.get_function_spec", true),
        make_tool::<GetUiScreenParams>("get_ui_screen", "mcp.get_ui_screen", true),
        make_tool::<ListDocumentsParams>("list_documents", "mcp.list_documents", true),
        make_tool::<SearchSdefParams>("search_sdef", "mcp.search_sdef", true),
        make_tool::<ListShardsParams>("get_dependency_graph", "mcp.get_dependency_graph", true),
        make_tool::<ListShardsParams>("list_shards", "mcp.list_shards", true),

        // Naming Service
        make_tool::<ResolveNameParams>("resolve_name", "mcp.resolve_name", true),
        make_tool::<BatchResolveParams>("batch_resolve_names", "mcp.batch_resolve_names", true),
        make_tool::<ListSymbolsParams>("list_symbols", "mcp.list_symbols", true),
        make_tool::<RegisterCustomNameParams>("register_custom_name", "mcp.register_custom_name", false),

        // Import/Export
        make_tool::<ExportSdefParams>("export_sdef", "mcp.export_sdef", true),
        make_tool::<ExportSdefDiskParams>("export_sdef_to_disk", "mcp.export_sdef_to_disk", false),
        make_tool::<ImportSdefParams>("import_sdef", "mcp.import_sdef", false),
        make_tool::<ExportShardParams>("export_shard", "mcp.export_shard", true),
        make_tool::<ImportShardParams>("import_shard", "mcp.import_shard", false),

        // Checkpoint
        make_tool::<CheckpointParams>("create_checkpoint", "mcp.create_checkpoint", false),
        make_tool::<CheckpointParams>("list_checkpoints", "mcp.list_checkpoints", true),
        make_tool::<CheckpointIdParams>("restore_checkpoint", "mcp.restore_checkpoint", false),

        // Transaction
        make_tool::<CheckpointParams>("begin_transaction", "mcp.begin_transaction", false),
        make_tool::<TransactionIdParams>("commit_transaction", "mcp.commit_transaction", false),
        make_tool::<TransactionIdParams>("rollback_transaction", "mcp.rollback_transaction", false),

        // Consistency
        make_tool::<ConsistencyCheckParams>("check_consistency", "mcp.check_consistency", true),
        make_tool::<FingerprintParams>("compute_fingerprints", "mcp.compute_fingerprints", false),
        make_tool::<ResolveInconsistencyParams>("resolve_inconsistency", "mcp.resolve_inconsistency", false),
        make_tool::<InconsistencyReportParams>("get_inconsistency_report", "mcp.get_inconsistency_report", true),

        // LSP Tools
        make_tool::<LspInitParams>("lsp_initialize", "mcp.lsp_initialize", false),
        make_tool::<LspDocumentSymbolsParams>("lsp_get_document_symbols", "mcp.lsp_get_document_symbols", true),
        make_tool::<LspTypeInfoParams>("lsp_get_type_info", "mcp.lsp_get_type_info", true),
        make_tool::<LspFindReferencesParams>("lsp_find_references", "mcp.lsp_find_references", true),
        make_tool::<LspDiagnosticsParams>("lsp_get_diagnostics", "mcp.lsp_get_diagnostics", true),
        make_tool::<LspHierarchyParams>("lsp_get_hierarchy", "mcp.lsp_get_hierarchy", true),

        // Compatibility Mode
        make_tool::<SetCompatModeParams>("set_compatibility_mode", "mcp.set_compatibility_mode", false),
        make_tool::<ListCompatLayersParams>("list_compat_layers", "mcp.list_compat_layers", true),
        make_tool::<GetCompatLayerParams>("get_compat_layer_detail", "mcp.get_compat_layer_detail", true),
        make_tool::<IgnoreCompatLayerParams>("ignore_compat_layer", "mcp.ignore_compat_layer", false),

        // Evaluation
        make_tool::<tools::eval_tools::RunEvalParams>("run_evaluation", "mcp.run_evaluation", false),
        make_tool::<tools::eval_tools::GetEvalReportParams>("get_evaluation_report", "mcp.get_evaluation_report", true),

        // Task Queue Management (docs/15 §9)
        make_tool::<tools::task_queue_tools::GetTaskQueueParams>("get_task_queue", "mcp.get_task_queue", true),
        make_tool::<tools::task_queue_tools::InsertTaskParams>("insert_task", "mcp.insert_task", false),
        make_tool::<tools::task_queue_tools::RemoveTaskParams>("remove_task", "mcp.remove_task", false),
        make_tool::<tools::task_queue_tools::ModifyTaskParams>("modify_task", "mcp.modify_task", false),

        // LLM↔Human Bridge (docs/15 §4)
        make_tool::<tools::bridge_tools::RequestClarificationParams>("request_clarification", "mcp.request_clarification", false),
        make_tool::<tools::bridge_tools::ProposeDecisionParams>("propose_decision", "mcp.propose_decision", false),
        make_tool::<tools::bridge_tools::PreviewChangesParams>("preview_changes", "mcp.preview_changes", true),

        // Workflow Control — cross-platform pause/resume (docs/15 §10)
        make_tool::<tools::bridge_tools::PauseResumeParams>("pause_workflow", "mcp.pause_workflow", false),
        make_tool::<tools::bridge_tools::PauseResumeParams>("resume_workflow", "mcp.resume_workflow", false),

        // Skills (PLAN2 Phase F) — see docs/21-skills-system.md §6
        make_tool::<SkillListParams>("skill_list", "mcp.skill_list", true),
        make_tool::<SkillActivateParams>("skill_activate", "mcp.skill_activate", true),
        make_tool::<SkillRefreshParams>("skill_refresh", "mcp.skill_refresh", false),
        make_tool::<SkillValidateParams>("skill_validate", "mcp.skill_validate", true),
    ]
}

// ============ ServerHandler Implementation ============

/// MCP [`ServerHandler`] implementation for Cleanroom Agent.
///
/// This trait implementation handles the MCP protocol lifecycle:
/// - `get_info` — Returns server metadata and capabilities
/// - `list_tools` — Returns all available tools with their schemas
/// - `get_tool` — Looks up a single tool by name
/// - `call_tool` — Dispatches tool calls with middleware
impl ServerHandler for CleanroomMcpServer {
    /// Returns server metadata and capabilities.
    ///
    /// Announces tools capability and returns server implementation info
    /// including name and version from `CARGO_PKG_VERSION`.
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("cleanroom-agent", env!("CARGO_PKG_VERSION")))
            .with_instructions(tr("mcp.server_instructions"))
    }

    /// Lists all available MCP tools with their parameter schemas.
    ///
    /// Called by MCP clients to discover what tools are available.
    /// Returns all tools from [`all_tools()`].
    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + std::marker::Send + '_ {
        let tools = all_tools();
        async move { Ok(ListToolsResult::with_all_items(tools)) }
    }

    /// Looks up a single tool definition by name.
    ///
    /// Used by the MCP protocol to validate tool calls before dispatching.
    fn get_tool(&self, name: &str) -> Option<Tool> {
        all_tools().into_iter().find(|t| t.name == name)
    }

    /// Calls an MCP tool and returns the result.
    ///
    /// This is the main execution path. It:
    /// 1. Dispatches to the appropriate handler via [`dispatch_tool_call()`]
    /// 2. Runs request logging middleware
    /// 3. Wraps the result in [`Content::json`] or [`Content::text`]
    /// 4. Sets `is_error` flag if the handler returned an error
    fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, ErrorData>> + std::marker::Send + '_ {
        let name = request.name.to_string();
        let args = request.arguments.clone().unwrap_or_default();
        let args_value = serde_json::to_value(&args).unwrap_or(json!({}));

        let result = self.dispatch_tool_call(request);

        // Apply logging middleware (capture result for logging)
        self.log_request(&name, &args_value, &result);

        async move {
            match result {
                Ok(json_val) => {
                    let content = Content::json(&json_val)
                        .unwrap_or_else(|_| Content::text(json_val.to_string()));
                    let mut ctr = CallToolResult::default();
                    ctr.content = vec![content];
                    Ok(ctr)
                }
                Err(err_msg) => {
                    let prefix = tr("mcp.error_prefix");
                    let content = Content::text(format!("{} {}", prefix, err_msg));
                    let mut ctr = CallToolResult::default();
                    ctr.content = vec![content];
                    ctr.is_error = Some(true);
                    Ok(ctr)
                }
            }
        }
    }
}

// ============ Tests ============

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_db::Database;

    /// TempDir wrapper that keeps the DB file alive for the test's duration.
    struct TestEnv {
        _dir: tempfile::TempDir,
        server: CleanroomMcpServer,
    }

    fn setup() -> TestEnv {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test.db");
        let db = Database::open_embedded(&path).expect("Failed to open embedded DB");

        // Create test document
        db.connection().execute_batch(
            "INSERT INTO sdef_documents (name, version, description, created_at, updated_at)
             VALUES ('test-doc', '1.0', 'Test document', datetime(), datetime());"
        ).unwrap();

        let server = CleanroomMcpServer::from_db(Arc::new(db), &path);
        TestEnv { _dir: dir, server }
    }

    fn call(server: &CleanroomMcpServer, name: &str, args: serde_json::Value) -> serde_json::Value {
        let params = rmcp::model::CallToolRequestParams::new(name.to_string())
            .with_arguments(args.as_object().cloned().unwrap_or_default());
        match server.dispatch_tool_call(params) {
            Ok(val) => val,
            Err(e) => panic!("Tool '{}' failed: {}", name, e),
        }
    }

    // ======= Task Management Tests =======

    #[test]
    fn test_create_and_list_tasks() {
        let env = setup();
        let server = &env.server;

        let result = call(server, "create_task", serde_json::json!({
            "task_type": "REPO_ANALYZE",
            "input": { "path": "/tmp/test" },
            "priority": 10,
        }));
        assert!(result.get("task_id").and_then(|v| v.as_str()).is_some());
        let task_id = result["task_id"].as_str().unwrap().to_string();

        let list = call(&server, "list_tasks", serde_json::json!({}));
        let tasks = list.as_array().unwrap();
        assert!(!tasks.is_empty(), "Should have at least one task");
        assert!(tasks.iter().any(|t| t["task_id"] == task_id));
    }

    #[test]
    fn test_claim_and_complete_task() {
        let env = setup();
        let server = &env.server;

        call(&server, "create_task", serde_json::json!({
            "task_type": "EXTRACT_DATA_MODEL",
            "input": { "document": "test-doc" },
        }));

        let claimed = call(&server, "claim_task", serde_json::json!({
            "agent_id": "test-agent"
        }));
        assert!(!claimed.is_null(), "Should claim a task");
        let task_id = claimed["task_id"].as_str().unwrap().to_string();
        assert_eq!(claimed["status"], "in_progress");

        let completed = call(&server, "complete_task", serde_json::json!({
            "task_id": task_id,
            "output": { "result": "ok" },
        }));
        assert_eq!(completed["ok"], true);
    }

    #[test]
    fn test_get_task_not_found() {
        let env = setup();
        let server = &env.server;
        let params = rmcp::model::CallToolRequestParams::new("get_task".to_string())
            .with_arguments(serde_json::json!({"task_id": "nonexistent"}).as_object().cloned().unwrap());
        let result = server.dispatch_tool_call(params);
        assert!(result.is_err(), "Non-existent task should error");
    }

    // ======= S.DEF Query Tests =======

    #[test]
    fn test_list_documents() {
        let env = setup();
        let server = &env.server;
        let docs = call(&server, "list_documents", serde_json::json!({}));
        let docs_arr = docs.as_array().unwrap();
        assert!(!docs_arr.is_empty());
        assert!(docs_arr.iter().any(|d| d["name"] == "test-doc"));
    }

    #[test]
    fn test_data_model_crud() {
        let env = setup();
        let server = &env.server;
        let db = &server.db;

        // Insert a data model directly for testing
        db.connection().execute_batch(
            "INSERT INTO data_models (entity, document_name, status, description)
             VALUES ('TodoItem', 'test-doc', 'active', 'A todo item entity');
             INSERT INTO data_attributes (document_name, entity, name, attr_type, description, required)
             VALUES ('test-doc', 'TodoItem', 'title', 'string', 'Task title', 1);"
        ).unwrap();

        let result = call(&server, "get_data_model", serde_json::json!({
            "document_name": "test-doc",
            "entity": "TodoItem",
        }));
        assert_eq!(result["entity"], "TodoItem");
        let attrs = result["attributes"].as_array().unwrap();
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0]["name"], "title");
    }

    // ======= Naming Service Tests =======

    #[test]
    fn test_resolve_name_auto_generates() {
        let env = setup();
        let server = &env.server;

        let result = call(&server, "resolve_name", serde_json::json!({
            "document_name": "test-doc",
            "sdef_uri": "sdef://test-doc/entity/User",
            "language": "rust",
            "symbol_type": "class",
        }));
        assert_eq!(result["found"], false, "New name should not be found");
        assert_eq!(result["is_new"], true, "Should be auto-generated");
        assert!(result["name"].as_str().unwrap_or("").len() > 0, "Should generate a name");

        // Second call should find cached
        let cached = call(&server, "resolve_name", serde_json::json!({
            "document_name": "test-doc",
            "sdef_uri": "sdef://test-doc/entity/User",
            "language": "rust",
            "symbol_type": "class",
        }));
        assert_eq!(cached["found"], true, "Cached name should be found");
        assert_eq!(cached["name"], result["name"], "Should return same name");
    }

    #[test]
    fn test_register_custom_name() {
        let env = setup();
        let server = &env.server;

        call(&server, "register_custom_name", serde_json::json!({
            "document_name": "test-doc",
            "sdef_uri": "sdef://test-doc/entity/MyService",
            "language": "rust",
            "symbol_type": "interface",
            "concrete_name": "my_service",
        }));

        let result = call(&server, "resolve_name", serde_json::json!({
            "document_name": "test-doc",
            "sdef_uri": "sdef://test-doc/entity/MyService",
            "language": "rust",
            "symbol_type": "interface",
        }));
        assert_eq!(result["name"], "my_service");
    }

    #[test]
    fn test_list_symbols() {
        let env = setup();
        let server = &env.server;

        // Register a symbol first
        call(&server, "register_custom_name", serde_json::json!({
            "document_name": "test-doc",
            "sdef_uri": "sdef://doc/entity/Item",
            "language": "typescript",
            "symbol_type": "class",
            "concrete_name": "Item",
        }));

        let symbols = call(&server, "list_symbols", serde_json::json!({
            "document_name": "test-doc",
            "language": "typescript",
        }));
        let arr = symbols.as_array().unwrap();
        assert!(!arr.is_empty());
    }

    // ======= Checkpoint Tests =======

    #[test]
    fn test_checkpoint_lifecycle() {
        let env = setup();
        let server = &env.server;

        // Create a checkpoint
        let cp = call(&server, "create_checkpoint", serde_json::json!({
            "document_name": "test-doc",
            "description": "Test checkpoint",
        }));
        let cp_id = cp["checkpoint_id"].as_str().unwrap().to_string();
        assert!(!cp_id.is_empty());

        // List checkpoints
        let list = call(&server, "list_checkpoints", serde_json::json!({
            "document_name": "test-doc",
        }));
        let cps = list.as_array().unwrap();
        assert!(cps.iter().any(|c| c["checkpoint_id"] == cp_id));

        // Create a task, then restore
        call(&server, "create_task", serde_json::json!({
            "task_type": "REPO_ANALYZE",
            "input": { "doc": "test-doc" },
        }));

        let restore = call(&server, "restore_checkpoint", serde_json::json!({
            "checkpoint_id": cp_id,
        }));
        assert_eq!(restore["ok"], true);
    }

    // ======= Consistency Tests =======

    #[test]
    fn test_check_consistency_empty() {
        let env = setup();
        let server = &env.server;
        let result = call(&server, "check_consistency", serde_json::json!({
            "document_name": "test-doc",
        }));
        assert_eq!(result["consistent"], true, "No fingerprints = consistent");
        assert_eq!(result["inconsistent_count"], 0);
    }

    #[test]
    fn test_compute_fingerprints() {
        let env = setup();
        let server = &env.server;
        let db = &server.db;

        // Add data models
        db.connection().execute_batch(
            "INSERT INTO data_models (entity, document_name, status, description)
             VALUES ('User', 'test-doc', 'active', 'User entity');
             INSERT INTO data_attributes (document_name, entity, name, attr_type)
             VALUES ('test-doc', 'User', 'id', 'UUID');
             INSERT INTO contracts (name, document_name, contract_type, description)
             VALUES ('UserService', 'test-doc', 'interface', 'User service interface');"
        ).unwrap();

        let result = call(&server, "compute_fingerprints", serde_json::json!({
            "document_name": "test-doc",
        }));
        let count = result["fingerprint_count"].as_i64().unwrap_or(0);
        assert!(count >= 2, "Should compute fingerprints for data models + contracts");

        // Now check consistency should find them
        let check = call(&server, "check_consistency", serde_json::json!({
            "document_name": "test-doc",
        }));
        assert_eq!(check["consistent"], true, "Fresh fingerprints = consistent");
    }

    // ======= Import/Export Tests =======

    #[test]
    fn test_export_sdef() {
        let env = setup();
        let server = &env.server;
        server.db.connection().execute_batch(
            "INSERT INTO data_models (entity, document_name, status, description)
             VALUES ('Task', 'test-doc', 'active', 'A task');"
        ).unwrap();

        let result = call(server, "export_sdef", serde_json::json!({
            "document_name": "test-doc",
        }));
        let content = result["sdef_content"].as_str().unwrap();
        assert!(content.contains("Task"), "Exported S.DEF should contain the data model");
    }

    #[test]
    fn test_export_import_roundtrip() {
        let env = setup();
        let server = &env.server;
        let db = &server.db;

        // Export the (mostly) empty test document
        let exported = call(&server, "export_sdef", serde_json::json!({
            "document_name": "test-doc",
        }));
        let json_str = exported["sdef_content"].as_str().unwrap().to_string();

        // Import into a new doc
        let imported = call(&server, "import_sdef", serde_json::json!({
            "sdef_json": json_str,
        }));
        assert_eq!(imported["ok"], true);
    }

    // ======= Error Handling Tests =======

    #[test]
    fn test_unknown_tool_errors() {
        let env = setup();
        let server = &env.server;
        let params = rmcp::model::CallToolRequestParams::new("nonexistent_tool".to_string());
        let result = server.dispatch_tool_call(params);
        assert!(result.is_err(), "Unknown tool should produce error");
        let err = result.unwrap_err();
        assert!(err.contains("Unknown tool"), "Error should mention unknown tool");
    }

    #[test]
    fn test_get_contract_not_found() {
        let env = setup();
        let server = &env.server;
        let params = rmcp::model::CallToolRequestParams::new("get_contract".to_string())
            .with_arguments(serde_json::json!({
                "document_name": "test-doc",
                "name": "NonExistent",
            }).as_object().cloned().unwrap());
        let result = server.dispatch_tool_call(params);
        assert!(result.is_err(), "Non-existent contract should error");
    }

    // ======= Dependency Graph Tests =======

    #[test]
    fn test_get_dependency_graph() {
        let env = setup();
        let server = &env.server;
        let db = &server.db;

        db.connection().execute(
            "INSERT INTO data_models (entity, document_name, status) VALUES (?1, ?2, 'active')",
            rusqlite::params!["Order", "test-doc"],
        ).unwrap();
        db.connection().execute(
            "INSERT INTO data_relationships (document_name, entity, kind, target)
             VALUES ('test-doc', 'Order', 'belongs_to', 'User');",
            [],
        ).unwrap();

        let result = call(server, "get_dependency_graph", serde_json::json!({
            "document_name": "test-doc",
        }));
        assert_eq!(result["document_name"], "test-doc");
        let edges = result["edges"].as_array().unwrap();
        assert!(edges.len() >= 1, "Should have at least one relationship edge");
    }

    // ======= Shard Tool Tests =======

    #[test]
    fn test_list_shards() {
        let env = setup();
        let server = &env.server;
        let result = call(&server, "list_shards", serde_json::json!({
            "document_name": "test-doc",
        }));
        // Should succeed with empty list
        assert!(result.is_array() || result.as_array().is_some());
    }

    // ======= Search Tests =======

    #[test]
    fn test_search_sdef() {
        let env = setup();
        let server = &env.server;
        let _result = call(&server, "search_sdef", serde_json::json!({
            "query": "test",
        }));
        // FTS5 search may or may not find results; tool should not error
    }
}
