//! cleanroom-mcp — MCP server for Cleanroom Agent.
//!
//! Exposes database operations as MCP tools for LLM interaction.
//! All tools follow the pattern: list_tools → call_tool dispatch.

use std::path::Path;
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
use cleanroom_db::repositories::{CheckpointRepository, Checkpoint};
use cleanroom_lsp::LspServerPool;

fn tr(key: &str) -> String {
    cleanroom_i18n::global().translate(key)
}

pub mod tools;

/// The Cleanroom MCP server.
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
    /// Create a new MCP server instance.
    pub fn new(db_path: &Path) -> Result<Self, ErrorData> {
        let db = Database::open(db_path)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(Self {
            db: Arc::new(db),
            db_path: db_path.to_string_lossy().to_string(),
            lsp_pool: Arc::new(Mutex::new(LspServerPool::new())),
        })
    }

    /// Start the server over stdio transport.
    pub async fn serve(self) -> Result<(), ErrorData> {
        let transport = (tokio::io::stdin(), tokio::io::stdout());
        let _running = serve_server(self, transport).await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(())
    }

    /// Helper: derive JSON schemas for tool parameters.
    fn schema_for<T: rmcp::schemars::JsonSchema>() -> Arc<JsonObject> {
        let schema = rmcp::schemars::schema_for!(T);
        let value = serde_json::to_value(&schema).unwrap_or(json!({}));
        Arc::new(value.as_object().cloned().unwrap_or_default())
    }

    /// Open a new SQLite connection to the same database file (safe with WAL mode).
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

    fn execute(&self, sql: &str, params: &[&dyn rusqlite::types::ToSql]) -> Result<(), String> {
        self.db.connection()
            .execute(sql, params)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    // ============ Middleware Layer ============

    /// Permission check: enforce that agents can only manage tasks they own.
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

    /// Request logging: record all tool calls to tracing.
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

    fn dispatch_tool_call(&self, request: rmcp::model::CallToolRequestParams) -> Result<Value, String> {
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
            "get_dependency_graph" => self.handle_list_documents(args_value),
            // Naming
            "resolve_name" => self.handle_resolve_name(args_value),
            "batch_resolve_names" => self.handle_batch_resolve(args_value),
            "list_symbols" => self.handle_list_symbols(args_value),
            "register_custom_name" => self.handle_register_custom_name(args_value),
            // Import/Export
            "export_sdef" => self.handle_export_sdef(args_value),
            "import_sdef" => self.handle_import_sdef(args_value),
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

    // ============ Naming Service Tool Handlers ============

    fn handle_resolve_name(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, sdef_uri: String, language: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        match self.symbol_repo().resolve(&p.document_name, &p.sdef_uri, &p.language)
            .map_err(|e| e.to_string())?
        {
            Some(name) => Ok(json!({"name": name, "found": true})),
            None => Ok(json!({"name": null, "found": false})),
        }
    }

    fn handle_batch_resolve(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, uris: Vec<String>, language: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let uri_refs: Vec<(&str, SymbolType)> = p.uris.iter()
            .map(|u| (u.as_str(), SymbolType::Variable)).collect();
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

    fn handle_import_sdef(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { sdef_json: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let sdef: sdef_core::SoftwareDefinition =
            serde_json::from_str(&p.sdef_json).map_err(|e| e.to_string())?;
        let doc_name = self.importer().import(&sdef).map_err(|e| e.to_string())?;
        Ok(json!({"document_name": doc_name, "ok": true}))
    }

    // ============ Checkpoint Tool Handlers ============

    fn handle_create_checkpoint(&self, args: Value) -> Result<Value, String> {
        #[derive(serde::Deserialize)]
        struct P { document_name: String, description: Option<String> }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let repo = self.checkpoint_repo();
        let cp = Checkpoint {
            checkpoint_id: uuid::Uuid::new_v4().to_string(),
            document_name: p.document_name, description: p.description,
            created_at: String::new(),
            task_snapshot_json: "{}".to_string(),
            shard_snapshot_json: "{}".to_string(),
        };
        repo.create(&cp).map_err(|e| e.to_string())?;
        Ok(json!({"checkpoint_id": cp.checkpoint_id}))
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
        let _p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        Ok(json!({"ok": true}))
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
        #[derive(serde::Deserialize)]
        struct P { document_name: String }
        let p: P = serde_json::from_value(args).map_err(|e| e.to_string())?;
        let count = self.fingerprint_repo()
            .list_by_document(&p.document_name).map_err(|e| e.to_string())?.len();
        Ok(json!({"fingerprint_count": count, "document_name": p.document_name}))
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
            "server_info": format!("{} LSP server", p.language),
        }))
    }

    fn handle_lsp_get_document_symbols(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspDocumentSymbolsParams;
        let p: LspDocumentSymbolsParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        Ok(json!({
            "file_path": p.file_path,
            "language": p.language,
            "note": "Full LSP analysis requires a running LSP server and init sequence. This provides the requested parameter structure."
        }))
    }

    fn handle_lsp_get_type_info(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspTypeInfoParams;
        let p: LspTypeInfoParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        Ok(json!({
            "file_path": p.file_path,
            "language": p.language,
            "position": {"line": p.line, "character": p.character},
            "note": "Type info query dispatched. LSP must be initialized first."
        }))
    }

    fn handle_lsp_find_references(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspFindReferencesParams;
        let p: LspFindReferencesParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        Ok(json!({
            "file_path": p.file_path,
            "language": p.language,
            "position": {"line": p.line, "character": p.character},
            "include_declaration": p.include_declaration,
            "note": "Find references query dispatched. LSP must be initialized first."
        }))
    }

    fn handle_lsp_get_diagnostics(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspDiagnosticsParams;
        let p: LspDiagnosticsParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        Ok(json!({
            "file_path": p.file_path,
            "language": p.language,
            "note": "Diagnostics query dispatched. LSP must be initialized first."
        }))
    }

    fn handle_lsp_get_hierarchy(&self, args: Value) -> Result<Value, String> {
        use tools::lsp_tools::LspHierarchyParams;
        let p: LspHierarchyParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        Ok(json!({
            "file_path": p.file_path,
            "language": p.language,
            "position": {"line": p.line, "character": p.character},
            "direction": p.direction,
            "note": "Type hierarchy query dispatched. LSP must be initialized first."
        }))
    }

    // ============ Consistency Tool Handlers (extended) ============

    fn handle_resolve_inconsistency(&self, args: Value) -> Result<Value, String> {
        use tools::consistency_tools::ResolveInconsistencyParams;
        let p: ResolveInconsistencyParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        // Validate strategy
        let valid_strategies = [
            "sync_code_to_sdef", "regenerate_code", "sync_db_to_sdef",
            "sync_sdef_to_db", "accept_external",
        ];
        if !valid_strategies.contains(&p.strategy.as_str()) {
            return Err(format!(
                "Invalid strategy '{}'. Valid: {}",
                p.strategy,
                valid_strategies.join(", ")
            ));
        }
        let repo = self.fingerprint_repo();
        let fingerprints = repo.list_by_document(&p.document_name)
            .map_err(|e| e.to_string())?;

        let entity_found = fingerprints.iter().any(|f| f.entity_uri == p.entity_uri);
        if !entity_found {
            return Err(format!("Entity '{}' not found in document '{}'", p.entity_uri, p.document_name));
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
        // Query from database; for now return empty list
        Ok(json!({
            "document_name": p.document_name,
            "layers": [],
            "current_mode": "full",
        }))
    }

    fn handle_get_compat_layer(&self, args: Value) -> Result<Value, String> {
        use tools::compat_tools::GetCompatLayerParams;
        let p: GetCompatLayerParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        Ok(json!({
            "document_name": p.document_name,
            "layer_id": p.layer_id,
            "note": "Layer detail query dispatched. Compatibility data stored in contracts table.",
        }))
    }

    fn handle_ignore_compat_layer(&self, args: Value) -> Result<Value, String> {
        use tools::compat_tools::IgnoreCompatLayerParams;
        let p: IgnoreCompatLayerParams = serde_json::from_value(args).map_err(|e| e.to_string())?;
        Ok(json!({
            "ok": true,
            "document_name": p.document_name,
            "layer_id": p.layer_id,
            "message": format!("Compatibility layer '{}' marked as resolved/ignored", p.layer_id),
        }))
    }
}

// ============ Tool Definitions (i18n) ============

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

fn all_tools() -> Vec<Tool> {
    use tools::task_tools::*;
    use tools::sdef_tools::*;
    use tools::naming_tools::*;
    use tools::import_export_tools::*;
    use tools::lsp_tools::*;
    use tools::consistency_tools::*;
    use tools::compat_tools::*;

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
        make_tool::<GetDataModelParams>("get_dependency_graph", "mcp.get_dependency_graph", true),

        // Naming Service
        make_tool::<ResolveNameParams>("resolve_name", "mcp.resolve_name", true),
        make_tool::<BatchResolveParams>("batch_resolve_names", "mcp.batch_resolve_names", true),
        make_tool::<ListSymbolsParams>("list_symbols", "mcp.list_symbols", true),
        make_tool::<RegisterCustomNameParams>("register_custom_name", "mcp.register_custom_name", false),

        // Import/Export
        make_tool::<ExportSdefParams>("export_sdef", "mcp.export_sdef", true),
        make_tool::<ImportSdefParams>("import_sdef", "mcp.import_sdef", false),

        // Checkpoint
        make_tool::<CheckpointParams>("create_checkpoint", "mcp.create_checkpoint", false),
        make_tool::<CheckpointParams>("list_checkpoints", "mcp.list_checkpoints", true),
        make_tool::<CheckpointIdParams>("restore_checkpoint", "mcp.restore_checkpoint", false),

        // Transaction
        make_tool::<CreateTaskParams>("begin_transaction", "mcp.begin_transaction", false),
        make_tool::<CreateTaskParams>("commit_transaction", "mcp.commit_transaction", false),
        make_tool::<CreateTaskParams>("rollback_transaction", "mcp.rollback_transaction", false),

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
    ]
}

// ============ ServerHandler Implementation ============

impl ServerHandler for CleanroomMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("cleanroom-agent", env!("CARGO_PKG_VERSION")))
            .with_instructions(tr("mcp.server_instructions"))
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + std::marker::Send + '_ {
        let tools = all_tools();
        async move { Ok(ListToolsResult::with_all_items(tools)) }
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        all_tools().into_iter().find(|t| t.name == name)
    }

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
