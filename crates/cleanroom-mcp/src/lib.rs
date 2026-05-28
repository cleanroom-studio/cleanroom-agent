//! cleanroom-mcp — MCP server for Cleanroom Agent.
//!
//! Exposes database operations as MCP tools for LLM interaction.
//! All tools follow the pattern: list_tools → call_tool dispatch.

use std::path::Path;
use std::sync::Arc;

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

pub mod tools;

/// The Cleanroom MCP server.
#[derive(Debug, Clone)]
pub struct CleanroomMcpServer {
    /// Database connection.
    pub db: Arc<Database>,
    /// Database file path for opening additional connections.
    pub db_path: String,
}

impl CleanroomMcpServer {
    /// Create a new MCP server instance.
    pub fn new(db_path: &Path) -> Result<Self, ErrorData> {
        let db = Database::open(db_path)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(Self {
            db: Arc::new(db),
            db_path: db_path.to_string_lossy().to_string(),
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

    // ============ Tool Dispatcher ============

    fn dispatch_tool_call(&self, request: rmcp::model::CallToolRequestParams) -> Result<Value, String> {
        let name = request.name.to_string();
        let args = request.arguments.unwrap_or_default();
        let args_value = serde_json::to_value(&args).unwrap_or(json!({}));

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
}

// ============ Tool Definitions ============

fn make_tool<T: rmcp::schemars::JsonSchema>(
    name: &'static str,
    description: &'static str,
    read_only: bool,
) -> Tool {
    let schema = CleanroomMcpServer::schema_for::<T>();
    Tool::new(name, description, schema)
        .with_annotations(ToolAnnotations::new().read_only(read_only))
}

fn all_tools() -> Vec<Tool> {
    use tools::task_tools::*;
    use tools::sdef_tools::*;
    use tools::naming_tools::*;
    use tools::import_export_tools::*;

    vec![
        // Task Management
        make_tool::<CreateTaskParams>("create_task", "创建新的分析或生成任务", false),
        make_tool::<ClaimTaskParams>("claim_task", "原子获取下一个待处理任务", false),
        make_tool::<UpdateProgressParams>("update_task_progress", "更新任务进度(0.0~1.0)", false),
        make_tool::<CompleteTaskParams>("complete_task", "标记任务完成并提交输出", false),
        make_tool::<FailTaskParams>("fail_task", "标记任务失败并记录错误", false),
        make_tool::<HeartbeatParams>("send_heartbeat", "发送任务心跳信号", false),
        make_tool::<CreateTaskParams>("get_task", "根据ID获取任务详情", true),
        make_tool::<ListTasksParams>("list_tasks", "列出任务(支持按状态/类型/责任人过滤)", true),

        // S.DEF Query
        make_tool::<GetDataModelParams>("get_data_model", "获取数据模型(含属性)", true),
        make_tool::<GetContractParams>("get_contract", "获取契约(接口/类/API)", true),
        make_tool::<ListDocumentsParams>("list_documents", "列出所有S.DEF文档", true),
        make_tool::<SearchSdefParams>("search_sdef", "FTS5全文搜索S.DEF文档", true),
        make_tool::<GetDataModelParams>("get_dependency_graph", "获取模块依赖图", true),

        // Naming Service
        make_tool::<ResolveNameParams>("resolve_name", "解析S.DEF URI → 代码名称", true),
        make_tool::<BatchResolveParams>("batch_resolve_names", "批量解析URI → 名称", true),
        make_tool::<ListSymbolsParams>("list_symbols", "列出已注册符号(按语言/类型过滤)", true),
        make_tool::<RegisterCustomNameParams>("register_custom_name", "手动注册自定义名称", false),

        // Import/Export
        make_tool::<ExportSdefParams>("export_sdef", "导出完整S.DEF(json/yaml)", true),
        make_tool::<ImportSdefParams>("import_sdef", "从JSON导入S.DEF到数据库", false),

        // Checkpoint
        make_tool::<CheckpointParams>("create_checkpoint", "创建全局检查点", false),
        make_tool::<CheckpointParams>("list_checkpoints", "列出文档的检查点", true),
        make_tool::<CheckpointIdParams>("restore_checkpoint", "恢复检查点", false),

        // Transaction
        make_tool::<CreateTaskParams>("begin_transaction", "开始数据库事务", false),
        make_tool::<CreateTaskParams>("commit_transaction", "提交事务", false),
        make_tool::<CreateTaskParams>("rollback_transaction", "回滚事务", false),

        // Consistency
        make_tool::<ConsistencyCheckParams>("check_consistency", "运行一致性检查", true),
        make_tool::<FingerprintParams>("compute_fingerprints", "重新计算并存储指纹", false),
    ]
}

// ============ ServerHandler Implementation ============

impl ServerHandler for CleanroomMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("cleanroom-agent", env!("CARGO_PKG_VERSION")))
            .with_instructions("S.DEF intelligent agent system. Use tools to manage tasks, query S.DEF definitions, resolve names, and manage consistency.")
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
        let result = self.dispatch_tool_call(request);
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
                    let content = Content::text(format!("Error: {}", err_msg));
                    let mut ctr = CallToolResult::default();
                    ctr.content = vec![content];
                    ctr.is_error = Some(true);
                    Ok(ctr)
                }
            }
        }
    }
}
