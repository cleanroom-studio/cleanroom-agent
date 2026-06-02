//! Producer Agent — analyzes code repositories and produces S.DEF documents.
//!
//! The Producer Agent is responsible for the "produce" phase of the Cleanroom
//! agent pipeline. It takes a source code repository and generates a complete
//! S.DEF (Software Definition Exchange Format) document describing the codebase.
//!
//! # Pipeline
//!
//! The producer has two execution modes (selected via [`ProducerConfig::mode`]):
//!
//! ## `Template` mode (default; pre-Phase-0.5)
//!
//! Runs the full template-driven analysis pipeline via
//! [`run_analysis_pipeline`]:
//! 1. Repository scanning via [`scan_repository`](crate::repo_scanner::scan_repository)
//! 2. Module partitioning via [`partition_files`](crate::module_partitioner::partition_files)
//! 3. Dependency graph construction via [`DependencyGraph`]
//! 4. IR to S.DEF mapping via [`SdefMapper`]
//! 5. Persistence to database
//!
//! ## `Llm` mode (Phase 0.5+)
//!
//! Each source file gets a `LlmAnalyzeFile` task scheduled in parallel
//! (no inter-file dependencies). The worker is the LLM itself — the
//! per-task handler is [`ProducerAgent::analyze_file_with_llm`], which
//! reads the file, builds a system prompt + user message, calls
//! [`crate::llm_loop::run_loop`], and persists the raw LLM output
//! (with token counts) to `output_json`. Phase 0.7 will add S.DEF parsing
//! to the output.
//!
//! ## `Both` mode
//!
//! Runs `Template` first, then schedules `Llm` tasks on top. Useful for
//! Phase 5 baseline comparison (the template output is the "before LLM"
//! baseline; the LLM output is the "after LLM" experiment).
//!
//! # Task Processing
//!
//! The agent claims and processes tasks from the database task queue.
//! Each task may represent a different phase of repository analysis.

use std::path::Path;
use std::sync::Arc;

use cleanroom_db::{
    Database, DbError, LlmCallLogRepository, Task, TaskRepository, TaskStatus, TaskType,
};
use cleanroom_meta_llm::MetaLlm;
use tracing::{info, instrument, warn};

use crate::llm_loop::{run_loop, LoopConfig, LoopContext, LoopOutcome};
use crate::llm_sdef_parser::{parse_llm_analyze_output, write_parsed_to_db};
use crate::producer_pipeline::{run_analysis_pipeline, run_analysis_pipeline_with_lsp};

/// Producer execution mode — controls how `analyze_repo` processes a repository.
///
/// Pre-Phase-0.5 the only option was `Template` (the hard-coded pipeline).
/// Phase 0.5 adds `Llm` and `Both`. See module-level docs for the full picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProducerMode {
    /// Run the pre-Phase-0.5 template-driven pipeline (`producer_pipeline.rs`).
    /// No LLM calls; pure static analysis. Phase 5 will use this as the
    /// baseline for "before LLM" comparison.
    Template,
    /// Schedule one `LlmAnalyzeFile` task per source file, each handled
    /// by `ProducerAgent::analyze_file_with_llm`. Requires the producer
    /// to be constructed with `.with_llm(...)`.
    Llm,
    /// Run `Template` first, then schedule `Llm` tasks on top. The two
    /// outputs land in separate `output_json` payloads, suitable for
    /// diffing in Phase 5.
    Both,
}

impl Default for ProducerMode {
    fn default() -> Self {
        Self::Template
    }
}

/// Producer configuration.
///
/// Contains settings for the producer agent's behavior during code analysis.
#[derive(Debug, Clone)]
pub struct ProducerConfig {
    /// List of programming languages the producer should recognize
    pub languages: Vec<String>,
    /// Whether to enable LSP (Language Server Protocol) for enhanced analysis
    pub lsp_enabled: bool,
    /// Execution mode. Defaults to `Template` (legacy, baseline-comparable).
    /// Switch to `Llm` (or `Both`) to drive analysis through
    /// `analyze_file_with_llm` instead of `producer_pipeline`.
    pub mode: ProducerMode,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            languages: vec![
                "typescript".to_string(),
                "javascript".to_string(),
                "python".to_string(),
                "rust".to_string(),
                "c".to_string(),
                "cpp".to_string(),
                "go".to_string(),
                "java".to_string(),
            ],
            lsp_enabled: false, // Disabled by default; enable with --lsp flag
            mode: ProducerMode::default(),
        }
    }
}

impl ProducerConfig {
    /// Convenience: build a config in `Llm` mode.
    pub fn llm() -> Self {
        Self {
            mode: ProducerMode::Llm,
            ..Self::default()
        }
    }

    /// Convenience: build a config in `Both` mode.
    pub fn both() -> Self {
        Self {
            mode: ProducerMode::Both,
            ..Self::default()
        }
    }
}

/// Producer Agent — analyzes code repositories and produces S.DEF documents.
///
/// The Producer Agent claims tasks from the database queue and executes
/// repository analysis via the producer pipeline. Each successful analysis
/// results in a complete S.DEF document stored in the database.
///
/// # Task Types
///
/// The producer handles the following task types:
/// - [`TaskType::RepoAnalyze`]: Full repository analysis (template pipeline)
/// - [`TaskType::LlmAnalyzeFile`]: LLM-driven per-file analysis (Phase 0.5+)
#[allow(dead_code)]
pub struct ProducerAgent {
    /// Producer configuration settings
    pub(crate) config: ProducerConfig,
    /// Database connection for task persistence
    db: Arc<Database>,
    /// Unique agent identifier for task claiming
    agent_id: String,
    /// LLM used for `LlmAnalyzeFile` tasks. None = LLM path disabled.
    /// Phase 0.5: optional; Phase 1: required for the default path.
    llm: Option<Arc<dyn MetaLlm>>,
    /// Loop config used for `LlmAnalyzeFile` tasks (token / cost limits).
    loop_config: LoopConfig,
    /// Optional Phase 0.9 audit logger. When set, every `run_loop`
    /// invocation will fire `LoopConfig::on_call_complete` to persist
    /// the call record to the `llm_call_log` table.
    llm_call_logger: Option<Arc<LlmCallLogRepository>>,
}

impl ProducerAgent {
    /// Create a new producer agent (no LLM by default).
    pub fn new(config: ProducerConfig, db: Arc<Database>) -> Self {
        let agent_id = format!("producer-{}", uuid::Uuid::new_v4());
        Self {
            config,
            db,
            agent_id,
            llm: None,
            loop_config: LoopConfig::default(),
            llm_call_logger: None,
        }
    }

    /// Attach an LLM for `LlmAnalyzeFile` task execution (Phase 0.5+).
    /// Without this, claims of `LlmAnalyzeFile` tasks fail with
    /// "LLM not configured".
    pub fn with_llm(mut self, llm: Arc<dyn MetaLlm>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the LLM loop config (token / iteration / cost guardrails).
    pub fn with_loop_config(mut self, cfg: LoopConfig) -> Self {
        self.loop_config = cfg;
        self
    }

    /// Attach an `llm_call_log` audit logger (Phase 0.9). Once set,
    /// every `run_loop` invocation will receive an `on_call_complete`
    /// callback that writes a row to the `llm_call_log` table.
    pub fn with_llm_call_logger(mut self, logger: Arc<LlmCallLogRepository>) -> Self {
        self.llm_call_logger = Some(logger);
        self
    }

    /// Get agent ID.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Whether an LLM is attached. `false` means LLM-driven tasks will fail
    /// with a clear error if claimed.
    pub fn has_llm(&self) -> bool {
        self.llm.is_some()
    }

    /// Claim and process a task.
    #[instrument(skip(self))]
    pub async fn process_next_task(&self) -> Result<Option<Task>, DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());

        if let Some(task) = repo.claim(&self.agent_id)? {
            info!(task_id = %task.task_id, task_type = ?task.task_type, "Processing task");

            match task.task_type {
                TaskType::RepoAnalyze => self.analyze_repo(&task).await?,
                TaskType::LlmAnalyzeFile => self.analyze_file_with_llm(&task).await?,
                _ => {
                    repo.complete(&task.task_id, "{}")?;
                }
            }

            return Ok(Some(task));
        }

        Ok(None)
    }

    /// Full repository analysis using the integrated pipeline.
    ///
    /// Dispatches to one of three strategies based on [`ProducerConfig::mode`]:
    /// - `Template`: legacy `producer_pipeline::run_analysis_pipeline` path
    ///   (LSP-aware via `lsp_enabled`).
    /// - `Llm`: scan the repo, build a `TaskPlan::llm_analysis_plan` with one
    ///   `LlmAnalyzeFile` task per source file, persist the plan, and report
    ///   a summary (file count) as the task output. The actual LLM work
    ///   happens later when `process_next_task` claims each `LlmAnalyzeFile`.
    /// - `Both`: run `Template` first, then schedule the LLM plan on top.
    ///
    /// The `LlmAnalyzeFile` tasks are NOT executed inline here — they sit in
    /// the queue and are processed by separate workers (or the same agent in
    /// a subsequent loop). This matches the existing task-queue contract
    /// where `RepoAnalyze` is a "high-level workflow" task and leaf tasks are
    /// claimed separately.
    async fn analyze_repo(&self, task: &Task) -> Result<(), DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());

        // Parse task input
        let input: serde_json::Value = serde_json::from_str(&task.input_json)
            .unwrap_or_else(|_| serde_json::json!({}));

        let repo_path = input.get("repo_path")
            .and_then(|v| v.as_str())
            .map(Path::new)
            .unwrap_or_else(|| Path::new("."));

        let project_name = input.get("project_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unnamed");

        info!(path = %repo_path.display(), project = %project_name, mode = ?self.config.mode, "Starting analyze_repo");

        // Update progress to 10% (template path only -- the LLM path has
        // its own progress reporting through the LlmAnalyzeFile tasks)
        repo.update_progress(&task.task_id, 0.1)?;

        match self.config.mode {
            ProducerMode::Template => self.analyze_repo_template(task, repo_path, project_name, &repo).await,
            ProducerMode::Llm => self.analyze_repo_llm(task, repo_path, project_name, &repo).await,
            ProducerMode::Both => {
                self.analyze_repo_template(task, repo_path, project_name, &repo).await?;
                self.analyze_repo_llm(task, repo_path, project_name, &repo).await
            }
        }
    }

    /// `Template` mode: legacy static-analysis pipeline. No LLM calls.
    async fn analyze_repo_template(
        &self,
        task: &Task,
        repo_path: &Path,
        project_name: &str,
        repo: &TaskRepository,
    ) -> Result<(), DbError> {
        let result = if self.config.lsp_enabled {
            run_analysis_pipeline_with_lsp(
                self.db.clone(),
                repo_path,
                project_name,
                "0.1.0",
                None,
                true,
            ).await?
        } else {
            run_analysis_pipeline(
                self.db.clone(),
                repo_path,
                project_name,
                "0.1.0",
                None,
            ).await?
        };

        repo.update_progress(&task.task_id, 0.9)?;

        let output = crate::producer_pipeline::result_to_json(&result);
        repo.complete(&task.task_id, &serde_json::to_string(&output).unwrap_or_default())?;

        info!(
            files = result.file_count,
            modules = result.module_count,
            data_models = result.sdef.data_models.as_ref().map(|v| v.len()).unwrap_or(0),
            "Template analyze_repo complete"
        );
        Ok(())
    }

    /// `Llm` mode: scan the repo, schedule one `LlmAnalyzeFile` task per
    /// source file. The tasks sit in the queue waiting for workers (or
    /// subsequent `process_next_task` calls in the same agent) to claim
    /// them. Reports the plan summary (file count) as the task output so
    /// downstream observers can see what was scheduled.
    async fn analyze_repo_llm(
        &self,
        task: &Task,
        repo_path: &Path,
        project_name: &str,
        repo: &TaskRepository,
    ) -> Result<(), DbError> {
        // 1. Scan the repo for source files. We filter to files with a
        //    detected language because the LLM prompt is language-agnostic
        //    today (Phase 0.2 will add per-language hint packs).
        let scan_config = crate::repo_scanner::ScanConfig {
            root: repo_path.to_path_buf(),
            exclude_patterns: vec![],
            include_dotfiles: false,
            max_file_size: 1_000_000,
        };
        let source_files: Vec<String> = crate::repo_scanner::scan_repository(&scan_config)
            .into_iter()
            .filter(|f| f.language.is_some())
            .map(|f| {
                f.relative_path
                    .to_string_lossy()
                    .replace('\\', "/") // normalize windows separators
            })
            .collect();
        info!(
            file_count = source_files.len(),
            "Llm analyze_repo: scan complete"
        );

        if source_files.is_empty() {
            let output = serde_json::json!({
                "mode": "llm",
                "scheduled_task_count": 0,
                "note": "no source files found in repo",
            });
            repo.complete(&task.task_id, &serde_json::to_string(&output).unwrap_or_default())?;
            return Ok(());
        }

        // 2. Build the LLM plan and create tasks.
        let plan = crate::scheduler::TaskPlan::llm_analysis_plan(
            project_name,
            &repo_path.to_string_lossy(),
            &source_files.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        );
        let scheduler = crate::scheduler::Scheduler::from_db((*self.db).clone());
        let created_ids = scheduler.create_from_plan(&plan)?;
        repo.update_progress(&task.task_id, 0.9)?;

        // 3. Mark the high-level task complete with a summary. The leaf
        //    `LlmAnalyzeFile` tasks are now in Pending state and will be
        //    processed by subsequent `process_next_task` calls.
        let output = serde_json::json!({
            "mode": "llm",
            "scheduled_task_count": created_ids.len(),
            "scheduled_task_ids": created_ids,
            "next_step": "call process_next_task() per file to drive the LLM",
        });
        repo.complete(&task.task_id, &serde_json::to_string(&output).unwrap_or_default())?;

        info!(
            project = %project_name,
            scheduled = created_ids.len(),
            "Llm analyze_repo complete: tasks are now Pending in the queue"
        );
        Ok(())
    }

    /// LLM-driven per-file analysis (Phase 0.5).
    ///
    /// Task input JSON shape (produced by `TaskPlan::llm_analysis_plan`):
    /// ```json
    /// {
    ///   "document": "<project name>",
    ///   "project_name": "<project name>",
    ///   "repo_path": "/abs/path/to/repo",
    ///   "file_path": "src/foo.rs"
    /// }
    /// ```
    ///
    /// Flow:
    /// 1. Read `<repo_path>/<file_path>` from disk
    /// 2. Build a system prompt + user message asking the LLM to emit S.DEF
    ///    entities (data models, contracts, functions) as JSON
    /// 3. Call `llm_loop::run_loop` (single-shot `MetaProvider::chat`)
    /// 4. Persist the **raw LLM output** to `output_json` for downstream
    ///    parsing (Phase 0.7+ will add S.DEF parsing + DB writes)
    /// 5. Surface the prompt / completion token counts in the output JSON
    ///    so `llm_call_log` (Phase 0.9) and the cost guardrail can read them
    ///
    /// For now, we do NOT call MCP tools. The LLM is a black box that
    /// produces S.DEF-shaped JSON; mcp_tool_bridge integration (Phase 0.5
    /// follow-up) is what unlocks ReAct + tool-calling here.
    #[instrument(skip(self, task))]
    pub async fn analyze_file_with_llm(&self, task: &Task) -> Result<(), DbError> {
        let llm = self.llm.as_ref().ok_or_else(|| {
            DbError::QueryFailed(
                "LlmAnalyzeFile task claimed but no LLM is attached. \
                 Construct the producer with `.with_llm(...)` to enable the LLM path."
                    .to_string(),
            )
        })?;

        // 1. Parse task input.
        let input: serde_json::Value = serde_json::from_str(&task.input_json)
            .unwrap_or_else(|_| serde_json::json!({}));
        let repo_path = input
            .get("repo_path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        let file_path = input
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                DbError::QueryFailed("LlmAnalyzeFile task missing input.file_path".to_string())
            })?;
        let document = input
            .get("document")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        // 2. Read the source file. Empty / unreadable files fail the task.
        let full_path = Path::new(repo_path).join(file_path);
        let source = std::fs::read_to_string(&full_path).map_err(|e| {
            DbError::QueryFailed(format!("read {} failed: {e}", full_path.display()))
        })?;
        if source.trim().is_empty() {
            warn!(file = %full_path.display(), "empty source file; skipping LLM call");
            let repo = TaskRepository::new(self.db.connection_arc());
            let output = serde_json::json!({
                "file_path": file_path,
                "skipped": true,
                "reason": "empty source",
            });
            repo.complete(&task.task_id, &serde_json::to_string(&output).unwrap_or_default())?;
            return Ok(());
        }

        // 3. Build prompts.
        let system_prompt = build_llm_analyze_file_system_prompt(document, file_path);
        let user_message = format!(
            "Analyze the following source file and emit a single JSON object that conforms to \
             the S.DEF `sdef_output` schema. Wrap your JSON between ```json ... ``` fences so \
             the downstream parser can extract it cleanly. Do not include commentary outside \
             the JSON fences.\n\nFile: {file_path}\n\n```\n{source}\n```"
        );

        // 4. Call the LLM.
        // Phase 0.9: build a per-call LoopConfig so we can attach the
        // `on_call_complete` audit-log hook without mutating the shared
        // `self.loop_config`. The hook (if `llm_call_logger` is set)
        // persists every LLM call to the `llm_call_log` table.
        let mut loop_cfg = self.loop_config.clone();
        if let Some(logger) = self.llm_call_logger.clone() {
            loop_cfg.on_call_complete = Some(Arc::new(move |log: cleanroom_db::LlmCallLog| {
                if let Err(e) = logger.create(&log) {
                    tracing::warn!(
                        error = %e,
                        call_id = %log.call_id,
                        "llm_call_log: failed to persist LlmCallLog"
                    );
                }
            }));
        }
        // Best-effort model name for the audit log: try `EVAL_MODEL`
        // (the same env var `build_llm_from_env` consults); fall back
        // to `"unknown"`. Phase 1+ will plumb the model through
        // `with_llm` so this is no longer needed.
        let model_name = std::env::var("EVAL_MODEL")
            .unwrap_or_else(|_| "unknown".to_string());
        let ctx = LoopContext::new(
            &task.task_id,
            "producer-llm-session",
            "cleanroom-producer",
            system_prompt,
            user_message,
        )
        .with_model(model_name);
        let outcome = run_loop(llm.clone(), ctx, &loop_cfg)
            .await
            .map_err(|e| DbError::QueryFailed(format!("LLM call failed: {e}")))?;

        // 5. Map LoopOutcome -> task output JSON.
        let repo = TaskRepository::new(self.db.connection_arc());
        match outcome {
            LoopOutcome::Done {
                result,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                // Phase 0.5 收尾: parse the LLM JSON into S.DEF entities
                // and persist them. Failures are non-fatal: the raw
                // output is still saved in `output_json` for replay.
                let sdef_repo = cleanroom_db::SdefRepository::new_with_arc(
                    self.db.connection_arc(),
                );
                let (parser_status, parse_summary) = match parse_llm_analyze_output(&result) {
                    Ok(entities) => match write_parsed_to_db(&sdef_repo, document, &entities) {
                        Ok(summary) => {
                            info!(
                                task_id = %task.task_id,
                                file = %file_path,
                                data_models = summary.data_models,
                                attributes = summary.attributes,
                                contracts = summary.contracts,
                                functions = summary.functions,
                                design_decisions = summary.design_decisions,
                                "LlmAnalyzeFile: parsed + persisted S.DEF entities"
                            );
                            (
                                "sdef-output/v0.1".to_string(),
                                Some(serde_json::json!({
                                    "data_models": summary.data_models,
                                    "attributes": summary.attributes,
                                    "contracts": summary.contracts,
                                    "functions": summary.functions,
                                    "design_decisions": summary.design_decisions,
                                })),
                            )
                        }
                        Err(e) => {
                            warn!(
                                task_id = %task.task_id,
                                file = %file_path,
                                error = %e,
                                "LlmAnalyzeFile: parsed but DB write failed"
                            );
                            (
                                format!("sdef-output/v0.1 (write failed: {e})"),
                                None,
                            )
                        }
                    },
                    Err(e) => {
                        warn!(
                            task_id = %task.task_id,
                            file = %file_path,
                            error = %e,
                            "LlmAnalyzeFile: LLM output could not be parsed"
                        );
                        (format!("sdef-output/v0.1 (parse failed: {e})"), None)
                    }
                };
                let output = serde_json::json!({
                    "file_path": file_path,
                    "document": document,
                    "raw_llm_output": result,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "schema_version": "sdef-output/draft",
                    "parser": parser_status,
                    "parse_summary": parse_summary,
                });
                repo.complete(&task.task_id, &serde_json::to_string(&output).unwrap_or_default())?;
                info!(
                    task_id = %task.task_id,
                    file = %file_path,
                    prompt = prompt_tokens,
                    completion = completion_tokens,
                    "LlmAnalyzeFile task completed"
                );
                Ok(())
            }
            LoopOutcome::Aborted { reason, .. } => Err(DbError::QueryFailed(format!(
                "LlmAnalyzeFile aborted: {reason}"
            ))),
            LoopOutcome::MaxIter { last_text, .. } => Err(DbError::QueryFailed(format!(
                "LlmAnalyzeFile hit max iterations; last text: {last_text}"
            ))),
            LoopOutcome::LlmRefused { reason, .. } => Err(DbError::QueryFailed(format!(
                "LlmAnalyzeFile refused: {reason}"
            ))),
        }
    }

    /// Send heartbeat for current task.
    pub async fn heartbeat(&self, task_id: &str) -> Result<(), DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());
        repo.heartbeat(task_id)
    }

    /// Self-contained repository-analysis flow (Phase 0.8 CLI entry point).
    ///
    /// 1. Inserts a `RepoAnalyze` task into the queue
    /// 2. Processes tasks in a loop until the queue is drained
    /// 3. Returns when no more `LlmAnalyzeFile` (or other) tasks are
    ///    claimable by this agent
    ///
    /// This is the orchestrator-free path; the CLI uses it to bypass
    /// the legacy `CleanroomAgent::run(RunMode::Produce)` flow which
    /// depends on the full Orchestrator infrastructure.
    pub async fn run_repo_analysis(
        &self,
        repo_path: &Path,
        project_name: &str,
    ) -> Result<usize, DbError> {
        let task_repo = TaskRepository::new(self.db.connection_arc());
        let task_id = uuid::Uuid::new_v4().to_string();
        let task = Task {
            task_id: task_id.clone(),
            task_type: TaskType::RepoAnalyze,
            status: TaskStatus::Pending,
            priority: 10,
            input_json: serde_json::json!({
                "document": project_name,
                "project_name": project_name,
                "repo_path": repo_path.to_string_lossy(),
            })
            .to_string(),
            output_json: None,
            error_message: None,
            assigned_to: None,
            progress: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 3,
            last_heartbeat: None,
            dependencies_json: "[]".to_string(),
            version: 1,
        };
        task_repo.create(&task)?;
        info!(
            project = %project_name,
            task_id = %task_id,
            "run_repo_analysis: scheduled RepoAnalyze task"
        );

        // Process tasks in a loop until nothing is claimable.
        let mut processed = 0usize;
        loop {
            match self.process_next_task().await? {
                None => break,
                Some(t) => {
                    info!(task_id = %t.task_id, task_type = ?t.task_type, "run_repo_analysis: processed task");
                    processed += 1;
                }
            }
        }
        info!(processed, "run_repo_analysis: queue drained");
        Ok(processed)
    }
}

/// Build the system prompt for an LlmAnalyzeFile task. Kept as a free function
/// (not a method) so it's easy to unit-test in isolation.
fn build_llm_analyze_file_system_prompt(document: &str, file_path: &str) -> String {
    format!(
        "You are a code analysis agent in the Cleanroom pipeline. Your job: take the source \
         code in the user message and emit a single JSON object that conforms to the S.DEF \
         `sdef_output` schema (data models, contracts, functions, design decisions).\n\
         \n\
         Project: {document}\n\
         File: {file_path}\n\
         \n\
         Rules:\n\
         - Emit only valid JSON. No prose outside the JSON fences.\n\
         - Use the field names defined in S.DEF schema v0.1 exactly.\n\
         - If a section has no relevant entities, omit it (don't emit empty arrays).\n\
         - For each entity, include a `description` field derived from the source.\n\
         - When unsure, prefer omitting the entity over guessing its structure.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_db::{Task, TaskStatus, TaskType};

    #[tokio::test]
    async fn test_process_next_task_no_tasks() {
        let db = Arc::new(Database::in_memory().unwrap());
        let agent = ProducerAgent::new(ProducerConfig::default(), db);
        let result = agent.process_next_task().await.unwrap();
        assert!(result.is_none(), "No tasks should be available");
    }

    #[test]
    fn test_producer_default_has_no_llm() {
        let db = Arc::new(Database::in_memory().unwrap());
        let agent = ProducerAgent::new(ProducerConfig::default(), db);
        assert!(!agent.has_llm(), "default producer must not have an LLM");
        assert_eq!(agent.agent_id().starts_with("producer-"), true);
    }

    #[test]
    fn test_system_prompt_includes_document_and_file() {
        let prompt = build_llm_analyze_file_system_prompt("my-proj", "src/foo.rs");
        assert!(prompt.contains("my-proj"), "document name missing");
        assert!(prompt.contains("src/foo.rs"), "file path missing");
        assert!(prompt.contains("S.DEF"), "should mention S.DEF");
    }

    /// Path to the workspace's `migrations/` directory from this crate's
    /// `CARGO_MANIFEST_DIR` (= `cleanroom-agent/crates/cleanroom-agent/`).
    /// Walk up two levels to reach the workspace root, then into `migrations/`.
    fn workspace_migrations_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent() // crates/
            .and_then(|p| p.parent()) // cleanroom-agent/
            .expect("cleanroom-agent crate layout has two parents")
            .join("migrations")
    }

    /// Insert a synthetic LlmAnalyzeFile task into the DB and assert that
    /// processing it without an LLM produces a clear "no LLM" error.
    /// (`process_next_task` returns `Ok(None)` for empty queues and
    /// `Err(DbError)` for the no-LLM case; we want the latter.)
    #[tokio::test]
    async fn test_llm_analyze_file_without_llm_fails() {
        // We need a file-based DB with all migrations applied so the
        // `task_type` CHECK constraint accepts `LLM_ANALYZE_FILE`.
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("llm-task-test.db");
        let db = Database::open_with_migrations_from(&db_path, Some(&workspace_migrations_dir()))
            .expect("open db");
        let db = Arc::new(db);

        let repo = TaskRepository::new(db.connection_arc());
        let task = Task {
            task_id: uuid::Uuid::new_v4().to_string(),
            task_type: TaskType::LlmAnalyzeFile,
            status: TaskStatus::Pending,
            priority: 8,
            input_json: serde_json::json!({
                "document": "test-doc",
                "repo_path": ".",
                "file_path": "src/lib.rs",
            })
            .to_string(),
            output_json: None,
            error_message: None,
            assigned_to: None,
            progress: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 2,
            last_heartbeat: None,
            dependencies_json: "[]".to_string(),
            version: 1,
        };
        repo.create(&task).expect("task create");

        let agent = ProducerAgent::new(ProducerConfig::default(), db);
        let result = agent.process_next_task().await;
        assert!(result.is_err(), "expected error when LLM not configured");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("no LLM") || err_msg.contains("LLM not configured"),
            "error message should mention missing LLM, got: {err_msg}",
        );
    }

    /// LlmAnalyzeFile with a `file_path` pointing at a non-existent file
    /// should fail with a clear "read failed" error (not silently succeed).
    /// We still don't have an LLM attached, so the test exercises the
    /// "LLM not configured" path -- documenting the shape of the error and
    /// the no-LLM-without-llm-fails behavior together.
    #[tokio::test]
    async fn test_llm_analyze_file_missing_input_path_fails() {
        // We can't easily attach a mock LLM in-process (MetaLlm requires 4
        // sub-traits; mocking them all is heavy), so we just exercise the
        // "no LLM" path here and rely on the end-to-end test in
        // `examples/` for the happy path. This is the documented Phase 0.5
        // scope; full mock LLM coverage is Phase 0.5 follow-up.
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("llm-missing-path-test.db");
        let db = Database::open_with_migrations_from(&db_path, Some(&workspace_migrations_dir()))
            .expect("open db");
        let db = Arc::new(db);

        let repo = TaskRepository::new(db.connection_arc());
        let task = Task {
            task_id: uuid::Uuid::new_v4().to_string(),
            task_type: TaskType::LlmAnalyzeFile,
            status: TaskStatus::Pending,
            priority: 8,
            input_json: serde_json::json!({
                "document": "test-doc",
                "repo_path": "/this/path/does/not/exist",
                "file_path": "missing.rs",
            })
            .to_string(),
            output_json: None,
            error_message: None,
            assigned_to: None,
            progress: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 2,
            last_heartbeat: None,
            dependencies_json: "[]".to_string(),
            version: 1,
        };
        repo.create(&task).expect("task create");

        let agent = ProducerAgent::new(ProducerConfig::default(), db);
        let result = agent.process_next_task().await;
        assert!(result.is_err());
    }

    // ========================================================================
    // Phase 0.5 wrap-up: ProducerMode dispatch + analyze_repo_llm tests
    // ========================================================================

    #[test]
    fn test_producer_mode_default_is_template() {
        assert_eq!(ProducerConfig::default().mode, ProducerMode::Template);
    }

    #[test]
    fn test_producer_config_llm_helper() {
        let cfg = ProducerConfig::llm();
        assert_eq!(cfg.mode, ProducerMode::Llm);
        // `llm()` doesn't strip the language list -- it only flips the mode.
        assert!(!cfg.languages.is_empty());
    }

    #[test]
    fn test_producer_config_both_helper() {
        let cfg = ProducerConfig::both();
        assert_eq!(cfg.mode, ProducerMode::Both);
    }

    /// LLM mode with a temp repo containing 2 source files: scheduling
    /// should create exactly 2 `LlmAnalyzeFile` tasks in the queue (no
    /// LLM call is made at scheduling time).
    #[tokio::test]
    async fn test_analyze_repo_llm_mode_creates_per_file_tasks() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo_dir = tmp.path().join("repo");
        std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir src");
        std::fs::write(repo_dir.join("src").join("a.rs"), "pub fn a() {}\n").expect("write a.rs");
        std::fs::write(repo_dir.join("src").join("b.py"), "def b():\n    pass\n").expect("write b.py");
        // A non-source file -- should NOT be scheduled (no detected language
        // by `repo_scanner::detect_language`). `.bin` is not in the
        // extension map so `language = None` and our `.filter()` drops it.
        std::fs::write(repo_dir.join("data.bin"), b"\x00\x01\x02").expect("write data.bin");

        // Spin up a fully-migrated DB.
        let db_path = tmp.path().join("llm-mode-test.db");
        let migrations_dir = workspace_migrations_dir();
        let db = Database::open_with_migrations_from(&db_path, Some(&migrations_dir)).expect("db");
        let db = Arc::new(db);

        let agent = ProducerAgent::new(ProducerConfig::llm(), db.clone());
        assert_eq!(agent.config.mode, ProducerMode::Llm);

        // Insert a RepoAnalyze task and run it.
        let task_repo = TaskRepository::new(db.connection_arc());
        let task_id = uuid::Uuid::new_v4().to_string();
        let task = Task {
            task_id: task_id.clone(),
            task_type: TaskType::RepoAnalyze,
            status: TaskStatus::Pending,
            priority: 10,
            input_json: serde_json::json!({
                "document": "llm-mode-test",
                "project_name": "llm-mode-test",
                "repo_path": repo_dir.to_string_lossy(),
            })
            .to_string(),
            output_json: None,
            error_message: None,
            assigned_to: None,
            progress: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 3,
            last_heartbeat: None,
            dependencies_json: "[]".to_string(),
            version: 1,
        };
        task_repo.create(&task).expect("create task");

        // Run via process_next_task (claims + processes the RepoAnalyze task).
        agent.process_next_task().await.expect("process_next_task ok");

        // High-level task should be Completed.
        let after = task_repo.get(&task_id).expect("get");
        assert_eq!(after.status, TaskStatus::Completed, "RepoAnalyze should complete");

        // Output should advertise the LLM mode + scheduled count.
        let output: serde_json::Value = serde_json::from_str(after.output_json.as_deref().unwrap()).expect("parse");
        assert_eq!(output["mode"], "llm");
        assert_eq!(output["scheduled_task_count"], 2, "2 source files => 2 LlmAnalyzeFile tasks");

        // Verify the 2 LlmAnalyzeFile tasks are in the queue, Pending, with
        // the correct input shape.
        let llm_tasks = task_repo
            .list(Some(TaskStatus::Pending), None, None)
            .expect("list pending")
            .into_iter()
            .filter(|t| t.task_type == TaskType::LlmAnalyzeFile)
            .collect::<Vec<_>>();
        assert_eq!(llm_tasks.len(), 2);
        let mut paths: Vec<String> = llm_tasks
            .iter()
            .map(|t| {
                serde_json::from_str::<serde_json::Value>(&t.input_json)
                    .ok()
                    .and_then(|v| v.get("file_path").and_then(|p| p.as_str()).map(String::from))
                    .unwrap_or_default()
            })
            .collect();
        paths.sort();
        assert_eq!(paths, vec!["src/a.rs", "src/b.py"]);
    }

    /// `Both` mode runs Template first, then schedules the LLM plan on
    /// top. The `RepoAnalyze` task is marked complete with the LLM
    /// phase's summary (since the LLM phase runs second and writes
    /// last).
    #[tokio::test]
    async fn test_analyze_repo_both_mode_runs_both_phases() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo_dir = tmp.path().join("repo");
        std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir src");
        std::fs::write(
            repo_dir.join("src").join("lib.rs"),
            "//! Sample lib\npub fn hello() -> &'static str { \"hi\" }\n",
        )
        .expect("write lib.rs");

        let db_path = tmp.path().join("both-mode-test.db");
        let migrations_dir = workspace_migrations_dir();
        let db = Database::open_with_migrations_from(&db_path, Some(&migrations_dir)).expect("db");
        let db = Arc::new(db);

        let agent = ProducerAgent::new(ProducerConfig::both(), db.clone());
        assert_eq!(agent.config.mode, ProducerMode::Both);

        let task_repo = TaskRepository::new(db.connection_arc());
        let task_id = uuid::Uuid::new_v4().to_string();
        let task = Task {
            task_id: task_id.clone(),
            task_type: TaskType::RepoAnalyze,
            status: TaskStatus::Pending,
            priority: 10,
            input_json: serde_json::json!({
                "document": "both-mode-test",
                "project_name": "both-mode-test",
                "repo_path": repo_dir.to_string_lossy(),
            })
            .to_string(),
            output_json: None,
            error_message: None,
            assigned_to: None,
            progress: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 3,
            last_heartbeat: None,
            dependencies_json: "[]".to_string(),
            version: 1,
        };
        task_repo.create(&task).expect("create task");

        agent.process_next_task().await.expect("process_next_task ok");

        let after = task_repo.get(&task_id).expect("get");
        assert_eq!(after.status, TaskStatus::Completed);
        let output: serde_json::Value = serde_json::from_str(after.output_json.as_deref().unwrap()).expect("parse");
        // LLM phase runs second and writes its summary last.
        assert_eq!(output["mode"], "llm");
        assert_eq!(output["scheduled_task_count"], 1, "1 source file => 1 LLM task");
    }
}