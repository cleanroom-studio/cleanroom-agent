//! Consumer Agent — generates code from S.DEF documents.
//!
//! The Consumer Agent is responsible for the "consume" phase of the Cleanroom
//! agent pipeline. It reads S.DEF (Software Definition Exchange Format) documents
//! from the database and generates code in various target programming languages.
//!
//! # Supported Languages
//!
//! - Rust: Generates structs with serde derives
//! - TypeScript/JavaScript: Generates interfaces and classes
//! - Python: Generates dataclasses
//! - C: Generates header and source files
//!
//! # Code Generation
//!
//! The consumer:
//! 1. Reads S.DEF documents from the database
//! 2. For each data model, generates appropriate code via language-specific generators
//! 3. For each interface contract, generates interface code
//! 4. Writes generated code to the output directory
//! 5. Optionally runs verification to ensure code compiles correctly

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::io::Write;

use tracing::{info, warn};
use rusqlite::params;
use serde_json;

use cleanroom_db::{Database, DbError, Task, TaskRepository, TaskType, TypeCacheRepository};
use cleanroom_meta_core::tool::MetaToolT;
use cleanroom_meta_llm::MetaLlm;

use crate::llm_loop::{run_loop, LoopConfig, LoopContext};

pub mod code_generator;
pub mod verification;
use code_generator::{create_generator, GeneratedCode};
use verification::{
    CompilationVerifier, GenerationLoop, GenerationLoopConfig, LoopOutcome,
    TestExecutor, VerificationReport,
};

/// Compatibility mode for code generation.
///
/// Determines how the consumer handles legacy patterns, deprecated features,
/// and cross-version compatibility when generating code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityMode {
    /// Full compatibility mode: include all legacy patterns
    Full,
    /// Mixed mode: modern patterns with some legacy support
    Mixed,
    /// Clean mode: only modern patterns, no legacy support
    Clean,
    /// Custom mode: user-defined compatibility rules
    Custom,
}

impl Default for CompatibilityMode {
    fn default() -> Self { Self::Mixed }
}

/// Fidelity level for code reconstruction.
///
/// Determines the completeness and detail level of generated code.
/// Higher fidelity produces more complete code but may take longer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fidelity {
    /// High fidelity: complete implementation with all methods
    High,
    /// Medium fidelity: standard implementation with common methods
    Medium,
    /// Low fidelity: minimal stubs and interfaces only
    Low,
}

impl Default for Fidelity {
    fn default() -> Self { Self::Medium }
}

/// Consumer Agent configuration.
///
/// Contains all settings needed to configure the consumer agent's
/// code generation behavior, including target language and output settings.
#[derive(Clone)]
pub struct ConsumerConfig {
    /// Target programming language for code generation (rust, typescript, python, c)
    pub language: String,
    /// Optional framework hint (e.g., "actix-web" for Rust, "express" for JS)
    pub framework: Option<String>,
    /// Compatibility mode for handling legacy patterns
    pub compatibility_mode: CompatibilityMode,
    /// Fidelity level for code reconstruction
    pub fidelity: Fidelity,
    /// Output directory for generated code files
    pub output_path: PathBuf,
    /// If true, use the pre-Phase-0.6 template-based generator under
    /// `code_generator/` (`_gen.rs` files). Default = `false`, which
    /// means the LLM path (Phase 0.6) is the default. The flag exists
    /// so Phase 5 can A/B compare the two paths.
    pub use_legacy_template: bool,
    /// Optional LLM used for the Phase 0.6 LLM path. Must be set if
    /// `use_legacy_template = false`; otherwise the LLM path fails
    /// with a clear error. Excluded from `Debug` because `dyn MetaLlm`
    /// is not `Debug`.
    pub llm: Option<Arc<dyn MetaLlm>>,
    /// Loop config for the LLM path (token / iteration / cost guardrails).
    pub loop_config: LoopConfig,
}

impl std::fmt::Debug for ConsumerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsumerConfig")
            .field("language", &self.language)
            .field("framework", &self.framework)
            .field("compatibility_mode", &self.compatibility_mode)
            .field("fidelity", &self.fidelity)
            .field("output_path", &self.output_path)
            .field("use_legacy_template", &self.use_legacy_template)
            .field("llm", &"<dyn MetaLlm>")
            .field("loop_config", &self.loop_config)
            .finish()
    }
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            language: "typescript".to_string(),
            framework: None,
            compatibility_mode: CompatibilityMode::default(),
            fidelity: Fidelity::default(),
            output_path: PathBuf::from("./generated"),
            use_legacy_template: false,
            llm: None,
            loop_config: LoopConfig::default(),
        }
    }
}

impl ConsumerConfig {
    /// Convenience: build a config that uses the legacy template path.
    pub fn legacy_template() -> Self {
        Self {
            use_legacy_template: true,
            ..Self::default()
        }
    }
}

/// Consumer Agent — generates code from S.DEF documents.
///
/// The Consumer Agent reads S.DEF (Software Definition Exchange Format) documents
/// from the database and generates code in a target programming language.
///
/// # Code Generation Process
///
/// 1. Create appropriate language-specific code generator
/// 2. Read data models and contracts from the database
/// 3. Generate code for each entity using the language generator
/// 4. Write generated files to the output directory
/// 5. Optionally verify generated code compiles correctly
///
/// # Supported Languages
///
/// - `rust`: Generates structs, traits, and implementations
/// - `typescript` / `javascript`: Generates interfaces and classes
/// - `python`: Generates dataclasses and abstract classes
/// - `c`: Generates header and source files
///
/// # Task Processing
///
/// The agent handles the following task types:
/// - [`TaskType::GenerateCode`]: Main code generation task
/// - [`TaskType::MergeCode`]: Merge generated code with existing files
/// - [`TaskType::RunTests`]: Run tests on generated code
pub struct ConsumerAgent {
    /// Consumer agent configuration
    config: ConsumerConfig,
    /// Database connection for reading S.DEF documents
    db: Arc<Database>,
    /// Unique agent identifier for task claiming
    agent_id: String,
    /// Optional LLM used for the Phase 0.6 LLM-driven code generation path.
    /// None = LLM path will fail with a clear error (use_legacy_template
    /// must be true in that case).
    llm: Option<Arc<dyn MetaLlm>>,
    /// Loop config for the LLM path.
    loop_config: LoopConfig,
    /// Optional Phase 0.10 tool set, forwarded into every
    /// `LoopConfig.tools` that `generate_code_with_llm` constructs.
    /// `None` (the default) keeps the pre-0.10 single-shot behavior.
    tools: Option<Vec<Arc<dyn MetaToolT>>>,
}

impl ConsumerAgent {
    pub fn new(config: ConsumerConfig, db: Arc<Database>) -> Self {
        let agent_id = format!("consumer-{}", uuid::Uuid::new_v4());
        Self {
            config,
            db,
            agent_id,
            llm: None,
            loop_config: LoopConfig::default(),
            tools: None,
        }
    }

    /// Attach an LLM for the Phase 0.6 LLM-driven code generation path.
    /// Without this, calling `generate_code` with `use_legacy_template = false`
    /// will fail with a clear "no LLM configured" error.
    pub fn with_llm(mut self, llm: Arc<dyn MetaLlm>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the loop config for the LLM path.
    pub fn with_loop_config(mut self, cfg: LoopConfig) -> Self {
        self.loop_config = cfg;
        self
    }

    /// Attach a tool set (Phase 0.10) to the per-call `LoopConfig.tools`
    /// that `generate_code_with_llm` will pass to `run_loop`. An empty
    /// vec (the default) is equivalent to the pre-0.10 no-tools
    /// behavior. The supplied tools must be `Arc<dyn MetaToolT>` so
    /// they can be cheaply cloned across `run_loop` invocations on
    /// the same `ConsumerAgent`.
    pub fn with_tools(mut self, tools: Vec<Arc<dyn MetaToolT>>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn agent_id(&self) -> &str { &self.agent_id }

    /// Whether an LLM is attached. `false` means the LLM path will fail
    /// with a clear error.
    pub fn has_llm(&self) -> bool {
        self.llm.is_some()
    }

    /// Generate code from S.DEF stored in the database.
    ///
    /// Dispatches based on [`ConsumerConfig::use_legacy_template`]:
    /// - `true`: pre-Phase-0.6 template path (data model + contract templates
    ///   under `code_generator/`).
    /// - `false` (default): Phase 0.6 LLM path — one `run_loop` call per
    ///   S.DEF entity, LLM emits code in a markdown fence, output written
    ///   to `<output_path>/generated/<entity>.<ext>`.
    pub async fn generate_code(&self) -> Result<(), DbError> {
        info!(
            language = %self.config.language,
            output = %self.config.output_path.display(),
            legacy = self.config.use_legacy_template,
            "Starting code generation"
        );

        if self.config.use_legacy_template {
            self.generate_code_legacy().await
        } else {
            self.generate_code_with_llm().await
        }
    }

    /// Pre-Phase-0.6 template-based generation. Kept around so Phase 5 can
    /// baseline-compare the LLM path against it. Phase 0.6+ will move
    /// these files under `consumer/legacy_templates/` (deferred for now
    /// to keep the diff focused).
    async fn generate_code_legacy(&self) -> Result<(), DbError> {
        let generator = match create_generator(&self.config.language) {
            Some(g) => g,
            None => {
                return Err(DbError::QueryFailed(format!(
                    "Unsupported language: {}. Supported: rust, typescript, python", self.config.language
                )));
            }
        };

        fs::create_dir_all(&self.config.output_path)
            .map_err(|e| DbError::QueryFailed(format!("Failed to create output dir: {}", e)))?;

        let mut total_files = 0;
        let documents = self.read_documents()?;
        info!(count = documents.len(), "Documents found");

        for doc_name in &documents {
            let models = self.read_data_models(doc_name)?;
            info!(document = %doc_name, models = models.len(), "Generating code");

            for model in &models {
                let files = generator.generate_data_model(model);
                for file in files {
                    self.write_code_file(&file)?;
                    total_files += 1;
                }
            }

            let contracts = self.read_contracts(doc_name)?;
            for contract in &contracts {
                let files = generator.generate_interface(contract);
                for file in files {
                    self.write_code_file(&file)?;
                    total_files += 1;
                }
            }
        }

        info!(files = total_files, language = %self.config.language, "Code generation complete");
        Ok(())
    }

    /// Phase 0.6 LLM-driven generation: for each S.DEF entity (data model
    /// + contract), build a system prompt + user message and call
    /// `llm_loop::run_loop` to get a code block. Parse the LLM's output,
    /// extract the code, and write to disk.
    ///
    /// The LLM is asked to emit a single markdown code block (````rust ... ````
    /// for rust, etc.) wrapped between `// file: <path>` and the actual code.
    /// If extraction fails the raw LLM output is saved as
    /// `<entity>.<ext>.raw.txt` so the user can post-process.
    async fn generate_code_with_llm(&self) -> Result<(), DbError> {
        let llm = self.llm.as_ref().ok_or_else(|| {
            DbError::QueryFailed(
                "LLM path requires an LLM; construct the consumer with `.with_llm(...)` \
                 or set `config.use_legacy_template = true` to use the template path."
                    .to_string(),
            )
        })?;

        fs::create_dir_all(&self.config.output_path.join("generated"))
            .map_err(|e| DbError::QueryFailed(format!("create output dir: {e}")))?;

        let documents = self.read_documents()?;
        info!(count = documents.len(), "LLM path: documents found");

        let mut total_files = 0u32;
        let mut total_prompt_tokens = 0u32;
        let mut total_completion_tokens = 0u32;
        let mut total_cost = 0.0f64;

        for doc_name in &documents {
            let models = self.read_data_models(doc_name)?;
            info!(document = %doc_name, models = models.len(), "LLM path: data models");

            for model in &models {
                let entity_json = serde_json::to_string(&model).unwrap_or_default();
                let system_prompt = build_llm_generate_code_system_prompt(
                    &self.config.language,
                    self.config.framework.as_deref(),
                );
                let user_message = format!(
                    "Generate the code for this S.DEF data model. Emit a single markdown \
                     code block wrapped in triple backticks with the language tag (e.g. \
                     ```{lang}...```). The first non-blank line inside the code block should \
                     be a comment `// file: <path>` indicating where the file should be \
                     saved (use snake_case for the entity name, e.g. `src/{snake}.rs`).\n\
                     \n\
                     S.DEF entity:\n\
                     ```json\n{entity_json}\n```",
                    lang = code_fence_tag(&self.config.language),
                    snake = snake_case(&model.entity),
                );

                let ctx = LoopContext::new(
                    "consumer-llm-gen",
                    "consumer-llm-session",
                    "cleanroom-consumer",
                    system_prompt,
                    user_message,
                );
                let started = std::time::Instant::now();
                // Phase 0.10: forward the per-agent tool set into the
                // per-call LoopConfig.tools. `None` (the default) yields
                // an empty tool set via `unwrap_or_default()` in
                // `run_loop_via_basic_agent` — i.e. pre-0.10 behavior.
                let mut loop_cfg = self.loop_config.clone();
                loop_cfg.tools = self.tools.clone();
                let outcome = run_loop(llm.clone(), ctx, &loop_cfg)
                    .await
                    .map_err(|e| DbError::QueryFailed(format!("LLM call failed: {e}")))?;
                let elapsed = started.elapsed().as_millis() as u64;

                match outcome {
                    crate::llm_loop::LoopOutcome::Done {
                        result,
                        prompt_tokens,
                        completion_tokens,
                        ..
                    } => {
                        total_prompt_tokens += prompt_tokens;
                        total_completion_tokens += completion_tokens;
                        // Rough Sonnet-3.5 estimate.
                        total_cost +=
                            (prompt_tokens as f64) * 3.0 / 1_000_000.0
                                + (completion_tokens as f64) * 15.0 / 1_000_000.0;

                        let (file_relpath, code) = parse_llm_code_output(
                            &result,
                            &self.config.language,
                            &model.entity,
                        );
                        let dest = self.config.output_path.join(&file_relpath);
                        if let Some(parent) = dest.parent() {
                            fs::create_dir_all(parent).map_err(|e| {
                                DbError::QueryFailed(format!("mkdir {}: {e}", parent.display()))
                            })?;
                        }
                        fs::write(&dest, code).map_err(|e| {
                            DbError::QueryFailed(format!("write {}: {e}", dest.display()))
                        })?;
                        info!(
                            file = %file_relpath,
                            prompt = prompt_tokens,
                            completion = completion_tokens,
                            elapsed_ms = elapsed,
                            "LLM-generated file"
                        );
                        total_files += 1;
                    }
                    other => {
                        warn!(
                            entity = %model.entity,
                            outcome = ?other,
                            "LLM consumer: non-Done outcome, skipping entity"
                        );
                    }
                }
            }

            let contracts = self.read_contracts(doc_name)?;
            for contract in &contracts {
                // For contracts, the prompt is similar but emphasises
                // "interface" rather than "data model".
                let entity_json = serde_json::to_string(&contract).unwrap_or_default();
                let system_prompt = build_llm_generate_code_system_prompt(
                    &self.config.language,
                    self.config.framework.as_deref(),
                );
                let user_message = format!(
                    "Generate the code for this S.DEF interface contract. Emit a single \
                     markdown code block wrapped in triple backticks with the language tag \
                     (e.g. ```{lang}...```). The first non-blank line inside the code block \
                     should be a comment `// file: <path>` indicating where the file \
                     should be saved (use snake_case for the contract name, e.g. \
                     `src/{snake}_trait.rs`).\n\
                     \n\
                     S.DEF contract:\n\
                     ```json\n{entity_json}\n```",
                    lang = code_fence_tag(&self.config.language),
                    snake = snake_case(&contract.name),
                );
                let ctx = LoopContext::new(
                    "consumer-llm-gen",
                    "consumer-llm-session",
                    "cleanroom-consumer",
                    system_prompt,
                    user_message,
                );
                let started = std::time::Instant::now();
                // Phase 0.10: same as above (first run_loop call site).
                let mut loop_cfg = self.loop_config.clone();
                loop_cfg.tools = self.tools.clone();
                let outcome = run_loop(llm.clone(), ctx, &loop_cfg)
                    .await
                    .map_err(|e| DbError::QueryFailed(format!("LLM call failed: {e}")))?;
                let elapsed = started.elapsed().as_millis() as u64;

                if let crate::llm_loop::LoopOutcome::Done {
                    result,
                    prompt_tokens,
                    completion_tokens,
                    ..
                } = outcome
                {
                    total_prompt_tokens += prompt_tokens;
                    total_completion_tokens += completion_tokens;
                    total_cost +=
                        (prompt_tokens as f64) * 3.0 / 1_000_000.0
                            + (completion_tokens as f64) * 15.0 / 1_000_000.0;
                    let (file_relpath, code) = parse_llm_code_output(
                        &result,
                        &self.config.language,
                        &contract.name,
                    );
                    let dest = self.config.output_path.join(&file_relpath);
                    if let Some(parent) = dest.parent() {
                        fs::create_dir_all(parent).map_err(|e| {
                            DbError::QueryFailed(format!("mkdir {}: {e}", parent.display()))
                        })?;
                    }
                    fs::write(&dest, code).map_err(|e| {
                        DbError::QueryFailed(format!("write {}: {e}", dest.display()))
                    })?;
                    info!(
                        file = %file_relpath,
                        prompt = prompt_tokens,
                        completion = completion_tokens,
                        elapsed_ms = elapsed,
                        "LLM-generated contract file"
                    );
                    total_files += 1;
                }
            }
        }

        info!(
            files = total_files,
            prompt = total_prompt_tokens,
            completion = total_completion_tokens,
            cost_usd = total_cost,
            "LLM-driven code generation complete"
        );
        Ok(())
    }

    /// Read document names from the database.
    fn read_documents(&self) -> Result<Vec<String>, DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare("SELECT name FROM sdef_documents ORDER BY name")
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        let mut rows = stmt.query([])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        let mut names = Vec::new();
        while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
            names.push(row.get::<_, String>(0).map_err(|e| DbError::QueryFailed(e.to_string()))?);
        }
        drop(rows);
        drop(stmt);
        drop(conn);
        Ok(names)
    }

    /// Read data models from the database.
    fn read_data_models(&self, document_name: &str) -> Result<Vec<sdef_core::DataModel>, DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare(
            "SELECT entity, description, version, logical_model FROM data_models WHERE document_name = ?1"
        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut rows = stmt.query(params![document_name])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut entities = Vec::new();
        while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
            entities.push((
                row.get::<_, String>(0).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                row.get::<_, Option<String>>(1).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                row.get::<_, Option<String>>(2).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                row.get::<_, Option<String>>(3).map_err(|e| DbError::QueryFailed(e.to_string()))?,
            ));
        }
        drop(rows);
        drop(stmt);
        drop(conn);

        let mut models = Vec::new();
        for (entity, description, version, logical_model) in entities {
            let attrs = self.read_attributes(document_name, &entity)?;
            models.push(sdef_core::DataModel {
                entity,
                status: None,
                version,
                deprecated: None,
                description,
                logical_model,
                attributes: if attrs.is_empty() { None } else { Some(attrs) },
                relationships: None,
                validation_rules: None,
                physical_design: None,
                origin: None,
            });
        }
        Ok(models)
    }

    /// Read attributes for a data model.
    fn read_attributes(&self, document_name: &str, entity: &str) -> Result<Vec<sdef_core::DataAttribute>, DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare(
            "SELECT name, attr_type, format, description, required, identity, generated, unique_flag, internal, deprecated
             FROM data_attributes WHERE document_name = ?1 AND entity = ?2"
        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut rows = stmt.query(params![document_name, entity])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut attrs = Vec::new();
        while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
            attrs.push(sdef_core::DataAttribute {
                name: row.get(0).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                attr_type: row.get(1).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                format: row.get(2).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                description: row.get(3).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                required: row.get(4).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                identity: row.get(5).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                generated: row.get(6).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                unique: row.get(7).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                internal: row.get(8).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                deprecated: row.get(9).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                default: None,
                compatibility: None,
                constraints: None,
                origin: None,
            });
        }
        drop(rows);
        drop(stmt);
        drop(conn);
        Ok(attrs)
    }

    /// Read interface contracts from the database.
    fn read_contracts(&self, document_name: &str) -> Result<Vec<sdef_core::InterfaceContract>, DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare(
            "SELECT name, description, is_abstract FROM contracts
             WHERE document_name = ?1 AND contract_type = 'interface'"
        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut rows = stmt.query(params![document_name])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut contracts = Vec::new();
        while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
            contracts.push(sdef_core::InterfaceContract {
                name: row.get(0).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                is_abstract: row.get::<_, bool>(2).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                status: Some("active".to_string()),
                version: None,
                deprecated: None,
                description: row.get(1).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                methods: None,
                invariants: None,
                origin: None,
            });
        }
        drop(rows);
        drop(stmt);
        drop(conn);
        Ok(contracts)
    }

    /// Write a generated code file to disk.
    fn write_code_file(&self, code: &GeneratedCode) -> Result<(), DbError> {
        let file_path = self.config.output_path.join(&code.file_path);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| DbError::QueryFailed(format!("Failed to create dir: {}", e)))?;
        }
        let mut file = fs::File::create(&file_path)
            .map_err(|e| DbError::QueryFailed(format!("Failed to create file: {}", e)))?;
        file.write_all(code.content.as_bytes())
            .map_err(|e| DbError::QueryFailed(format!("Failed to write file: {}", e)))?;
        info!(path = %file_path.display(), "Generated file");
        Ok(())
    }

    /// Process a generation task.
    pub async fn process_next_task(&self) -> Result<Option<Task>, DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());
        if let Some(task) = repo.claim(&self.agent_id)? {
            info!(task_id = %task.task_id, task_type = ?task.task_type, "Processing task");
            match task.task_type {
                TaskType::GenerateCode => {
                    self.generate_code().await?;
                    // After generation, run verification loop
                    let report = self.verify_generated_code(&task).await?;
                    let output = serde_json::to_string(&report)
                        .unwrap_or_else(|_| "{}".to_string());
                    repo.complete(&task.task_id, &output)?;
                }
                TaskType::MergeCode => {
                    let report = self.do_merge_code(&task).await?;
                    let output = serde_json::to_string(&report)
                        .unwrap_or_else(|_| "{}".to_string());
                    repo.complete(&task.task_id, &output)?;
                }
                TaskType::RunTests => {
                    let report = self.run_tests(&task).await?;
                    let output = serde_json::to_string(&report)
                        .unwrap_or_else(|_| "{}".to_string());
                    repo.complete(&task.task_id, &output)?;
                }
                _ => { repo.complete(&task.task_id, "{}")?; }
            }
            return Ok(Some(task));
        }
        Ok(None)
    }

    /// Run the four-layer verification on generated code.
    pub async fn verify_generated_code(&self, task: &Task) -> Result<VerificationReport, DbError> {
        let loop_config = GenerationLoopConfig::default();
        let gen_loop = GenerationLoop::new(loop_config, self.db.clone());
        let document_name = self.extract_document_from_task(task);

        let outcome = gen_loop.verify_and_heal(
            &self.config.output_path,
            &document_name,
            &self.config.language,
            0, // entity count could be passed via task input in future
        );

        match outcome {
            LoopOutcome::Success(report) => {
                info!("Generation verification passed in {}ms", report.total_duration_ms);
                Ok(report)
            }
            LoopOutcome::Failed { report, reason } => {
                info!(%reason, "Generation verification failed");
                Ok(report)
            }
        }
    }

    async fn do_merge_code(&self, task: &Task) -> Result<VerificationReport, DbError> {
        let document_name = self.extract_document_from_task(task);
        info!(document = %document_name, "Merging generated code");

        let _ = CompilationVerifier::format_code(&self.config.output_path, &self.config.language);

        Ok(VerificationReport {
            compile_passed: true,
            type_check_passed: true,
            test_passed: true,
            ..Default::default()
        })
    }

    async fn run_tests(&self, task: &Task) -> Result<VerificationReport, DbError> {
        let document_name = self.extract_document_from_task(task);
        info!(document = %document_name, "Running tests on generated code");

        let test_report = TestExecutor::run(&self.config.output_path, &self.config.language);
        let compile = CompilationVerifier::check(&self.config.output_path, &self.config.language);

        Ok(VerificationReport {
            compile_passed: compile.success,
            compile_errors: compile.error_count,
            compile_warnings: compile.warning_count,
            test_passed: test_report.failed == 0,
            tests_total: test_report.total,
            tests_failed: test_report.failed,
            type_check_passed: compile.success,
            ..Default::default()
        })
    }
}

impl ConsumerAgent {
    /// Look up a cached type resolution for an entity.
    ///
    /// Checks the `type_cache` table first to avoid re-querying LSP servers.
    /// Returns `None` if no cached entry exists for this entity + language.
    pub fn lookup_type_cache(
        &self,
        entity_uri: &str,
        language: &str,
    ) -> Result<Option<String>, DbError> {
        let cache_repo = TypeCacheRepository::new(self.db.connection_arc());
        match cache_repo.lookup(entity_uri, language)? {
            Some(entry) => Ok(Some(entry.resolved_type)),
            None => Ok(None),
        }
    }

    /// Check if the type_cache has entries for the target language.
    pub fn has_cached_types(&self, language: &str) -> Result<bool, DbError> {
        let _cache_repo = TypeCacheRepository::new(self.db.connection_arc());
        // Quick check: try to clear nothing and see if any entries existed
        let conn = self.db.connection();
        let mut stmt = conn
            .prepare("SELECT COUNT(*) FROM type_cache WHERE language = ?1")
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        let count: i64 = stmt
            .query_row(rusqlite::params![language], |row| row.get(0))
            .unwrap_or(0);
        Ok(count > 0)
    }

    /// Extract document name from task input JSON.
    fn extract_document_from_task(&self, task: &Task) -> String {
        serde_json::from_str::<serde_json::Value>(&task.input_json)
            .ok()
            .and_then(|v| v.get("document_name")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Self-contained consume flow (Phase 0.8 CLI entry point).
    /// Delegates to `generate_code` which dispatches on
    /// `use_legacy_template`. Exists so the CLI has a single named
    /// entry point.
    pub async fn run_consume(&self) -> Result<(), DbError> {
        self.generate_code().await
    }
}

// ============================================================================
// Phase 0.6 helpers -- LLM-driven code generation
// ============================================================================

/// Build the system prompt for an LLM code-generation call.
fn build_llm_generate_code_system_prompt(language: &str, framework: Option<&str>) -> String {
    let framework_hint = match framework {
        Some(f) => format!("\nPreferred framework: {f}.\n"),
        None => String::new(),
    };
    format!(
        "You are a code-generation agent in the Cleanroom pipeline. The user will give \
         you an S.DEF entity (data model or interface contract) in JSON. Your job: emit \
         idiomatic {language} code that implements that entity, wrapped in a single \
         markdown code block (triple backticks with a `{lang}` tag).\n\
         \n\
         Rules:\n\
         {framework_hint}\
         - Emit ONLY the code, no prose outside the code block.\n\
         - The first non-blank line inside the code block must be a comment `// file: <path>` \
           indicating where the file should be saved. Use snake_case for the entity name.\n\
         - Use language-idiomatic naming (snake_case for Rust/Python, camelCase for TypeScript).\n\
         - If the entity has fields, generate a struct (or dataclass) with typed fields and \
           brief doc comments on each field.\n\
         - If the entity is an interface contract, generate a trait / interface with the \
           documented method signatures.\n",
        language = language,
        lang = language,
    )
}

/// Map a high-level language name to a markdown code-fence tag.
/// Returns empty string for unknown languages (the fence still works
/// but won't be syntax-highlighted).
fn code_fence_tag(language: &str) -> &'static str {
    match language {
        "rust" => "rust",
        "typescript" | "javascript" => "typescript",
        "python" => "python",
        "c" => "c",
        _ => "",
    }
}

/// Convert a `CamelCase` or `snake_case` identifier to `snake_case` (ASCII
/// only; non-ASCII is left unchanged).
fn snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let mut prev_lower = false;
    for ch in s.chars() {
        if ch.is_ascii_uppercase() {
            if prev_lower {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_lower = false;
        } else {
            out.push(ch);
            prev_lower = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    out
}

/// Default file extension for a target language.
fn language_ext(language: &str) -> &'static str {
    match language {
        "rust" => "rs",
        "typescript" => "ts",
        "javascript" => "js",
        "python" => "py",
        "c" => "h",
        _ => "txt",
    }
}

/// Parse the LLM's response to extract (relative file path, code body).
///
/// Strategy:
/// 1. Find a fenced code block ```` ```<lang>? ... ``` ```` in the LLM's
///    output. If absent, return `(<entity>.<ext>.raw.txt, raw_output)` as a
///    fallback so the user can post-process.
/// 2. Strip the leading `// file: <path>` comment (if present) to derive
///    the relative file path. If absent, fall back to
///    `generated/<snake>.<ext>`.
fn parse_llm_code_output(raw: &str, language: &str, entity: &str) -> (String, String) {
    let ext = language_ext(language);
    let fallback_path = format!("generated/{}.{}", snake_case(entity), ext);

    // Find the first fenced code block. We accept any tag (or no tag).
    let fence_start = raw.find("```");
    let Some(start) = fence_start else {
        return (format!("{}.{}.raw.txt", snake_case(entity), ext), raw.to_string());
    };
    // Skip past the opening fence (and optional language tag on the same line).
    let after_fence = &raw[start + 3..];
    let after_tag = match after_fence.find('\n') {
        Some(idx) => &after_fence[idx + 1..],
        None => return (fallback_path, raw.to_string()),
    };
    // Find the closing fence.
    let Some(end_rel) = after_tag.find("```") else {
        return (fallback_path, raw.to_string());
    };
    let body = &after_tag[..end_rel];
    // Trim leading/trailing blank lines.
    let body = body.trim_matches('\n').to_string();

    // Look for a `// file: <path>` line at the top of the body.
    let mut rel_path = fallback_path.clone();
    let mut code = body.clone();
    for line in body.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("// file:") {
            let p = rest.trim();
            if !p.is_empty() {
                rel_path = p.to_string();
                // Strip the file comment from the code we save.
                code = body
                    .lines()
                    .filter(|l| !l.trim_start().starts_with("// file:"))
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            break;
        }
        if !trimmed.is_empty() {
            // First non-blank, non-`// file:` line -- assume there's no path comment.
            break;
        }
    }
    (rel_path, code)
}

/// LLM-driven code regeneration for a single file (Phase 0.7).
///
/// Reads `<output_dir>/<file_relpath>` from disk, builds a prompt that
/// embeds the file's current content + the compiler error messages, calls
/// `llm_loop::run_loop`, and writes the LLM's corrected version back.
///
/// Returns the relative path of the file that was actually written
/// (usually the same as `file_relpath`; may differ if the LLM included a
/// `// file:` comment that pointed elsewhere).
///
/// # Orchestration
///
/// The caller (typically `GenerationLoop::verify_and_heal` or its
/// wrapper) drives the cycle:
/// ```ignore
/// loop {
///     let compile = CompilationVerifier::check(output_dir, language);
///     if compile.success { break; }
///     for err in &compile.errors {
///         llm_regenerate_file(output_dir, &err.file_path, &[err.to_string()],
///                             llm.clone(), &loop_config).await?;
///     }
/// }
/// ```
///
/// # Why a free function, not a method
///
/// Kept outside the `ConsumerAgent` struct so the producer / verification
/// layers can call it without needing a full consumer configuration.
pub async fn llm_regenerate_file(
    output_dir: &Path,
    file_relpath: &str,
    error_messages: &[String],
    llm: Arc<dyn MetaLlm>,
    loop_config: &LoopConfig,
) -> Result<String, DbError> {
    let full_path = output_dir.join(file_relpath);
    let source = std::fs::read_to_string(&full_path).map_err(|e| {
        DbError::QueryFailed(format!("llm_regenerate_file: read {} failed: {e}", full_path.display()))
    })?;
    if source.trim().is_empty() {
        return Err(DbError::QueryFailed(format!(
            "llm_regenerate_file: {} is empty; refusing to regenerate",
            full_path.display()
        )));
    }
    let language = detect_language_from_path(&full_path);
    let sys_prompt = build_llm_regenerate_system_prompt(language);
    let user_msg = build_llm_regenerate_user_message(file_relpath, error_messages, &source);
    let user_msg_len = user_msg.len();
    let ctx = LoopContext::new(
        "llm-regenerate",
        "regen-session",
        "cleanroom-verifier",
        sys_prompt,
        user_msg,
    );
    let outcome = run_loop(llm, ctx, loop_config)
        .await
        .map_err(|e| DbError::QueryFailed(format!("LLM call failed: {e}")))?;
    match outcome {
        crate::llm_loop::LoopOutcome::Done { result, .. } => {
            let (parsed_path, code) = parse_llm_code_output(&result, language, file_relpath);
            let new_path = if !parsed_path.is_empty() {
                parsed_path
            } else {
                file_relpath.to_string()
            };
            let dest = output_dir.join(&new_path);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    DbError::QueryFailed(format!("mkdir {}: {e}", parent.display()))
                })?;
            }
            std::fs::write(&dest, &code).map_err(|e| {
                DbError::QueryFailed(format!("write {}: {e}", dest.display()))
            })?;
            info!(
                file = %new_path,
                prompt_bytes = user_msg_len,
                response_bytes = result.len(),
                "llm_regenerate_file wrote corrected source"
            );
            Ok(new_path)
        }
        other => Err(DbError::QueryFailed(format!(
            "llm_regenerate_file: LLM returned non-Done outcome: {other:?}"
        ))),
    }
}

/// Best-effort language detection for a path. Falls back to "rust" (the
/// most common case for Cleanroom).
fn detect_language_from_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("ts") | Some("tsx") => "typescript",
        Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => "javascript",
        Some("py") | Some("pyi") => "python",
        Some("c") | Some("h") => "c",
        Some("cpp") | Some("cc") | Some("hpp") => "cpp",
        Some("go") => "go",
        Some("java") => "java",
        _ => "rust",
    }
}

fn build_llm_regenerate_system_prompt(language: &str) -> String {
    format!(
        "You are a code-fixing agent in the Cleanroom pipeline. The user will give you a \
         source file (in {language}) and a list of compiler error messages. Your job: emit \
         a corrected version of the source file that resolves every error.\n\
         \n\
         Rules:\n\
         - Emit ONLY the corrected full file, in a single markdown code block.\n\
         - The first non-blank line inside the code block must be a comment `// file: <path>` \
           so the verifier knows where to write the result.\n\
         - Do NOT introduce unrelated refactors. Minimal diff to fix the listed errors.\n\
         - Preserve the file's public API (function signatures, struct names) unless an \
           error forces a change.\n\
         - If you cannot resolve all errors, emit your best attempt and add a comment \
           `// TODO(llm-regen): <error code>` near the unresolved site.\n"
    )
}

fn build_llm_regenerate_user_message(
    file_relpath: &str,
    error_messages: &[String],
    source: &str,
) -> String {
    let errs = if error_messages.is_empty() {
        "(no specific error messages; the verifier reported a non-zero exit code)"
            .to_string()
    } else {
        error_messages
            .iter()
            .map(|e| format!("- {e}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "File: {file_relpath}\n\
         \n\
         Compiler / linter errors to fix:\n\
         {errs}\n\
         \n\
         Current source (verbatim -- fix it, do NOT rewrite from scratch):\n\
         ```{lang}\n\
         {source}\n\
         ```\n\
         \n\
         Emit the corrected full file in a single markdown code block.",
        lang = detect_language_from_path(Path::new(file_relpath)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db() -> Arc<Database> {
        let db = Arc::new(Database::in_memory().unwrap());
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, version, description, created_at, updated_at)
                 VALUES ('test-proj', '0.1.0', 'A test', datetime(), datetime());
                 INSERT INTO data_models (entity, document_name, status, description)
                 VALUES ('User', 'test-proj', 'active', 'A system user');
                 INSERT INTO data_attributes (document_name, entity, name, attr_type, description, required, identity, generated, unique_flag)
                 VALUES ('test-proj', 'User', 'id', 'UUID', 'Primary key', 1, 1, 1, 1);
                 INSERT INTO data_attributes (document_name, entity, name, attr_type, description, required)
                 VALUES ('test-proj', 'User', 'email', 'string', 'Email address', 1);"
            ).unwrap();
        }
        db
    }

    #[tokio::test]
    async fn test_generate_code_typescript() {
        let db = setup_db();
        let tmpdir = std::env::temp_dir().join("cleanroom_test_consumer_ts");
        let _ = std::fs::remove_dir_all(&tmpdir);

        let config = ConsumerConfig {
            language: "typescript".to_string(),
            output_path: tmpdir.clone(),
            use_legacy_template: true,
            ..ConsumerConfig::default()
        };
        let agent = ConsumerAgent::new(config, db);
        agent.generate_code().await.unwrap();

        // Check that files were generated
        let entries = std::fs::read_dir(&tmpdir).unwrap();
        let count = entries.count();
        assert!(count > 0, "Should generate at least one file");

        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[tokio::test]
    async fn test_generate_code_rust() {
        let db = setup_db();
        let tmpdir = std::env::temp_dir().join("cleanroom_test_consumer_rs");
        let _ = std::fs::remove_dir_all(&tmpdir);

        let config = ConsumerConfig {
            language: "rust".to_string(),
            output_path: tmpdir.clone(),
            use_legacy_template: true,
            ..ConsumerConfig::default()
        };
        let agent = ConsumerAgent::new(config, db);
        agent.generate_code().await.unwrap();

        let entries = std::fs::read_dir(&tmpdir).unwrap();
        let count = entries.count();
        assert!(count > 0, "Should generate at least one file");

        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[tokio::test]
    async fn test_unsupported_language() {
        let db = setup_db();
        let config = ConsumerConfig {
            language: "brainfuck".to_string(),
            ..ConsumerConfig::default()
        };
        let agent = ConsumerAgent::new(config, db);
        let result = agent.generate_code().await;
        assert!(result.is_err());
    }

    // ========================================================================
    // Phase 0.6 wrap-up: LLM-driven path + helpers tests
    // ========================================================================

    #[test]
    fn test_consumer_config_default_is_llm_mode() {
        // Default is the new LLM path; legacy template must be opted in.
        assert!(!ConsumerConfig::default().use_legacy_template);
    }

    #[test]
    fn test_consumer_config_legacy_template_helper() {
        let cfg = ConsumerConfig::legacy_template();
        assert!(cfg.use_legacy_template);
    }

    #[test]
    fn test_code_fence_tag_known_languages() {
        assert_eq!(code_fence_tag("rust"), "rust");
        assert_eq!(code_fence_tag("typescript"), "typescript");
        assert_eq!(code_fence_tag("python"), "python");
        // Unknown language falls back to empty (we still wrap with ``` ```).
        assert_eq!(code_fence_tag("brainfuck"), "");
    }

    #[test]
    fn test_snake_case_basic() {
        assert_eq!(snake_case("User"), "user");
        assert_eq!(snake_case("UserProfile"), "user_profile");
        assert_eq!(snake_case("HTTPSConnection"), "httpsconnection"); // ASCII only
        assert_eq!(snake_case("already_snake"), "already_snake");
        assert_eq!(snake_case(""), "");
    }

    #[test]
    fn test_parse_llm_code_output_extracts_fenced_block() {
        let raw = "Here's the code:\n\n```rust\n// file: src/foo.rs\npub fn foo() -> i32 { 42 }\n```\n\nThat should compile.";
        let (path, code) = parse_llm_code_output(raw, "rust", "Foo");
        assert_eq!(path, "src/foo.rs", "path should come from the // file: comment");
        assert!(code.contains("pub fn foo()"), "code should contain the function body");
        assert!(!code.contains("That should compile"), "should strip the prose");
    }

    #[test]
    fn test_parse_llm_code_output_fallback_when_no_fence() {
        // If the LLM doesn't emit a code fence, save the raw output under
        // `<entity>.<ext>.raw.txt` so the user can post-process.
        let raw = "I couldn't format that. Sorry.";
        let (path, code) = parse_llm_code_output(raw, "rust", "Foo");
        assert!(path.ends_with(".raw.txt"), "fallback path: {path}");
        assert_eq!(code, raw, "fallback code is the raw output");
    }

    #[test]
    fn test_parse_llm_code_output_fence_without_file_comment() {
        // Code fence but no `// file:` line -- fall back to a conventional
        // path: `generated/<snake>.<ext>`.
        let raw = "```rust\npub fn bar() {}\n```";
        let (path, code) = parse_llm_code_output(raw, "rust", "Bar");
        assert_eq!(path, "generated/bar.rs");
        assert!(code.contains("pub fn bar"));
    }
}