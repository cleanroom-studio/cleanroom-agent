//! CleanroomAgent — top-level agent entry point for cleanroom-agent.
//!
//! Coordinates the Produce/Consume pipelines and prompt engineering infrastructure.
//! LLM interaction is delegated to [`crate::llm_loop`], which wraps `autoagents`'s
//! `ChatProvider` trait (Phase 0 选定的 LLM framework)。
//!
//! Uses cleanroom-prompt for structured prompt engineering:
//! role definition, context budgeting, tool orchestration, etc.
//!
//! # Run Modes
//!
//! The agent supports three primary execution modes:
//!
//! - **Produce Mode**: Analyzes a code repository and outputs an S.DEF document
//! - **Consume Mode**: Reads an S.DEF document and generates code in a target language
//! - **Resume Mode**: Restores a paused workflow from a checkpoint
//!
//! # Example
//!
//! ```no_run
//! use cleanroom_agent::{CleanroomAgent, AgentConfig, RunMode};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = AgentConfig::producer(PathBuf::from("state.db"));
//! let agent = CleanroomAgent::new(config)?;
//!
//! agent.run(RunMode::Produce {
//!     repo_path: PathBuf::from("./my-repo"),
//!     output_path: PathBuf::from("./output"),
//!     project_name: "my-project".to_string(),
//! }).await?;
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use cleanroom_db::Database;
use cleanroom_prompt::{
    AgentType, ContextBudget, ContextItem, FewShotManager, FidelityLevel, GeneratedPrompt,
    PromptBuilder, SystemPromptConfig, default_tool_descriptions,
    load_from_database,
};
use tracing::{info, instrument};

use crate::consumer::{ConsumerAgent, ConsumerConfig, CompatibilityMode, Fidelity};
use crate::llm_loop::LoopConfig;
use crate::orchestrator::{Orchestrator, OrchestratorConfig};
use crate::producer::{ProducerAgent, ProducerConfig};

/// Run mode for the CleanroomAgent.
///
/// Determines the execution mode and required parameters for the agent:
/// - [`Produce`](RunMode::Produce): Analyze a code repository and generate S.DEF
/// - [`Consume`](RunMode::Consume): Generate code from an S.DEF document
/// - [`Resume`](RunMode::Resume): Resume a paused workflow from checkpoint
#[derive(Debug, Clone)]
pub enum RunMode {
    /// Analyze a code repository and output S.DEF.
    ///
    /// # Fields
    /// - `repo_path`: Path to the source code repository
    /// - `output_path`: Directory for S.DEF output
    /// - `project_name`: Name for the generated S.DEF document
    Produce {
        /// Path to the source code repository to analyze
        repo_path: PathBuf,
        /// Directory where S.DEF documents will be written
        output_path: PathBuf,
        /// Name for the generated S.DEF document
        project_name: String,
    },
    /// Read S.DEF and generate code in a target language.
    ///
    /// # Fields
    /// - `sdef_path`: Path to the S.DEF input file
    /// - `output_path`: Directory for generated code output
    /// - `language`: Target programming language (rust, typescript, python, etc.)
    /// - `framework`: Optional target framework hint
    /// - `compat_mode`: Compatibility mode for code generation
    /// - `fidelity`: Fidelity level for reconstruction
    Consume {
        /// Path to the S.DEF input document
        sdef_path: PathBuf,
        /// Directory where generated code will be written
        output_path: PathBuf,
        /// Target programming language for code generation
        language: String,
        /// Optional framework hint (e.g., "actix-web" for Rust)
        framework: Option<String>,
        /// Compatibility mode handling legacy patterns
        compat_mode: CompatibilityMode,
        /// Fidelity level for code reconstruction
        fidelity: Fidelity,
    },
    /// Resume a paused workflow from a checkpoint.
    ///
    /// # Fields
    /// - `document`: Document name or identifier to resume tasks for
    /// - `retry_failed`: Whether to also retry tasks that previously failed
    Resume {
        /// Document name or identifier to resume tasks for
        document: String,
        /// Whether to retry tasks that previously failed
        retry_failed: bool,
    },
}

/// Configuration for the top-level CleanroomAgent.
///
/// Contains all settings needed to initialize and run the Cleanroom agent,
/// including database connection, LLM provider, and prompt engineering settings.
///
/// # Example
///
/// ```no_run
/// use cleanroom_agent::AgentConfig;
/// use std::path::PathBuf;
///
/// let config = AgentConfig::consumer(
///     PathBuf::from("state.db"),
///     "rust".to_string()
/// );
/// ```
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Path to the SQLite database for state persistence
    pub db_path: PathBuf,
    /// LLM model name (e.g. "gemini-2.5-flash"). Uses environment if None.
    pub model_name: Option<String>,
    /// Agent name for identification and logging
    pub agent_name: String,
    /// Prompt building configuration for LLM interaction
    pub prompt_config: SystemPromptConfig,
    /// Whether to auto-load few-shot examples from completed tasks in DB
    pub load_few_shot: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        let prompt_config = SystemPromptConfig {
            agent_type: AgentType::Producer,
            include_tools: true,
            tool_descriptions: default_tool_descriptions(),
            ..Default::default()
        };
        Self {
            db_path: PathBuf::from("state.db"),
            model_name: Some("gemini-2.5-flash".to_string()),
            agent_name: "cleanroom-agent".to_string(),
            prompt_config,
            load_few_shot: false,
        }
    }
}

impl AgentConfig {
    /// Create a configuration preset for Producer mode.
    ///
    /// Producer mode analyzes code repositories and generates S.DEF documents.
    /// This preset configures the agent with appropriate prompt settings
    /// for the code analysis role.
    ///
    /// # Arguments
    /// * `db_path` - Path to the SQLite database for state persistence
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cleanroom_agent::AgentConfig;
    /// use std::path::PathBuf;
    ///
    /// let config = AgentConfig::producer(PathBuf::from("state.db"));
    /// ```
    pub fn producer(db_path: PathBuf) -> Self {
        let prompt_config = SystemPromptConfig {
            agent_type: AgentType::Producer,
            include_tools: true,
            tool_descriptions: default_tool_descriptions(),
            ..Default::default()
        };
        Self {
            db_path,
            model_name: None,
            agent_name: "cleanroom-agent".to_string(),
            prompt_config,
            load_few_shot: false,
        }
    }

    /// Create a configuration preset for Consumer mode.
    ///
    /// Consumer mode reads S.DEF documents and generates code in a target language.
    /// This preset configures the agent with appropriate prompt settings
    /// for the code generation role.
    ///
    /// # Arguments
    /// * `db_path` - Path to the SQLite database for state persistence
    /// * `language` - Target programming language for code generation
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cleanroom_agent::AgentConfig;
    /// use std::path::PathBuf;
    ///
    /// let config = AgentConfig::consumer(
    ///     PathBuf::from("state.db"),
    ///     "rust".to_string()
    /// );
    /// ```
    pub fn consumer(db_path: PathBuf, language: String) -> Self {
        let prompt_config = SystemPromptConfig {
            agent_type: AgentType::Consumer,
            target_language: Some(language),
            fidelity: FidelityLevel::ProductionEquivalent,
            compatibility_mode: Some("full".to_string()),
            include_tools: true,
            tool_descriptions: default_tool_descriptions(),
            ..Default::default()
        };
        Self {
            db_path,
            model_name: None,
            agent_name: "cleanroom-agent".to_string(),
            prompt_config,
            load_few_shot: false,
        }
    }
}

/// The top-level Cleanroom Agent.
///
/// Coordinates the Produce/Consume pipelines and prompt engineering infrastructure.
/// LLM interaction is delegated to [`crate::llm_loop`], which wraps `autoagents`'s
/// `ChatProvider` trait (Phase 0 选定的 framework)。
///
/// This is the main entry point for the Cleanroom agent system. It supports
/// three run modes: Produce (code → S.DEF), Consume (S.DEF → code), and
/// Resume (restore workflow from checkpoint).
///
/// # Architecture
///
/// The agent coordinates with:
/// - [`Orchestrator`]: Task orchestration and workflow management
/// - [`ProducerAgent`]: Code repository analysis and S.DEF generation
/// - [`ConsumerAgent`]: S.DEF consumption and code generation
/// - [`Database`]: SQLite persistence for state, tasks, and S.DEF documents
///
/// # Example
///
/// ```no_run
/// use cleanroom_agent::{CleanroomAgent, AgentConfig, RunMode};
/// use std::path::PathBuf;
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = AgentConfig::producer(PathBuf::from("state.db"));
/// let agent = CleanroomAgent::new(config)?;
///
/// agent.run(RunMode::Produce {
///     repo_path: PathBuf::from("./my-repo"),
///     output_path: PathBuf::from("./output"),
///     project_name: "my-project".to_string(),
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub struct CleanroomAgent {
    /// Database connection for state persistence and S.DEF storage
    pub db: Arc<Database>,
    /// Prompt builder for structured LLM interaction
    pub prompt_builder: PromptBuilder,
    /// Few-shot example manager loaded from database on startup
    pub few_shot: FewShotManager,
    /// Agent configuration
    config: AgentConfig,
}

impl CleanroomAgent {
    /// Create a new CleanroomAgent instance.
    ///
    /// Initializes the database connection, prompt builder, and optionally
    /// the LLM agent based on environment configuration.
    ///
    /// # Arguments
    /// * `config` - Agent configuration including DB path and prompt settings
    ///
    /// # Returns
    /// Returns a `Result` containing the new agent instance or a database error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cleanroom_agent::{CleanroomAgent, AgentConfig};
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = AgentConfig::producer(PathBuf::from("state.db"));
    /// let agent = CleanroomAgent::new(config)?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip_all)]
    pub fn new(config: AgentConfig) -> Result<Self, cleanroom_db::DbError> {
        let db = Database::open(&config.db_path)?;

        // Build the prompt builder
        let prompt_builder = PromptBuilder::new(config.prompt_config.clone())
            .with_budget(ContextBudget::default())
            .with_tools(default_tool_descriptions());

        // Load few-shot examples from existing completed tasks
        let few_shot = if config.load_few_shot {
            load_from_database(&db, 100)
        } else {
            FewShotManager::new(10, 100)
        };

        // Generate the system prompt from config (kept for future tool-call sessions)
        let _system_prompt = cleanroom_prompt::build_system_prompt(&config.prompt_config);

        // LLM agent construction is delegated to crate::llm_loop::run_loop, which
        // builds an autoagents LLM via LLMBuilder at call time. We no longer hold
        // a long-lived LLM agent on CleanroomAgent.

        Ok(Self {
            db: Arc::new(db),
            prompt_builder,
            few_shot,
            config,
        })
    }

    /// Build a task-specific prompt with context, few-shot examples, and orchestration.
    ///
    /// Constructs a prompt for LLM interaction by combining:
    /// - The task instruction
    /// - Context items from dependency tasks
    /// - Working set context
    /// - Relevant few-shot examples from the database
    ///
    /// # Arguments
    /// * `task_instruction` - Instructions for the specific task
    /// * `task_type` - Optional task type for example selection
    /// * `dependency_context` - Context from dependent task results
    /// * `working_set` - Current working set of context items
    pub fn build_task_prompt(
        &self,
        task_instruction: &str,
        task_type: Option<&cleanroom_db::TaskType>,
        dependency_context: &[ContextItem],
        working_set: &[ContextItem],
    ) -> GeneratedPrompt {
        self.prompt_builder.build(task_instruction, task_type, dependency_context, working_set)
    }

    /// Record a successful task for future few-shot examples.
    ///
    /// After a task completes successfully, call this to add it to the
    /// few-shot example pool for future task reference.
    ///
    /// # Arguments
    /// * `task_type` - Type of the completed task
    /// * `language` - Optional language context
    /// * `input` - Task input JSON
    /// * `output` - Task output JSON
    /// * `tool_trace` - Sequence of tools used during task execution
    pub fn record_example(
        &mut self,
        task_type: &cleanroom_db::TaskType,
        language: Option<&str>,
        input: serde_json::Value,
        output: serde_json::Value,
        tool_trace: Vec<String>,
    ) {
        self.few_shot.record(task_type, language, input, output, tool_trace);
    }

    /// Reload few-shot examples from the database.
    ///
    /// Re-reads completed tasks from the database to update the
    /// few-shot example pool. Useful after long-running operations
    /// where new successful examples may have been recorded.
    pub fn reload_few_shot(&mut self) {
        self.few_shot = load_from_database(&self.db, 100);
    }

    /// Run the agent in the specified mode.
    ///
    /// Executes the agent according to the provided [`RunMode`]:
    /// - [`RunMode::Produce`]: Analyze repository and generate S.DEF
    /// - [`RunMode::Consume`]: Read S.DEF and generate code
    /// - [`RunMode::Resume`]: Restore workflow from checkpoint
    ///
    /// # Arguments
    /// * `mode` - The execution mode and its parameters
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cleanroom_agent::{CleanroomAgent, AgentConfig, RunMode};
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = AgentConfig::producer(PathBuf::from("state.db"));
    /// let agent = CleanroomAgent::new(config)?;
    ///
    /// agent.run(RunMode::Produce {
    ///     repo_path: PathBuf::from("./my-repo"),
    ///     output_path: PathBuf::from("./output"),
    ///     project_name: "my-project".to_string(),
    /// }).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self))]
    pub async fn run(&self, mode: RunMode) -> anyhow::Result<()> {
        match mode {
            RunMode::Produce {
                repo_path,
                output_path,
                project_name,
            } => self.run_producer(repo_path, output_path, project_name).await,
            RunMode::Consume {
                sdef_path,
                output_path,
                language,
                framework,
                compat_mode,
                fidelity,
            } => {
                self.run_consumer(sdef_path, output_path, &language, framework.as_deref(), compat_mode, fidelity)
                    .await
            }
            RunMode::Resume {
                document,
                retry_failed,
            } => self.run_resume(&document, retry_failed).await,
        }
    }

    /// Run in produce mode: analyze repo → S.DEF.
    async fn run_producer(
        &self,
        repo_path: PathBuf,
        output_path: PathBuf,
        project_name: String,
    ) -> anyhow::Result<()> {
        let config = OrchestratorConfig {
            repo_path,
            output_path,
            db_path: self.config.db_path.clone(),
            project_name: project_name.clone(),
            checkpoint_interval_secs: 600,
            agent_idle_timeout_secs: 300,
            num_producers: 1,
            num_consumers: 0,
            num_reviewers: 0,
        };
        let orchestrator = Orchestrator::new(config)?;
        orchestrator.start_workflow().await?;

        let producer = ProducerAgent::new(ProducerConfig::default(), orchestrator.db().clone());
        while let Ok(Some(task)) = producer.process_next_task().await {
            info!(task_id = %task.task_id, "Processed task");
        }
        info!(project = %project_name, "Production complete");
        Ok(())
    }

    /// Run in consume mode: S.DEF → code.
    async fn run_consumer(
        &self,
        sdef_path: PathBuf,
        output_path: PathBuf,
        language: &str,
        framework: Option<&str>,
        compat_mode: CompatibilityMode,
        fidelity: Fidelity,
    ) -> anyhow::Result<()> {
        let sdef_content = std::fs::read_to_string(&sdef_path)?;
        let sdef: sdef_core::SoftwareDefinition = serde_json::from_str(&sdef_content)?;

        let importer = cleanroom_db::export_import::SdefImporter::new(
            rusqlite::Connection::open(&self.config.db_path)?,
        );
        importer.import(&sdef)?;

        let config = ConsumerConfig {
            language: language.to_string(),
            framework: framework.map(|s| s.to_string()),
            compatibility_mode: compat_mode,
            fidelity,
            output_path,
            use_legacy_template: false,
            llm: None,
            loop_config: LoopConfig::default(),
        };
        let consumer = ConsumerAgent::new(config, self.db.clone());
        consumer.generate_code().await?;

        info!("Consumption complete");
        Ok(())
    }

    /// Run in resume mode: restore workflow state.
    async fn run_resume(&self, document: &str, retry_failed: bool) -> anyhow::Result<()> {
        use crate::scheduler::Scheduler;

        let scheduler = Scheduler::new(self.db.clone());
        let repo = cleanroom_db::TaskRepository::new(self.db.connection_arc());

        let all_tasks = repo.list(None, None, None)?;
        let doc_tasks: Vec<_> = all_tasks
            .iter()
            .filter(|t| t.input_json.contains(document))
            .collect();

        if doc_tasks.is_empty() {
            info!(document = %document, "No tasks found for document");
            return Ok(());
        }

        for task in doc_tasks.iter().filter(|t| {
            matches!(t.status, cleanroom_db::TaskStatus::InProgress | cleanroom_db::TaskStatus::Assigned)
        }) {
            repo.update_status(&task.task_id, cleanroom_db::TaskStatus::Pending)?;
        }

        if retry_failed {
            scheduler.retry_failed_tasks()?;
        }

        let pending_count = doc_tasks
            .iter()
            .filter(|t| t.status == cleanroom_db::TaskStatus::Pending)
            .count();
        info!(document = %document, pending = %pending_count, "Workflow resumable");

        Ok(())
    }

    /// Get a reference to the database.
    pub fn db(&self) -> &Arc<Database> {
        &self.db
    }

    /// Get the configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }
}

impl std::fmt::Debug for CleanroomAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CleanroomAgent")
            .field("db_path", &self.config.db_path)
            .field("few_shot_count", &self.few_shot.total_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_prompt::Priority;

    #[test]
    fn test_agent_config_producer() {
        let config = AgentConfig::producer(PathBuf::from(":memory:"));
        assert_eq!(config.agent_name, "cleanroom-agent");
        assert!(config.prompt_config.include_tools);
    }

    #[test]
    fn test_agent_config_consumer() {
        let config = AgentConfig::consumer(PathBuf::from(":memory:"), "rust".into());
        assert!(matches!(config.prompt_config.agent_type, AgentType::Consumer));
        assert_eq!(config.prompt_config.target_language, Some("rust".into()));
    }

    #[test]
    fn test_build_task_prompt() {
        let config = AgentConfig::producer(PathBuf::from("state.db"));
        // Skip DB open — test prompt building in isolation
        let prompt_builder = PromptBuilder::new(config.prompt_config)
            .with_tools(default_tool_descriptions());

        let deps = vec![
            ContextItem::new("sdef://ref/User", "struct User { id: u64 }", Priority::High),
        ];
        let prompt = prompt_builder.build(
            "Analyze User entity",
            None,
            &deps,
            &[],
        );
        assert!(prompt.text.contains("ANALYSIS agent"));
        assert!(prompt.text.contains("User"));
        assert!(prompt.estimated_tokens > 0);
    }
}
