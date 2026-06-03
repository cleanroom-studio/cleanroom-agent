//! CLI Commands — All user-facing operations for the Cleanroom Agent.
//!
//! This module implements the command dispatcher and individual command handlers.
//! All user-facing output uses i18n via [`tr_global!()`] for internationalization.
//!
//! # Architecture
//!
//! - [`Commands`] — Clap enum defining all available subcommands
//! - [`run()`] — Top-level dispatcher that routes to command handlers
//! - Individual handler functions — One per command (e.g., [`produce_command()`])
//!
//! # Command Categories
//!
//! ## Pipeline Commands
//! - [`Commands::Produce`] — Analyze repository → output S.DEF
//! - [`Commands::Consume`] — Read S.DEF → generate code
//!
//! ## Server Commands
//! - [`Commands::Serve`] — Start MCP server for external integrations
//!
//! ## Workflow Commands
//! - [`Commands::Resume`] — Resume workflow from checkpoint
//!
//! ## Database Commands
//! - [`Commands::Inspect`] — Inspect database state and consistency
//! - [`Commands::Export`] — Export S.DEF document to JSON/YAML
//! - [`Commands::Import`] — Import S.DEF document from JSON/YAML
//! - [`Commands::Migrate`] — Run database migrations
//!
//! ## Analysis Commands
//! - [`Commands::Upgrade`] — Analyze version differences and breaking changes
//!
//! # Internationalization
//!
//! All output strings are wrapped with [`tr_global!()`] macro to support
//! multiple languages. Translation keys follow the pattern `cli.<command>_<action>`.
//!
//! # Example
//!
//! ```bash
//! cleanroom produce --repo ./my-project --output ./sdef
//! cleanroom consume --sdef ./sdef/sdef.json --output ./gen --language typescript
//! cleanroom inspect --check-type coverage
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{Result, Context};
use clap::{Subcommand, ValueEnum};
use cleanroom_agent::{
    AgentConfig, CleanroomAgent, RunMode,
    CompatibilityMode, Fidelity, CompletenessValidator, format_report,
    VersionUpgradeAnalyzer, consumer::ConsumeScope,
    consumer::{ConsumerAgent, ConsumerConfig},
    llm_loop::LoopConfig,
    producer::{ProducerAgent, ProducerConfig, ProducerMode},
};
use cleanroom_db::Database;
use cleanroom_i18n::tr_global;
use cleanroom_meta_llm::MetaLlm;
use tracing::info;

/// CLI execution mode (Phase 0.8). Maps 1:1 to the producer/consumer
/// internal mode enums. `Both` runs both pipelines and writes a diff
/// report under `<output>/_diff_report.txt`.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliMode {
    /// LLM-driven (default in 0.8): per-file LLM analysis, per-entity
    /// LLM code generation. Requires an LLM API key.
    Llm,
    /// Template/legacy: the pre-Phase-0.5 static-analysis pipeline.
    /// No LLM calls. Used as the Phase 5 baseline.
    Template,
    /// Run both modes; the second pass's output goes to
    /// `<output>/_both_<mode>/` and a diff report is written to
    /// `<output>/_diff_report.txt`.
    Both,
}

/// Build an LLM from env vars + optional CLI overrides. Returns a clear
/// error with "how to configure" guidance if no API key is found
/// anywhere.
pub fn build_llm_from_env(model: Option<&str>, api_key: Option<&str>) -> Result<Arc<dyn MetaLlm>> {
    use cleanroom_meta_llm::backends::openai::OpenAiProvider;
    use cleanroom_meta_llm::builder::MetaBuilder;
    let key = match api_key {
        Some(k) => k.to_string(),
        None => std::env::var("MINIMAX_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "no LLM API key found. Set one of:\n  \
                     - MINIMAX_API_KEY (recommended for MiniMax-M3)\n  \
                     - OPENAI_API_KEY\n  \
                     - ANTHROPIC_API_KEY\n\n\
                     Or pass `--api-key <KEY>` on the command line,\n\
                     or use `--mode template` to skip the LLM entirely."
                )
            })?,
    };
    if std::env::var_os("OPENAI_BASE_URL").is_none() {
        std::env::set_var("OPENAI_BASE_URL", "https://api.minimaxi.com/v1");
    }
    let model_name = model
        .map(|s| s.to_string())
        .or_else(|| std::env::var("EVAL_MODEL").ok())
        .unwrap_or_else(|| "MiniMax-M3".to_string());
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.minimaxi.com/v1".to_string());
    let llm: Arc<OpenAiProvider> = MetaBuilder::<OpenAiProvider>::new()
        .api_key(key)
        .base_url(base_url)
        .model(model_name.clone())
        .max_tokens(1024)
        .temperature(0.0)
        .build()?;
    info!(
        model = %model_name,
        "build_llm_from_env: constructed OpenAiProvider"
    );
    Ok(llm)
}

/// Build a LoopConfig from optional CLI overrides, falling back to defaults.
pub fn loop_config_from_opts(
    max_iterations: Option<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    cost_limit_usd: Option<f64>,
) -> LoopConfig {
    let mut cfg = LoopConfig::default();
    if let Some(n) = max_iterations {
        cfg.max_iterations = n;
    }
    if let Some(n) = max_tokens {
        cfg.max_tokens_per_call = n;
    }
    if let Some(t) = temperature {
        cfg.temperature = t;
    }
    cfg.cost_limit_usd = cost_limit_usd;
    cfg
}

/// Pipeline command: analyze code repository and produce S.DEF output.
///
/// # Process
///
/// 1. Scans the repository for source files
/// 2. Runs LLM-powered analysis via ADK
/// 3. Extracts data models, functions, contracts, and architecture
/// 4. Writes S.DEF JSON to the output directory
///
/// # Output Structure
///
/// ```
/// output/
/// └── sdef.json          # Complete S.DEF document
/// ```
///
/// # Example
///
/// ```bash
/// cleanroom produce --repo ./my-project --output ./sdef-output --name my-project
/// ```
#[derive(Subcommand)]
pub enum Commands {
    /// Production mode: analyze code repository → output S.DEF
    Produce {
        #[arg(long)]
        repo: String,
        #[arg(long, default_value = "./sdef-output")]
        output: String,
        #[arg(long)]
        exclude: Option<String>,
        #[arg(long)]
        name: Option<String>,
        /// LLM model (e.g. gemini-2.5-flash)
        #[arg(long)]
        model: Option<String>,
        /// API key for LLM provider
        #[arg(long)]
        api_key: Option<String>,
        /// Execution mode (Phase 0.8): `llm` (default; one LlmAnalyzeFile
        /// task per source file), `template` (legacy static-analysis
        /// pipeline; Phase 5 baseline), or `both` (run both, write a
        /// diff report to `<output>/_diff_report.txt`).
        #[arg(long, value_enum, default_value_t = CliMode::Llm)]
        mode: CliMode,
        /// LLM loop tuning (only honored when `--mode llm` or `both`).
        #[arg(long)]
        max_iterations: Option<u32>,
        /// LLM loop tuning (max_tokens per LLM call).
        #[arg(long)]
        max_tokens: Option<u32>,
        /// LLM sampling temperature (0.0 = deterministic, 1.0 = creative).
        #[arg(long)]
        temperature: Option<f32>,
        /// Hard cap on total estimated USD cost for the run; abort with
        /// `Aborted` if exceeded.
        #[arg(long)]
        cost_limit_usd: Option<f64>,
    },

    /// Consumption mode: read S.DEF and generate code in target language.
    ///
    /// # Process
    ///
    /// 1. Loads S.DEF document from JSON/YAML
    /// 2. Applies compatibility filtering based on `compat_mode`
    /// 3. Triggers code generation via LLM for specified language/framework
    /// 4. Validates completeness of generated output
    ///
    /// # Compatibility Modes
    ///
    /// - `full` — Include all legacy elements, 100% backward compatibility
    /// - `mixed` — Include compat layers but mark deprecated (default)
    /// - `clean` — Current version only, strip all legacy code
    ///
    /// # Fidelity Levels
    ///
    /// - `high` — Maximum detail, all optional fields populated
    /// - `medium` — Balanced detail (default)
    /// - `low` — Minimal representation, essential fields only
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom consume --sdef ./sdef/sdef.json --output ./generated \
    ///   --language typescript --framework react --compat-mode mixed
    /// ```
    Consume {
        #[arg(long)]
        sdef: String,
        #[arg(long, default_value = "./output")]
        output: String,
        #[arg(long)]
        language: String,
        #[arg(long)]
        framework: Option<String>,
        #[arg(long, default_value = "mixed")]
        compat_mode: String,
        #[arg(long, default_value = "medium")]
        fidelity: String,
        /// Phase 1.3: scope of the consume pass. `whole` (default)
        /// translates every entity in the S.DEF; `module=<name>`
        /// filters down to entities whose `logical_model` contains
        /// `<name>`; `function=<name>` filters to a single
        /// function spec. The form is one flag with a
        /// `kind=value` payload.
        #[arg(long, default_value = "whole")]
        scope: String,
        /// Phase 1.2: path to the *target* project (the codebase
        /// the generated code will be dropped into). When given,
        /// we read its `Cargo.toml` / `package.json` /
        /// `pyproject.toml` and inject the inferred package name,
        /// version, and dep set into the LLM prompt. Optional —
        /// the LLM still works without it (it just guesses at
        /// imports).
        #[arg(long)]
        target_dir: Option<String>,
        /// LLM model (e.g. gemini-2.5-flash)
        #[arg(long)]
        model: Option<String>,
        /// API key for LLM provider
        #[arg(long)]
        api_key: Option<String>,
        /// Execution mode (Phase 0.8): `llm` (default; one run_loop
        /// call per S.DEF entity), `template` (legacy static-analysis
        /// pipeline; Phase 5 baseline), or `both` (run both, write a
        /// diff report to `<output>/_diff_report.txt`).
        #[arg(long, value_enum, default_value_t = CliMode::Llm)]
        mode: CliMode,
        /// LLM loop tuning (only honored when `--mode llm` or `both`).
        #[arg(long)]
        max_iterations: Option<u32>,
        /// LLM loop tuning (max_tokens per LLM call).
        #[arg(long)]
        max_tokens: Option<u32>,
        /// LLM sampling temperature (0.0 = deterministic, 1.0 = creative).
        #[arg(long)]
        temperature: Option<f32>,
        /// Hard cap on total estimated USD cost for the run.
        #[arg(long)]
        cost_limit_usd: Option<f64>,
    },

    /// Start MCP server for external tool integrations.
    ///
    /// The Model Context Protocol (MCP) server allows external tools like
    /// editors, IDEs, and other agents to interact with the Cleanroom Agent.
    ///
    /// # Transport Types
    ///
    /// - `stdio` — Standard input/output (default, for local IDE integration)
    /// - `tcp://127.0.0.1:0` — TCP transport with OS-assigned port (cross-platform,
    ///   enables CLI task queue management). Port is written to `<tmp>/cleanroom.port`.
    /// - `tcp://127.0.0.1:12345` — TCP transport on a specific port
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom serve --transport stdio
    /// cleanroom serve --transport tcp://127.0.0.1:0
    /// cleanroom serve --transport tcp://127.0.0.1:9000
    /// ```
    Serve {
        #[arg(long, default_value = "stdio")]
        transport: String,
    },

    /// Resume a previously interrupted workflow from checkpoint.
    ///
    /// This command restarts the agent from the last saved checkpoint,
    /// allowing recovery from crashes or interrupted operations.
    ///
    /// # Arguments
    ///
    /// - `document` — Name of the S.DEF document to resume
    /// - `--retry-failed` — If set, also retry tasks that previously failed
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom resume --document my-project
    /// cleanroom resume --document my-project --retry-failed
    /// ```
    Resume {
        #[arg(long)]
        document: String,
        /// Resume failed tasks too
        #[arg(long)]
        retry_failed: bool,
    },

    /// Inspect database state: consistency, coverage, or task progress.
    ///
    /// Provides diagnostic information about the database and S.DEF state.
    ///
    /// # Check Types
    ///
    /// - `consistency` — Check fingerprint mismatches between S.DEF, DB, and code hashes (default)
    /// - `coverage` — Report counts of data models, attributes, contracts, functions, symbols
    /// - `progress` — Show task status breakdown (pending, running, completed, failed)
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom inspect --check-type consistency
    /// cleanroom inspect --check-type coverage
    /// cleanroom inspect --check-type progress
    /// cleanroom inspect --queue
    /// ```
    Inspect {
        #[arg(long, default_value = "consistency")]
        check_type: String,
        /// Show full task queue (requires running MCP server)
        #[arg(long)]
        queue: bool,
    },

    /// Phase 0.9: inspect the LLM call log (one row per `run_loop`).
    ///
    /// Every LLM call the agent makes (Producer `LlmAnalyzeFile`,
    /// future Consumer `LlmGenerateCode`, etc.) is appended to the
    /// `llm_call_log` table via `LoopConfig::on_call_complete`. This
    /// command reads that table back so you can:
    /// - audit token / cost usage per task
    /// - debug prompt regressions (compare `prompt_tokens` /
    ///   `completion_tokens` across runs)
    /// - replay the conversation in a future UI / dashboard
    ///
    /// # Filters
    ///
    /// - `--task-id <UUID>` — list every call for a single task (oldest first)
    /// - `--recent N` — list the N most recent calls across all tasks (newest first)
    /// - `--agent-type <producer|consumer|meta>` — further filter
    /// - `--format text|json` — human-readable (default) or newline-delimited JSON
    ///
    /// If neither `--task-id` nor `--recent` is supplied, defaults to
    /// `--recent 20`.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Recent 20 calls across all tasks
    /// cleanroom inspect llm-log
    ///
    /// # All calls for one task
    /// cleanroom inspect llm-log --task-id 1b31f20a-c1a7-43d8-9359-1f7417f3d7f0
    ///
    /// # Recent 5 calls in JSON for piping into jq
    /// cleanroom inspect llm-log --recent 5 --format json | jq
    /// ```
    /// Phase 0.9: list the LLM call log (one row per `run_loop`).
    ///
    /// Reachable as either `llm-log` (canonical) or `inspect-llm-log` (alias
    /// for the original PLAN.md call site). Defaults to the 20 most
    /// recent calls. See `docs/11-prompt-engineering.md §10` for the
    /// full prompt-engineering workflow built on top of this log.
    #[clap(alias = "inspect-llm-log")]
    LlmLog {
        #[arg(long, conflicts_with = "recent")]
        task_id: Option<String>,
        #[arg(long)]
        recent: Option<usize>,
        #[arg(long)]
        agent_type: Option<String>,
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Export S.DEF document to JSON or YAML file.
    ///
    /// Reads a document from the database and serializes it to the specified
    /// format for backup, sharing, or version control.
    ///
    /// # Arguments
    ///
    /// - `document` — Name of the S.DEF document to export
    /// - `--output` — Output file path (default: `./sdef-output/sdef.json`)
    /// - `--format` — Format: `json` (default) or `yaml`/`yml`
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom export --document my-project --output ./backup.json
    /// cleanroom export --document my-project --output ./backup.yaml --format yaml
    /// ```
    Export {
        #[arg(long)]
        document: String,
        #[arg(long, default_value = "./sdef-output/sdef.json")]
        output: String,
        #[arg(long, default_value = "json")]
        format: String,
    },

    /// Import S.DEF document from JSON or YAML file.
    ///
    /// Parses a S.DEF file and loads it into the database, making it available
    /// for code generation and other operations.
    ///
    /// # Arguments
    ///
    /// - `file` — Path to S.DEF file (JSON or YAML)
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom import --file ./sdef-input.json
    /// ```
    Import {
        #[arg(long)]
        file: String,
    },

    /// Run database migrations.
    ///
    /// Applies or rolls back schema migrations for the Cleanroom Agent database.
    ///
    /// # Directions
    ///
    /// - `up` — Apply pending migrations (default)
    /// - `down` — Rollback last migration (not supported)
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom migrate --direction up
    /// ```
    Migrate {
        #[arg(long, default_value = "up")]
        direction: String,
    },

    /// Analyze version differences and breaking changes between git tags.
    ///
    /// Compares two git tags or commits and produces a detailed report of:
    /// - Added, modified, and deleted files
    /// - Breaking API changes
    /// - Deprecated entities
    /// - New compatibility layers
    /// - Suggested migration paths
    ///
    /// # Arguments
    ///
    /// - `old_version` — Git ref (tag/commit) for the older version
    /// - `new_version` — Git ref (tag/commit) for the newer version
    /// - `repo` — Path to the git repository
    /// - `--document` — Optional S.DEF document name to scope analysis
    /// - `--apply` — If true, apply detected changes to the database
    ///
    /// # Output
    ///
    /// The report includes counts and details for:
    /// - Files added/modified/deleted
    /// - Breaking changes (API incompatibilities)
    /// - Deprecated entities
    /// - New compatibility layers needed
    /// - Suggested migrations
    ///
    /// # Example
    ///
    /// ```bash
    /// # Analyze without applying
    /// cleanroom upgrade --old-version v1.0 --new-version v2.0 --repo ./my-project
    ///
    /// # Analyze and apply changes to database
    /// cleanroom upgrade --old-version v1.0 --new-version v2.0 --repo ./my-project --apply
    /// ```
    Upgrade {
        #[arg(long)]
        old_version: String,
        #[arg(long)]
        new_version: String,
        #[arg(long)]
        repo: String,
        #[arg(long)]
        document: Option<String>,
        /// Apply detected changes to the database
        #[arg(long, default_value = "false")]
        apply: bool,
    },

    /// Run evaluation against benchmark projects.
    ///
    /// Evaluates the quality of the Cleanroom Agent's analysis and code generation
    /// by running against known benchmark projects. Produces a quality report with
    /// coverage, accuracy, fidelity, and operational metrics.
    ///
    /// # Example
    ///
    /// ```bash
    /// # Run all built-in benchmarks
    /// cleanroom evaluate
    ///
    /// # Run a specific benchmark
    /// cleanroom evaluate --benchmark redis
    ///
    /// # Output report to a file
    /// cleanroom evaluate --output ./report.json
    /// ```
    Evaluate {
        /// Specific benchmark project to evaluate (omit for all built-in)
        #[arg(long)]
        benchmark: Option<String>,
        /// Output path for the evaluation report (default: stdout)
        #[arg(long)]
        output: Option<String>,
    },

    /// Manage Skills (PLAN2 Phase G).
    ///
    /// Browse, activate, validate, and refresh the Skills index. Skills
    /// are the LLM behavior-spec mechanism defined in
    /// `docs/21-skills-system.md` §6.
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom skill list
    /// cleanroom skill show rust-analysis
    /// cleanroom skill activate rust-analysis
    /// cleanroom skill validate .cleanroom/skills/my-skill/SKILL.md
    /// ```
    Skill {
        #[command(subcommand)]
        command: SkillCommand,
    },

    /// Manage Staging workspaces (PLAN2 Phase G).
    ///
    /// Inspect / abort staged file operations that an LLM has not yet
    /// committed to the source tree.
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom staging status <task_id>
    /// cleanroom staging commit <task_id> --target /path/to/repo
    /// cleanroom staging abort <task_id>
    /// ```
    Staging {
        #[command(subcommand)]
        command: StagingCommand,
    },

    /// Manage the task queue of a running agent.
    ///
    /// Insert, remove, or modify pending tasks in a running `cleanroom serve`
    /// process. Requires the MCP server to be running with TCP transport.
    ///
    /// # Example
    ///
    /// ```bash
    /// # Show task queue
    /// cleanroom task list
    ///
    /// # Insert a new task
    /// cleanroom task insert --type EXTRACT_DATA_MODEL --input '{"module":"payment"}' --priority 6
    ///
    /// # Remove a pending task
    /// cleanroom task remove t-006
    ///
    /// # Modify a pending task
    /// cleanroom task modify t-005 --priority 9
    ///
    /// # Reprioritize tasks
    /// cleanroom task reprioritize t-008 10 t-007 8
    /// ```
    #[command(subcommand)]
    Task(TaskCommand),

    /// Control the workflow lifecycle of a running agent.
    ///
    /// Pause, resume, or check the status of a running workflow.
    #[command(subcommand)]
    Workflow(WorkflowCommand),
}

/// Task queue subcommands.
#[derive(clap::Subcommand)]
enum TaskCommand {
    /// List all tasks in the queue.
    List {
        /// Filter by status
        #[arg(long)]
        status: Option<String>,
        /// Filter by task type
        #[arg(long)]
        task_type: Option<String>,
    },
    /// Insert a new task into the queue.
    Insert {
        /// Task type (e.g. EXTRACT_DATA_MODEL)
        #[arg(long)]
        r#type: String,
        /// JSON input for the task
        #[arg(long)]
        input: String,
        /// Priority (higher = earlier)
        #[arg(long, default_value = "5")]
        priority: i32,
        /// Task ID to depend on
        #[arg(long)]
        after: Option<String>,
        /// Max retry count
        #[arg(long, default_value = "3")]
        max_retries: i32,
    },
    /// Remove a pending task.
    Remove {
        /// Task ID to remove
        task_id: String,
    },
    /// Modify a pending task's properties.
    Modify {
        /// Task ID to modify
        task_id: String,
        /// New priority
        #[arg(long)]
        priority: Option<i32>,
        /// New input JSON
        #[arg(long)]
        input: Option<String>,
        /// New max retries
        #[arg(long)]
        max_retries: Option<i32>,
    },
    /// Reprioritize multiple tasks.
    Reprioritize {
        /// Pairs of task_id + new_priority (e.g. "t-001 10 t-002 8")
        pairs: Vec<String>,
    },
}

/// Workflow lifecycle subcommands.
#[derive(clap::Subcommand)]
enum WorkflowCommand {
    /// Pause the running workflow (wait for current tasks to finish).
    Pause,
    /// Resume a paused workflow.
    Resume,
    /// Show workflow status.
    Status,
}

/// Skill subcommands (PLAN2 Phase G.1).
#[derive(Subcommand)]
pub enum SkillCommand {
    /// List all visible skills (Tier 1 catalog).
    List {
        /// Optional scope filter (`builtin` / `project-cleanroom` /
        /// `project-agents` / `user-cleanroom` / `user-agents`).
        #[arg(long)]
        scope: Option<String>,
        /// Filter by `applies-to` task type.
        #[arg(long)]
        task_type: Option<String>,
    },
    /// Show a single skill's metadata (Tier 1 + light body).
    Show {
        /// Skill name.
        name: String,
    },
    /// Activate a skill: print the Tier 2 instruction block.
    Activate {
        /// Skill name.
        name: String,
        /// Override the skill's `token-budget`.
        #[arg(long)]
        token_budget: Option<u32>,
    },
    /// Validate a `SKILL.md` file against the spec.
    Validate {
        /// Absolute or relative path to the SKILL.md file.
        path: String,
    },
    /// Re-scan the filesystem for skill changes.
    Refresh {
        /// Optional root path. Default: current working directory.
        #[arg(long)]
        path: Option<String>,
    },
}

/// Staging subcommands (PLAN2 Phase G.2).
#[derive(Subcommand)]
pub enum StagingCommand {
    /// Show the manifest of a staging workspace (printed from local cache;
    /// not yet wired to SQLite).
    Status {
        /// Task ID of the staging workspace.
        task_id: String,
    },
    /// Commit a staging workspace to its target directory.
    Commit {
        /// Task ID of the staging workspace.
        task_id: String,
        /// Target source-tree directory.
        #[arg(long)]
        target: String,
    },
    /// Abort (drop) a staging workspace.
    Abort {
        /// Task ID of the staging workspace.
        task_id: String,
    },
}

/// Dispatches a CLI command to its corresponding handler.
///
/// This is the top-level entry point called from `main.rs`. It routes the
/// parsed [`Commands`] enum to the appropriate function based on variant.
///
/// # Arguments
///
/// - `command` — The parsed command enum from Clap
/// - `db_path` — Path to the SQLite database file
///
/// # Errors
///
/// Returns an error if the underlying handler fails. Error types vary by command.
pub fn run(command: Commands, db_path: &str) -> Result<()> {
    match command {
        Commands::Produce { repo, output, exclude: _, name, model, api_key, mode, max_iterations, max_tokens, temperature, cost_limit_usd } => {
            produce_command(&repo, &output, db_path, name, model, api_key, mode, max_iterations, max_tokens, temperature, cost_limit_usd)
        }
        Commands::Consume { sdef, output, language, framework, compat_mode, fidelity, scope, target_dir, model, api_key, mode, max_iterations: _, max_tokens: _, temperature: _, cost_limit_usd: _ } => {
            // Phase 0.8: Consume currently ignores the 4 loop-tuning
            // flags (they apply to Producer's LLM path). The `mode`
            // flag IS honored: llm / template / both.
            consume_command(&sdef, &output, &language, framework.as_deref(), &compat_mode, &fidelity, &scope, &target_dir, db_path, model, api_key, mode)
        }
        Commands::Serve { transport } => {
            serve_command(&transport, db_path)
        }
        Commands::Resume { document, retry_failed } => {
            resume_command(&document, retry_failed, db_path)
        }
        Commands::Inspect { check_type, queue } => {
            if queue {
                inspect_queue_command(db_path)
            } else {
                inspect_command(&check_type, db_path)
            }
        }
        Commands::LlmLog { task_id, recent, agent_type, format } => {
            llm_log_command(
                task_id.as_deref(),
                recent,
                agent_type.as_deref(),
                &format,
                db_path,
            )
        }
        Commands::Export { document, output, format } => {
            export_command(&document, &output, &format, db_path)
        }
        Commands::Import { file } => {
            import_command(&file, db_path)
        }
        Commands::Migrate { direction } => {
            migrate_command(&direction, db_path)
        }
        Commands::Upgrade { old_version, new_version, repo, document, apply } => {
            upgrade_command(&old_version, &new_version, &repo, document.as_deref(), apply, db_path)
        }
        Commands::Evaluate { benchmark, output } => {
            evaluate_command(benchmark.as_deref(), output.as_deref(), db_path)
        }
        Commands::Task(task) => {
            task_dispatch(task, db_path)
        }
        Commands::Workflow(workflow) => {
            workflow_dispatch(workflow, db_path)
        }
        Commands::Skill { command } => {
            skill_dispatch(command, db_path)
        }
        Commands::Staging { command } => {
            staging_dispatch(command, db_path)
        }
    }
}

/// Dispatch task subcommands via MCP client over TCP.
fn task_dispatch(cmd: TaskCommand, _db_path: &str) -> Result<()> {
    use crate::mcp_client::call_mcp_tool_sync;

    let addr = std::env::var("CLEANROOM_ADDR")
        .ok()
        .or_else(|| crate::mcp_client::discover_address().ok());

    let addr = addr.as_deref();

    match cmd {
        TaskCommand::List { status, task_type } => {
            let mut args = serde_json::json!({});
            if let Some(s) = status {
                args["filter_status"] = serde_json::json!([s]);
            }
            if let Some(t) = task_type {
                args["filter_type"] = serde_json::Value::String(t);
            }
            let result = call_mcp_tool_sync("get_task_queue", args, addr)?;
            println!("{}", serde_json::to_string_pretty(&result).unwrap_or_else(|_| format!("{:?}", result)));
        }
        TaskCommand::Insert { r#type, input, priority, after, max_retries } => {
            let input_val: serde_json::Value = serde_json::from_str(&input)
                .map_err(|e| anyhow::anyhow!("Invalid JSON input: {}", e))?;
            let mut args = serde_json::json!({
                "task_type": r#type,
                "priority": priority,
                "input": input_val,
                "max_retries": max_retries,
            });
            if let Some(ref aid) = after {
                args["after_task_id"] = serde_json::Value::String(aid.clone());
            }
            let result = call_mcp_tool_sync("insert_task", args, addr)?;
            println!("{}", serde_json::to_string_pretty(&result).unwrap_or_else(|_| format!("{:?}", result)));
        }
        TaskCommand::Remove { task_id } => {
            let args = serde_json::json!({"task_id": task_id});
            let result = call_mcp_tool_sync("remove_task", args, addr)?;
            println!("{}", serde_json::to_string_pretty(&result).unwrap_or_else(|_| format!("{:?}", result)));
        }
        TaskCommand::Modify { task_id, priority, input, max_retries } => {
            let mut args = serde_json::json!({"task_id": task_id});
            if let Some(p) = priority { args["priority"] = p.into(); }
            if let Some(ref i) = input {
                let val: serde_json::Value = serde_json::from_str(i)
                    .map_err(|e| anyhow::anyhow!("Invalid JSON input: {}", e))?;
                args["input"] = val;
            }
            if let Some(r) = max_retries { args["max_retries"] = r.into(); }
            let result = call_mcp_tool_sync("modify_task", args, addr)?;
            println!("{}", serde_json::to_string_pretty(&result).unwrap_or_else(|_| format!("{:?}", result)));
        }
        TaskCommand::Reprioritize { pairs } => {
            if pairs.len() % 2 != 0 {
                anyhow::bail!("Reprioritize requires pairs of task_id + priority (got {} args)", pairs.len());
            }
            for chunk in pairs.chunks(2) {
                let task_id = &chunk[0];
                let priority: i32 = chunk[1].parse()
                    .map_err(|e| anyhow::anyhow!("Invalid priority '{}': {}", chunk[1], e))?;
                let args = serde_json::json!({"task_id": task_id, "priority": priority});
                let result = call_mcp_tool_sync("modify_task", args, addr)?;
                println!("{}: {}", task_id, serde_json::to_string_pretty(&result).unwrap_or_else(|_| format!("{:?}", result)));
            }
        }
    }
    Ok(())
}

/// Dispatch workflow subcommands — pause/resume via MCP, status via PID + port file.
fn workflow_dispatch(cmd: WorkflowCommand, _db_path: &str) -> Result<()> {
    use crate::mcp_client::call_mcp_tool_sync;

    let addr = std::env::var("CLEANROOM_ADDR")
        .ok()
        .or_else(|| crate::mcp_client::discover_address().ok());

    match cmd {
        WorkflowCommand::Pause => {
            let addr = addr.ok_or_else(|| anyhow::anyhow!(
                "Cannot connect to server. Is `cleanroom serve --transport tcp://` running?"
            ))?;
            let result = call_mcp_tool_sync("pause_workflow", serde_json::json!({}), Some(&addr))?;
            let paused = result.get("paused").and_then(|v| v.as_bool()).unwrap_or(false);
            if paused {
                println!("Workflow pause requested — agents will finish current tasks then stop.");
            } else {
                println!("Workflow was already paused.");
            }
        }
        WorkflowCommand::Resume => {
            let addr = addr.ok_or_else(|| anyhow::anyhow!(
                "Cannot connect to server. Is `cleanroom serve --transport tcp://` running?"
            ))?;
            let result = call_mcp_tool_sync("resume_workflow", serde_json::json!({}), Some(&addr))?;
            let resumed = result.get("resumed").and_then(|v| v.as_bool()).unwrap_or(false);
            if resumed {
                println!("Workflow resumed — agents will continue claiming tasks.");
            } else {
                println!("Workflow was not paused.");
            }
        }
        WorkflowCommand::Status => {
            let pid_path = cleanroom_agent::pid_file_path();
            let port_path = cleanroom_agent::port_file_path();

            let pid_str = std::fs::read_to_string(&pid_path).ok();
            let port_str = std::fs::read_to_string(&port_path).ok();

            let running = if let Some(ref addr) = addr {
                // Best liveness check: can we connect via TCP?
                call_mcp_tool_sync("get_task_queue", serde_json::json!({}), Some(addr)).is_ok()
            } else {
                false
            };

            if running {
                let pid = pid_str.as_deref().unwrap_or("?");
                let port = port_str.as_deref().unwrap_or("?");
                println!("cleanroom agent is running (PID: {}, port: {})", pid.trim(), port.trim());
                if let Some(ref addr) = addr {
                    match call_mcp_tool_sync("get_task_queue", serde_json::json!({}), Some(addr)) {
                        Ok(val) => {
                            let tasks = val.as_array().map(|a| a.len()).unwrap_or(0);
                            println!("{} tasks in queue.", tasks);
                            println!("Use `cleanroom inspect --queue` for details.");
                        }
                        Err(_) => {}
                    }
                }
            } else if pid_str.is_some() {
                println!("cleanroom agent is NOT running (stale PID file at {})", pid_path.display());
            } else {
                println!("cleanroom agent is NOT running (no PID file at {})", pid_path.display());
            }
        }
    }
    Ok(())
}

/// `inspect --queue` via MCP client.
fn inspect_queue_command(_db_path: &str) -> Result<()> {
    use crate::mcp_client::call_mcp_tool_sync;

    let addr = std::env::var("CLEANROOM_ADDR")
        .ok()
        .or_else(|| crate::mcp_client::discover_address().ok());

    let addr = addr.as_deref();

    let result = call_mcp_tool_sync("get_task_queue", serde_json::json!({}), addr)?;
    let tasks = result.as_array()
        .ok_or_else(|| anyhow::anyhow!("Expected array response, got: {}", result))?;

    if tasks.is_empty() {
        println!("No tasks in queue.");
        return Ok(());
    }

    println!("═══ Task Queue ═══");
    println!("{:<14} {:<20} {:>8} {:<16} {}",
        "ID", "Type", "Priority", "Status", "Assigned To");
    println!("{:-<14} {:-<20} {:-<8} {:-<16} {:-<12}", "", "", "", "", "");

    for t in tasks {
        let id = t["task_id"].as_str().unwrap_or("?");
        let ty = t["task_type"].as_str().unwrap_or("?");
        let pri = t["priority"].as_i64().unwrap_or(0);
        let status = t["status"].as_str().unwrap_or("?");
        let assigned = t["assigned_to"].as_str().unwrap_or("-");
        let status_icon = match status {
            "completed" => "✅",
            "in_progress" => "🔄",
            "pending" => "⏳",
            "failed" | "failed_permanently" => "❌",
            _ => "?",
        };
        println!("{:<14} {:<20} {:>8} {} {:<14} {}",
            id, ty, pri, status_icon, status, assigned);
    }
    println!();
    println!("Use `cleanroom task <subcommand>` to manage the queue.");
    Ok(())
}

fn set_api_key(key: Option<String>) {
    if let Some(k) = key {
        if std::env::var("GOOGLE_API_KEY").is_err() {
            std::env::set_var("GOOGLE_API_KEY", k);
        }
    }
}

// ============================================================================
// PLAN2 Phase G: Skill / Staging CLI subcommands
// ============================================================================

/// Dispatch `cleanroom skill ...` subcommands. Talks directly to the
/// `cleanroom-skill` crate (no DB required for these).
fn skill_dispatch(cmd: SkillCommand, _db_path: &str) -> Result<()> {
    use cleanroom_skill::{
        build_skill_catalog_block, select_skill_prompt_block, load_skill_index_strict, validate_skill_dir, SelectionPolicy,
    };

    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    match cmd {
        SkillCommand::List { scope, task_type } => {
            let idx = load_skill_index_strict(&root)
                .map_err(|e| anyhow::anyhow!("load_skill_index: {e}"))?;
            let filtered: Vec<_> = idx
                .summaries()
                .into_iter()
                .filter(|s| match &scope {
                    Some(scope_str) => format!("{:?}", s.scope).eq_ignore_ascii_case(scope_str),
                    None => true,
                })
                .filter(|s| match &task_type {
                    Some(tt) => idx
                        .find_by_name(&s.name)
                        .map(|d| d.applies_to_task(tt))
                        .unwrap_or(false),
                    None => true,
                })
                .collect();
            if filtered.is_empty() {
                println!("(no skills found)");
            } else {
                println!("{} skill(s):", filtered.len());
                for s in filtered {
                    println!(
                        "  - {} [{:?}]  priority={}  token_budget={}",
                        s.name, s.scope, s.priority, s.token_budget
                    );
                    println!("    {}", s.description);
                }
            }
        }
        SkillCommand::Show { name } => {
            let idx = load_skill_index_strict(&root)
                .map_err(|e| anyhow::anyhow!("load_skill_index: {e}"))?;
            let skill = idx
                .find_by_name(&name)
                .ok_or_else(|| anyhow::anyhow!("skill not found: {name}"))?;
            println!("# {}", skill.name);
            println!("scope: {:?}", skill.scope);
            println!("priority: {}", skill.priority);
            println!("token_budget: {}", skill.token_budget);
            println!("path: {}", skill.path.display());
            println!("hash: {}", skill.hash);
            if !skill.allowed_tools.is_empty() {
                println!("allowed-tools: {}", skill.allowed_tools.join(", "));
            }
            if !skill.allowed_paths.is_empty() {
                println!("allowed-paths: {}", skill.allowed_paths.join(", "));
            }
            if !skill.applies_to.is_empty() {
                println!("applies-to: {}", skill.applies_to.join(", "));
            }
            println!("\n--- description ---\n{}", skill.description);
            println!("\n--- body (first 2000 chars) ---\n{}",
                skill.body.chars().take(2000).collect::<String>());
        }
        SkillCommand::Activate { name, token_budget } => {
            let idx = load_skill_index_strict(&root)
                .map_err(|e| anyhow::anyhow!("load_skill_index: {e}"))?;
            let policy = SelectionPolicy {
                top_k: 1,
                min_score: 0.0,
                ..Default::default()
            };
            let (block, summary) = select_skill_prompt_block(
                &idx,
                &name,
                &policy,
                token_budget.map(|b| b as usize).unwrap_or(4096),
            )
            .ok_or_else(|| anyhow::anyhow!("skill not found: {name}"))?;
            println!("Activated: {} (id={})", summary.name, summary.id);
            println!("--- Tier 2 prompt block ---\n{block}");
        }
        SkillCommand::Validate { path } => {
            let p = std::path::PathBuf::from(&path);
            let report = validate_skill_dir(&p)
                .map_err(|e| anyhow::anyhow!("validate: {e}"))?;
            if report.is_valid() {
                println!("✓ valid ({} warning(s))", report.warnings.len());
            } else {
                println!("✗ invalid ({} error(s), {} warning(s))", report.errors.len(), report.warnings.len());
            }
            for issue in report.issues() {
                println!("  [{}] {}", format!("{:?}", issue.level).to_lowercase(), issue.message);
            }
        }
        SkillCommand::Refresh { path } => {
            let p = path
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| root.clone());
            let idx = load_skill_index_strict(&p)
                .map_err(|e| anyhow::anyhow!("refresh: {e}"))?;
            println!("Refreshed: {} skill(s) under {}", idx.len(), p.display());
            for s in idx.summaries() {
                println!("  - {} [{:?}]", s.name, s.scope);
            }
        }
    }
    Ok(())
}

/// Dispatch `cleanroom staging ...` subcommands. Currently local-only
/// (no DB persistence yet — the SQLite manifest table is added in
/// a follow-up Phase E task).
fn staging_dispatch(cmd: StagingCommand, _db_path: &str) -> Result<()> {
    match cmd {
        StagingCommand::Status { task_id } => {
            println!("(note) staging status is currently local — no DB-backed manifest yet");
            println!("task_id: {task_id}");
            println!("hint: the staging workspace lives at /tmp/cleanroom-staging-{task_id}-* during the run");
        }
        StagingCommand::Commit { task_id, target } => {
            println!("(note) staging commit currently requires a live LLM run; this is a placeholder");
            println!("task_id: {task_id}");
            println!("target: {target}");
        }
        StagingCommand::Abort { task_id } => {
            println!("(note) staging abort is a no-op outside an LLM run");
            println!("task_id: {task_id}");
        }
    }
    Ok(())
}

/// Handler for the `produce` command.
///
/// Scans the repository, runs LLM analysis via ADK, and outputs S.DEF JSON.
fn produce_command(
    repo: &str, output: &str, db_path: &str,
    name: Option<String>, model: Option<String>, api_key: Option<String>,
    mode: CliMode,
    max_iterations: Option<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    cost_limit_usd: Option<f64>,
) -> Result<()> {
    set_api_key(api_key.clone());
    use tokio::runtime::Runtime;
    let project_name = name.unwrap_or_else(|| {
        Path::new(repo).file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string())
    });

    // Validate LLM env if we need one.
    if !matches!(mode, CliMode::Template) {
        // Try to build the LLM up-front so we fail fast with a clear
        // "how to configure" message.
        let _ = build_llm_from_env(model.as_deref(), api_key.as_deref())?;
    }

    let rt = Runtime::new().context(tr_global!("cli.error_runtime"))?;
    rt.block_on(async {
        // Phase 0.8: for `both` mode we run the producer twice (once
        // template, once llm) and write a diff report. Each pass
        // gets its own subdir under the requested `output`.
        match mode {
            CliMode::Llm => {
                run_produce_one_pass(
                    repo, output, db_path, &project_name,
                    model.as_deref(), api_key.as_deref(),
                    ProducerMode::Llm,
                    max_iterations, max_tokens, temperature, cost_limit_usd,
                ).await?;
            }
            CliMode::Template => {
                run_produce_one_pass(
                    repo, output, db_path, &project_name,
                    None, None,
                    ProducerMode::Template,
                    None, None, None, None,
                ).await?;
            }
            CliMode::Both => {
                // Template pass: into <output>/_template
                let template_output = format!("{output}/_template");
                run_produce_one_pass(
                    repo, &template_output, db_path, &project_name,
                    None, None,
                    ProducerMode::Template,
                    None, None, None, None,
                ).await?;
                // LLM pass: into <output>/_llm
                let llm_output = format!("{output}/_llm");
                run_produce_one_pass(
                    repo, &llm_output, db_path, &project_name,
                    model.as_deref(), api_key.as_deref(),
                    ProducerMode::Llm,
                    max_iterations, max_tokens, temperature, cost_limit_usd,
                ).await?;
                // Diff report
                let report_path = format!("{output}/_diff_report.txt");
                write_produce_diff_report(&template_output, &llm_output, &report_path)?;
                println!("== both-mode diff report: {report_path}");
            }
        }

        println!("{}", tr_global!("cli.produce_complete", &project_name));
        Ok(())
    })
}

/// Absolute path to the workspace's `migrations/` directory.
///
/// We walk up from the **running binary's location** rather than
/// `env!("CARGO_MANIFEST_DIR")`, because `cargo run` runs the binary
/// out of `target/debug/` and `env!("CARGO MANIFEST_DIR")` (when
/// captured at build time) sometimes ends up pointing inside the
/// target tree after a stale incremental build, which then
/// produces a spurious `target/debug/migrations` candidate that
/// silently swallows the real one.
///
/// The directory layout at build / run time is:
///
/// ```text
///   cleanroom-agent/
///     target/
///       debug/
///         cleanroom-cli          <-- this is what we resolve from
///     crates/
///       cleanroom-cli/
///         src/
///     migrations/                 <-- what we want
///       001_initial_schema.sql
///       002_sdef_storage.sql
///       ...
/// ```
///
/// so we walk up **three** levels from the binary (debug/ → target/ →
/// cleanroom-agent/) and then join `migrations/`. The `migrations/`
/// directory lives inside the `cleanroom-agent/` Cargo workspace
/// (not the surrounding repo root — `cleanroom-agent` is its own
/// workspace; see its root `Cargo.toml` `[workspace]` block).
pub fn cli_migrations_dir() -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap_or_else(|_| {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("cleanroom-cli"))
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    });
    // exe is `cleanroom-agent/target/<profile>/cleanroom-cli`
    // (or `cleanroom-cli.exe` on Windows). Walk up three levels to
    // land on `cleanroom-agent/`, then join `migrations/`.
    let cleanroom_agent = exe
        .parent() // <profile>/
        .and_then(|p| p.parent()) // target/
        .and_then(|p| p.parent()) // cleanroom-agent/
        .unwrap_or_else(|| std::path::Path::new("."));
    cleanroom_agent.join("migrations")
}

/// Single-pass variant of the produce flow. Opens the DB, builds
/// `ProducerConfig` + `LoopConfig` per `mode`, attaches the LLM
/// (when needed), and runs `ProducerAgent::run_repo_analysis`.
async fn run_produce_one_pass(
    repo: &str, output: &str, db_path: &str, project_name: &str,
    model: Option<&str>, api_key: Option<&str>,
    mode: ProducerMode,
    max_iterations: Option<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    cost_limit_usd: Option<f64>,
) -> Result<()> {
    use cleanroom_db::Database;
    let db = Arc::new(Database::open_with_migrations_from(
        Path::new(db_path),
        Some(&cli_migrations_dir()),
    )?);
    let producer_config = match mode {
        ProducerMode::Llm => ProducerConfig::llm(),
        ProducerMode::Template => ProducerConfig::default(),
        ProducerMode::Both => ProducerConfig::both(),
    };
    let mut producer = ProducerAgent::new(producer_config, db.clone());
    if matches!(mode, ProducerMode::Llm | ProducerMode::Both) {
        let llm = build_llm_from_env(model, api_key)?;
        producer = producer.with_llm(llm);
    }
    if max_iterations.is_some()
        || max_tokens.is_some()
        || temperature.is_some()
        || cost_limit_usd.is_some()
    {
        let cfg = loop_config_from_opts(
            max_iterations, max_tokens, temperature, cost_limit_usd,
        );
        producer = producer.with_loop_config(cfg);
    }
    // Phase 0.9: wire the LLM call log so every `run_loop` invocation
    // in this pass appends a row to `llm_call_log`. The repository
    // reuses the same `Arc<Mutex<Connection>>` the producer already
    // holds so we never open a second connection.
    let log_repo = Arc::new(cleanroom_db::LlmCallLogRepository::new_with_arc(
        db.connection_arc(),
    ));
    producer = producer.with_llm_call_logger(log_repo);
    println!("== produce mode={mode:?} output={output}");
    let processed = producer
        .run_repo_analysis(Path::new(repo), project_name)
        .await
        .map_err(|e| anyhow::anyhow!("produce failed: {e}"))?;
    println!("== processed {processed} tasks");
    // Auto-export the resulting S.DEF (Phase 0.8: makes `produce --output X`
    // actually leave an S.DEF file on disk for downstream `consume --sdef`).
    // Failures are non-fatal: the pipeline still ran, we just couldn't
    // serialize; the user can re-run `cleanroom-cli export --document ...`.
    let sdef_path = format!("{output}/{project_name}.json");
    match export_command(project_name, &sdef_path, "json", db_path) {
        Ok(()) => println!("== exported sdef to {sdef_path}"),
        Err(e) => eprintln!(
            "warn: post-produce export failed: {e}\n\
             (use `cleanroom-cli export --document {project_name} --output {sdef_path}` manually)"
        ),
    }
    Ok(())
}

/// Write a simple text diff report for `both` mode: list files
/// generated in each pass, mark identical / different / only-in-one.
fn write_produce_diff_report(
    template_dir: &str,
    llm_dir: &str,
    report_path: &str,
) -> Result<()> {
    use std::collections::BTreeSet;
    let tpl = std::path::Path::new(template_dir);
    let llm = std::path::Path::new(llm_dir);
    let mut all_files: BTreeSet<String> = BTreeSet::new();
    let mut template_files: Vec<String> = Vec::new();
    let mut llm_files: Vec<String> = Vec::new();

    fn walk(p: &Path, prefix: &str, out: &mut Vec<String>) -> std::io::Result<()> {
        if !p.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(p)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if entry.file_type()?.is_dir() {
                walk(&entry.path(), &format!("{prefix}{name}/"), out)?;
            } else {
                out.push(format!("{prefix}{name}"));
            }
        }
        Ok(())
    }
    walk(tpl, "", &mut template_files).ok();
    walk(llm, "", &mut llm_files).ok();
    for f in &template_files {
        all_files.insert(f.clone());
    }
    for f in &llm_files {
        all_files.insert(f.clone());
    }

    let mut report = String::new();
    report.push_str("# Both-mode diff report (template vs LLM producer output)\n\n");
    for f in &all_files {
        let in_t = template_files.contains(f);
        let in_l = llm_files.contains(f);
        if in_t && in_l {
            let t_content = std::fs::read_to_string(tpl.join(f)).unwrap_or_default();
            let l_content = std::fs::read_to_string(llm.join(f)).unwrap_or_default();
            if t_content == l_content {
                report.push_str(&format!("  IDENTICAL: {f}\n"));
            } else {
                report.push_str(&format!("  DIFFERENT: {f}  ({} vs {} bytes)\n",
                    t_content.len(), l_content.len()));
            }
        } else if in_t {
            report.push_str(&format!("  ONLY-IN-TEMPLATE: {f}\n"));
        } else {
            report.push_str(&format!("  ONLY-IN-LLM: {f}\n"));
        }
    }
    if let Some(parent) = std::path::Path::new(report_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(report_path, report)?;
    Ok(())
}

/// Handler for the `consume` command.
///
/// Loads S.DEF, generates code via LLM, and validates output completeness.
/// Parse the `--scope` CLI flag (a `kind=value` string) into a
/// [`ConsumeScope`]. Accepts:
/// - `"whole"`  (default) → `WholeProject`
/// - `"module=<name>"`   → `Module(name)`
/// - `"function=<name>"` → `Function(name)`
///
/// Unknown / malformed values fall back to `WholeProject` with a
/// `tracing::warn!` so the consume pass still runs.
fn parse_consume_scope(s: &str) -> ConsumeScope {
    let s = s.trim();
    if s.is_empty() || s == "whole" {
        return ConsumeScope::WholeProject;
    }
    if let Some(rest) = s.strip_prefix("module=") {
        if rest.is_empty() {
            tracing::warn!("--scope 'module=' is missing a name; defaulting to WholeProject");
            return ConsumeScope::WholeProject;
        }
        return ConsumeScope::Module(rest.to_string());
    }
    if let Some(rest) = s.strip_prefix("function=") {
        if rest.is_empty() {
            tracing::warn!("--scope 'function=' is missing a name; defaulting to WholeProject");
            return ConsumeScope::WholeProject;
        }
        return ConsumeScope::Function(rest.to_string());
    }
    tracing::warn!(
        scope = s,
        "--scope value is not 'whole', 'module=<name>', or 'function=<name>'; \
         defaulting to WholeProject"
    );
    ConsumeScope::WholeProject
}

fn consume_command(
    sdef: &str, output: &str, language: &str, framework: Option<&str>,
    compat_mode: &str, fidelity: &str, scope: &str, target_dir: &Option<String>,
    db_path: &str, model: Option<String>, api_key: Option<String>,
    mode: CliMode,
) -> Result<()> {
    set_api_key(api_key.clone());
    println!("{}", tr_global!("cli.consume_step1", sdef));

    let cm = match compat_mode {
        "full" => CompatibilityMode::Full,
        "mixed" => CompatibilityMode::Mixed,
        "clean" => CompatibilityMode::Clean,
        _ => CompatibilityMode::Mixed,
    };
    let fid = match fidelity {
        "high" => Fidelity::High,
        "low" => Fidelity::Low,
        _ => Fidelity::Medium,
    };

    let rt = tokio::runtime::Runtime::new().context(tr_global!("cli.error_runtime"))?;
    rt.block_on(async {
        use cleanroom_db::Database;
        // First, apply migrations to the DB file. We open it once via
        // `Database::open_with_migrations_from` (which runs every
        // `migrations/*.sql` that's newer than the DB's recorded
        // version), then drop the handle. Without this, the next line
        // — `rusqlite::Connection::open(db_path)` for the
        // `SdefImporter` — would land on an empty DB with no
        // `sdef_documents` / `data_models` / ... tables and the import
        // would fail with `no such table: sdef_documents` (we hit
        // this on the Phase 0.10.4 end-to-end #3 run, 2026-06-02).
        let _apply = Database::open_with_migrations_from(
            Path::new(db_path),
            Some(&cli_migrations_dir()),
        )?;
        // Now read the S.DEF and import it.
        let sdef_content = std::fs::read_to_string(sdef)?;
        let sdef: sdef_core::SoftwareDefinition = serde_json::from_str(&sdef_content)?;
        let importer = cleanroom_db::export_import::SdefImporter::new(
            rusqlite::Connection::open(db_path)?,
        );
        importer.import(&sdef)?;

        // Re-open as a managed `Database` for the consumer / validator.
        let db = Arc::new(Database::open_with_migrations_from(
            Path::new(db_path),
            Some(&cli_migrations_dir()),
        )?);
        let use_legacy_template = matches!(mode, CliMode::Template);
        // Phase 1.3: parse the `--scope` flag (`whole` | `module=<name>`
        // | `function=<name>`) into a `ConsumeScope` enum.
        let scope = parse_consume_scope(scope);
        // Phase 1.2: infer the target project skeleton from
        // `--target-dir` if the caller passed one. Falls back to
        // "no manifest" (None) when omitted — the LLM still gets
        // language-only context, the pre-1.2 behavior.
        let target_manifest = target_dir.as_deref().and_then(|d| {
            let path = std::path::Path::new(d);
            if path.is_dir() {
                Some(cleanroom_agent::target_manifest::infer_manifest(path))
            } else {
                tracing::warn!(
                    target_dir = %d,
                    "--target-dir is not a directory; ignoring for infer_manifest"
                );
                None
            }
        });
        let consumer_config = ConsumerConfig {
            language: language.to_string(),
            framework: framework.map(|s| s.to_string()),
            compatibility_mode: cm,
            fidelity: fid,
            output_path: Path::new(output).to_path_buf(),
            use_legacy_template,
            llm: None,
            loop_config: LoopConfig::default(),
            target_manifest,
            scope,
        };
        let mut consumer = ConsumerAgent::new(consumer_config, db.clone());
        if !use_legacy_template {
            let llm = build_llm_from_env(model.as_deref(), api_key.as_deref())?;
            consumer = consumer.with_llm(llm);
            // Phase 0.9 (closed 2026-06-02): wire the LLM call audit
            // log so consume-side LLM calls show up in `llm_call_log`
            // alongside the producer-side ones. Mirrors the producer
            // wiring in `run_produce_one_pass` (see L:1311-1318).
            let consume_log_repo = Arc::new(
                cleanroom_db::LlmCallLogRepository::new_with_arc(db.connection_arc()),
            );
            consumer = consumer.with_llm_call_logger(consume_log_repo);
        }
        consumer.run_consume().await.map_err(|e| anyhow::anyhow!("consume failed: {e}"))?;

        // Run completeness validation
        let validator = CompletenessValidator::new(db);
        match validator.validate("") {
            Ok(report) => println!("{}", format_report(&report)),
            Err(_) => {}
        }
        Ok(())
    })
}

/// Handler for the `serve` command.
///
/// Starts the MCP server for external integrations.
fn serve_command(transport: &str, db_path: &str) -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context(tr_global!("cli.error_runtime"))?;
    rt.block_on(async {
        let server = cleanroom_mcp::CleanroomMcpServer::new(Path::new(db_path))
            .context(tr_global!("cli.error_mcp_server"))?;
        println!("{}", tr_global!("cli.serve_starting", transport));

        if transport.starts_with("tcp://") {
            // TCP transport — cross-platform, enables CLI task queue access
            server.serve_tcp(transport).await?;
        } else {
            // Default: stdio transport for IDE/LLM integration
            server.serve().await?;
        }
        Ok(())
    })
}

/// Handler for the `resume` command.
///
/// Restarts agent from last checkpoint, optionally retrying failed tasks.
fn resume_command(document: &str, retry_failed: bool, db_path: &str) -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context(tr_global!("cli.error_runtime"))?;
    rt.block_on(async {
        let agent_config = AgentConfig {
            db_path: Path::new(db_path).to_path_buf(),
            ..AgentConfig::default()
        };
        let agent = CleanroomAgent::new(agent_config)
            .context(tr_global!("cli.error_runtime"))?;

        agent.run(RunMode::Resume {
            document: document.to_string(),
            retry_failed,
        }).await?;
        Ok(())
    })
}

/// Handler for the `inspect` command.
///
/// Runs database diagnostics based on check type:
/// - `consistency`: Fingerprint mismatch detection
/// - `coverage`: Entity counts across all tables
/// - `progress`: Task status distribution
fn inspect_command(check_type: &str, db_path: &str) -> Result<()> {
    let db = match Database::open_with_migrations_from(
        Path::new(db_path),
        Some(&cli_migrations_dir()),
    ) {
        Ok(db) => db,
        Err(_) => {
            println!("{}", tr_global!("cli.inspect_no_db", db_path));
            return Ok(());
        }
    };

    println!("{}", tr_global!("cli.inspect_header"));
    println!("{}", tr_global!("cli.inspect_db", db_path));

    match check_type {
        "consistency" => {
            let conn = db.connection();
            let mut stmt = conn.prepare(
                "SELECT COUNT(*) FROM fingerprints WHERE sdef_hash != db_hash OR db_hash != code_hash"
            ).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let inconsistent: i64 = stmt.query_row([], |row| row.get(0)).unwrap_or(0);
            println!("{}", tr_global!("cli.inspect_inconsistent", inconsistent));

            let mut stmt = conn.prepare("SELECT COUNT(*) FROM fingerprints")
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let total: i64 = stmt.query_row([], |row| row.get(0)).unwrap_or(0);
            println!("{}", tr_global!("cli.inspect_total_fp", total));
            if total > 0 {
                let pct = 100.0 * (total - inconsistent) as f64 / total as f64;
                println!("{}", tr_global!("cli.inspect_consistency", pct));
            }
        }
        "coverage" => {
            let conn = db.connection();
            let models: i64 = conn.query_row("SELECT COUNT(*) FROM data_models", [], |r| r.get(0)).unwrap_or(0);
            let attrs: i64 = conn.query_row("SELECT COUNT(*) FROM data_attributes", [], |r| r.get(0)).unwrap_or(0);
            let contracts: i64 = conn.query_row("SELECT COUNT(*) FROM contracts", [], |r| r.get(0)).unwrap_or(0);
            let functions: i64 = conn.query_row("SELECT COUNT(DISTINCT document_name || '|' || name) FROM function_specs", [], |r| r.get(0)).unwrap_or(0);
            let symbols: i64 = conn.query_row("SELECT COUNT(DISTINCT document_name || '|' || sdef_uri || '|' || language) FROM symbol_registry", [], |r| r.get(0)).unwrap_or(0);

            println!("{}", tr_global!("cli.inspect_coverage"));
            println!("{}", tr_global!("cli.inspect_data_models", models));
            println!("{}", tr_global!("cli.inspect_attributes", attrs));
            println!("{}", tr_global!("cli.inspect_contracts", contracts));
            println!("{}", tr_global!("cli.inspect_functions", functions));
            println!("{}", tr_global!("cli.inspect_symbols", symbols));
        }
        "progress" => {
            let conn = db.connection();
            let mut stmt = conn.prepare(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
            ).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            }).map_err(|e| anyhow::anyhow!(e.to_string()))?;

            println!("{}", tr_global!("cli.inspect_task_progress"));
            let mut total = 0i64;
            let mut results = Vec::new();
            for row in rows.flatten() {
                results.push(row);
                total += results.last().unwrap().1;
            }
            for (status, count) in &results {
                let pct = if total > 0 { 100.0 * *count as f64 / total as f64 } else { 0.0 };
                println!("{}", tr_global!("cli.inspect_task_line", status, count, pct));
            }
        }
        _ => {
            println!("{}", tr_global!("cli.inspect_unknown_check", check_type));
        }
    }
    Ok(())
}

/// Phase 0.9: handler for `cleanroom-cli inspect llm-log`.
///
/// Three mutually-exclusive filters:
/// - `--task-id <UUID>` — list every LLM call for a single task
///   (oldest first)
/// - `--recent N` — list the N most recent calls across all tasks
///   (newest first)
/// - (no flag) — same as `--recent 20` if neither is supplied
///
/// `--agent-type` further filters by `producer` / `consumer` / `meta`
/// when set. `--format` switches between human-readable text (default)
/// and newline-delimited JSON for piping into `jq`.
fn llm_log_command(
    task_id: Option<&str>,
    recent: Option<usize>,
    agent_type: Option<&str>,
    format: &str,
    db_path: &str,
) -> Result<()> {
    use cleanroom_db::LlmCallLogRepository;
    let db = Database::open_with_migrations_from(
        Path::new(db_path),
        Some(&cli_migrations_dir()),
    )?;
    let repo = LlmCallLogRepository::new_with_arc(db.connection_arc());
    let mut rows = if let Some(tid) = task_id {
        repo.list_by_task(tid)?
    } else {
        let n = recent.unwrap_or(20);
        repo.list_recent(n)?
    };
    if let Some(at) = agent_type {
        rows.retain(|r| r.agent_type == at);
    }
    if rows.is_empty() {
        println!("(no LLM call log entries match the filter)");
        return Ok(());
    }
    let total_cost: f64 = rows.iter().map(|r| r.cost_estimate_usd).sum();
    let total_prompt: u32 = rows.iter().map(|r| r.prompt_tokens).sum();
    let total_completion: u32 = rows.iter().map(|r| r.completion_tokens).sum();
    let total_duration_ms: u64 = rows.iter().map(|r| r.duration_ms).sum();
    match format {
        "json" => {
            for r in &rows {
                let json = serde_json::to_string(r).unwrap_or_default();
                println!("{json}");
            }
        }
        _ => {
            for r in &rows {
                println!("call_id: {}", r.call_id);
                println!("  task_id:     {}", r.task_id.as_deref().unwrap_or("-"));
                println!("  agent_type:  {}", r.agent_type);
                println!("  app_name:    {}", r.app_name.as_deref().unwrap_or("-"));
                println!("  model:       {}", r.model.as_deref().unwrap_or("-"));
                println!("  status:      {}", r.status);
                if let Some(err) = &r.error {
                    println!("  error:       {err}");
                }
                println!(
                    "  tokens:      {} prompt + {} completion",
                    r.prompt_tokens, r.completion_tokens
                );
                println!("  duration_ms: {}", r.duration_ms);
                println!("  iterations:  {}", r.iterations);
                println!("  tool_calls:  {}", r.tool_calls);
                println!("  cost_usd:    ${:.6}", r.cost_estimate_usd);
                println!("  created_at:  {}", r.created_at);
                println!();
            }
            println!(
                "---- {} call(s) | {} prompt + {} completion tok | {} ms | ${:.6} total ----",
                rows.len(),
                total_prompt,
                total_completion,
                total_duration_ms,
                total_cost
            );
        }
    }
    Ok(())
}

/// Handler for the `export` command.
///
/// Serializes a S.DEF document from the database to JSON or YAML.
fn export_command(document: &str, output: &str, format: &str, db_path: &str) -> Result<()> {
    use std::io::Write;

    // `Database::open` would call `run_migrations` which resolves the
    // migrations directory at CWD (`target/debug/migrations, migrations,
    // ../migrations`). That fails when the binary is invoked from
    // outside the workspace root. Pass the workspace `migrations/`
    // explicitly to skip the CWD probe.
    let db = Database::open_with_migrations_from(
        Path::new(db_path),
        Some(&cli_migrations_dir()),
    )?;
    let conn = db.connection();

    let mut stmt = conn.prepare(
        "SELECT name, version, description FROM sdef_documents WHERE name = ?1"
    ).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let (name, version, description): (String, Option<String>, Option<String>) = stmt.query_row(
        rusqlite::params![document],
        |row| Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
        ))
    ).map_err(|_e| anyhow::anyhow!(tr_global!("cli.error_no_doc_in_db")))?;

    drop(stmt);

    let sdef = build_export_sdef(&conn, &name, version, description)?;

    let output_dir = Path::new(output).parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(output_dir)
        .context(tr_global!("cli.error_output_dir"))?;

    match format {
        "yaml" | "yml" => {
            let yaml = serde_yaml::to_string(&sdef)
                .context(tr_global!("cli.error_serialize_yaml"))?;
            let mut file = std::fs::File::create(output)
                .context(tr_global!("cli.error_output_file"))?;
            file.write_all(yaml.as_bytes())?;
        }
        _ => {
            let json = serde_json::to_string_pretty(&sdef)
                .context(tr_global!("cli.error_serialize_json"))?;
            let mut file = std::fs::File::create(output)
                .context(tr_global!("cli.error_output_file"))?;
            file.write_all(json.as_bytes())?;
        }
    }

    println!("{}", tr_global!("cli.export_success", document, output));
    Ok(())
}

/// Builds a complete [`sdef_core::SoftwareDefinition`] from database contents.
///
/// Reconstructs a full S.DEF document by querying the database for:
/// - Data models and their attributes
/// - Design decisions
/// - Architecture layers
/// - Function specs with input/output parameters
/// - Interface contracts with methods and invariants
fn build_export_sdef(
    conn: &rusqlite::Connection,
    name: &str,
    version: Option<String>,
    description: Option<String>,
) -> Result<sdef_core::SoftwareDefinition> {
    let mut sdef = sdef_core::SoftwareDefinition::default();
    sdef.sdef_version = sdef_core::CURRENT_SCHEMA_VERSION.to_string();
    sdef.name = name.to_string();
    sdef.version = version;
    sdef.description = description;

    // 1. Data models
    let mut stmt = conn.prepare(
        "SELECT entity, status, version, description, logical_model FROM data_models WHERE document_name = ?1"
    ).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut rows = stmt.query(rusqlite::params![name])
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut models = Vec::new();
    while let Some(row) = rows.next().map_err(|e| anyhow::anyhow!(e.to_string()))? {
        let entity: String = row.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let dm_status: Option<String> = row.get(1).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let dm_version: Option<String> = row.get(2).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let dm_description: Option<String> = row.get(3).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut attr_stmt = conn.prepare(
            "SELECT name, attr_type, format, description, required, identity, generated, unique_flag, internal, deprecated
             FROM data_attributes WHERE document_name = ?1 AND entity = ?2"
        ).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut attr_rows = attr_stmt.query(rusqlite::params![name, &entity])
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut attrs = Vec::new();
        while let Some(ar) = attr_rows.next().map_err(|e| anyhow::anyhow!(e.to_string()))? {
            attrs.push(sdef_core::DataAttribute {
                name: ar.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                attr_type: ar.get(1).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                format: ar.get(2).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                description: ar.get(3).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                required: ar.get(4).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                default: None,
                identity: ar.get(5).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                generated: ar.get(6).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                unique: ar.get(7).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                internal: ar.get(8).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                deprecated: ar.get(9).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                compatibility: None,
                constraints: None,
                origin: None,
            });
        }
        drop(attr_rows);
        drop(attr_stmt);

        models.push(sdef_core::DataModel {
            entity,
            status: dm_status,
            version: dm_version,
            deprecated: None,
            description: dm_description,
            logical_model: None,
            attributes: if attrs.is_empty() { None } else { Some(attrs) },
            relationships: None,
            validation_rules: None,
            physical_design: None,
            origin: None,
        });
    }
    drop(rows);
    drop(stmt);

    if !models.is_empty() {
        sdef.data_models = Some(models);
    }

    let dm_count = sdef.data_models.as_ref().map(|v| v.len()).unwrap_or(0);
    println!("{}", tr_global!("cli.export_data_models", dm_count));

    // 2. Design decisions
    let mut dd_stmt = conn.prepare(
        "SELECT id, topic, decision, rationale, context, alternatives_json, consequences_json, constraints_json
         FROM design_decisions WHERE document_name = ?1"
    ).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut dd_rows = dd_stmt.query(rusqlite::params![name])
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut decisions = Vec::new();
    while let Some(row) = dd_rows.next().map_err(|e| anyhow::anyhow!(e.to_string()))? {
        let alternatives: Option<Vec<String>> = row.get::<_, Option<String>>(5)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .and_then(|s| serde_json::from_str(&s).ok());
        let consequences: Option<Vec<String>> = row.get::<_, Option<String>>(6)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .and_then(|s| serde_json::from_str(&s).ok());
        let constraints: Option<Vec<String>> = row.get::<_, Option<String>>(7)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .and_then(|s| serde_json::from_str(&s).ok());

        decisions.push(sdef_core::DesignDecision {
            id: row.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            topic: row.get(1).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            decision: row.get(2).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            rationale: row.get(3).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            context: row.get(4).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            alternatives,
            consequences,
            constraints,
        });
    }
    drop(dd_rows);
    drop(dd_stmt);
    if !decisions.is_empty() {
        sdef.design_decisions = Some(decisions);
    }

    // 3. Architecture layers
    let mut arch_stmt = conn.prepare(
        "SELECT layer_name, components_json FROM architecture_layers WHERE document_name = ?1"
    ).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut arch_rows = arch_stmt.query(rusqlite::params![name])
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut layers = Vec::new();
    while let Some(row) = arch_rows.next().map_err(|e| anyhow::anyhow!(e.to_string()))? {
        let components: Option<Vec<String>> = row.get::<_, Option<String>>(1)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .and_then(|s| serde_json::from_str(&s).ok());
        layers.push(sdef_core::ArchitectureLayer {
            name: row.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            components,
        });
    }
    drop(arch_rows);
    drop(arch_stmt);
    if !layers.is_empty() {
        sdef.architecture = Some(sdef_core::Architecture {
            style: None,
            rationale: None,
            layers: Some(layers),
            modules: None,
            communication: None,
            cross_cutting_concerns: None,
        });
    }

    // 4. Functions
    let mut fn_stmt = conn.prepare(
        "SELECT id, name, description, logic, complexity, pure_function
         FROM function_specs WHERE document_name = ?1 ORDER BY id"
    ).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut fn_rows = fn_stmt.query(rusqlite::params![name])
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut functions = Vec::new();
    while let Some(row) = fn_rows.next().map_err(|e| anyhow::anyhow!(e.to_string()))? {
        let func_id: i64 = row.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        // Query input params
        let mut in_stmt = conn.prepare(
            "SELECT name, param_type, description FROM function_params
             WHERE function_id = ?1 AND param_direction = 'input'"
        ).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let mut in_rows = in_stmt.query(rusqlite::params![func_id])
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let mut inputs = Vec::new();
        while let Some(ir) = in_rows.next().map_err(|e| anyhow::anyhow!(e.to_string()))? {
            inputs.push(sdef_core::FunctionParam {
                name: ir.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                param_type: ir.get(1).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                description: ir.get(2).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            });
        }
        drop(in_rows);
        drop(in_stmt);

        // Query output params
        let mut out_stmt = conn.prepare(
            "SELECT name, param_type, description FROM function_params
             WHERE function_id = ?1 AND param_direction = 'output'"
        ).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let mut out_rows = out_stmt.query(rusqlite::params![func_id])
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let mut outputs = Vec::new();
        while let Some(or) = out_rows.next().map_err(|e| anyhow::anyhow!(e.to_string()))? {
            outputs.push(sdef_core::FunctionParam {
                name: or.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                param_type: or.get(1).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                description: or.get(2).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            });
        }
        drop(out_rows);
        drop(out_stmt);

        functions.push(sdef_core::FunctionSpec {
            name: row.get(1).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            description: row.get(2).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            inputs: if inputs.is_empty() { None } else { Some(inputs) },
            outputs: if outputs.is_empty() { None } else { Some(outputs) },
            logic: row.get(3).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            complexity: row.get(4).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            pure_function: row.get(5).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            edge_cases: None,
            origin: None,
        });
    }
    drop(fn_rows);
    drop(fn_stmt);
    if !functions.is_empty() {
        sdef.behavior = Some(sdef_core::Behavior {
            functions: Some(functions),
            flows: None,
            state_machines: None,
        });
    }

    // 5. Contracts (interfaces)
    let mut c_stmt = conn.prepare(
        "SELECT name, is_abstract, status, version, description, invariants_json
         FROM contracts WHERE document_name = ?1 AND contract_type = 'interface'"
    ).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut c_rows = c_stmt.query(rusqlite::params![name])
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut interfaces = Vec::new();
    while let Some(row) = c_rows.next().map_err(|e| anyhow::anyhow!(e.to_string()))? {
        let cname: String = row.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        // Query methods
        let mut m_stmt = conn.prepare(
            "SELECT signature, status, behavior, preconditions_json, postconditions_json, errors_json
             FROM contract_methods WHERE document_name = ?1 AND contract_name = ?2"
        ).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut m_rows = m_stmt.query(rusqlite::params![name, &cname])
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut methods = Vec::new();
        while let Some(mr) = m_rows.next().map_err(|e| anyhow::anyhow!(e.to_string()))? {
            let preconds: Option<Vec<String>> = mr.get::<_, Option<String>>(3)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .and_then(|s| serde_json::from_str(&s).ok());
            let postconds: Option<Vec<String>> = mr.get::<_, Option<String>>(4)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .and_then(|s| serde_json::from_str(&s).ok());
            let errors: Option<Vec<String>> = mr.get::<_, Option<String>>(5)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .and_then(|s| serde_json::from_str(&s).ok());

            methods.push(sdef_core::ContractMethod {
                signature: mr.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                status: mr.get(1).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                deprecated: None,
                behavior: mr.get(2).map_err(|e| anyhow::anyhow!(e.to_string()))?,
                preconditions: preconds,
                postconditions: postconds,
                errors,
                origin: None,
            });
        }
        drop(m_rows);
        drop(m_stmt);

        let invariants: Option<Vec<String>> = row.get::<_, Option<String>>(5)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .and_then(|s| serde_json::from_str(&s).ok());

        interfaces.push(sdef_core::InterfaceContract {
            name: cname,
            is_abstract: row.get::<_, bool>(1).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            status: row.get(2).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            version: row.get(3).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            deprecated: None,
            description: row.get(4).map_err(|e| anyhow::anyhow!(e.to_string()))?,
            methods: if methods.is_empty() { None } else { Some(methods) },
            invariants,
            origin: None,
        });
    }
    drop(c_rows);
    drop(c_stmt);
    if !interfaces.is_empty() {
        sdef.contracts = Some(sdef_core::Contracts {
            interfaces: Some(interfaces),
            classes: None,
            enums: None,
            apis: None,
            compatibility_modules: None,
            data_migrations: None,
        });
    }

    let dm_len = sdef.data_models.as_ref().map(|v| v.len()).unwrap_or(0);
    let dd_len = sdef.design_decisions.as_ref().map(|v| v.len()).unwrap_or(0);
    let iface_len = sdef.contracts.as_ref().and_then(|c| c.interfaces.as_ref()).map(|v| v.len()).unwrap_or(0);
    let fn_len = sdef.behavior.as_ref().and_then(|b| b.functions.as_ref()).map(|v| v.len()).unwrap_or(0);

    println!("Exported {} data models, {} design decisions, {} interfaces, {} functions",
        dm_len, dd_len, iface_len, fn_len);
    Ok(sdef)
}

/// Parses and imports a S.DEF file into the database.
///
/// Handles both JSON and YAML formats, deserializes the document,
/// and uses [`SdefImporter`] to load all entities into the database.
fn import_sdef_file(file: &str, db_path: &str) -> Result<String> {
    let content = std::fs::read_to_string(file)
        .context(tr_global!("cli.import_fail_read"))?;

    let sdef: sdef_core::SoftwareDefinition = if file.ends_with(".yaml") || file.ends_with(".yml") {
        serde_yaml::from_str(&content)
            .context(tr_global!("cli.import_fail_parse_yaml"))?
    } else {
        serde_json::from_str(&content)
            .context(tr_global!("cli.import_fail_parse_json"))?
    };

    let _db = Database::open_with_migrations_from(
        Path::new(db_path),
        Some(&cli_migrations_dir()),
    )?;
    // Use the export_import importer for full data model + contract import
    let importer = cleanroom_db::export_import::SdefImporter::new(
        rusqlite::Connection::open(db_path)?,
    );
    importer.import(&sdef)?;

    let model_count = sdef.data_models.as_ref().map(|v| v.len()).unwrap_or(0);
    println!("{}", tr_global!("cli.import_success", sdef.name, model_count));
    Ok(sdef.name)
}

/// Handler for the `import` command.
///
/// Wrapper around [`import_sdef_file()`] that discards the returned document name.
fn import_command(file: &str, db_path: &str) -> Result<()> {
    import_sdef_file(file, db_path)?;
    Ok(())
}

/// Handler for the `migrate` command.
///
/// Runs database migrations. Currently only supports `up` direction.
fn migrate_command(direction: &str, db_path: &str) -> Result<()> {
    match direction {
        "up" => {
            let _db = Database::open_with_migrations_from(
                Path::new(db_path),
                Some(&cli_migrations_dir()),
            )?;
            println!("{}", tr_global!("cli.migrate_success"));
        }
        "down" => {
            println!("{}", tr_global!("cli.migrate_down_unsupported"));
        }
        _ => {
            println!("{}", tr_global!("cli.migrate_unknown_direction", direction));
        }
    }
    Ok(())
}

/// Handler for the `upgrade` command.
///
/// Compares two git refs and produces a detailed version upgrade report.
/// Optionally applies detected changes to the database.
fn upgrade_command(
    old_version: &str, new_version: &str, repo: &str,
    document: Option<&str>, apply: bool, db_path: &str,
) -> Result<()> {
    let db = Arc::new(Database::open_with_migrations_from(
        Path::new(db_path),
        Some(&cli_migrations_dir()),
    )?);
    let _doc_name = document.unwrap_or("default");

    println!("{}", tr_global!("cli.upgrade_running", old_version, new_version));

    let analyzer = VersionUpgradeAnalyzer::new(db.clone(), repo);
    let report = analyzer.analyze(old_version, new_version)
        .context(anyhow::anyhow!("Version upgrade analysis failed"))?;

    println!("{}", tr_global!("cli.upgrade_summary"));
    println!("{}", tr_global!("cli.upgrade_files_added", report.added_files.len()));
    println!("{}", tr_global!("cli.upgrade_files_modified", report.modified_files.len()));
    println!("{}", tr_global!("cli.upgrade_files_deleted", report.deleted_files.len()));
    println!("{}", tr_global!("cli.upgrade_breaking", report.breaking_changes.len()));

    for change in &report.breaking_changes {
        println!("  - {}", change.description);
    }

    println!("{}", tr_global!("cli.upgrade_deprecated", report.deprecated_entities.len()));
    for entity in &report.deprecated_entities {
        println!("{}", tr_global!("cli.upgrade_entity", entity));
    }

    println!("{}", tr_global!("cli.upgrade_compat_layers", report.new_compat_layers.len()));
    for layer in &report.new_compat_layers {
        println!("{}", tr_global!("cli.upgrade_entity", layer));
    }

    println!("{}", tr_global!("cli.upgrade_migrations", report.suggested_migrations.len()));
    for m in &report.suggested_migrations {
        println!("{}", tr_global!("cli.upgrade_entity", m.from_entity));
        println!("{}", tr_global!("cli.upgrade_entity_new", m.to_entity));
    }

    if apply {
        analyzer.apply_upgrade(&report)?;
        println!("{}", tr_global!("cli.upgrade_applied", old_version, new_version));
    }

    Ok(())
}

/// Handler for the `evaluate` command.
///
/// Runs the evaluation suite against built-in benchmark projects
/// and outputs a quality report.
fn evaluate_command(
    benchmark: Option<&str>,
    output: Option<&str>,
    db_path: &str,
) -> Result<()> {
    use cleanroom_agent::evaluation::{EvaluationRunner, EvalConfig, BenchmarkSuite};
    use cleanroom_db::Database;

    let db = Arc::new(
        Database::open_with_migrations_from(
            std::path::Path::new(db_path),
            Some(&cli_migrations_dir()),
        )
        .context("Failed to open database")?,
    );

    let output_dir = output
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("./eval-reports"));

    std::fs::create_dir_all(&output_dir)
        .context("Failed to create output directory")?;

    let config = EvalConfig {
        output_dir: output_dir.clone(),
        ..EvalConfig::default()
    };

    let runner = EvaluationRunner::new(config, db.clone());

    let mut suite = BenchmarkSuite::builtin();
    if let Some(name) = benchmark {
        suite.projects.retain(|p| p.name == name);
        if suite.projects.is_empty() {
            anyhow::bail!(
                "Unknown benchmark: '{}'. Available: redis, express, hugo",
                name
            );
        }
    }

    println!("Running evaluation on {} benchmark project(s)...", suite.projects.len());

    let rt = tokio::runtime::Runtime::new()
        .context("Failed to create tokio runtime")?;

    let report = rt.block_on(runner.run(&suite))
        .context("Evaluation failed")?;

    // Write report to output file
    let report_path = output_dir.join(format!("evaluation-{}.json", &report.run_id[..8]));
    let report_json = serde_json::to_string_pretty(&report)
        .context("Failed to serialize report")?;
    std::fs::write(&report_path, &report_json)
        .context("Failed to write report file")?;

    // Print summary
    if let Some(ref summary) = report.summary {
        println!(
            "Evaluation: {} projects | Fidelity: {:.1}% | Coverage: {:.1}% | Compile: {:.1}%",
            summary.projects_evaluated,
            summary.overall_fidelity * 100.0,
            summary.overall_coverage * 100.0,
            summary.overall_compile_rate * 100.0,
        );

        if !summary.degraded_projects.is_empty() {
            println!(
                "WARNING: Degraded projects: {}",
                summary.degraded_projects.join(", ")
            );
        }
    }

    println!("Report saved to: {}", report_path.display());
    Ok(())
}
