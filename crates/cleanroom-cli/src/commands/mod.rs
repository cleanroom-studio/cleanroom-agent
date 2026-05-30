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

use std::path::Path;
use std::sync::Arc;
use anyhow::{Result, Context};
use clap::Subcommand;
use cleanroom_agent::{
    AgentConfig, CleanroomAgent, RunMode,
    CompatibilityMode, Fidelity, CompletenessValidator, format_report,
    VersionUpgradeAnalyzer,
};
use cleanroom_db::Database;
use cleanroom_i18n::tr_global;

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
        /// LLM model (e.g. gemini-2.5-flash)
        #[arg(long)]
        model: Option<String>,
        /// API key for LLM provider
        #[arg(long)]
        api_key: Option<String>,
    },

    /// Start MCP server for external tool integrations.
    ///
    /// The Model Context Protocol (MCP) server allows external tools like
    /// editors, IDEs, and other agents to interact with the Cleanroom Agent.
    ///
    /// # Transport Types
    ///
    /// - `stdio` — Standard input/output (default, for local integration)
    ///
    /// # Example
    ///
    /// ```bash
    /// cleanroom serve --transport stdio
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
    /// ```
    Inspect {
        #[arg(long, default_value = "consistency")]
        check_type: String,
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
        Commands::Produce { repo, output, exclude: _, name, model, api_key } => {
            produce_command(&repo, &output, db_path, name, model, api_key)
        }
        Commands::Consume { sdef, output, language, framework, compat_mode, fidelity, model, api_key } => {
            consume_command(&sdef, &output, &language, framework.as_deref(), &compat_mode, &fidelity, db_path, model, api_key)
        }
        Commands::Serve { transport } => {
            serve_command(&transport, db_path)
        }
        Commands::Resume { document, retry_failed } => {
            resume_command(&document, retry_failed, db_path)
        }
        Commands::Inspect { check_type } => {
            inspect_command(&check_type, db_path)
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
    }
}

fn set_api_key(key: Option<String>) {
    if let Some(k) = key {
        if std::env::var("GOOGLE_API_KEY").is_err() {
            std::env::set_var("GOOGLE_API_KEY", k);
        }
    }
}

/// Handler for the `produce` command.
///
/// Scans the repository, runs LLM analysis via ADK, and outputs S.DEF JSON.
fn produce_command(
    repo: &str, output: &str, db_path: &str,
    name: Option<String>, model: Option<String>, api_key: Option<String>,
) -> Result<()> {
    set_api_key(api_key);
    use tokio::runtime::Runtime;
    let project_name = name.unwrap_or_else(|| {
        Path::new(repo).file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string())
    });

    let rt = Runtime::new().context(tr_global!("cli.error_runtime"))?;
    rt.block_on(async {
        let agent_config = AgentConfig {
            db_path: Path::new(db_path).to_path_buf(),
            model_name: model,
            agent_name: "cleanroom-producer".to_string(),
            ..AgentConfig::default()
        };
        let agent = CleanroomAgent::new(agent_config)
            .context(tr_global!("cli.error_orchestrator"))?;

        let pn = project_name.clone();
        agent.run(RunMode::Produce {
            repo_path: Path::new(repo).to_path_buf(),
            output_path: Path::new(output).to_path_buf(),
            project_name,
        }).await?;

        println!("{}", tr_global!("cli.produce_complete", pn));
        Ok(())
    })
}

/// Handler for the `consume` command.
///
/// Loads S.DEF, generates code via LLM, and validates output completeness.
fn consume_command(
    sdef: &str, output: &str, language: &str, framework: Option<&str>,
    compat_mode: &str, fidelity: &str, db_path: &str,
    model: Option<String>, api_key: Option<String>,
) -> Result<()> {
    set_api_key(api_key);
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
        let agent_config = AgentConfig {
            db_path: Path::new(db_path).to_path_buf(),
            model_name: model,
            agent_name: "cleanroom-consumer".to_string(),
            ..AgentConfig::default()
        };
        let agent = CleanroomAgent::new(agent_config)
            .context(tr_global!("cli.error_orchestrator"))?;

        agent.run(RunMode::Consume {
            sdef_path: Path::new(sdef).to_path_buf(),
            output_path: Path::new(output).to_path_buf(),
            language: language.to_string(),
            framework: framework.map(|s| s.to_string()),
            compat_mode: cm,
            fidelity: fid,
        }).await?;

        // Run completeness validation
        let validator = CompletenessValidator::new(agent.db().clone());
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
        server.serve().await?;
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
    let db = match Database::open(Path::new(db_path)) {
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

/// Handler for the `export` command.
///
/// Serializes a S.DEF document from the database to JSON or YAML.
fn export_command(document: &str, output: &str, format: &str, db_path: &str) -> Result<()> {
    use std::io::Write;

    let db = Database::open(Path::new(db_path))?;
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

    let _db = Database::open(Path::new(db_path))?;
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
            let _db = Database::open(Path::new(db_path))?;
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
    let db = Arc::new(Database::open(Path::new(db_path))?);
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
