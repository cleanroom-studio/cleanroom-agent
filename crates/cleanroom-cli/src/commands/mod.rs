//! CLI commands.

use std::path::Path;
use anyhow::{Result, Context};
use clap::Subcommand;
use cleanroom_agent::{Orchestrator, OrchestratorConfig, ProducerAgent, ProducerConfig, ConsumerAgent, ConsumerConfig, CompatibilityMode, Fidelity, PipelineResult};
use cleanroom_db::{Database, TaskRepository, TaskStatus, TaskType};

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
    },

    /// Consumption mode: read S.DEF → generate code
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
    },

    /// MCP server mode
    Serve {
        #[arg(long, default_value = "stdio")]
        transport: String,
    },

    /// Resume workflow from checkpoint
    Resume {
        /// Document name to resume
        #[arg(long)]
        document: String,
        /// Resume failed tasks too
        #[arg(long, default_value = "false")]
        retry_failed: bool,
    },

    /// Inspect database/S.DEF state
    Inspect {
        #[arg(long, default_value = "consistency")]
        check_type: String,
    },

    /// Database migration
    Migrate {
        #[arg(long, default_value = "up")]
        direction: String,
    },
}

pub fn run(command: Commands, db_path: &str) -> Result<()> {
    match command {
        Commands::Produce { repo, output, exclude: _, name } => {
            produce_command(&repo, &output, db_path, name)
        }
        Commands::Consume { sdef, output, language, framework, compat_mode, fidelity } => {
            consume_command(&sdef, &output, &language, framework.as_deref(), &compat_mode, &fidelity, db_path)
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
        Commands::Migrate { direction } => {
            migrate_command(&direction, db_path)
        }
    }
}

fn produce_command(repo: &str, output: &str, db_path: &str, name: Option<String>) -> Result<()> {
    use tokio::runtime::Runtime;
    let project_name = name.unwrap_or_else(|| {
        Path::new(repo).file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string())
    });

    let rt = Runtime::new().context("Failed to create Tokio runtime")?;
    rt.block_on(async {
        let config = OrchestratorConfig {
            repo_path: Path::new(repo).to_path_buf(),
            output_path: Path::new(output).to_path_buf(),
            db_path: Path::new(db_path).to_path_buf(),
            checkpoint_interval_secs: 600,
            agent_idle_timeout_secs: 300,
        };
        let orchestrator = Orchestrator::new(config).context("Failed to create orchestrator")?;
        orchestrator.start_workflow().await?;

        let producer = ProducerAgent::new(ProducerConfig::default(), orchestrator.db().clone());
        while let Ok(Some(task)) = producer.process_next_task().await {
            println!("Processed task: {}", task.task_id);
        }
        println!("Production completed for '{}'", project_name);
        Ok(())
    })
}

fn consume_command(sdef: &str, output: &str, language: &str, framework: Option<&str>, compat_mode: &str, fidelity: &str, db_path: &str) -> Result<()> {
    let compat = match compat_mode {
        "full" => CompatibilityMode::Full,
        "mixed" => CompatibilityMode::Mixed,
        "clean" => CompatibilityMode::Clean,
        "custom" => CompatibilityMode::Custom,
        _ => CompatibilityMode::Mixed,
    };
    let fid = match fidelity {
        "high" => Fidelity::High,
        "medium" => Fidelity::Medium,
        "low" => Fidelity::Low,
        _ => Fidelity::Medium,
    };
    let config = ConsumerConfig {
        language: language.to_string(),
        framework: framework.map(String::from),
        compatibility_mode: compat,
        fidelity: fid,
        output_path: Path::new(output).to_path_buf(),
    };
    let db = Database::open(Path::new(db_path))?;
    let consumer = ConsumerAgent::new(config, std::sync::Arc::new(db));
    println!("Consume: sdef={}, output={}, language={}", sdef, output, language);
    println!("Consumer agent created: {}", consumer.agent_id());
    Ok(())
}

fn serve_command(transport: &str, db_path: &str) -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context("Failed to create Tokio runtime")?;
    rt.block_on(async {
        let server = cleanroom_mcp::CleanroomMcpServer::new(Path::new(db_path))
            .context("Failed to create MCP server")?;
        println!("MCP server starting with {} transport...", transport);
        server.serve().await?;
        Ok(())
    })
}

fn resume_command(document: &str, retry_failed: bool, db_path: &str) -> Result<()> {
    let db = Database::open(Path::new(db_path))?;
    let repo = TaskRepository::new(db.connection_arc());

    // Find all tasks for this document
    let all_tasks = repo.list(None, None, None).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Filter by document name in input_json
    let doc_tasks: Vec<_> = all_tasks.iter().filter(|t| {
        t.input_json.contains(document)
    }).collect();

    if doc_tasks.is_empty() {
        println!("No tasks found for document '{}'", document);
        println!("Try: inspect document to see available documents");
        return Ok(());
    }

    // Separate completed, pending, and failed tasks
    let pending: Vec<_> = doc_tasks.iter().filter(|t| t.status == TaskStatus::Pending).collect();
    let in_progress: Vec<_> = doc_tasks.iter().filter(|t| matches!(t.status, TaskStatus::InProgress | TaskStatus::Assigned)).collect();
    let failed: Vec<_> = doc_tasks.iter().filter(|t| t.status == TaskStatus::Failed).collect();
    let completed: Vec<_> = doc_tasks.iter().filter(|t| t.status == TaskStatus::Completed).collect();

    println!("=== Workflow Summary for '{}' ===", document);
    println!("  Total tasks:      {}", doc_tasks.len());
    println!("  Completed:        {}", completed.len());
    println!("  In progress:      {}", in_progress.len());
    println!("  Pending:          {}", pending.len());
    println!("  Failed:           {}", failed.len());

    // Reset in_progress tasks back to pending
    for task in &in_progress {
        repo.update_status(&task.task_id, TaskStatus::Pending)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        println!("  Reset '{}' to pending", task.task_id);
    }

    // Optionally reset failed tasks
    if retry_failed {
        for task in &failed {
            repo.update_status(&task.task_id, TaskStatus::Pending)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            println!("  Retrying '{}'", task.task_id);
        }
    }

    println!("\nReady to resume. Run `cleanroom produce` to continue processing.");
    Ok(())
}

fn inspect_command(check_type: &str, db_path: &str) -> Result<()> {
    let db = Database::open(Path::new(db_path))?;
    println!("=== Cleanroom Inspector ===");
    println!("Database: {}", db_path);

    match check_type {
        "consistency" => {
            // Check for inconsistent fingerprints
            let conn = db.connection();
            let mut stmt = conn.prepare(
                "SELECT COUNT(*) FROM fingerprints WHERE sdef_hash != db_hash OR db_hash != code_hash"
            ).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let inconsistent: i64 = stmt.query_row([], |row| row.get(0))
                .unwrap_or(0);
            println!("Inconsistent fingerprints: {}", inconsistent);

            let mut stmt = conn.prepare(
                "SELECT COUNT(*) FROM fingerprints"
            ).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let total: i64 = stmt.query_row([], |row| row.get(0))
                .unwrap_or(0);
            println!("Total fingerprints: {}", total);
            if total > 0 {
                let pct = 100.0 * (total - inconsistent) as f64 / total as f64;
                println!("Consistency: {:.1}%", pct);
            }
        }
        "coverage" => {
            // Count data models and attributes
            let conn = db.connection();
            let models: i64 = conn.query_row("SELECT COUNT(*) FROM data_models", [], |r| r.get(0)).unwrap_or(0);
            let attrs: i64 = conn.query_row("SELECT COUNT(*) FROM data_attributes", [], |r| r.get(0)).unwrap_or(0);
            let contracts: i64 = conn.query_row("SELECT COUNT(*) FROM contracts", [], |r| r.get(0)).unwrap_or(0);
            let functions: i64 = conn.query_row("SELECT COUNT(*) FROM function_specs", [], |r| r.get(0)).unwrap_or(0);
            let symbols: i64 = conn.query_row("SELECT COUNT(*) FROM symbol_registry", [], |r| r.get(0)).unwrap_or(0);

            println!("S.DEF coverage:");
            println!("  Data models:    {}", models);
            println!("  Attributes:     {}", attrs);
            println!("  Contracts:      {}", contracts);
            println!("  Functions:      {}", functions);
            println!("  Symbols:        {}", symbols);
        }
        "progress" => {
            let conn = db.connection();
            let mut stmt = conn.prepare(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
            ).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            }).map_err(|e| anyhow::anyhow!(e.to_string()))?;

            println!("Task progress:");
            let mut total = 0i64;
            let mut results = Vec::new();
            for row in rows.flatten() {
                results.push(row);
                total += results.last().unwrap().1;
            }
            for (status, count) in &results {
                let pct = if total > 0 { 100.0 * *count as f64 / total as f64 } else { 0.0 };
                println!("  {:<20}: {:>4} ({:.1}%)", status, count, pct);
            }
        }
        _ => {
            println!("Unknown check type: {}", check_type);
        }
    }
    Ok(())
}

fn migrate_command(direction: &str, db_path: &str) -> Result<()> {
    match direction {
        "up" => {
            let _db = Database::open(Path::new(db_path))?;
            println!("Migrations applied successfully");
            Ok(())
        }
        "down" => {
            println!("Down migration not supported in this version");
            Ok(())
        }
        _ => {
            println!("Unknown migration direction: {}", direction);
            Ok(())
        }
    }
}