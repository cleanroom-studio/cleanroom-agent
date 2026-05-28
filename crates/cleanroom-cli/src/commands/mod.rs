//! CLI commands.

use std::path::Path;
use anyhow::{Result, Context};
use clap::Subcommand;
use cleanroom_agent::{Orchestrator, OrchestratorConfig, ProducerAgent, ProducerConfig, ConsumerAgent, ConsumerConfig, CompatibilityMode, Fidelity};
use cleanroom_db::Database;

#[derive(Subcommand)]
pub enum Commands {
    /// Production mode: analyze code repository → output S.DEF
    Produce {
        /// Repository path
        #[arg(long)]
        repo: String,

        /// Output directory
        #[arg(long, default_value = "./sdef-output")]
        output: String,

        /// Exclude patterns (gitignore-like)
        #[arg(long)]
        exclude: Option<String>,

        /// Project name
        #[arg(long)]
        name: Option<String>,
    },

    /// Consumption mode: read S.DEF → generate code
    Consume {
        /// S.DEF file or directory
        #[arg(long)]
        sdef: String,

        /// Output directory
        #[arg(long, default_value = "./output")]
        output: String,

        /// Target language
        #[arg(long)]
        language: String,

        /// Target framework
        #[arg(long)]
        framework: Option<String>,

        /// Compatibility mode
        #[arg(long, default_value = "mixed")]
        compat_mode: String,

        /// Reconstruction fidelity
        #[arg(long, default_value = "medium")]
        fidelity: String,
    },

    /// MCP server mode
    Serve {
        /// Transport (stdio/http)
        #[arg(long, default_value = "stdio")]
        transport: String,
    },

    /// Resume workflow
    Resume {
        /// Workflow ID
        #[arg(long)]
        workflow_id: String,
    },

    /// Inspect database/S.DEF state
    Inspect {
        /// Check type
        #[arg(long, default_value = "consistency")]
        check_type: String,
    },

    /// Database migration
    Migrate {
        /// Direction
        #[arg(long, default_value = "up")]
        direction: String,
    },
}

pub fn run(command: Commands, db_path: &str) -> Result<()> {
    match command {
        Commands::Produce { repo, output, exclude: _, name: _ } => {
            produce_command(&repo, &output, db_path)
        }
        Commands::Consume { sdef, output, language, framework, compat_mode, fidelity } => {
            consume_command(&sdef, &output, &language, framework.as_deref(), &compat_mode, &fidelity, db_path)
        }
        Commands::Serve { transport } => {
            serve_command(&transport, db_path)
        }
        Commands::Resume { workflow_id: _ } => {
            println!("Resume command not yet implemented");
            Ok(())
        }
        Commands::Inspect { check_type } => {
            inspect_command(&check_type, db_path)
        }
        Commands::Migrate { direction } => {
            migrate_command(&direction, db_path)
        }
    }
}

fn produce_command(repo: &str, output: &str, db_path: &str) -> Result<()> {
    use tokio::runtime::Runtime;
    
    let rt = Runtime::new().context("Failed to create Tokio runtime")?;
    
    rt.block_on(async {
        let config = OrchestratorConfig {
            repo_path: Path::new(repo).to_path_buf(),
            output_path: Path::new(output).to_path_buf(),
            db_path: Path::new(db_path).to_path_buf(),
            checkpoint_interval_secs: 600,
            agent_idle_timeout_secs: 300,
        };
        
        let orchestrator = Orchestrator::new(config)
            .context("Failed to create orchestrator")?;
        
        orchestrator.start_workflow().await?;
        
        // Run producer agent
        let producer = ProducerAgent::new(ProducerConfig::default(), orchestrator.db().clone());
        
        while let Ok(Some(task)) = producer.process_next_task().await {
            println!("Processed task: {}", task.task_id);
        }
        
        println!("Production completed");
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
    
    println!("Consume command: sdef={}, output={}, language={}", sdef, output, language);
    println!("Note: Full consume functionality not yet implemented");
    
    Ok(())
}

fn serve_command(transport: &str, db_path: &str) -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context("Failed to create Tokio runtime")?;
    
    rt.block_on(async {
        let server = cleanroom_mcp::CleanroomMcpServer::new(Path::new(db_path))
            .context("Failed to create MCP server")?;
        
        println!("Starting MCP server with {} transport", transport);
        server.serve().await?;
        Ok(())
    })
}

fn inspect_command(check_type: &str, db_path: &str) -> Result<()> {
    let db = Database::open(Path::new(db_path))?;
    
    match check_type {
        "consistency" => {
            println!("Consistency check:");
            println!("Note: Full consistency check not yet implemented");
        }
        "coverage" => {
            println!("Coverage check:");
            println!("Note: Full coverage check not yet implemented");
        }
        "progress" => {
            println!("Progress check:");
            println!("Note: Full progress check not yet implemented");
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
            let db = Database::open(Path::new(db_path))?;
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