//! Cleanroom Agent CLI entry point.

use anyhow::Result;
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

mod commands;

#[derive(Parser)]
#[command(name = "cleanroom")]
#[command(about = "Cleanroom Agent — S.DEF intelligent agent system")]
struct Cli {
    #[command(subcommand)]
    command: commands::Commands,

    /// Database path
    #[arg(long, default_value = "state.db")]
    db: String,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cli = Cli::try_parse_from(&args)?;

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(&cli.log_level))
        .init();

    info!("cleanroom-agent v{}", env!("CARGO_PKG_VERSION"));

    commands::run(cli.command, &cli.db)
}