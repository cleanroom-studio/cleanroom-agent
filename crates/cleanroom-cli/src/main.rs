//! Cleanroom Agent CLI — Command-line interface for the S.DEF intelligent agent system.
//!
//! This binary provides a user-facing CLI for all Cleanroom Agent operations,
//! including code analysis (Produce), code generation (Consume), MCP server mode,
//! workflow resumption, database inspection, and more.
//!
//! # Usage
//!
//! ```bash
//! # Analyze a repository and output S.DEF
//! cleanroom produce --repo ./my-project --output ./sdef-output
//!
//! # Generate code from S.DEF
//! cleanroom consume --sdef ./sdef-output/sdef.json --output ./generated --language typescript
//!
//! # Start MCP server
//! cleanroom serve --transport stdio
//!
//! # Inspect database consistency
//! cleanroom inspect --check-type consistency
//!
//! # Export S.DEF to JSON
//! cleanroom export --document my-project --output ./sdef-export.json
//! ```
//!
//! # Global Flags
//!
//! - `--db`: Path to SQLite database (default: `state.db`)
//! - `--log-level`: Tracing log level (default: `info`)
//! - `--lang`: UI language — `en`, `zh`, or `auto` (default: `auto`)
//!
//! # Exit Codes
//!
//! - `0`: Success
//! - `1`: General error
//! - `2`: Parse error (invalid arguments)

use anyhow::Result;
use clap::Parser;
use cleanroom_i18n::{init, Lang};
use tracing::info;
use tracing_subscriber::EnvFilter;

mod commands;
mod progress;

/// Cleanroom Agent CLI arguments.
#[derive(Parser)]
#[command(name = "cleanroom")]
#[command(about = "Cleanroom Agent — S.DEF intelligent agent system")]
struct Cli {
    /// The subcommand to execute.
    #[command(subcommand)]
    command: commands::Commands,

    /// Path to the SQLite database file.
    #[arg(long, default_value = "state.db")]
    db: String,

    /// Tracing log level (e.g., `info`, `debug`, `warn`).
    #[arg(long, default_value = "info")]
    log_level: String,

    /// UI language: `en` (English), `zh` (中文), or `auto` (detect from environment).
    #[arg(long, default_value = "auto")]
    lang: String,
}

/// CLI entry point.
///
/// Initializes internationalization, sets up logging, and dispatches
/// to the appropriate command handler.
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cli = Cli::try_parse_from(&args)?;

    // Initialize i18n
    let lang = if cli.lang == "auto" {
        Lang::from_env()
    } else {
        Lang::from_str(&cli.lang)
    };
    init(lang);

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(&cli.log_level))
        .init();

    info!("cleanroom-agent v{}", env!("CARGO_PKG_VERSION"));

    commands::run(cli.command, &cli.db)
}
