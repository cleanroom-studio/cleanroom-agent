//! `examples/produce_llm.rs`
//!
//! Phase 0.5 end-to-end demo: drive the **full Producer LLM pipeline** by
//! running `analyze_repo` in `Llm` mode on a real temp repo with multiple
//! source files, then claiming and processing every `LlmAnalyzeFile` task.
//!
//! Flow:
//! 1. Build the LLM (same way `eval_meta` / `eval_llm_loop` /
//!    `llm_analyze_file` do).
//! 2. Stage a temp repo with **3 source files** in 2 languages.
//! 3. Spin up a fully-migrated DB (so the `LlmAnalyzeFile` task_type is
//!    accepted by the CHECK constraint).
//! 4. Build a `ProducerAgent` in `Llm` mode and insert a `RepoAnalyze` task.
//! 5. Call `process_next_task()` -- this invokes the LLM-mode
//!    `analyze_repo` path which scans the repo and schedules one
//!    `LlmAnalyzeFile` task per source file.
//! 6. Loop `process_next_task()` until all `LlmAnalyzeFile` tasks are
//!    Completed; print a per-file summary (LLM output, token counts, cost).
//!
//! This is the canonical "Producer in LLM mode" smoke test. Use it to
//! regression-check the full pipeline when changing
//! `analyze_file_with_llm`, the scheduler, or the producer config.
//!
//! ## Usage
//!
//! ```bash
//! # Pre-req: drop a `.env` with MINIMAX_API_KEY / ANTHROPIC_API_KEY /
//! #          OPENAI_API_KEY
//!
//! cargo run --manifest-path cleanroom-agent/Cargo.toml \
//!   -p cleanroom-agent --example produce_llm
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cleanroom_agent::producer::{ProducerAgent, ProducerConfig};
use cleanroom_db::{
    Database, Task, TaskRepository, TaskStatus, TaskType,
};
use cleanroom_meta_llm::backends::openai::OpenAiProvider;
use cleanroom_meta_llm::builder::MetaBuilder;
use cleanroom_meta_llm::MetaLlm;

const FILE_A_RUST: &str = r#"//! Account model
pub struct Account {
    pub id: String,
    pub email: String,
    pub balance_cents: i64,
}
impl Account {
    pub fn credit(&mut self, cents: u64) {
        self.balance_cents += cents as i64;
    }
}
"#;

const FILE_B_RUST: &str = r#"//! Transfer model
pub struct Transfer {
    pub from_id: String,
    pub to_id: String,
    pub amount_cents: i64,
    pub timestamp: i64,
}
pub fn validate(t: &Transfer) -> bool {
    t.from_id != t.to_id && t.amount_cents > 0
}
"#;

const FILE_C_PYTHON: &str = r#"""Order record used by the checkout service."""
from dataclasses import dataclass

@dataclass
class Order:
    order_id: str
    customer_id: str
    total_cents: int
    paid: bool = False
"#;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 0. Tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,cleanroom_meta=warn")),
        )
        .init();

    // 1. Build the LLM.
    let api_key = std::env::var("MINIMAX_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .map_err(|_| "MINIMAX_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY not set")?;
    if std::env::var_os("OPENAI_BASE_URL").is_none() {
        std::env::set_var("OPENAI_BASE_URL", "https://api.minimaxi.com/v1");
    }
    let model = std::env::var("EVAL_MODEL").unwrap_or_else(|_| "MiniMax-M3".to_string());
    let llm: Arc<OpenAiProvider> = MetaBuilder::<OpenAiProvider>::new()
        .api_key(api_key)
        .base_url(std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.minimaxi.com/v1".into()))
        .model(model.clone())
        .max_tokens(512)
        .temperature(0.0)
        .build()?;
    let llm_dyn: Arc<dyn MetaLlm> = llm;

    // 2. Stage a temp repo with 3 source files.
    let tmpdir = tempfile::tempdir()?;
    let repo_dir = tmpdir.path().join("repo");
    std::fs::create_dir_all(repo_dir.join("src"))?;
    write_file(&repo_dir.join("src").join("account.rs"), FILE_A_RUST)?;
    write_file(&repo_dir.join("src").join("transfer.rs"), FILE_B_RUST)?;
    write_file(&repo_dir.join("src").join("order.py"), FILE_C_PYTHON)?;
    // Non-source file -- should be skipped by the language filter.
    std::fs::write(repo_dir.join("data.bin"), b"\x00\x01")?;
    let repo_path_str = repo_dir.to_string_lossy().to_string();

    // 3. Spin up a fully-migrated DB.
    let db_path = tmpdir.path().join("produce_llm.db");
    let migrations_dir = workspace_migrations_dir();
    let db = Database::open_with_migrations_from(&db_path, Some(&migrations_dir))?;
    let db = Arc::new(db);

    // 4. Build a producer in Llm mode and insert a RepoAnalyze task.
    let agent = ProducerAgent::new(ProducerConfig::llm(), db.clone())
        .with_llm(llm_dyn);
    assert!(agent.has_llm(), "agent must have LLM attached");

    let task_repo = TaskRepository::new(db.connection_arc());
    let high_level_id = uuid::Uuid::new_v4().to_string();
    let high_level_task = Task {
        task_id: high_level_id.clone(),
        task_type: TaskType::RepoAnalyze,
        status: TaskStatus::Pending,
        priority: 10,
        input_json: serde_json::json!({
            "document": "produce-llm-demo",
            "project_name": "produce-llm-demo",
            "repo_path": repo_path_str,
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
    task_repo.create(&high_level_task)?;

    println!("== produce_llm (mode=Llm) ==");
    println!("provider: openai");
    println!("model:    {model}");
    println!("repo:     {repo_path_str}");
    println!("task_id:  {high_level_id}");
    println!();

    // 5. Run the high-level task: schedules N LlmAnalyzeFile tasks.
    let started = std::time::Instant::now();
    agent.process_next_task().await?;
    let after = task_repo.get(&high_level_id)?;
    let scheduled_count = serde_json::from_str::<serde_json::Value>(after.output_json.as_deref().unwrap_or("{}"))?
        .get("scheduled_task_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    println!("== RepoAnalyze done in {}ms; scheduled {scheduled_count} LlmAnalyzeFile tasks ==", started.elapsed().as_millis());
    println!();

    // 6. Loop until no more tasks are claimable.
    let mut per_file: Vec<(String, String, u32, u32, f64)> = Vec::new();
    let mut processed = 0u32;
    loop {
        match agent.process_next_task().await? {
            None => break,
            Some(t) => {
                if t.task_type != TaskType::LlmAnalyzeFile {
                    continue;
                }
                let after = task_repo.get(&t.task_id)?;
                if after.status == TaskStatus::Completed {
                    let out: serde_json::Value = serde_json::from_str(
                        after.output_json.as_deref().unwrap_or("{}"),
                    )?;
                    let path = out
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string();
                    let prompt = out
                        .get("prompt_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let completion = out
                        .get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let raw = out
                        .get("raw_llm_output")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    // $3 / 1M input + $15 / 1M output (Sonnet 3.5 estimate).
                    let cost = (prompt as f64) * 3.0 / 1_000_000.0
                        + (completion as f64) * 15.0 / 1_000_000.0;
                    per_file.push((path, raw, prompt, completion, cost));
                    processed += 1;
                }
            }
        }
    }
    let total_elapsed_ms = started.elapsed().as_millis();

    // 7. Print per-file summary.
    per_file.sort_by(|a, b| a.0.cmp(&b.0));
    println!("== processed {processed} files in {total_elapsed_ms}ms ==");
    let mut total_prompt = 0u32;
    let mut total_completion = 0u32;
    let mut total_cost = 0.0;
    for (path, raw, prompt, completion, cost) in &per_file {
        total_prompt += prompt;
        total_completion += completion;
        total_cost += cost;
        let preview: String = raw.chars().take(160).collect();
        println!();
        println!("  --- {path} ---");
        println!("  prompt={prompt} completion={completion} cost=${cost:.4}");
        println!("  output preview: {preview}{}", if raw.chars().count() > 160 { "..." } else { "" });
    }
    println!();
    println!("== totals ==");
    println!("files:        {processed}");
    println!("prompt tok:   {total_prompt}");
    println!("completion:   {total_completion}");
    println!("total cost:   ${total_cost:.4}");
    println!("elapsed:      {total_elapsed_ms}ms");

    if processed == 0 {
        return Err("no LlmAnalyzeFile tasks were processed".into());
    }
    Ok(())
}

fn write_file(path: &Path, content: &str) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(content.as_bytes())?;
    Ok(())
}

fn workspace_migrations_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .and_then(|p| p.parent()) // cleanroom-agent/
        .expect("cleanroom-agent crate layout has two parents")
        .join("migrations")
}
