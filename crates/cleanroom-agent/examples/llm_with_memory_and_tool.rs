//! `examples/llm_with_memory_and_tool.rs`
//!
//! Phase 0.10 end-to-end #2 demo: drive the Producer's LLM path
//! (`ProducerAgent::analyze_file_with_llm`) with **two** consecutive calls,
//! wired up with:
//! - **One tool** (`echo`) exposed via `McpToolBridge::new` so the LLM sees
//!   a `MetaToolT` in its tool list
//! - **Sliding-window memory** (`window_size = 4`) so the second call can
//!   see the first call's user/assistant messages in its history
//!
//! The two consecutive `LlmAnalyzeFile` tasks share the same
//! `ProducerAgent` (which carries the memory `Arc<Mutex<...>>` across
//! calls), and the same tool list. We then query `llm_call_log` to verify:
//! - both calls were logged
//! - the second call's `memory_messages_at_call` is `> 0` (proves memory
//!   accumulates across calls)
//! - `tool_calls = 0` for both (the LLM is free to call `echo`, but for
//!   the source-analysis task the structured JSON answer is the right
//!   output, so we don't *require* it to call the tool — we just verify
//!   the wiring doesn't crash if it does)
//!
//! ## Usage
//!
//! ```bash
//! # Pre-req: drop a `.env` with one of:
//! #   MINIMAX_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
//!
//! cargo run --manifest-path cleanroom-agent/Cargo.toml \
//!   -p cleanroom-agent --example llm_with_memory_and_tool
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use cleanroom_agent::llm_loop::{LoopConfig, MemoryConfig};
use cleanroom_agent::mcp_tool_bridge::McpToolBridge;
use cleanroom_agent::producer::{ProducerAgent, ProducerConfig};
use cleanroom_db::repositories::llm_call_log_repository::LlmCallLogRepository;
use cleanroom_db::{Database, Task, TaskRepository, TaskStatus, TaskType};
use cleanroom_meta_llm::backends::openai::OpenAiProvider;
use cleanroom_meta_llm::builder::MetaBuilder;
use cleanroom_meta_llm::MetaLlm;

use std::io::Write;

/// A small Rust source file the LLM should be able to summarize. We
/// reuse the same shape as `examples/llm_analyze_file.rs` so the two
/// demos exercise the same parser path.
const SAMPLE_RUST_SOURCE: &str = r#"//! Tiny sample crate used by `examples/llm_with_memory_and_tool.rs`.
//!
//! Exercises a few patterns the LLM should pick up:
//! - A `User` data model with typed fields.
//! - A `UserStore` interface contract with two methods.

/// Represents a registered user in the system.
pub struct User {
    /// Stable unique identifier (UUID v4).
    pub id: String,
    /// Email address (RFC 5322).
    pub email: String,
}

/// Storage interface for `User` records.
pub trait UserStore {
    /// Fetch a user by id; returns `None` if not found.
    fn get(&self, id: &str) -> Option<User>;
    /// Insert or update a user; returns the persisted record.
    fn put(&mut self, user: User) -> User;
}

/// Validate that `email` looks like a syntactically valid email address.
pub fn validate_email(email: &str) -> bool {
    let mut parts = email.splitn(2, '@');
    let local = parts.next().unwrap_or("");
    let domain = parts.next().unwrap_or("");
    !local.is_empty() && !domain.is_empty() && domain.contains('.')
}
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

    // 1. Build the LLM (same pattern as `llm_analyze_file`).
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
        .base_url(
            std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.minimaxi.com/v1".into()),
        )
        .model(model.clone())
        .max_tokens(1024)
        .temperature(0.0)
        .build()?;
    let llm_dyn: Arc<dyn MetaLlm> = llm;

    // 2. Set up a temp DB with all migrations applied.
    let tmpdir = tempfile::tempdir()?;
    let db_path = tmpdir.path().join("llm_with_memory_and_tool.db");
    let migrations_dir = workspace_migrations_dir();
    let db = Database::open_with_migrations_from(&db_path, Some(&migrations_dir))?;
    let db = Arc::new(db);

    // 3. Write a small source file to a temp repo dir.
    let repo_dir = tmpdir.path().join("repo");
    std::fs::create_dir_all(repo_dir.join("src"))?;
    let src_path = repo_dir.join("src").join("user.rs");
    {
        let mut f = std::fs::File::create(&src_path)?;
        f.write_all(SAMPLE_RUST_SOURCE.as_bytes())?;
    }
    let rel_path = "src/user.rs";
    let repo_path_str = repo_dir.to_string_lossy().to_string();

    // 4. Insert TWO `LlmAnalyzeFile` tasks so the second call can see
    // the first call's history via the sliding-window memory.
    let task_repo = TaskRepository::new(db.connection_arc());
    let mut task_ids = Vec::new();
    for i in 0..2 {
        let task_id = uuid::Uuid::new_v4().to_string();
        let task = Task {
            task_id: task_id.clone(),
            task_type: TaskType::LlmAnalyzeFile,
            status: TaskStatus::Pending,
            priority: 8,
            input_json: serde_json::json!({
                "document": "llm-mem-tool-demo",
                "project_name": "llm-mem-tool-demo",
                "repo_path": repo_path_str,
                "file_path": rel_path,
                // Tiny variant so the second call's `user` message differs
                // from the first — easier to eyeball in the LLM log.
                "run_index": i,
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
        task_repo.create(&task)?;
        task_ids.push(task_id);
    }

    // 5. Build a Producer with:
    //    - the LLM
    //    - one tool (`echo` via `McpToolBridge`)
    //    - a `SlidingWindow { window_size: 4 }` memory shared across calls
    let echo_dispatch: Arc<dyn Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync> =
        Arc::new(|args: serde_json::Value| {
            // Echo back the args so the LLM can verify the tool round-trip
            // if it decides to call us.
            Ok(serde_json::json!({
                "tool": "echo",
                "echoed_args": args,
                "note": "this is the echo tool wired in by llm_with_memory_and_tool.rs",
            }))
        });
    let echo_bridge: Arc<dyn cleanroom_meta_core::tool::MetaToolT> = Arc::new(
        McpToolBridge::new(
            "echo",
            "Echoes back the JSON arguments the LLM passes. Useful as a wiring test for tool injection.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Free-form text to echo back."}
                },
                "required": [],
                "additionalProperties": true,
            }),
            echo_dispatch,
        ),
    );

    let llm_call_logger = Arc::new(LlmCallLogRepository::new_with_arc(db.connection_arc()));
    let agent = ProducerAgent::new(ProducerConfig::default(), db.clone())
        .with_llm(llm_dyn)
        .with_tools(vec![echo_bridge])
        .with_memory_sliding_window(4)
        .with_llm_call_logger(llm_call_logger.clone())
        .with_loop_config(LoopConfig {
            max_iterations: 4,
            max_tokens_per_call: 1024,
            temperature: 0.0,
            tool_timeout_secs: 30,
            cost_limit_usd: Some(0.10),
            on_call_complete: None,
            tools: None,
            memory: MemoryConfig::None, // The agent's own `memory_config` wins
        });
    assert!(agent.has_llm(), "agent must have LLM attached for this demo");

    println!("== llm_with_memory_and_tool ==");
    println!("provider: openai");
    println!("model:    {model}");
    println!("repo:     {repo_path_str}");
    println!("file:     {rel_path}");
    println!("task_ids: {:?}", task_ids);
    println!("tools:    [echo]");
    println!("memory:   SlidingWindow {{ window_size: 4 }}");
    println!();

    // 6. Process the two tasks sequentially. We process them in a loop
    //    because each call mutates the shared memory + appends to
    //    `llm_call_log`. Note: `process_next_task` returns the task as
    //    it was when *claimed* (status = `InProgress`), so we re-fetch
    //    by id afterwards to see the post-completion status. This is
    //    the same pattern `examples/llm_analyze_file.rs` uses.
    let started = std::time::Instant::now();
    for (i, task_id) in task_ids.iter().enumerate() {
        println!("---- run {i} (task_id={task_id}) ----");
        let _outcome = agent
            .process_next_task()
            .await?
            .expect("task should be claimable (we just inserted it)");
        // Re-fetch by id to see the post-completion status.
        let claimed = task_repo.get(task_id)?;
        println!(
            "  status:    {:?}",
            claimed.status
        );
        if let Some(err) = &claimed.error_message {
            println!("  error:     {err}");
        }
        if claimed.status != TaskStatus::Completed {
            return Err(format!(
                "task {task_id} did not complete (status = {:?})",
                claimed.status
            )
            .into());
        }
    }
    let elapsed_ms = started.elapsed().as_millis();
    println!();
    println!("== both tasks finished in {elapsed_ms}ms ==");
    println!();

    // 7. Inspect `llm_call_log` to verify:
    //    - both calls were logged
    //    - second call's `memory_messages_at_call` is `> 0`
    //    - both calls' `tool_calls` is `0` (the LLM may or may not call
    //      `echo`; either is fine — we just verify the wiring doesn't
    //      crash and the counts make sense)
    let total = llm_call_logger.count()?;
    println!("llm_call_log total rows: {total}");
    if total < 2 {
        return Err(format!(
            "expected >= 2 LlmCallLog rows after 2 task runs, got {total}"
        )
        .into());
    }
    let recent = llm_call_logger.list_recent(10)?;
    println!();
    println!("== llm_call_log (most recent first) ==");
    for (i, row) in recent.iter().enumerate() {
        println!(
            "  [{}] call_id={} task_id={} model={} status={:?} \
             tokens={}p+{}c tool_calls={} iter={} \
             memory_messages_at_call={} cost=${:.6} duration_ms={}",
            i,
            row.call_id,
            row.task_id.as_deref().unwrap_or("?"),
            row.model.as_deref().unwrap_or("?"),
            row.status,
            row.prompt_tokens,
            row.completion_tokens,
            row.tool_calls,
            row.iterations,
            row.memory_messages_at_call,
            row.cost_estimate_usd,
            row.duration_ms,
        );
    }
    println!();

    // 8. Verify the SECOND call saw the FIRST call's history in its memory.
    //    `list_recent` returns newest-first, so the *last* element of the
    //    returned `Vec` is the *first* call (oldest). The element just
    //    before it (index `len-2`) is the second call.
    let second = recent
        .get(recent.len().saturating_sub(2))
        .ok_or("missing second call row")?;
    if second.memory_messages_at_call <= 0 {
        return Err(format!(
            "second call should have memory_messages_at_call > 0 (sliding window size 4), \
             got {} — memory did not accumulate across calls",
            second.memory_messages_at_call
        )
        .into());
    }
    println!(
        "OK: second call memory_messages_at_call = {} (> 0, sliding window wired correctly)",
        second.memory_messages_at_call
    );

    // The first call should have memory_messages_at_call = 0 (no prior
    // history to recall).
    let first = recent
        .last()
        .ok_or("missing first call row")?;
    if first.memory_messages_at_call != 0 {
        return Err(format!(
            "first call should have memory_messages_at_call = 0 (no prior history), got {}",
            first.memory_messages_at_call
        )
        .into());
    }
    println!(
        "OK: first  call memory_messages_at_call = 0 (no prior history, as expected)"
    );

    println!();
    println!("== llm_with_memory_and_tool: all checks passed ==");
    Ok(())
}

fn workspace_migrations_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .and_then(|p| p.parent()) // cleanroom-agent/
        .expect("cleanroom-agent crate layout has two parents")
        .join("migrations")
}
