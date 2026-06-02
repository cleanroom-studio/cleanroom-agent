//! `examples/llm_analyze_file.rs`
//!
//! Phase 0.5 end-to-end demo: drive the Producer's LLM path
//! (`ProducerAgent::analyze_file_with_llm`) against a real source file and
//! MiniMax-M3.
//!
//! Flow:
//! 1. Build an `OpenAiProvider` LLM via `MetaBuilder` (the same way
//!    `eval_meta` / `eval_llm_loop` do).
//! 2. Spin up a temp SQLite DB with all migrations applied (so the
//!    `LLM_ANALYZE_FILE` task_type is accepted by the CHECK constraint).
//! 3. Write a small Rust source file to a temp repo dir.
//! 4. Insert a synthetic `LlmAnalyzeFile` task with the right `input_json`.
//! 5. Build a `ProducerAgent` with `.with_llm(...)` attached.
//! 6. Call `process_next_task()` and print the result.
//!
//! This is the "smallest possible" end-to-end check that the Phase 0.5 LLM
//! path works: LLM is invoked, raw output is persisted, token counts are
//! captured (now non-zero thanks to `UsageCapturingLlm`).
//!
//! ## Usage
//!
//! ```bash
//! # Pre-req: drop a `.env` with one of:
//! #   MINIMAX_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
//!
//! cargo run --manifest-path cleanroom-agent/Cargo.toml \
//!   -p cleanroom-agent --example llm_analyze_file
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use cleanroom_agent::llm_loop::LoopConfig;
use cleanroom_agent::producer::{ProducerAgent, ProducerConfig};
use cleanroom_db::{Database, Task, TaskRepository, TaskStatus, TaskType};
use cleanroom_meta_llm::backends::openai::OpenAiProvider;
use cleanroom_meta_llm::builder::MetaBuilder;
use cleanroom_meta_llm::MetaLlm;

use std::io::Write;

const SAMPLE_RUST_SOURCE: &str = r#"//! Tiny sample crate used by `examples/llm_analyze_file.rs`.
//!
//! Exercises a few patterns the LLM should pick up:
//! - A `User` data model with typed fields and a doc comment.
//! - A `UserStore` interface contract with two methods.
//! - A free function (`validate_email`) with a Rustdoc example.

use std::collections::HashMap;

/// Represents a registered user in the system.
pub struct User {
    /// Stable unique identifier (UUID v4).
    pub id: String,
    /// Email address (RFC 5322).
    pub email: String,
    /// Display name shown in the UI.
    pub display_name: String,
    /// Unix epoch seconds of account creation.
    pub created_at: i64,
}

/// Storage interface for `User` records.
pub trait UserStore {
    /// Fetch a user by id; returns `None` if not found.
    fn get(&self, id: &str) -> Option<User>;
    /// Insert or update a user; returns the persisted record.
    fn put(&mut self, user: User) -> User;
}

/// Validate that `email` looks like a syntactically valid email address.
///
/// # Example
/// ```
/// assert!(validate_email("a@b.io"));
/// assert!(!validate_email("not-an-email"));
/// ```
pub fn validate_email(email: &str) -> bool {
    let mut parts = email.splitn(2, '@');
    let local = parts.next().unwrap_or("");
    let domain = parts.next().unwrap_or("");
    !local.is_empty() && !domain.is_empty() && domain.contains('.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct InMemoryStore(HashMap<String, User>);

    impl UserStore for InMemoryStore {
        fn get(&self, id: &str) -> Option<User> {
            self.0.get(id).cloned()
        }
        fn put(&mut self, user: User) -> User {
            self.0.insert(user.id.clone(), user.clone());
            user
        }
    }

    #[test]
    fn email_validator_accepts_well_formed() {
        assert!(validate_email("alice@example.com"));
    }

    #[test]
    fn store_round_trip() {
        let mut s = InMemoryStore(HashMap::new());
        let u = User {
            id: "u1".into(),
            email: "alice@example.com".into(),
            display_name: "Alice".into(),
            created_at: 0,
        };
        s.put(u.clone());
        assert_eq!(s.get("u1").unwrap().email, "alice@example.com");
    }
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
        .max_tokens(1024)
        .temperature(0.0)
        .build()?;
    let llm_dyn: Arc<dyn MetaLlm> = llm;

    // 2. Set up a temp DB with all migrations applied.
    let tmpdir = tempfile::tempdir()?;
    let db_path = tmpdir.path().join("llm_analyze_file.db");
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

    // 4. Insert an LlmAnalyzeFile task.
    let repo = TaskRepository::new(db.connection_arc());
    let task_id = uuid::Uuid::new_v4().to_string();
    let task = Task {
        task_id: task_id.clone(),
        task_type: TaskType::LlmAnalyzeFile,
        status: TaskStatus::Pending,
        priority: 8,
        input_json: serde_json::json!({
            "document": "llm-analyze-file-demo",
            "project_name": "llm-analyze-file-demo",
            "repo_path": repo_path_str,
            "file_path": rel_path,
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
    repo.create(&task)?;

    // 5. Build a Producer with the LLM attached and a generous loop config.
    let agent = ProducerAgent::new(ProducerConfig::default(), db.clone())
        .with_llm(llm_dyn)
        .with_loop_config(LoopConfig {
            max_iterations: 4,
            max_tokens_per_call: 1024,
            temperature: 0.0,
            tool_timeout_secs: 30,
            cost_limit_usd: Some(0.05),
            on_call_complete: None,
            tools: None,
        });
    assert!(agent.has_llm(), "agent must have LLM attached for this demo");

    println!("== llm_analyze_file ==");
    println!("provider: openai");
    println!("model:    {model}");
    println!("repo:     {repo_path_str}");
    println!("file:     {rel_path}");
    println!("task_id:  {task_id}");
    println!();

    let started = std::time::Instant::now();
    let outcome = agent.process_next_task().await?;
    let elapsed_ms = started.elapsed().as_millis();

    println!("== process_next_task finished in {elapsed_ms}ms ==");
    println!("claimed task: {outcome:?}");

    // 6. Read the task back and show the output_json.
    let claimed = repo.get(&task_id)?;
    println!("status:    {:?}", claimed.status);
    if let Some(err) = &claimed.error_message {
        println!("error:     {err}");
    }
    let output: serde_json::Value = serde_json::from_str(
        claimed.output_json.as_deref().unwrap_or("{}"),
    )?;
    println!();
    println!("== task output_json ==");
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_else(|_| output.to_string())
    );

    if claimed.status != TaskStatus::Completed {
        return Err(format!("task did not complete (status = {:?})", claimed.status).into());
    }
    let raw = output
        .get("raw_llm_output")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if raw.is_empty() {
        return Err("task completed but raw_llm_output is empty".into());
    }
    println!();
    println!("== raw LLM output (first 600 chars) ==");
    println!("{}", &raw.chars().take(600).collect::<String>());
    if raw.chars().count() > 600 {
        println!("... ({} more chars truncated)", raw.chars().count() - 600);
    }

    Ok(())
}

fn workspace_migrations_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .and_then(|p| p.parent()) // cleanroom-agent/
        .expect("cleanroom-agent crate layout has two parents")
        .join("migrations")
}
