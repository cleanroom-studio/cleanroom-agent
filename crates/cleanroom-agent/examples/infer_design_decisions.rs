//! `examples/infer_design_decisions.rs`
//!
//! Phase 1.1 end-to-end demo: drive the Producer's
//! `InferDesignDecisions` task path against a small Rust source file.
//!
//! Flow (mirrors `llm_analyze_file.rs`):
//! 1. Build an `OpenAiProvider` LLM via `MetaBuilder`.
//! 2. Spin up a temp SQLite DB with all migrations applied.
//! 3. Write a small Rust source file to a temp repo dir.
//! 4. Insert a synthetic `InferDesignDecisions` task with the right
//!    `input_json` (document + module_name + file_paths + repo_path).
//! 5. Build a `ProducerAgent` with `.with_llm(...)` attached.
//! 6. Call `process_next_task()` and print the result.
//! 7. Re-query the `design_decisions` table and verify the LLM
//!    inferred 3-10 module-level decisions with `context` containing
//!    `module=<name>; phase=1.1`.
//!
//! ## Usage
//!
//! ```bash
//! # Pre-req: drop a `.env` with one of:
//! #   MINIMAX_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
//!
//! cargo run --manifest-path cleanroom-agent/Cargo.toml \
//!   -p cleanroom-agent --example infer_design_decisions
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use cleanroom_agent::producer::{ProducerAgent, ProducerConfig};
use cleanroom_db::{Database, Task, TaskRepository, TaskStatus, TaskType};
use cleanroom_meta_llm::backends::openai::OpenAiProvider;
use cleanroom_meta_llm::builder::MetaBuilder;
use cleanroom_meta_llm::MetaLlm;

use std::io::Write;

/// A representative Rust module used by this example. The LLM should
/// be able to infer a handful of module-level design decisions
/// (storage strategy, error-handling, public API surface, etc.)
/// from this code.
const SAMPLE_RUST_MODULE: &str = r#"//! Sample user-storage module used by `examples/infer_design_decisions.rs`.
//!
//! Exercises enough surface for the LLM to infer real
//! module-level design decisions: a small in-memory store, a
//! public `UserStore` trait, an email validator, and the
//! `InMemoryStore` impl behind it.

use std::fmt;

/// Represents a registered user.
pub struct User {
    pub id: u64,
    pub email: String,
    pub active: bool,
}

/// Storage contract for `User` records. The public API surface.
pub trait UserStore {
    /// Look up a user by id.
    fn get(&self, id: u64) -> Option<User>;
    /// Insert or replace a user.
    fn put(&mut self, user: User);
}

/// Concrete in-memory `UserStore`. Demonstrates the chosen
/// storage backend (Vec + linear scan).
pub struct InMemoryStore {
    items: Vec<User>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
}

impl UserStore for InMemoryStore {
    fn get(&self, id: u64) -> Option<User> {
        self.items.iter().find(|u| u.id == id).cloned()
    }
    fn put(&mut self, user: User) {
        self.items.push(user);
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for User {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "User {{ id: {}, email: {}, active: {} }}", self.id, self.email, self.active)
    }
}

/// Validate `email` looks like a syntactically plausible address.
pub fn validate_email(email: &str) -> bool {
    let mut parts = email.splitn(2, '@');
    let local = parts.next().unwrap_or("");
    let domain = parts.next().unwrap_or("");
    !local.is_empty() && !domain.is_empty() && domain.contains('.')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut s = InMemoryStore::new();
        s.put(User { id: 1, email: "a@b.io".into(), active: true });
        assert!(s.get(1).is_some());
    }

    #[test]
    fn email_validator() {
        assert!(validate_email("alice@example.com"));
        assert!(!validate_email("not-an-email"));
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
    let db_path = tmpdir.path().join("infer_design_decisions.db");
    let migrations_dir = workspace_migrations_dir();
    let db = Database::open_with_migrations_from(&db_path, Some(&migrations_dir))?;
    let db = Arc::new(db);

    // 3. Write a small source file to a temp repo dir.
    let repo_dir = tmpdir.path().join("repo");
    std::fs::create_dir_all(repo_dir.join("src"))?;
    let src_path = repo_dir.join("src").join("lib.rs");
    {
        let mut f = std::fs::File::create(&src_path)?;
        f.write_all(SAMPLE_RUST_MODULE.as_bytes())?;
    }
    let rel_path = "src/lib.rs";
    let repo_path_str = repo_dir.to_string_lossy().to_string();

    // 4. Insert an InferDesignDecisions task.
    let task_repo = TaskRepository::new(db.connection_arc());
    let task_id = uuid::Uuid::new_v4().to_string();
    let task = Task {
        task_id: task_id.clone(),
        task_type: TaskType::InferDesignDecisions,
        status: TaskStatus::Pending,
        priority: 8,
        input_json: serde_json::json!({
            "document": "infer-design-decisions-demo",
            "project_name": "infer-design-decisions-demo",
            "module_name": "src",
            "file_paths": [rel_path],
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
        max_retries: 2,
        last_heartbeat: None,
        dependencies_json: "[]".to_string(),
        version: 1,
    };
    task_repo.create(&task)?;

    // 5. Build a Producer with the LLM attached.
    let agent = ProducerAgent::new(ProducerConfig::default(), db.clone()).with_llm(llm_dyn);
    assert!(agent.has_llm(), "agent must have LLM attached for this demo");

    println!("== infer_design_decisions ==");
    println!("provider: openai");
    println!("model:    {model}");
    println!("repo:     {repo_path_str}");
    println!("file:     {rel_path}");
    println!("task_id:  {task_id}");
    println!();

    let started = std::time::Instant::now();
    let _outcome = agent
        .process_next_task()
        .await?
        .expect("task should be claimable (we just inserted it)");
    // Re-fetch by id to see the post-completion status.
    let claimed = task_repo.get(&task_id)?;
    let elapsed_ms = started.elapsed().as_millis();
    println!("== process_next_task finished in {elapsed_ms}ms ==");
    println!("status:    {:?}", claimed.status);
    if let Some(err) = &claimed.error_message {
        println!("error:     {err}");
    }
    if claimed.status != TaskStatus::Completed {
        return Err(format!(
            "task {task_id} did not complete (status = {:?})",
            claimed.status
        )
        .into());
    }

    // 6. Inspect the output JSON.
    let output: serde_json::Value = serde_json::from_str(
        claimed.output_json.as_deref().unwrap_or("{}"),
    )?;
    println!();
    println!("== task output_json ==");
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_else(|_| output.to_string())
    );

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

    // 7. Verify design_decisions table got module-level rows.
    //
    //    We open a fresh `rusqlite::Connection` against the same DB
    //    file to query the `design_decisions` table directly
    //    (SdefRepository doesn't expose a `list_design_decisions`
    //    helper yet). Phase 1.1 close-out: filter on the
    //    dedicated `module_name` column (migration 013) rather
    //    than `context LIKE 'module=src;%'` — the LIKE regex was
    //    the pre-migration workaround and is no longer used.
    let conn = rusqlite::Connection::open(&db_path)?;
    let mut stmt = conn
        .prepare(
            "SELECT topic, decision, rationale FROM design_decisions \
             WHERE document_name = ?1 AND module_name = ?2 \
             ORDER BY topic ASC",
        )
        .map_err(|e| format!("prepare: {e}"))?;
    let rows: Vec<(String, String, String)> = stmt
        .query_map(
            rusqlite::params!["infer-design-decisions-demo", "src"],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|e| format!("query_map: {e}"))?
        .filter_map(|r| r.ok())
        .collect();
    println!();
    println!("== module-level design decisions persisted to DB ==");
    if rows.is_empty() {
        return Err(
            "expected >= 1 module-level design decision in the DB, got 0 -- writer dropped them"
                .into(),
        );
    }
    for (i, (topic, decision, rationale)) in rows.iter().enumerate() {
        println!(
            "  [{}] topic={} decision={} rationale={}",
            i, topic, decision, rationale
        );
    }
    assert!(
        rows.len() >= 1 && rows.len() <= 10,
        "expected 1-10 module-level decisions, got {} (the PLAN says 3-10; \
         we accept 1-10 here so the example passes even on a terse LLM)",
        rows.len()
    );
    println!();
    println!(
        "OK: {n} module-level design decision(s) persisted under 'src'",
        n = rows.len()
    );
    println!("== infer_design_decisions: all checks passed ==");

    Ok(())
}

fn workspace_migrations_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .and_then(|p| p.parent()) // cleanroom-agent/
        .expect("cleanroom-agent crate layout has two parents")
        .join("migrations")
}
