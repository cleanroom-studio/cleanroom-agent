//! End-to-end check for migration `007_llm_task_types.sql`.
//!
//! Inserts one task of each new LLM-driven `task_type` and reads it back. If
//! migration 007's CHECK-constraint extension didn't take effect, the
//! `INSERT` would fail with `constraint failed`.
//!
//! We use `migrations::run_pending_at` with an explicit path derived from
//! `env!("CARGO_MANIFEST_DIR")` instead of the CWD-based default, because
//! `cargo test` runs the test binary with CWD = this crate's directory,
//! not the workspace root.

use std::path::PathBuf;

use cleanroom_db::{Database, Task, TaskRepository, TaskStatus, TaskType};

/// Absolute path to the workspace's `migrations/` directory, resolved at
/// compile time. `CARGO_MANIFEST_DIR` for the `cleanroom-db` crate points
/// at `cleanroom-agent/crates/cleanroom-db`, so we walk up two levels.
fn workspace_migrations_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .and_then(|p| p.parent()) // cleanroom-agent/
        .expect("cleanroom-db crate layout has two parents")
        .join("migrations")
}

fn make_task(task_type: TaskType) -> Task {
    Task {
        task_id: format!("t-{:?}", task_type),
        task_type,
        status: TaskStatus::Pending,
        priority: 5,
        input_json: "{}".to_string(),
        output_json: None,
        error_message: None,
        assigned_to: None,
        progress: 0.0,
        created_at: "2026-06-01T00:00:00Z".to_string(),
        started_at: None,
        completed_at: None,
        retry_count: 0,
        max_retries: 3,
        last_heartbeat: None,
        dependencies_json: "[]".to_string(),
        version: 1,
    }
}

#[test]
fn test_migration_007_accepts_new_llm_task_types() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("007-check.db");
    // Use the explicit-migrations-dir constructor so the test isn't sensitive
    // to the CWD that `cargo test` happens to use.
    let db = Database::open_with_migrations_from(&path, Some(&workspace_migrations_dir()))
        .expect("open db with full migration sequence");

    let repo = TaskRepository::new(db.connection_arc());

    // Both `INSERT`s must succeed. If migration 007 hadn't been applied, the
    // CHECK constraint on `task_type` would reject these strings and
    // `repo.create` would return an error.
    repo.create(&make_task(TaskType::LlmAnalyzeFile))
        .expect("insert LlmAnalyzeFile");
    repo.create(&make_task(TaskType::LlmGenerateCode))
        .expect("insert LlmGenerateCode");

    // Round-trip: read them back and confirm the strings survive serialization.
    let analyze = repo
        .get(&format!("t-{:?}", TaskType::LlmAnalyzeFile))
        .expect("fetch LlmAnalyzeFile");
    assert_eq!(analyze.task_type, TaskType::LlmAnalyzeFile);
    assert_eq!(analyze.task_type.as_str(), "LLM_ANALYZE_FILE");

    let generate = repo
        .get(&format!("t-{:?}", TaskType::LlmGenerateCode))
        .expect("fetch LlmGenerateCode");
    assert_eq!(generate.task_type, TaskType::LlmGenerateCode);
    assert_eq!(generate.task_type.as_str(), "LLM_GENERATE_CODE");
}
