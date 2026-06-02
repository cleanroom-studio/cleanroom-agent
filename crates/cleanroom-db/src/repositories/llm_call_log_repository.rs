//! `llm_call_log` repository — append-only history of LLM calls.
//!
//! Phase 0.9: every `llm_loop::run_loop` (and `run_loop_via_basic_agent`)
//! invocation fires a callback that constructs a [`LlmCallLog`] and
//! hands it to a [`LlmCallLogRepository`]. `cleanroom-cli inspect llm-log`
//! reads back from this table to support debugging, cost auditing, and
//! prompt-engineering iteration.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use crate::error::{DbError, DbResult};

/// One row in `llm_call_log`. Mirrors the migration `008_llm_call_log.sql`
/// columns. All fields are public so callers (typically the
/// `LoopConfig::on_call_complete` hook) can build a record without
/// needing a builder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmCallLog {
    pub call_id: String,
    pub task_id: Option<String>,
    pub session_id: Option<String>,
    pub agent_type: String,
    pub app_name: Option<String>,
    pub model: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub duration_ms: u64,
    pub iterations: u32,
    pub tool_calls: u32,
    pub cost_estimate_usd: f64,
    /// One of: `completed` / `aborted` / `max_iter` / `refused` / `failed`.
    pub status: String,
    pub error: Option<String>,
    pub created_at: String,
}

/// Status values allowed in the `status` CHECK constraint.
pub const STATUS_COMPLETED: &str = "completed";
pub const STATUS_ABORTED: &str = "aborted";
pub const STATUS_MAX_ITER: &str = "max_iter";
pub const STATUS_REFUSED: &str = "refused";
pub const STATUS_FAILED: &str = "failed";

/// Repository for `llm_call_log`. Append-only: no `update` / `delete`
/// methods are exposed by design (audit log).
pub struct LlmCallLogRepository {
    conn: Arc<Mutex<Connection>>,
}

impl LlmCallLogRepository {
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    pub fn new_with_arc(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Append one row. `call_id` must be unique; the caller is expected
    /// to generate it (e.g. `format!("call-{}", uuid::Uuid::new_v4())`).
    pub fn create(&self, rec: &LlmCallLog) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO llm_call_log (
                call_id, task_id, session_id, agent_type, app_name, model,
                prompt_tokens, completion_tokens, duration_ms, iterations,
                tool_calls, cost_estimate_usd, status, error, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)"#,
            params![
                rec.call_id,
                rec.task_id,
                rec.session_id,
                rec.agent_type,
                rec.app_name,
                rec.model,
                rec.prompt_tokens,
                rec.completion_tokens,
                rec.duration_ms,
                rec.iterations,
                rec.tool_calls,
                rec.cost_estimate_usd,
                rec.status,
                rec.error,
                rec.created_at,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Fetch one call by `call_id`. Returns `DbError::NotFound` if absent.
    pub fn get_by_id(&self, call_id: &str) -> DbResult<LlmCallLog> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT call_id, task_id, session_id, agent_type, app_name, model,
                        prompt_tokens, completion_tokens, duration_ms, iterations,
                        tool_calls, cost_estimate_usd, status, error, created_at
                 FROM llm_call_log WHERE call_id = ?1",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        stmt.query_row(params![call_id], Self::row_to_record)
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                    resource: "llm_call_log",
                    field: "call_id",
                    value: call_id.to_string(),
                },
                _ => DbError::QueryFailed(e.to_string()),
            })
    }

    /// List all calls for a given `task_id`, oldest first.
    pub fn list_by_task(&self, task_id: &str) -> DbResult<Vec<LlmCallLog>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT call_id, task_id, session_id, agent_type, app_name, model,
                        prompt_tokens, completion_tokens, duration_ms, iterations,
                        tool_calls, cost_estimate_usd, status, error, created_at
                 FROM llm_call_log WHERE task_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        let rows = stmt
            .query_map(params![task_id], Self::row_to_record)
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// List the `n` most recent calls (across all tasks), newest first.
    pub fn list_recent(&self, n: usize) -> DbResult<Vec<LlmCallLog>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT call_id, task_id, session_id, agent_type, app_name, model,
                        prompt_tokens, completion_tokens, duration_ms, iterations,
                        tool_calls, cost_estimate_usd, status, error, created_at
                 FROM llm_call_log ORDER BY created_at DESC LIMIT ?1",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        let rows = stmt
            .query_map(params![n as i64], Self::row_to_record)
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Total row count.
    pub fn count(&self) -> DbResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM llm_call_log", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(|e| DbError::QueryFailed(e.to_string()))
    }

    /// Sum `cost_estimate_usd` for a given `agent_type` (e.g. `"producer"`).
    /// Used by docs/11 §"Cost attribution" recipes.
    pub fn total_cost_by_agent(&self, agent_type: &str) -> DbResult<f64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT COALESCE(SUM(cost_estimate_usd), 0.0) FROM llm_call_log WHERE agent_type = ?1",
            params![agent_type],
            |row| row.get::<_, f64>(0),
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))
    }

    fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<LlmCallLog> {
        Ok(LlmCallLog {
            call_id: row.get(0)?,
            task_id: row.get(1)?,
            session_id: row.get(2)?,
            agent_type: row.get(3)?,
            app_name: row.get(4)?,
            model: row.get(5)?,
            prompt_tokens: row.get(6)?,
            completion_tokens: row.get(7)?,
            duration_ms: row.get(8)?,
            iterations: row.get(9)?,
            tool_calls: row.get(10)?,
            cost_estimate_usd: row.get(11)?,
            status: row.get(12)?,
            error: row.get(13)?,
            created_at: row.get(14)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Database;

    fn repo() -> LlmCallLogRepository {
        let db = Database::in_memory().expect("in-memory db");
        LlmCallLogRepository::new_with_arc(db.connection_arc())
    }

    fn sample(call_id: &str, task_id: &str, status: &str) -> LlmCallLog {
        LlmCallLog {
            call_id: call_id.to_string(),
            task_id: Some(task_id.to_string()),
            session_id: Some("session-1".to_string()),
            agent_type: "producer".to_string(),
            app_name: Some("cleanroom-producer".to_string()),
            model: Some("MiniMax-M3".to_string()),
            prompt_tokens: 522,
            completion_tokens: 1024,
            duration_ms: 29_400,
            iterations: 1,
            tool_calls: 0,
            cost_estimate_usd: 0.0169,
            status: status.to_string(),
            error: None,
            created_at: "2026-06-02T12:00:00Z".to_string(),
        }
    }

    #[test]
    fn create_and_get_by_id() {
        let repo = repo();
        let rec = sample("call-001", "task-A", STATUS_COMPLETED);
        repo.create(&rec).expect("create");
        let got = repo.get_by_id("call-001").expect("get");
        assert_eq!(got.prompt_tokens, 522);
        assert_eq!(got.status, "completed");
    }

    #[test]
    fn get_by_id_missing_returns_not_found() {
        let repo = repo();
        let err = repo.get_by_id("nope").expect_err("should miss");
        assert!(matches!(err, DbError::NotFound { .. }));
    }

    #[test]
    fn list_by_task_returns_only_matching_rows_in_order() {
        let repo = repo();
        repo.create(&sample("c1", "task-A", STATUS_COMPLETED)).unwrap();
        repo.create(&sample("c2", "task-B", STATUS_COMPLETED)).unwrap();
        repo.create(&sample("c3", "task-A", STATUS_ABORTED)).unwrap();
        let rows = repo.list_by_task("task-A").expect("list");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].call_id, "c1");
        assert_eq!(rows[1].call_id, "c3");
        assert_eq!(rows[1].status, "aborted");
    }

    #[test]
    fn list_recent_orders_newest_first() {
        let repo = repo();
        repo.create(&sample("old", "t", STATUS_COMPLETED)).unwrap();
        repo.create(&sample("new", "t", STATUS_COMPLETED)).unwrap();
        let rows = repo.list_recent(10).expect("recent");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].call_id, "new");
        assert_eq!(rows[1].call_id, "old");
    }

    #[test]
    fn list_recent_respects_limit() {
        let repo = repo();
        for i in 0..5 {
            repo.create(&sample(&format!("c{i}"), "t", STATUS_COMPLETED)).unwrap();
        }
        let rows = repo.list_recent(2).expect("recent limit 2");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn count_and_total_cost_by_agent() {
        let repo = repo();
        repo.create(&sample("a", "t", STATUS_COMPLETED)).unwrap();
        repo.create(&sample("b", "t", STATUS_COMPLETED)).unwrap();
        assert_eq!(repo.count().unwrap(), 2);
        let cost = repo.total_cost_by_agent("producer").unwrap();
        assert!((cost - 0.0338).abs() < 1e-6);
        // Unknown agent_type -> 0.0, not an error.
        assert_eq!(repo.total_cost_by_agent("consumer").unwrap(), 0.0);
    }

    #[test]
    fn status_field_rejects_unknown_values_at_db_level() {
        // The CHECK constraint rejects anything outside the 5 known
        // statuses. This is the schema's guarantee; we exercise it
        // here to lock down the contract that the hook must respect.
        let repo = repo();
        let mut rec = sample("c", "t", STATUS_COMPLETED);
        rec.status = "nonsense".to_string();
        let err = repo.create(&rec).expect_err("should fail");
        assert!(matches!(err, DbError::QueryFailed(_)));
    }
}
