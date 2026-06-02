//! Embedded SQL schema for initial database setup.
//! Generated from migrations/001_initial_schema.sql

/// The complete initial schema SQL.
pub const INITIAL_SCHEMA_SQL: &str = include_str!("../../../migrations/001_initial_schema.sql");

/// Phase 0.9: LLM call log table (one row per `run_loop` invocation).
/// Appended after `INITIAL_SCHEMA_SQL` so in-memory test databases also
/// have the `llm_call_log` table; production databases apply it via
/// `migrations::run_pending_at(conn, dir)` from `008_llm_call_log.sql`.
pub const LLM_CALL_LOG_SCHEMA_SQL: &str = include_str!("../../../migrations/008_llm_call_log.sql");
