//! Embedded SQL schema for initial database setup.
//! Generated from migrations/001_initial_schema.sql

/// The complete initial schema SQL.
pub const INITIAL_SCHEMA_SQL: &str = include_str!("../../../migrations/001_initial_schema.sql");

/// Phase 0.9: LLM call log table (one row per `run_loop` invocation).
/// Appended after `INITIAL_SCHEMA_SQL` so in-memory test databases also
/// have the `llm_call_log` table; production databases apply it via
/// `migrations::run_pending_at(conn, dir)` from `008_llm_call_log.sql`.
pub const LLM_CALL_LOG_SCHEMA_SQL: &str = include_str!("../../../migrations/008_llm_call_log.sql");

/// Phase 0.10: add `memory_messages_at_call` column to `llm_call_log`.
/// Tracks how many history messages `run_loop` prepended to each
/// LLM call (from `MemoryProvider::recall()`). Production DBs apply
/// this via `migrations::run_pending_at(conn, dir)` from
/// `009_llm_call_log_memory.sql`; in-memory test DBs get it
/// appended here.
pub const LLM_CALL_LOG_MEMORY_SCHEMA_SQL: &str =
    include_str!("../../../migrations/009_llm_call_log_memory.sql");

/// Phase 1.1 (close-out): add `module_name` column to
/// `design_decisions` so per-module decisions can be queried
/// cleanly (without the pre-1.1 `context LIKE 'module=<name>;%'`
/// regex workaround). Production DBs apply this via
/// `migrations::run_pending_at` from
/// `013_design_decisions_module_name.sql`; in-memory test DBs
/// get it appended here.
pub const DESIGN_DECISIONS_MODULE_NAME_SCHEMA_SQL: &str =
    include_str!("../../../migrations/013_design_decisions_module_name.sql");
