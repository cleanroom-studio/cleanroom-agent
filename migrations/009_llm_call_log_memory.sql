-- ============================================
-- Cleanroom Agent - LLM call log: add memory_messages_at_call column
-- Version: 009
-- ============================================
--
-- Phase 0.10: the LLM call log records how many memory-history messages
-- were prepended to each LLM call (via `MemoryProvider::recall()`).
-- Production DBs that were initialized under migration 008 get the
-- column via this `ALTER TABLE`. In-memory test databases get it
-- through the embedded schema (see `embedded_schema.rs`).

ALTER TABLE llm_call_log
    ADD COLUMN memory_messages_at_call INTEGER NOT NULL DEFAULT 0;
