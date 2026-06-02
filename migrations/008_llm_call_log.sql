-- ============================================
-- Cleanroom Agent - LLM Call Log
-- Version: 008
-- ============================================
--
-- Phase 0.9: persist one row per LLM call so
-- `cleanroom-cli inspect llm-log` can replay / inspect history.
-- Driven by `LoopConfig::on_call_complete` hook, fired from
-- `llm_loop::run_loop` and `llm_loop::run_loop_via_basic_agent`.
--
-- Schema is intentionally append-only (no UPDATE); rows reflect the raw
-- call event. The `status` field is one of:
--   * completed   - LLM returned a usable response
--   * aborted     - cost limit exceeded / caller-aborted
--   * max_iter    - hit `LoopConfig::max_iterations`
--   * refused     - LLM returned empty text (refused to answer)
--   * failed      - transport / parse error before getting a response

CREATE TABLE llm_call_log (
    call_id          TEXT PRIMARY KEY,
    task_id          TEXT,
    session_id       TEXT,
    agent_type       TEXT NOT NULL,                       -- 'producer' / 'consumer' / 'meta'
    app_name         TEXT,
    model            TEXT,
    prompt_tokens    INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    duration_ms      INTEGER NOT NULL DEFAULT 0,
    iterations       INTEGER NOT NULL DEFAULT 1,
    tool_calls       INTEGER NOT NULL DEFAULT 0,
    cost_estimate_usd REAL NOT NULL DEFAULT 0.0,
    status           TEXT NOT NULL CHECK (status IN (
                        'completed', 'aborted', 'max_iter', 'refused', 'failed'
                    )),
    error            TEXT,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_llm_call_log_task_id     ON llm_call_log(task_id);
CREATE INDEX IF NOT EXISTS idx_llm_call_log_created_at  ON llm_call_log(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_log_status      ON llm_call_log(status);
CREATE INDEX IF NOT EXISTS idx_llm_call_log_agent_type  ON llm_call_log(agent_type);
