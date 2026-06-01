-- ============================================
-- Cleanroom Agent - LLM Task Types
-- Version: 007
-- ============================================
--
-- Phase 0.4: extend the `tasks.task_type` CHECK constraint to allow the
-- two new LLM-driven task types introduced alongside `llm_loop::run_loop`:
--   * `LLM_ANALYZE_FILE`  -- leaf-level Producer task (replaces
--                              `REPO_ANALYZE` once the pipeline is LLM-driven)
--   * `LLM_GENERATE_CODE` -- leaf-level Consumer task (replaces
--                              `GENERATE_CODE`)
--
-- SQLite CHECK constraints cannot be altered in place; we follow the same
-- rebuild-table pattern used by migration 005 for `agents`. No foreign key
-- references `tasks.task_id`, so a full rebuild is safe.

-- Disable FKs while we rebuild; no FK points at tasks anyway, but be safe.
PRAGMA foreign_keys = OFF;

-- Rebuild `tasks` with the extended CHECK constraint.
CREATE TABLE tasks_new (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL CHECK (task_type IN (
        'REPO_ANALYZE', 'EXTRACT_METADATA', 'EXTRACT_ARCHITECTURE',
        'EXTRACT_DATA_MODEL', 'EXTRACT_MODULE', 'EXTRACT_UI',
        'EXTRACT_TESTS', 'INFER_DESIGN_DECISIONS', 'VALIDATE_SHARD',
        'GENERATE_CODE', 'RUN_TESTS', 'MERGE_CODE', 'IMPORT_SDEF', 'EXPORT_SDEF',
        'VALIDATE_DATA_MODEL', 'VALIDATE_CROSS_FILE', 'ROUNDTRIP_VERIFY',
        -- Phase 0.4 additions:
        'LLM_ANALYZE_FILE', 'LLM_GENERATE_CODE'
    )),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'assigned', 'in_progress', 'completed',
                          'failed', 'retrying', 'failed_permanently')),
    priority INTEGER NOT NULL DEFAULT 5,
    input_json TEXT NOT NULL DEFAULT '{}',
    output_json TEXT,
    error_message TEXT,
    assigned_to TEXT,
    progress REAL NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 1),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    last_heartbeat TIMESTAMP,
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    version INTEGER NOT NULL DEFAULT 1
);

INSERT OR IGNORE INTO tasks_new SELECT * FROM tasks;
DROP TABLE tasks;
ALTER TABLE tasks_new RENAME TO tasks;

-- Re-create the indexes that were on the original `tasks` table.
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_to ON tasks(assigned_to);
CREATE INDEX IF NOT EXISTS idx_tasks_type_status ON tasks(task_type, status);

PRAGMA foreign_keys = ON;
