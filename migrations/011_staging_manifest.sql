-- ============================================
-- Cleanroom Agent - Staging Manifest
-- Version: 011
-- ============================================
--
-- PLAN2 Phase E.2: per-task record of every `staging.*` write the LLM
-- makes, so an orchestrator can resume after a crash and the verifier
-- can reason about the diff between staging and the live source tree.
-- Backed by `cleanroom_staging::manifest::StagingEntry` (in
-- `cleanroom-staging/src/manifest.rs`).
--
-- Append-only. Rows are inserted in the order the LLM makes tool calls
-- and consulted during verification. Deletion happens on `staging.commit`
-- or `staging.abort`.

CREATE TABLE staging_manifest (
    task_id      TEXT NOT NULL,            -- foreign key concept: links to a per-task staging workspace
    file_path    TEXT NOT NULL,            -- relative path under the staging root (POSIX-style, no `..`)
    content_hash TEXT NOT NULL,            -- SHA-256 hex of the staged content (empty string for delete ops)
    op           TEXT NOT NULL,            -- write | edit | delete
    created_at   INTEGER NOT NULL,         -- Unix seconds
    PRIMARY KEY (task_id, file_path, created_at)
);

CREATE INDEX idx_staging_manifest_task ON staging_manifest(task_id);
CREATE INDEX idx_staging_manifest_path ON staging_manifest(file_path);
