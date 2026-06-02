-- ============================================
-- Cleanroom Agent - Skill Index
-- Version: 010
-- ============================================
--
-- PLAN2 Phase E.1: persist the Skills index so it survives restarts and
-- can be queried by the MCP tools / CLI without re-scanning the
-- filesystem on every call. Backed by
-- `cleanroom_skill::db_cache::SkillCacheRepository` (in
-- `cleanroom-skill/src/db_cache.rs`).
--
-- Schema mirrors the fields of `cleanroom_skill::model::SkillDocument`:
--   * name + scope form the UNIQUE constraint (project-local collisions
--     are resolved by the `scope` priority order in `model::SkillScope`).
--   * content_hash (SHA-256) is used to compute the stable `id`.
--   * frontmatter is the raw YAML blob (for re-parse without filesystem
--     access).
--   * allowed_tools / denied_tools / applies_to are stored as JSON arrays
--     for cheap retrieval (reconstructed back into Vec<String> on read).
--   * FTS5 virtual table enables fast keyword search by name / description /
--     body.

CREATE TABLE skill_index (
    id              TEXT PRIMARY KEY,                 -- name + 12-char content-hash prefix
    name            TEXT NOT NULL,
    scope           TEXT NOT NULL,                    -- builtin | project-cleanroom | project-agents | user-cleanroom | user-agents
    path            TEXT NOT NULL,                    -- absolute filesystem path to the SKILL.md
    description     TEXT NOT NULL,
    content_hash    TEXT NOT NULL,                    -- SHA-256 hex
    last_modified   INTEGER,                          -- Unix seconds
    frontmatter     TEXT NOT NULL,                    -- raw YAML between the `---` fences
    body            TEXT NOT NULL,                    -- markdown body after the closing fence
    allowed_tools   TEXT,                             -- JSON array (Vec<String>)
    denied_tools    TEXT,                             -- JSON array
    applies_to      TEXT,                             -- JSON array (TaskType list)
    token_budget    INTEGER NOT NULL DEFAULT 4096,
    priority        TEXT NOT NULL DEFAULT 'normal',  -- low | normal | high
    sdef_shard_uri  TEXT,                             -- sdef://cleanroom/skills/<name>
    created_at      INTEGER NOT NULL,                 -- Unix seconds
    updated_at      INTEGER NOT NULL,
    UNIQUE(name, scope)
);

CREATE INDEX idx_skill_index_scope ON skill_index(scope);
CREATE INDEX idx_skill_index_name  ON skill_index(name);
CREATE INDEX idx_skill_index_priority ON skill_index(priority);

-- FTS5 virtual table for fast text search across the catalog.
-- The 'content=' option keeps the FTS table a thin index over skill_index;
-- rebuilding is done via INSERT triggers (not declared here for brevity).
CREATE VIRTUAL TABLE IF NOT EXISTS skill_index_fts USING fts5(
    name,
    description,
    body,
    content='skill_index',
    content_rowid='rowid'
);
