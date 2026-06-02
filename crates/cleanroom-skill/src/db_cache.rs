//! SQLite cache layer for skills.
//!
//! See `PLAN2.md` §E.3 for the schema and operations.

use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection};

use crate::error::SkillResult;
use crate::model::SkillDocument;

/// Thin wrapper around a SQLite connection. Caller supplies the `Connection`
/// (typically the same one used by `cleanroom-db`); we just create the
/// `skill_index` table if it doesn't exist and run migrations.
pub struct SkillCacheRepository {
    conn: Arc<Mutex<Connection>>,
}

impl SkillCacheRepository {
    /// Open a cache backed by a file.
    pub fn open(path: &Path) -> SkillResult<Self> {
        let conn = Connection::open(path)?;
        let me = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        me.migrate()?;
        Ok(me)
    }

    /// Open an in-memory cache (used in tests).
    pub fn in_memory() -> SkillResult<Self> {
        let conn = Connection::open_in_memory()?;
        let me = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        me.migrate()?;
        Ok(me)
    }

    /// Use an existing `Arc<Mutex<Connection>>` (so we can share a DB
    /// with `cleanroom-db`).
    pub fn with_connection(conn: Arc<Mutex<Connection>>) -> SkillResult<Self> {
        let me = Self { conn };
        me.migrate()?;
        Ok(me)
    }

    fn migrate(&self) -> SkillResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS skill_index (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                scope           TEXT NOT NULL,
                path            TEXT NOT NULL,
                description     TEXT NOT NULL,
                content_hash    TEXT NOT NULL,
                last_modified   INTEGER,
                frontmatter     TEXT NOT NULL,
                body            TEXT NOT NULL,
                allowed_tools   TEXT,
                denied_tools    TEXT,
                applies_to      TEXT,
                token_budget    INTEGER NOT NULL DEFAULT 4096,
                priority        TEXT NOT NULL DEFAULT 'normal',
                sdef_shard_uri  TEXT,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL,
                UNIQUE(name, scope)
            );
            CREATE INDEX IF NOT EXISTS idx_skill_index_scope ON skill_index(scope);
            CREATE INDEX IF NOT EXISTS idx_skill_index_name  ON skill_index(name);
            "#,
        )?;
        Ok(())
    }

    pub fn upsert(&self, skill: &SkillDocument) -> SkillResult<()> {
        let now = chrono::Utc::now().timestamp();
        let allowed_tools_json = serde_json::to_string(&skill.allowed_tools).unwrap_or_else(|_| "[]".into());
        let denied_tools_json = serde_json::to_string(&skill.denied_tools).unwrap_or_else(|_| "[]".into());
        let applies_to_json = serde_json::to_string(&skill.applies_to).unwrap_or_else(|_| "[]".into());
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"
            INSERT INTO skill_index
              (id, name, scope, path, description, content_hash, last_modified,
               frontmatter, body, allowed_tools, denied_tools, applies_to,
               token_budget, priority, sdef_shard_uri, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
            ON CONFLICT(name, scope) DO UPDATE SET
              path            = excluded.path,
              description     = excluded.description,
              content_hash    = excluded.content_hash,
              last_modified   = excluded.last_modified,
              frontmatter     = excluded.frontmatter,
              body            = excluded.body,
              allowed_tools   = excluded.allowed_tools,
              denied_tools    = excluded.denied_tools,
              applies_to      = excluded.applies_to,
              token_budget    = excluded.token_budget,
              priority        = excluded.priority,
              sdef_shard_uri  = excluded.sdef_shard_uri,
              updated_at      = excluded.updated_at
            "#,
            params![
                skill.id,
                skill.name,
                format!("{:?}", skill.scope),
                skill.path.to_string_lossy().to_string(),
                skill.description,
                skill.hash,
                skill.last_modified,
                format!("{:?}", skill.x_cleanroom_or_empty()),
                skill.body,
                allowed_tools_json,
                denied_tools_json,
                applies_to_json,
                skill.token_budget,
                skill.priority,
                skill.sdef_shard_uri,
                now,
                now,
            ],
        )?;
        Ok(())
    }

    pub fn get(&self, name: &str, scope: &str) -> SkillResult<Option<SkillDocument>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, scope, path, description, content_hash, last_modified, body, \
                    allowed_tools, denied_tools, applies_to, token_budget, priority, sdef_shard_uri \
             FROM skill_index WHERE name = ?1 AND scope = ?2",
        )?;
        let mut rows = stmt.query(params![name, scope])?;
        if let Some(row) = rows.next()? {
            // Lightweight deserialization — full x_cleanroom reconstruction
            // is deferred (the body is enough to inject).
            let _id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let _scope: String = row.get(2)?;
            let path: String = row.get(3)?;
            let description: String = row.get(4)?;
            let hash: String = row.get(5)?;
            let last_modified: Option<i64> = row.get(6)?;
            let body: String = row.get(7)?;
            let allowed_tools_json: String = row.get(8)?;
            let denied_tools_json: String = row.get(9)?;
            let _applies_to: String = row.get(10)?;
            let token_budget: u32 = row.get(11)?;
            let priority: String = row.get(12)?;
            let sdef_shard_uri: Option<String> = row.get(13)?;
            let allowed_tools: Vec<String> = serde_json::from_str(&allowed_tools_json).unwrap_or_default();
            let denied_tools: Vec<String> = serde_json::from_str(&denied_tools_json).unwrap_or_default();
            let doc = SkillDocument {
                id: format!("{name}+{hash}"),
                name: name.clone(),
                description,
                license: None,
                compatibility: None,
                tags: vec![],
                allowed_tools,
                denied_tools,
                allowed_paths: vec![],
                staging: None,
                output_schema: None,
                gates: vec![],
                divergence_spec: None,
                applies_to: vec![],
                token_budget,
                priority,
                trigger: false,
                body,
                path: std::path::PathBuf::from(path),
                scope: crate::model::SkillScope::Builtin,
                hash,
                last_modified,
                sdef_shard_uri,
                metadata: std::collections::HashMap::new(),
            };
            Ok(Some(doc))
        } else {
            Ok(None)
        }
    }

    pub fn list(&self, scope: Option<&str>) -> SkillResult<Vec<SkillDocument>> {
        let conn = self.conn.lock().unwrap();
        let (sql, params_vec): (&str, Vec<rusqlite::types::Value>) = if let Some(s) = scope {
            (
                "SELECT name, path, description, content_hash, last_modified, body, \
                        allowed_tools, token_budget, priority, sdef_shard_uri \
                 FROM skill_index WHERE scope = ?1 ORDER BY priority DESC, name ASC",
                vec![rusqlite::types::Value::Text(s.to_string())],
            )
        } else {
            (
                "SELECT name, path, description, content_hash, last_modified, body, \
                        allowed_tools, token_budget, priority, sdef_shard_uri \
                 FROM skill_index ORDER BY priority DESC, name ASC",
                vec![],
            )
        };
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(params_vec.iter()), |row| {
            let name: String = row.get(0)?;
            let path: String = row.get(1)?;
            let description: String = row.get(2)?;
            let hash: String = row.get(3)?;
            let last_modified: Option<i64> = row.get(4)?;
            let body: String = row.get(5)?;
            let allowed_tools_json: String = row.get(6)?;
            let token_budget: u32 = row.get(7)?;
            let priority: String = row.get(8)?;
            let sdef_shard_uri: Option<String> = row.get(9)?;
            let allowed_tools: Vec<String> =
                serde_json::from_str(&allowed_tools_json).unwrap_or_default();
            Ok(SkillDocument {
                id: format!("{name}+{hash}"),
                name: name.clone(),
                description,
                license: None,
                compatibility: None,
                tags: vec![],
                allowed_tools,
                denied_tools: vec![],
                allowed_paths: vec![],
                staging: None,
                output_schema: None,
                gates: vec![],
                divergence_spec: None,
                applies_to: vec![],
                token_budget,
                priority,
                trigger: false,
                body,
                path: std::path::PathBuf::from(path),
                scope: crate::model::SkillScope::Builtin,
                hash,
                last_modified,
                sdef_shard_uri,
                metadata: std::collections::HashMap::new(),
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn delete(&self, name: &str, scope: &str) -> SkillResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM skill_index WHERE name = ?1 AND scope = ?2",
            params![name, scope],
        )?;
        Ok(())
    }
}

/// Helper used by `upsert` to serialize the x_cleanroom block as a debug
/// string (lightweight, schema-version-tolerant).
trait SkillXCleanroomDebug {
    fn x_cleanroom_or_empty(&self) -> String;
}
impl SkillXCleanroomDebug for SkillDocument {
    fn x_cleanroom_or_empty(&self) -> String {
        format!(
            "{{allowed_tools:{:?}, denied:{:?}, paths:{:?}, staging:{:?}, gates:{:?}}}",
            self.allowed_tools,
            self.denied_tools,
            self.allowed_paths,
            self.staging,
            self.gates
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SkillScope;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn fake_skill(name: &str) -> SkillDocument {
        SkillDocument {
            id: format!("{name}+h"),
            name: name.to_string(),
            description: "d".into(),
            license: None,
            compatibility: None,
            tags: vec![],
            allowed_tools: vec!["fs.read_file".into()],
            denied_tools: vec![],
            allowed_paths: vec![],
            staging: None,
            output_schema: None,
            gates: vec![],
            divergence_spec: None,
            applies_to: vec![],
            token_budget: 4096,
            priority: "normal".into(),
            trigger: false,
            body: "body".into(),
            path: PathBuf::from("/x"),
            scope: SkillScope::Builtin,
            hash: "h".into(),
            last_modified: None,
            sdef_shard_uri: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn upsert_and_get() {
        let repo = SkillCacheRepository::in_memory().unwrap();
        let mut s = fake_skill("a");
        s.scope = SkillScope::Builtin;
        repo.upsert(&s).unwrap();
        let got = repo.get("a", "Builtin").unwrap().expect("found");
        assert_eq!(got.name, "a");
        assert_eq!(got.allowed_tools, vec!["fs.read_file"]);
    }
}
