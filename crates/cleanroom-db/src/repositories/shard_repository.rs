//! Shard repository for shard CRUD operations.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tracing::instrument;

use crate::error::{DbError, DbResult};

/// Shard status enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShardStatus {
    Pending,
    Generating,
    Generated,
    Validating,
    Validated,
    CodeGenerated,
    Tested,
    Failed,
}

impl ShardStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Generating => "generating",
            Self::Generated => "generated",
            Self::Validating => "validating",
            Self::Validated => "validated",
            Self::CodeGenerated => "code_generated",
            Self::Tested => "tested",
            Self::Failed => "failed",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "generating" => Some(Self::Generating),
            "generated" => Some(Self::Generated),
            "validating" => Some(Self::Validating),
            "validated" => Some(Self::Validated),
            "code_generated" => Some(Self::CodeGenerated),
            "tested" => Some(Self::Tested),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

/// Shard model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    pub shard_id: String,
    pub document_name: String,
    pub sdef_uri: String,
    pub section_type: String,
    pub file_path: Option<String>,
    pub status: ShardStatus,
    pub content_hash: Option<String>,
    pub token_estimate: Option<i32>,
    pub version: i32,
    pub created_at: String,
    pub updated_at: String,
}

/// Shard repository.
pub struct ShardRepository {
    conn: Mutex<Connection>,
}

impl ShardRepository {
    /// Create a new shard repository.
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Mutex::new(conn),
        }
    }

    /// Create a new shard.
    #[instrument(skip_all)]
    pub fn create(&self, shard: &Shard) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO shards (
                shard_id, document_name, sdef_uri, section_type, file_path,
                status, content_hash, token_estimate, version
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"#,
            params![
                shard.shard_id,
                shard.document_name,
                shard.sdef_uri,
                shard.section_type,
                shard.file_path,
                shard.status.as_str(),
                shard.content_hash,
                shard.token_estimate,
                shard.version,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Get a shard by ID.
    #[instrument(skip_all)]
    pub fn get(&self, shard_id: &str) -> DbResult<Shard> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT shard_id, document_name, sdef_uri, section_type, file_path,
                   status, content_hash, token_estimate, version, created_at, updated_at
                   FROM shards WHERE shard_id = ?1"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![shard_id], |row| {
            let status_str: String = row.get(5)?;
            Ok(Shard {
                shard_id: row.get(0)?,
                document_name: row.get(1)?,
                sdef_uri: row.get(2)?,
                section_type: row.get(3)?,
                file_path: row.get(4)?,
                status: ShardStatus::from_str(&status_str).unwrap_or(ShardStatus::Pending),
                content_hash: row.get(6)?,
                token_estimate: row.get(7)?,
                version: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "shard",
                field: "shard_id",
                value: shard_id.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    /// Get a shard by URI.
    #[instrument(skip_all)]
    pub fn get_by_uri(&self, sdef_uri: &str) -> DbResult<Shard> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT shard_id, document_name, sdef_uri, section_type, file_path,
                   status, content_hash, token_estimate, version, created_at, updated_at
                   FROM shards WHERE sdef_uri = ?1"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![sdef_uri], |row| {
            let status_str: String = row.get(5)?;
            Ok(Shard {
                shard_id: row.get(0)?,
                document_name: row.get(1)?,
                sdef_uri: row.get(2)?,
                section_type: row.get(3)?,
                file_path: row.get(4)?,
                status: ShardStatus::from_str(&status_str).unwrap_or(ShardStatus::Pending),
                content_hash: row.get(6)?,
                token_estimate: row.get(7)?,
                version: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "shard",
                field: "sdef_uri",
                value: sdef_uri.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    /// Update shard status.
    #[instrument(skip_all)]
    pub fn update_status(&self, shard_id: &str, status: ShardStatus) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute(
                "UPDATE shards SET status = ?1, updated_at = CURRENT_TIMESTAMP WHERE shard_id = ?2",
                params![status.as_str(), shard_id],
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        if rows == 0 {
            return Err(DbError::NotFound {
                resource: "shard",
                field: "shard_id",
                value: shard_id.to_string(),
            });
        }
        Ok(())
    }

    /// Update shard content hash.
    #[instrument(skip_all)]
    pub fn update_content_hash(&self, shard_id: &str, content_hash: &str) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute(
                "UPDATE shards SET content_hash = ?1, updated_at = CURRENT_TIMESTAMP WHERE shard_id = ?2",
                params![content_hash, shard_id],
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        if rows == 0 {
            return Err(DbError::NotFound {
                resource: "shard",
                field: "shard_id",
                value: shard_id.to_string(),
            });
        }
        Ok(())
    }

    /// List shards by document.
    #[instrument(skip_all)]
    pub fn list_by_document(&self, document_name: &str) -> DbResult<Vec<Shard>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT shard_id, document_name, sdef_uri, section_type, file_path,
                   status, content_hash, token_estimate, version, created_at, updated_at
                   FROM shards WHERE document_name = ?1 ORDER BY created_at"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let shards = stmt
            .query_map(params![document_name], |row| {
                let status_str: String = row.get(5)?;
                Ok(Shard {
                    shard_id: row.get(0)?,
                    document_name: row.get(1)?,
                    sdef_uri: row.get(2)?,
                    section_type: row.get(3)?,
                    file_path: row.get(4)?,
                    status: ShardStatus::from_str(&status_str).unwrap_or(ShardStatus::Pending),
                    content_hash: row.get(6)?,
                    token_estimate: row.get(7)?,
                    version: row.get(8)?,
                    created_at: row.get(9)?,
                    updated_at: row.get(10)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(shards)
    }

    /// List shards by status.
    #[instrument(skip_all)]
    pub fn list_by_status(&self, status: ShardStatus) -> DbResult<Vec<Shard>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT shard_id, document_name, sdef_uri, section_type, file_path,
                   status, content_hash, token_estimate, version, created_at, updated_at
                   FROM shards WHERE status = ?1 ORDER BY created_at"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let shards = stmt
            .query_map(params![status.as_str()], |row| {
                let status_str: String = row.get(5)?;
                Ok(Shard {
                    shard_id: row.get(0)?,
                    document_name: row.get(1)?,
                    sdef_uri: row.get(2)?,
                    section_type: row.get(3)?,
                    file_path: row.get(4)?,
                    status: ShardStatus::from_str(&status_str).unwrap_or(ShardStatus::Pending),
                    content_hash: row.get(6)?,
                    token_estimate: row.get(7)?,
                    version: row.get(8)?,
                    created_at: row.get(9)?,
                    updated_at: row.get(10)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(shards)
    }

    /// Delete a shard.
    #[instrument(skip_all)]
    pub fn delete(&self, shard_id: &str) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute("DELETE FROM shards WHERE shard_id = ?1", params![shard_id])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        if rows == 0 {
            return Err(DbError::NotFound {
                resource: "shard",
                field: "shard_id",
                value: shard_id.to_string(),
            });
        }
        Ok(())
    }
}

// Note: Repository-level unit tests require a shared connection since
// ShardRepository takes owned Connection. Integration tests in cleanroom-mcp
// cover shard CRUD operations end-to-end via MCP tools (list_shards,
// export_shard, import_shard).