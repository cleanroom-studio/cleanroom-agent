//! Fingerprint repository for consistency checking.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::instrument;

use crate::error::{DbError, DbResult};

/// Fingerprint model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint {
    pub entity_uri: String,
    pub document_name: String,
    pub entity_type: String,
    pub sdef_hash: Option<String>,
    pub db_hash: Option<String>,
    pub code_hash: Option<String>,
    pub code_path: Option<String>,
    pub last_checked_at: String,
    pub last_consistent_at: Option<String>,
}

/// Fingerprint repository.
pub struct FingerprintRepository {
    conn: Arc<Mutex<Connection>>,
}

impl FingerprintRepository {
    /// Create a new fingerprint repository.
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    /// Create from an existing Arc-wrapped connection.
    pub fn from_arc(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Upsert a fingerprint.
    #[instrument(skip_all)]
    pub fn upsert(&self, fp: &Fingerprint) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO fingerprints (
                entity_uri, document_name, entity_type, sdef_hash, db_hash, code_hash,
                code_path, last_checked_at, last_consistent_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, CURRENT_TIMESTAMP, ?8)
               ON CONFLICT(document_name, entity_uri) DO UPDATE SET
                   sdef_hash = ?4,
                   db_hash = ?5,
                   code_hash = ?6,
                   code_path = ?7,
                   last_checked_at = CURRENT_TIMESTAMP,
                   last_consistent_at = ?8"#,
            params![
                fp.entity_uri,
                fp.document_name,
                fp.entity_type,
                fp.sdef_hash,
                fp.db_hash,
                fp.code_hash,
                fp.code_path,
                fp.last_consistent_at,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Get a fingerprint by entity URI.
    #[instrument(skip_all)]
    pub fn get(&self, document_name: &str, entity_uri: &str) -> DbResult<Fingerprint> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT entity_uri, document_name, entity_type, sdef_hash, db_hash,
                   code_hash, code_path, last_checked_at, last_consistent_at
                   FROM fingerprints WHERE document_name = ?1 AND entity_uri = ?2"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![document_name, entity_uri], |row| {
            Ok(Fingerprint {
                entity_uri: row.get(0)?,
                document_name: row.get(1)?,
                entity_type: row.get(2)?,
                sdef_hash: row.get(3)?,
                db_hash: row.get(4)?,
                code_hash: row.get(5)?,
                code_path: row.get(6)?,
                last_checked_at: row.get(7)?,
                last_consistent_at: row.get(8)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "fingerprint",
                field: "entity_uri",
                value: entity_uri.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    /// List all fingerprints for a document.
    #[instrument(skip_all)]
    pub fn list_by_document(&self, document_name: &str) -> DbResult<Vec<Fingerprint>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT entity_uri, document_name, entity_type, sdef_hash, db_hash,
                   code_hash, code_path, last_checked_at, last_consistent_at
                   FROM fingerprints WHERE document_name = ?1"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let fps = stmt
            .query_map(params![document_name], |row| {
                Ok(Fingerprint {
                    entity_uri: row.get(0)?,
                    document_name: row.get(1)?,
                    entity_type: row.get(2)?,
                    sdef_hash: row.get(3)?,
                    db_hash: row.get(4)?,
                    code_hash: row.get(5)?,
                    code_path: row.get(6)?,
                    last_checked_at: row.get(7)?,
                    last_consistent_at: row.get(8)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(fps)
    }

    /// List inconsistent fingerprints (where hashes don't match).
    #[instrument(skip_all)]
    pub fn list_inconsistent(&self, document_name: &str) -> DbResult<Vec<Fingerprint>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT entity_uri, document_name, entity_type, sdef_hash, db_hash,
                   code_hash, code_path, last_checked_at, last_consistent_at
                   FROM fingerprints
                   WHERE document_name = ?1
                     AND (sdef_hash != db_hash OR db_hash != code_hash OR sdef_hash != code_hash)"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let fps = stmt
            .query_map(params![document_name], |row| {
                Ok(Fingerprint {
                    entity_uri: row.get(0)?,
                    document_name: row.get(1)?,
                    entity_type: row.get(2)?,
                    sdef_hash: row.get(3)?,
                    db_hash: row.get(4)?,
                    code_hash: row.get(5)?,
                    code_path: row.get(6)?,
                    last_checked_at: row.get(7)?,
                    last_consistent_at: row.get(8)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(fps)
    }

    /// Update only the code hash (for code generation).
    #[instrument(skip_all)]
    pub fn update_code_hash(&self, document_name: &str, entity_uri: &str, code_hash: &str, code_path: &str) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"UPDATE fingerprints SET code_hash = ?1, code_path = ?2, last_checked_at = CURRENT_TIMESTAMP
               WHERE document_name = ?3 AND entity_uri = ?4"#,
            params![code_hash, code_path, document_name, entity_uri],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }
}