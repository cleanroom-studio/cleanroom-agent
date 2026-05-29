//! Checkpoint repository for workflow recovery.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tracing::instrument;

use crate::error::{DbError, DbResult};

/// Checkpoint model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub checkpoint_id: String,
    pub document_name: String,
    pub description: Option<String>,
    pub created_at: String,
    pub task_snapshot_json: String,
    pub shard_snapshot_json: String,
}

/// Checkpoint repository.
pub struct CheckpointRepository {
    conn: Mutex<Connection>,
}

impl CheckpointRepository {
    /// Create a new checkpoint repository.
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Mutex::new(conn),
        }
    }

    /// Create a checkpoint.
    #[instrument(skip_all)]
    pub fn create(&self, checkpoint: &Checkpoint) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO checkpoints (
                checkpoint_id, document_name, description, task_snapshot_json, shard_snapshot_json
            ) VALUES (?1, ?2, ?3, ?4, ?5)"#,
            params![
                checkpoint.checkpoint_id,
                checkpoint.document_name,
                checkpoint.description,
                checkpoint.task_snapshot_json,
                checkpoint.shard_snapshot_json,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Get a checkpoint by ID.
    #[instrument(skip_all)]
    pub fn get(&self, checkpoint_id: &str) -> DbResult<Checkpoint> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT checkpoint_id, document_name, description, created_at,
                   task_snapshot_json, shard_snapshot_json
                   FROM checkpoints WHERE checkpoint_id = ?1"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![checkpoint_id], |row| {
            Ok(Checkpoint {
                checkpoint_id: row.get(0)?,
                document_name: row.get(1)?,
                description: row.get(2)?,
                created_at: row.get(3)?,
                task_snapshot_json: row.get(4)?,
                shard_snapshot_json: row.get(5)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "checkpoint",
                field: "checkpoint_id",
                value: checkpoint_id.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    /// List checkpoints for a document.
    #[instrument(skip_all)]
    pub fn list(&self, document_name: &str) -> DbResult<Vec<Checkpoint>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT checkpoint_id, document_name, description, created_at,
                   task_snapshot_json, shard_snapshot_json
                   FROM checkpoints WHERE document_name = ?1
                   ORDER BY created_at DESC"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let checkpoints = stmt
            .query_map(params![document_name], |row| {
                Ok(Checkpoint {
                    checkpoint_id: row.get(0)?,
                    document_name: row.get(1)?,
                    description: row.get(2)?,
                    created_at: row.get(3)?,
                    task_snapshot_json: row.get(4)?,
                    shard_snapshot_json: row.get(5)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(checkpoints)
    }

    /// Delete a checkpoint.
    #[instrument(skip_all)]
    pub fn delete(&self, checkpoint_id: &str) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute("DELETE FROM checkpoints WHERE checkpoint_id = ?1", params![checkpoint_id])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        if rows == 0 {
            return Err(DbError::NotFound {
                resource: "checkpoint",
                field: "checkpoint_id",
                value: checkpoint_id.to_string(),
            });
        }
        Ok(())
    }
}