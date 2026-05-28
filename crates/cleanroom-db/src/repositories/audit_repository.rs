//! Audit repository for audit log operations.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::instrument;

use crate::error::{DbError, DbResult};

/// Audit entry model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: Option<i64>,
    pub timestamp: Option<String>,
    pub actor: String,
    pub action: String,
    pub resource_type: String,
    pub resource_id: String,
    pub old_value_json: Option<String>,
    pub new_value_json: Option<String>,
}

/// Audit repository (write-only).
pub struct AuditRepository {
    conn: Mutex<Connection>,
}

impl AuditRepository {
    /// Create a new audit repository.
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Mutex::new(conn),
        }
    }

    /// Write an audit entry.
    #[instrument(skip_all, fields(actor = %entry.actor, action = %entry.action, resource = %entry.resource_id))]
    pub fn write(&self, entry: &AuditEntry) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO audit_log (
                actor, action, resource_type, resource_id, old_value_json, new_value_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"#,
            params![
                entry.actor,
                entry.action,
                entry.resource_type,
                entry.resource_id,
                entry.old_value_json,
                entry.new_value_json,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Query audit log entries.
    #[instrument(skip_all)]
    pub fn query(
        &self,
        resource_type: Option<&str>,
        resource_id: Option<&str>,
        limit: Option<i32>,
    ) -> DbResult<Vec<AuditEntry>> {
        let conn = self.conn.lock().unwrap();

        let mut query = String::from(
            "SELECT id, timestamp, actor, action, resource_type, resource_id, old_value_json, new_value_json
             FROM audit_log WHERE 1=1",
        );

        if resource_type.is_some() {
            query.push_str(" AND resource_type = ?");
        }
        if resource_id.is_some() {
            query.push_str(" AND resource_id = ?");
        }

        query.push_str(" ORDER BY timestamp DESC");

        if let Some(l) = limit {
            query.push_str(&format!(" LIMIT {}", l));
        }

        let mut stmt = conn
            .prepare(&query)
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let entries = stmt
            .query_map(
                rusqlite::params_from_iter(
                    resource_type
                        .into_iter()
                        .chain(resource_id.into_iter()),
                ),
                |row| {
                    Ok(AuditEntry {
                        id: row.get(0)?,
                        timestamp: row.get(1)?,
                        actor: row.get(2)?,
                        action: row.get(3)?,
                        resource_type: row.get(4)?,
                        resource_id: row.get(5)?,
                        old_value_json: row.get(6)?,
                        new_value_json: row.get(7)?,
                    })
                },
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }
}