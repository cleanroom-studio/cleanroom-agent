//! Two-phase commit for crash-safe multi-step operations.

use std::sync::Arc;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument, warn};

use cleanroom_db::{Database, DbError};

/// Phase of a prepared transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionPhase {
    Pending,
    Prepared,
    Committed,
    RolledBack,
    Failed,
}

impl TransactionPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Prepared => "prepared",
            Self::Committed => "committed",
            Self::RolledBack => "rolled_back",
            Self::Failed => "failed",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "prepared" => Some(Self::Prepared),
            "committed" => Some(Self::Committed),
            "rolled_back" => Some(Self::RolledBack),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

/// A single change in a transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRecord {
    /// Resource type (e.g. "data_model", "task", "symbol").
    pub resource_type: String,
    /// Resource identifier.
    pub resource_id: String,
    /// Operation: "create", "update", "delete".
    pub operation: String,
    /// Previous value (for undo).
    pub old_value: Option<serde_json::Value>,
    /// New value (for redo).
    pub new_value: Option<serde_json::Value>,
}

/// Transaction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionResult {
    pub transaction_id: String,
    pub status: TransactionPhase,
    pub change_count: usize,
}

/// Two-phase commit manager.
pub struct TwoPhaseCommit {
    db: Arc<Database>,
}

impl TwoPhaseCommit {
    /// Create a new transaction manager.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Phase 1: Prepare — save changes to the prepared_transactions table.
    #[instrument(skip(self, changes))]
    pub fn prepare(&self, changes: Vec<ChangeRecord>) -> Result<TransactionResult, DbError> {
        if changes.is_empty() {
            return Err(DbError::TransactionError("No changes to prepare".to_string()));
        }

        let transaction_id = uuid::Uuid::new_v4().to_string();
        let changes_json = serde_json::to_string(&changes)
            .map_err(|e| DbError::TransactionError(format!("Serialization failed: {}", e)))?;

        let conn = self.db.connection();
        conn.execute(
            r#"INSERT INTO prepared_transactions 
               (transaction_id, phase, changes_json, status, prepared_at)
               VALUES (?1, 'prepare', ?2, 'prepared', CURRENT_TIMESTAMP)"#,
            params![transaction_id, changes_json],
        ).map_err(|e| DbError::TransactionError(e.to_string()))?;

        info!(%transaction_id, count = changes.len(), "Transaction prepared");

        Ok(TransactionResult {
            transaction_id,
            status: TransactionPhase::Prepared,
            change_count: changes.len(),
        })
    }

    /// Phase 2a: Commit — apply all changes and mark as committed.
    #[instrument(skip(self))]
    pub fn commit(&self, transaction_id: &str) -> Result<(), DbError> {
        let (changes_json, status): (String, String);
        {
            let conn = self.db.connection();
            let result = conn.query_row(
                "SELECT changes_json, status FROM prepared_transactions WHERE transaction_id = ?1",
                params![transaction_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            ).map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                    resource: "prepared_transaction",
                    field: "transaction_id",
                    value: transaction_id.to_string(),
                },
                _ => DbError::TransactionError(e.to_string()),
            })?;
            (changes_json, status) = result;
        }

        let phase = TransactionPhase::from_str(&status).unwrap_or(TransactionPhase::Pending);
        if phase != TransactionPhase::Prepared {
            return Err(DbError::TransactionError(format!(
                "Transaction {} is in '{:?}' state, expected 'prepared'",
                transaction_id, phase
            )));
        }

        // Apply changes (outside conn scope to avoid deadlock)
        let changes: Vec<ChangeRecord> = serde_json::from_str(&changes_json)
            .map_err(|e| DbError::TransactionError(format!("Deserialization failed: {}", e)))?;
        self.apply_changes(&changes)?;

        // Mark as committed
        {
            let conn = self.db.connection();
            conn.execute(
                r#"UPDATE prepared_transactions 
                   SET status = 'committed', phase = 'commit', committed_at = CURRENT_TIMESTAMP
                   WHERE transaction_id = ?1"#,
                params![transaction_id],
            ).map_err(|e| DbError::TransactionError(e.to_string()))?;
        }

        info!(%transaction_id, "Transaction committed");
        Ok(())
    }

    /// Phase 2b: Rollback — revert all changes and mark as rolled back.
    #[instrument(skip(self))]
    pub fn rollback(&self, transaction_id: &str) -> Result<(), DbError> {
        let (changes_json, status): (String, String);
        {
            let conn = self.db.connection();
            let result = conn.query_row(
                "SELECT changes_json, status FROM prepared_transactions WHERE transaction_id = ?1",
                params![transaction_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            ).map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                    resource: "prepared_transaction",
                    field: "transaction_id",
                    value: transaction_id.to_string(),
                },
                _ => DbError::TransactionError(e.to_string()),
            })?;
            (changes_json, status) = result;
        }

        let phase = TransactionPhase::from_str(&status).unwrap_or(TransactionPhase::Pending);
        if phase != TransactionPhase::Prepared {
            return Err(DbError::TransactionError(format!(
                "Transaction {} is in '{:?}' state, cannot rollback",
                transaction_id, phase
            )));
        }

        // Undo changes (outside conn scope to avoid deadlock)
        let changes: Vec<ChangeRecord> = serde_json::from_str(&changes_json)
            .map_err(|e| DbError::TransactionError(format!("Deserialization failed: {}", e)))?;
        self.undo_changes(&changes)?;

        // Mark as rolled back
        {
            let conn = self.db.connection();
            conn.execute(
                r#"UPDATE prepared_transactions 
                   SET status = 'rolled_back', phase = 'rollback', rollback_at = CURRENT_TIMESTAMP
                   WHERE transaction_id = ?1"#,
                params![transaction_id],
            ).map_err(|e| DbError::TransactionError(e.to_string()))?;
        }

        info!(%transaction_id, "Transaction rolled back");
        Ok(())
    }

    /// Recover dangling prepared transactions on startup.
    #[instrument(skip(self))]
    pub fn recover(&self) -> Result<Vec<String>, DbError> {
        let pending: Vec<(String, String, String)>;
        {
            let conn = self.db.connection();
            let mut stmt = conn.prepare(
                "SELECT transaction_id, phase, changes_json FROM prepared_transactions 
                 WHERE status = 'prepared'"
            ).map_err(|e| DbError::TransactionError(e.to_string()))?;

            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            }).map_err(|e| DbError::TransactionError(e.to_string()))?;

            let mut tmp = Vec::new();
            for row in rows {
                if let Ok(r) = row {
                    tmp.push(r);
                }
            }
            pending = tmp;
        }

        if pending.is_empty() {
            info!("No dangling transactions to recover");
            return Ok(Vec::new());
        }

        let mut recovered = Vec::new();
        for (tx_id, phase_str, _) in &pending {
            // If phase is 'commit', try to complete; if 'rollback', undo; otherwise auto-rollback
            let action = match phase_str.as_str() {
                "commit" => "committing",
                "rollback" => "rolling back",
                _ => "auto-rolling back",
            };
            warn!(%tx_id, %action, "Recovering dangling transaction");

            match phase_str.as_str() {
                "commit" => {
                    if let Err(e) = self.commit(tx_id) {
                        warn!(%tx_id, error = %e, "Recovery commit failed");
                    }
                }
                "rollback" => {
                    if let Err(e) = self.rollback(tx_id) {
                        warn!(%tx_id, error = %e, "Recovery rollback failed");
                    }
                }
                _ => {
                    if let Err(e) = self.rollback(tx_id) {
                        warn!(%tx_id, error = %e, "Recovery auto-rollback failed");
                    }
                }
            }
            recovered.push(tx_id.clone());
        }

        info!(count = recovered.len(), "Transaction recovery complete");
        Ok(recovered)
    }

    /// List pending transactions.
    pub fn list_pending(&self) -> Result<Vec<String>, DbError> {
        let mut ids = Vec::new();
        {
            let conn = self.db.connection();
            let mut stmt = conn.prepare(
                "SELECT transaction_id FROM prepared_transactions WHERE status = 'prepared'"
            ).map_err(|e| DbError::TransactionError(e.to_string()))?;

            let rows = stmt.query_map([], |row| row.get::<_, String>(0))
                .map_err(|e| DbError::TransactionError(e.to_string()))?;
            for row in rows {
                if let Ok(id) = row {
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }

    /// Apply changes (insert/update records).
    fn apply_changes(&self, changes: &[ChangeRecord]) -> Result<(), DbError> {
        let conn = self.db.connection();
        for change in changes {
            match change.operation.as_str() {
                "create" => {
                    if let Some(val) = &change.new_value {
                        conn.execute(
                            "INSERT INTO audit_log (actor, action, resource_type, resource_id, new_value_json)
                             VALUES ('system', 'create', ?1, ?2, ?3)",
                            params![change.resource_type, change.resource_id, serde_json::to_string(val).ok()],
                        ).ok();
                    }
                }
                "update" => {
                    if let Some(val) = &change.new_value {
                        conn.execute(
                            "INSERT INTO audit_log (actor, action, resource_type, resource_id, new_value_json)
                             VALUES ('system', 'update', ?1, ?2, ?3)",
                            params![change.resource_type, change.resource_id, serde_json::to_string(val).ok()],
                        ).ok();
                    }
                }
                "delete" => {}
                _ => {
                    warn!(op = %change.operation, "Unknown change operation");
                }
            }
        }
        Ok(())
    }

    /// Undo changes (reverse insert/update).
    fn undo_changes(&self, changes: &[ChangeRecord]) -> Result<(), DbError> {
        let conn = self.db.connection();
        for change in changes.iter().rev() {
            match change.operation.as_str() {
                "create" => {
                    conn.execute(
                        "INSERT INTO audit_log (actor, action, resource_type, resource_id, old_value_json)
                         VALUES ('system', 'rollback_create', ?1, ?2, ?3)",
                        params![change.resource_type, change.resource_id, 
                                change.new_value.as_ref().and_then(|v| serde_json::to_string(v).ok())],
                    ).ok();
                }
                "update" => {
                    if let Some(val) = &change.old_value {
                        conn.execute(
                            "INSERT INTO audit_log (actor, action, resource_type, resource_id, old_value_json)
                             VALUES ('system', 'rollback_update', ?1, ?2, ?3)",
                            params![change.resource_type, change.resource_id, serde_json::to_string(val).ok()],
                        ).ok();
                    }
                }
                "delete" => {
                    if let Some(val) = &change.old_value {
                        conn.execute(
                            "INSERT INTO audit_log (actor, action, resource_type, resource_id, old_value_json)
                             VALUES ('system', 'rollback_delete', ?1, ?2, ?3)",
                            params![change.resource_type, change.resource_id, serde_json::to_string(val).ok()],
                        ).ok();
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Arc<Database>, TwoPhaseCommit) {
        let db = Arc::new(Database::in_memory().unwrap());
        let tpc = TwoPhaseCommit::new(db.clone());
        (db, tpc)
    }

    #[test]
    fn test_prepare_empty_changes() {
        let (_, tpc) = setup();
        let result = tpc.prepare(vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No changes"));
    }

    #[test]
    fn test_full_two_phase_cycle() {
        let (_, tpc) = setup();

        let changes = vec![ChangeRecord {
            resource_type: "test".to_string(),
            resource_id: "test-001".to_string(),
            operation: "create".to_string(),
            old_value: None,
            new_value: Some(serde_json::json!({"key": "value"})),
        }];

        // Phase 1: Prepare
        let result = tpc.prepare(changes).unwrap();
        assert_eq!(result.status, TransactionPhase::Prepared);
        assert_eq!(result.change_count, 1);

        // Phase 2: Commit
        tpc.commit(&result.transaction_id).unwrap();
    }

    #[test]
    fn test_prepare_then_rollback() {
        let (_, tpc) = setup();

        let changes = vec![ChangeRecord {
            resource_type: "test".to_string(),
            resource_id: "test-002".to_string(),
            operation: "update".to_string(),
            old_value: Some(serde_json::json!({"status": "old"})),
            new_value: Some(serde_json::json!({"status": "new"})),
        }];

        let result = tpc.prepare(changes).unwrap();
        tpc.rollback(&result.transaction_id).unwrap();

        // Should not be able to commit again
        let commit_result = tpc.commit(&result.transaction_id);
        assert!(commit_result.is_err());
    }

    #[test]
    fn test_double_commit_fails() {
        let (_, tpc) = setup();

        let changes = vec![ChangeRecord {
            resource_type: "test".to_string(),
            resource_id: "test-003".to_string(),
            operation: "delete".to_string(),
            old_value: Some(serde_json::json!({"id": 1})),
            new_value: None,
        }];

        let result = tpc.prepare(changes).unwrap();
        tpc.commit(&result.transaction_id).unwrap();

        // Second commit should fail
        let second = tpc.commit(&result.transaction_id);
        assert!(second.is_err());
    }

    #[test]
    fn test_recover_no_dangling() {
        let (_, tpc) = setup();
        let recovered = tpc.recover().unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_list_pending() {
        let (_, tpc) = setup();

        let changes = vec![ChangeRecord {
            resource_type: "test".to_string(),
            resource_id: "test-pending".to_string(),
            operation: "create".to_string(),
            old_value: None,
            new_value: Some(serde_json::json!({"data": "value"})),
        }];

        tpc.prepare(changes).unwrap();
        let pending = tpc.list_pending().unwrap();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_committed_not_in_pending() {
        let (_, tpc) = setup();
        let changes = vec![ChangeRecord {
            resource_type: "test".to_string(),
            resource_id: "test-committed".to_string(),
            operation: "create".to_string(),
            old_value: None,
            new_value: Some(serde_json::json!({"done": true})),
        }];

        let result = tpc.prepare(changes).unwrap();
        tpc.commit(&result.transaction_id).unwrap();
        let pending = tpc.list_pending().unwrap();
        assert!(pending.is_empty());
    }
}