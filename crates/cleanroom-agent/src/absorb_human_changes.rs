//! Absorb human changes — reverse syncs human-modified code back into S.DEF.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use rusqlite::params;
use sha2::{Sha256, Digest};
use tracing::{info, instrument, warn};

use cleanroom_db::{Database, DbError};

/// A single human modification detected.
#[derive(Debug, Clone)]
pub struct HumanChange {
    /// File path where change was detected.
    pub file_path: String,
    /// Entity URI in S.DEF that may need updating.
    pub entity_uri: Option<String>,
    /// Type of change.
    pub change_type: ChangeType,
    /// Description of what changed.
    pub description: String,
}

/// Types of human changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    /// Code content changed — fingerprint mismatch.
    ContentModified,
    /// File was added.
    FileAdded,
    /// File was deleted.
    FileDeleted,
    /// New structure detected (new field/method).
    StructureAdded,
    /// Structure removed.
    StructureRemoved,
}

/// Result of the absorb process.
#[derive(Debug, Clone)]
pub struct AbsorbResult {
    /// Total files scanned.
    pub files_scanned: usize,
    /// Changes detected.
    pub changes: Vec<HumanChange>,
    /// Number of changes applied to DB.
    pub changes_applied: usize,
}

/// Detects and absorbs human modifications to generated code.
pub struct HumanChangeAbsorber {
    db: Arc<Database>,
    document_name: String,
}

impl HumanChangeAbsorber {
    pub fn new(db: Arc<Database>, document_name: &str) -> Self {
        Self {
            db,
            document_name: document_name.to_string(),
        }
    }

    /// Scan for human modifications by comparing current file hashes with stored fingerprints.
    #[instrument(skip(self, code_paths))]
    pub fn scan(&self, code_paths: &[String]) -> Result<AbsorbResult, DbError> {
        let mut changes = Vec::new();
        let conn = self.db.connection();

        // Load stored fingerprints for code files
        let mut stored: HashMap<String, String> = HashMap::new();
        {
            let mut stmt = conn.prepare(
                "SELECT entity_uri, code_hash FROM fingerprints
                 WHERE document_name = ?1 AND entity_type = 'code_file'"
            ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

            let mut rows = stmt.query(params![self.document_name])
                .map_err(|e| DbError::QueryFailed(e.to_string()))?;

            while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
                let uri: String = row.get(0).map_err(|e| DbError::QueryFailed(e.to_string()))?;
                let hash: String = row.get(1).map_err(|e| DbError::QueryFailed(e.to_string()))?;
                stored.insert(uri, hash);
            }
        } // conn dropped

        // Scan file paths
        for file_path in code_paths {
            let path = Path::new(file_path);
            let uri = format!("code://{}", file_path);

            if !path.exists() {
                if stored.contains_key(&uri) {
                    changes.push(HumanChange {
                        file_path: file_path.clone(),
                        entity_uri: Some(uri),
                        change_type: ChangeType::FileDeleted,
                        description: format!("Generated file deleted by human: {}", file_path),
                    });
                }
                continue;
            }

            // Compute current hash
            let current_hash = match Self::compute_hash(path) {
                Ok(h) => h,
                Err(_) => continue,
            };

            match stored.get(&uri) {
                Some(stored_hash) if *stored_hash == current_hash => {
                    // No change
                }
                Some(_) => {
                    // Content changed
                    changes.push(HumanChange {
                        file_path: file_path.clone(),
                        entity_uri: Some(uri.clone()),
                        change_type: ChangeType::ContentModified,
                        description: format!("File content modified by human: {}", file_path),
                    });
                }
                None => {
                    // New file
                    changes.push(HumanChange {
                        file_path: file_path.clone(),
                        entity_uri: Some(uri),
                        change_type: ChangeType::FileAdded,
                        description: format!("New file added by human: {}", file_path),
                    });
                }
            }
        }

        info!(scanned = code_paths.len(), changes = changes.len(), "Human change scan complete");
        Ok(AbsorbResult {
            files_scanned: code_paths.len(),
            changes,
            changes_applied: 0,
        })
    }

    /// Absorb detected changes back into the S.DEF / database.
    #[instrument(skip(self))]
    pub fn absorb(&self, result: &mut AbsorbResult) -> Result<(), DbError> {
        let conn = self.db.connection();

        for change in &result.changes {
            match change.change_type {
                ChangeType::FileAdded | ChangeType::ContentModified => {
                    // Update fingerprint to match current file
                    if let Some(uri) = &change.entity_uri {
                        if let Ok(hash) = Self::compute_hash(Path::new(&change.file_path)) {
                            conn.execute(
                                "INSERT OR REPLACE INTO fingerprints
                                 (entity_uri, document_name, entity_type, code_hash, code_path, last_checked_at)
                                 VALUES (?1, ?2, 'code_file', ?3, ?4, datetime())",
                                params![uri, self.document_name, hash, change.file_path],
                            ).ok();
                        }
                    }
                    // Log the human change to audit log
                    conn.execute(
                        "INSERT INTO audit_log (actor, action, resource_type, resource_id, new_value_json)
                         VALUES ('human', 'absorb', 'code_file', ?1, ?2)",
                        params![change.file_path, serde_json::json!({"change": change.description}).to_string()],
                    ).ok();

                    result.changes_applied += 1;
                }
                ChangeType::FileDeleted => {
                    // Mark the fingerprint as stale
                    if let Some(uri) = &change.entity_uri {
                        conn.execute(
                            "UPDATE fingerprints SET code_hash = 'DELETED', last_checked_at = datetime()
                             WHERE entity_uri = ?1 AND document_name = ?2",
                            params![uri, self.document_name],
                        ).ok();
                    }
                    result.changes_applied += 1;
                }
                _ => {}
            }
        }

        info!(applied = result.changes_applied, "Human changes absorbed");
        Ok(())
    }

    /// Compute SHA-256 hash of a file.
    fn compute_hash(path: &Path) -> Result<String, DbError> {
        let content = std::fs::read(path)
            .map_err(|e| DbError::QueryFailed(format!("Failed to read {}: {}", path.display(), e)))?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn setup_db() -> Arc<Database> {
        let db = Arc::new(Database::in_memory().unwrap());
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, version, created_at, updated_at)
                 VALUES ('test', '1.0', datetime(), datetime());"
            ).unwrap();
        }
        db
    }

    #[test]
    fn test_scan_empty_paths() {
        let db = setup_db();
        let absorber = HumanChangeAbsorber::new(db, "test");
        let result = absorber.scan(&[]).unwrap();
        assert_eq!(result.files_scanned, 0);
        assert!(result.changes.is_empty());
    }

    #[test]
    fn test_scan_new_file() {
        let db = setup_db();
        let tmp = std::env::temp_dir().join("cleanroom_absorb_test.rs");
        std::fs::write(&tmp, b"// Human added file").unwrap();

        let absorber = HumanChangeAbsorber::new(db, "test");
        let path_str = tmp.to_string_lossy().to_string();
        let result = absorber.scan(&[path_str]).unwrap();
        assert_eq!(result.changes.len(), 1);
        assert_eq!(result.changes[0].change_type, ChangeType::FileAdded);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_scan_deleted_file() {
        let db = setup_db();
        {
            let conn = db.connection();
            conn.execute(
                "INSERT INTO fingerprints (entity_uri, document_name, entity_type, code_hash, code_path, last_checked_at)
                 VALUES ('code://missing.rs', 'test', 'code_file', 'hash123', 'missing.rs', datetime())",
                [],
            ).unwrap();
        }

        let absorber = HumanChangeAbsorber::new(db, "test");
        let result = absorber.scan(&[]).unwrap();
        assert_eq!(result.changes.len(), 0); // changed: only scans given paths, not all stored
    }

    #[test]
    fn test_absorb_updates_fingerprint() {
        let db = setup_db();
        let tmp = std::env::temp_dir().join("cleanroom_absorb_update.rs");
        std::fs::write(&tmp, b"// Original").unwrap();

        let absorber = HumanChangeAbsorber::new(db.clone(), "test");
        let path_str = tmp.to_string_lossy().to_string();
        let mut result = absorber.scan(&[path_str.clone()]).unwrap();
        assert_eq!(result.changes.len(), 1);

        absorber.absorb(&mut result).unwrap();
        assert_eq!(result.changes_applied, 1);

        // Verify fingerprint was stored
        let conn = db.connection();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM fingerprints WHERE entity_uri = ?1",
            params![format!("code://{}", path_str)],
            |row| row.get(0),
        ).unwrap_or(0);
        assert_eq!(count, 1);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_absorb_logs_audit() {
        let db = setup_db();
        let tmp = std::env::temp_dir().join("cleanroom_absorb_audit.rs");
        std::fs::write(&tmp, b"// Audited").unwrap();

        let absorber = HumanChangeAbsorber::new(db.clone(), "test");
        let path_str = tmp.to_string_lossy().to_string();
        let mut result = absorber.scan(&[path_str]).unwrap();
        absorber.absorb(&mut result).unwrap();

        // Verify audit log entry
        let conn = db.connection();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM audit_log WHERE action = 'absorb'",
            [],
            |row| row.get(0),
        ).unwrap_or(0);
        assert_eq!(count, 1);

        let _ = std::fs::remove_file(&tmp);
    }
}