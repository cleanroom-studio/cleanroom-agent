//! Consistency checker background loop — periodically runs consistency checks.

use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn, error};

use cleanroom_db::{Database, DbError, FingerprintRepository};
use crate::completeness::CompletenessValidator;

/// Configuration for the background consistency checker.
#[derive(Debug, Clone)]
pub struct ConsistencyCheckerConfig {
    /// How often to run checks.
    pub interval: Duration,
    /// Documents to check.
    pub document_names: Vec<String>,
    /// Whether to auto-fix inconsistencies.
    pub auto_fix: bool,
}

impl Default for ConsistencyCheckerConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(300), // 5 minutes
            document_names: vec![],
            auto_fix: false,
        }
    }
}

/// Background consistency checker.
pub struct ConsistencyChecker {
    db: Arc<Database>,
    config: ConsistencyCheckerConfig,
}

impl ConsistencyChecker {
    pub fn new(db: Arc<Database>, config: ConsistencyCheckerConfig) -> Self {
        Self { db, config }
    }

    /// Run a single consistency check cycle.
    pub fn run_once(&self) -> Result<usize, DbError> {
        let mut total_inconsistencies = 0;

        for doc_name in &self.config.document_names {
            // 1. Check fingerprints for inconsistencies
            let fp_repo = FingerprintRepository::from_arc(self.db.connection_arc());
            match fp_repo.list_inconsistent(doc_name) {
                Ok(inconsistencies) => {
                    let count = inconsistencies.len();
                    total_inconsistencies += count;
                    if count > 0 {
                        warn!(
                            document = %doc_name,
                            inconsistencies = count,
                            "Inconsistencies detected"
                        );
                    } else {
                        info!(document = %doc_name, "All fingerprints consistent");
                    }
                }
                Err(e) => {
                    warn!(document = %doc_name, error = %e, "Failed to check fingerprints");
                }
            }

            // 2. Run completeness validation
            let validator = CompletenessValidator::new(self.db.clone());
            match validator.validate(doc_name) {
                Ok(report) => {
                    if report.overall_score.overall < 0.5 {
                        warn!(
                            document = %doc_name,
                            score = report.overall_score.overall,
                            "Low completeness score"
                        );
                    }
                }
                Err(e) => {
                    warn!(document = %doc_name, error = %e, "Completeness check failed");
                }
            }
        }

        info!(documents = self.config.document_names.len(), inconsistencies = total_inconsistencies, "Consistency check cycle complete");
        Ok(total_inconsistencies)
    }

    /// Run the consistency check loop in a background thread.
    /// Returns a handle that can be used to stop the loop.
    pub fn run_loop(self) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("consistency-checker".into())
            .spawn(move || {
                info!(
                    interval_secs = self.config.interval.as_secs(),
                    "Consistency checker loop started"
                );

                loop {
                    if let Err(e) = self.run_once() {
                        error!(error = %e, "Consistency check cycle failed");
                    }

                    std::thread::sleep(self.config.interval);
                }
            })
            .expect("Failed to spawn consistency checker thread")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_db::Database;

    #[test]
    fn test_run_once_no_documents() {
        let db = Arc::new(Database::in_memory().unwrap());
        let checker = ConsistencyChecker::new(db, ConsistencyCheckerConfig::default());
        let result = checker.run_once().unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_run_once_with_document() {
        let db = Arc::new(Database::in_memory().unwrap());
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, version, created_at, updated_at)
                 VALUES ('test-doc', '1.0', datetime(), datetime());"
            ).unwrap();
        }

        let config = ConsistencyCheckerConfig {
            document_names: vec!["test-doc".to_string()],
            ..ConsistencyCheckerConfig::default()
        };
        let checker = ConsistencyChecker::new(db, config);
        let result = checker.run_once().unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_run_once_with_inconsistency() {
        let db = Arc::new(Database::in_memory().unwrap());
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, version, created_at, updated_at)
                 VALUES ('test', '1.0', datetime(), datetime());
                 INSERT INTO fingerprints (entity_uri, document_name, entity_type, sdef_hash, db_hash, code_hash, last_checked_at)
                 VALUES ('entity://test', 'test', 'data_model', 'abc', 'def', 'ghi', datetime());"
            ).unwrap();
        }

        let config = ConsistencyCheckerConfig {
            document_names: vec!["test".to_string()],
            ..ConsistencyCheckerConfig::default()
        };
        let checker = ConsistencyChecker::new(db, config);
        let result = checker.run_once().unwrap();
        assert!(result > 0, "Should detect the fingerprint mismatch");
    }
}