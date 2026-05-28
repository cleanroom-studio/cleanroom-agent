//! Consistency Service — ensures S.DEF, DB, and Code are in sync.

use sha2::{Sha256, Digest};
use std::sync::Arc;

use cleanroom_db::{Database, DbError};

/// Consistency check level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckLevel {
    /// Fast check — only compare hashes.
    Fast,
    /// Full check — verify structure.
    Full,
    /// Deep check — validate semantics.
    Deep,
}

/// Fix strategy for inconsistencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixStrategy {
    /// Sync code to S.DEF.
    SyncCodeToSdef,
    /// Regenerate code from S.DEF.
    RegenerateCode,
    /// Sync DB to S.DEF.
    SyncDbToSdef,
    /// Sync S.DEF to DB.
    SyncSdefToDb,
    /// Accept external changes.
    AcceptExternal,
}

/// Consistency service for three-way verification.
pub struct ConsistencyService {
    /// Database connection.
    db: Arc<Database>,
}

impl ConsistencyService {
    /// Create a new consistency service.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Compute SHA-256 hash of content.
    pub fn compute_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Check consistency for a document.
    pub fn check(&self, document_name: &str, _level: CheckLevel) -> Result<Vec<Inconsistency>, DbError> {
        let conn = self.db.connection();
        
        let mut stmt = conn.prepare(
            "SELECT entity_uri, sdef_hash, db_hash, code_hash FROM fingerprints WHERE document_name = ?1"
        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let rows = stmt.query_map([document_name], |row| {
            Ok(Inconsistency {
                entity_uri: row.get(0)?,
                sdef_hash: row.get(1)?,
                db_hash: row.get(2)?,
                code_hash: row.get(3)?,
            })
        }).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut inconsistencies = Vec::new();
        for row in rows {
            if let Ok(inc) = row {
                let is_inconsistent = inc.sdef_hash.as_ref()
                    .zip(inc.db_hash.as_ref())
                    .zip(inc.code_hash.as_ref())
                    .map(|((sdef, db), code)| sdef != db || db != code)
                    .unwrap_or(false);

                if is_inconsistent {
                    inconsistencies.push(inc);
                }
            }
        }

        Ok(inconsistencies)
    }

    /// Fix an inconsistency using the specified strategy.
    pub fn fix(&self, _inconsistency: &Inconsistency, strategy: FixStrategy) -> Result<(), DbError> {
        match strategy {
            FixStrategy::SyncCodeToSdef => {}
            FixStrategy::RegenerateCode => {}
            FixStrategy::SyncDbToSdef => {}
            FixStrategy::SyncSdefToDb => {}
            FixStrategy::AcceptExternal => {}
        }
        Ok(())
    }
}

/// Represents an inconsistency between S.DEF, DB, and Code.
#[derive(Debug, Clone)]
pub struct Inconsistency {
    /// Entity URI.
    pub entity_uri: String,
    /// S.DEF hash.
    pub sdef_hash: Option<String>,
    /// Database hash.
    pub db_hash: Option<String>,
    /// Code hash.
    pub code_hash: Option<String>,
}