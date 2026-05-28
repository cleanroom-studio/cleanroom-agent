//! Compatibility resolver — filters S.DEF entities based on compatibility mode.

use rusqlite::params;
use std::sync::Arc;
use tracing::info;

use cleanroom_db::{Database, DbError};

/// Compatibility mode for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityMode {
    /// Full — include all legacy elements, 100% compatibility.
    Full,
    /// Mixed — include compatibility layers but mark as deprecated.
    Mixed,
    /// Clean — only current version, remove all compatibility layers.
    Clean,
    /// Custom — user-defined rules applied externally.
    Custom,
}

impl CompatibilityMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "full" => Self::Full,
            "mixed" => Self::Mixed,
            "clean" => Self::Clean,
            _ => Self::Mixed,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Mixed => "mixed",
            Self::Clean => "clean",
            Self::Custom => "custom",
        }
    }
}

/// Which entities to include based on compatibility mode.
#[derive(Debug, Clone)]
pub struct InclusionFilter {
    pub include_active: bool,
    pub include_deprecated: bool,
    pub include_legacy: bool,
    pub include_compat_modules: bool,
    pub mark_deprecated: bool,
}

impl InclusionFilter {
    pub fn from_mode(mode: CompatibilityMode) -> Self {
        match mode {
            CompatibilityMode::Full => Self {
                include_active: true,
                include_deprecated: true,
                include_legacy: true,
                include_compat_modules: true,
                mark_deprecated: false,
            },
            CompatibilityMode::Mixed => Self {
                include_active: true,
                include_deprecated: true,
                include_legacy: true,
                include_compat_modules: true,
                mark_deprecated: true,
            },
            CompatibilityMode::Clean => Self {
                include_active: true,
                include_deprecated: false,
                include_legacy: false,
                include_compat_modules: false,
                mark_deprecated: false,
            },
            CompatibilityMode::Custom => Self {
                include_active: true,
                include_deprecated: false,
                include_legacy: false,
                include_compat_modules: false,
                mark_deprecated: false,
            },
        }
    }
}

/// Resolved compatibility configuration for the consumer.
pub struct CompatibilityResolver {
    db: Arc<Database>,
    mode: CompatibilityMode,
    filter: InclusionFilter,
}

impl CompatibilityResolver {
    pub fn new(db: Arc<Database>, mode: CompatibilityMode) -> Self {
        let filter = InclusionFilter::from_mode(mode);
        Self { db, mode, filter }
    }

    pub fn mode(&self) -> CompatibilityMode { self.mode }
    pub fn filter(&self) -> &InclusionFilter { &self.filter }

    /// Get data model names filtered by compatibility status.
    pub fn resolve_data_models(&self, document_name: &str) -> Result<Vec<String>, DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare(
            "SELECT entity, status FROM data_models WHERE document_name = ?1"
        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut rows = stmt.query(params![document_name])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut result = Vec::new();
        while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
            let entity: String = row.get(0).map_err(|e| DbError::QueryFailed(e.to_string()))?;
            let status: String = row.get(1).map_err(|e| DbError::QueryFailed(e.to_string()))?;

            if self.should_include(&status) {
                result.push(entity);
            }
        }
        Ok(result)
    }

    /// Check if an entity with given status should be included.
    pub fn should_include(&self, status: &str) -> bool {
        match status {
            "active" => self.filter.include_active,
            "deprecated" => self.filter.include_deprecated,
            "legacy" => self.filter.include_legacy,
            _ => self.filter.include_active,
        }
    }

    /// Get the deprecation annotation for a status.
    pub fn deprecation_annotation(&self, status: &str) -> Option<&'static str> {
        if !self.filter.mark_deprecated { return None; }
        match status {
            "deprecated" => Some("#[deprecated]"),
            "legacy" => Some("#[deprecated(note = \"legacy\")]"),
            _ => None,
        }
    }

    /// Describe the active filtering as text.
    pub fn describe(&self) -> String {
        let action = match self.mode {
            CompatibilityMode::Full => "including all (full compatibility)",
            CompatibilityMode::Mixed => "including all, marking deprecated",
            CompatibilityMode::Clean => "excluding legacy/deprecated (clean)",
            CompatibilityMode::Custom => "custom filter applied",
        };
        format!("Compatibility mode: {} — {}", self.mode.as_str(), action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db() -> Arc<Database> {
        let db = Arc::new(Database::in_memory().unwrap());
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, version, created_at, updated_at)
                 VALUES ('test', '1.0', datetime(), datetime());
                 INSERT INTO data_models (entity, document_name, status, description)
                 VALUES ('User', 'test', 'active', 'Current user model');
                 INSERT INTO data_models (entity, document_name, status, description)
                 VALUES ('UserV1', 'test', 'deprecated', 'Legacy user model');
                 INSERT INTO data_models (entity, document_name, status, description)
                 VALUES ('OldOrder', 'test', 'legacy', 'Old order system');",
            ).unwrap();
        }
        db
    }

    #[test]
    fn test_full_mode_includes_all() {
        let db = setup_db();
        let resolver = CompatibilityResolver::new(db, CompatibilityMode::Full);
        let models = resolver.resolve_data_models("test").unwrap();
        assert_eq!(models.len(), 3);
    }

    #[test]
    fn test_clean_mode_excludes_deprecated() {
        let db = setup_db();
        let resolver = CompatibilityResolver::new(db, CompatibilityMode::Clean);
        let models = resolver.resolve_data_models("test").unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "User");
    }

    #[test]
    fn test_should_include() {
        let db = setup_db();
        let clean = CompatibilityResolver::new(db.clone(), CompatibilityMode::Clean);
        assert!(clean.should_include("active"));
        assert!(!clean.should_include("deprecated"));
        assert!(!clean.should_include("legacy"));

        let full = CompatibilityResolver::new(db, CompatibilityMode::Full);
        assert!(full.should_include("active"));
        assert!(full.should_include("deprecated"));
        assert!(full.should_include("legacy"));
    }

    #[test]
    fn test_deprecation_annotation() {
        let db = setup_db();
        let mixed = CompatibilityResolver::new(db, CompatibilityMode::Mixed);
        assert_eq!(mixed.deprecation_annotation("deprecated"), Some("#[deprecated]"));
        assert_eq!(mixed.deprecation_annotation("legacy"), Some("#[deprecated(note = \"legacy\")]"));
        assert_eq!(mixed.deprecation_annotation("active"), None);

        let full = CompatibilityResolver::new(Arc::new(Database::in_memory().unwrap()), CompatibilityMode::Full);
        assert_eq!(full.deprecation_annotation("deprecated"), None);
    }

    #[test]
    fn test_mixed_mode_filter() {
        let db = setup_db();
        let resolver = CompatibilityResolver::new(db, CompatibilityMode::Mixed);
        let filter = resolver.filter();
        assert!(filter.include_active);
        assert!(filter.include_deprecated);
        assert!(filter.mark_deprecated);
    }

    #[test]
    fn test_describe() {
        let db = setup_db();
        let full = CompatibilityResolver::new(db.clone(), CompatibilityMode::Full);
        assert!(full.describe().contains("full compatibility"));

        let clean = CompatibilityResolver::new(db, CompatibilityMode::Clean);
        assert!(clean.describe().contains("excluding legacy"));
    }
}