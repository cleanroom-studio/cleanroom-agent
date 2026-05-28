//! Version upgrade analysis — detects breaking changes and generates S.DEF updates.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use tracing::{info, instrument};

use cleanroom_db::{Database, DbError};

/// Result of version upgrade analysis.
#[derive(Debug, Clone)]
pub struct VersionUpgradeReport {
    pub old_version: String,
    pub new_version: String,
    pub added_files: Vec<String>,
    pub modified_files: Vec<String>,
    pub deleted_files: Vec<String>,
    pub breaking_changes: Vec<BreakingChange>,
    pub deprecated_entities: Vec<String>,
    pub new_compat_layers: Vec<String>,
    pub suggested_migrations: Vec<SuggestedMigration>,
}

/// A breaking change detected during upgrade analysis.
#[derive(Debug, Clone)]
pub struct BreakingChange {
    pub entity: String,
    pub change_type: ChangeType,
    pub description: String,
    pub old: Option<String>,
    pub new: Option<String>,
}

/// Type of change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    FieldRemoved,
    FieldTypeChanged,
    FieldRenamed,
    MethodSignatureChanged,
    InterfaceRemoved,
    EntityRemoved,
    TypeChanged,
    Other,
}

/// Suggested data migration based on diff analysis.
#[derive(Debug, Clone)]
pub struct SuggestedMigration {
    pub from_entity: String,
    pub to_entity: String,
    pub reason: String,
}

/// Version upgrade analyzer.
pub struct VersionUpgradeAnalyzer {
    db: Arc<Database>,
    repo_path: String,
}

impl VersionUpgradeAnalyzer {
    pub fn new(db: Arc<Database>, repo_path: &str) -> Self {
        Self {
            db,
            repo_path: repo_path.to_string(),
        }
    }

    /// Run the full version upgrade analysis.
    #[instrument(skip(self))]
    pub fn analyze(&self, old_version: &str, new_version: &str) -> Result<VersionUpgradeReport, DbError> {
        // 1. Get git diff between versions
        let diff_output = self.git_diff(old_version, new_version)?;
        let files = self.parse_diff_files(&diff_output);

        // 2. Classify files
        let added_files = self.filter_by_status(&files, "A");
        let modified_files = self.filter_by_status(&files, "M");
        let deleted_files = self.filter_by_status(&files, "D");

        // 3. Extract detailed changes from modified files
        let mut breaking_changes = Vec::new();
        for file in &modified_files {
            let file_diff = self.git_diff_file(old_version, new_version, file)?;
            let changes = self.analyze_breaking_changes(file, &file_diff);
            breaking_changes.extend(changes);
        }

        // 4. Identify deprecated entities (modified files → old entity is deprecated)
        let deprecated_entities: Vec<String> = modified_files.iter()
            .filter_map(|f| Path::new(f).file_stem())
            .map(|s| s.to_string_lossy().to_string())
            .collect();

        // 5. Detect new compatibility layers
        let new_compat_layers: Vec<String> = added_files.iter()
            .filter(|f| {
                let low = f.to_lowercase();
                low.contains("compat") || low.contains("legacy") || low.contains("deprecated")
            })
            .cloned()
            .collect();

        // 6. Suggest data migrations
        let mut suggested_migrations = Vec::new();
        for file in &modified_files {
            let stem = Path::new(file).file_stem().map(|s| s.to_string_lossy()).unwrap_or_default();
            if let Some(deleted) = deleted_files.iter().find(|d| {
                Path::new(d).file_stem().map(|s| s.to_string_lossy()) == Some(stem.clone())
            }) {
                let del_stem = Path::new(deleted).file_stem().map(|s| s.to_string_lossy()).unwrap_or_default();
                suggested_migrations.push(SuggestedMigration {
                    from_entity: del_stem.to_string(),
                    to_entity: stem.to_string(),
                    reason: format!("Entity renamed from '{}' to '{}'", del_stem, stem),
                });
            }
        }

        let report = VersionUpgradeReport {
            old_version: old_version.to_string(),
            new_version: new_version.to_string(),
            added_files,
            modified_files,
            deleted_files,
            breaking_changes,
            deprecated_entities,
            new_compat_layers,
            suggested_migrations,
        };

        info!(
            added = report.added_files.len(),
            modified = report.modified_files.len(),
            deleted = report.deleted_files.len(),
            breaking = report.breaking_changes.len(),
            "Version upgrade analysis complete"
        );

        Ok(report)
    }

    /// Run git diff between two version references.
    fn git_diff(&self, old: &str, new: &str) -> Result<String, DbError> {
        let output = Command::new("git")
            .args(["-C", &self.repo_path, "diff", "--name-status", old, new])
            .output()
            .map_err(|e| DbError::QueryFailed(format!("Git diff failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DbError::QueryFailed(format!("Git error: {}", stderr)));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Run git diff for a specific file.
    fn git_diff_file(&self, old: &str, new: &str, file: &str) -> Result<String, DbError> {
        let output = Command::new("git")
            .args(["-C", &self.repo_path, "diff", old, new, "--", file])
            .output()
            .map_err(|e| DbError::QueryFailed(format!("Git diff file failed: {}", e)))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Parse git diff --name-status output into (status, file_path) tuples.
    fn parse_diff_files(&self, output: &str) -> Vec<(String, String)> {
        output.lines()
            .filter_map(|line| {
                if line.trim().is_empty() { return None; }
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect()
    }

    fn filter_by_status(&self, files: &[(String, String)], status: &str) -> Vec<String> {
        files.iter()
            .filter(|(s, _)| s == status)
            .map(|(_, f)| f.clone())
            .collect()
    }

    /// Analyze breaking changes from a file diff.
    fn analyze_breaking_changes(&self, file: &str, diff: &str) -> Vec<BreakingChange> {
        let mut changes = Vec::new();
        let stem = Path::new(file).file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        for line in diff.lines() {
            // Detect removed lines with "fn " or "def " or "function "
            if line.starts_with('-') && (line.contains("fn ") || line.contains("def ") || line.contains("function ")) {
                changes.push(BreakingChange {
                    entity: stem.clone(),
                    change_type: ChangeType::MethodSignatureChanged,
                    description: format!("Method changed: {}", line.trim_start_matches('-').trim()),
                    old: Some(line.trim_start_matches('-').to_string()),
                    new: None,
                });
            }

            // Detect type annotation changes
            if line.starts_with('-') && line.contains(":") && line.contains("->") {
                changes.push(BreakingChange {
                    entity: stem.clone(),
                    change_type: ChangeType::TypeChanged,
                    description: format!("Return type changed: {}", line.trim()),
                    old: Some(line.trim_start_matches('-').to_string()),
                    new: None,
                });
            }

            // Detect removed fields in structs/interfaces
            if line.starts_with('-') && line.trim().starts_with("pub ") {
                changes.push(BreakingChange {
                    entity: stem.clone(),
                    change_type: ChangeType::FieldRemoved,
                    description: format!("Field removed: {}", line.trim()),
                    old: Some(line.trim_start_matches('-').to_string()),
                    new: None,
                });
            }
        }

        changes
    }

    /// Apply the version upgrade report to the database.
    #[instrument(skip(self))]
    pub fn apply_upgrade(&self, report: &VersionUpgradeReport) -> Result<(), DbError> {
        let conn = self.db.connection();

        // Mark deprecated entities in data_models
        for entity in &report.deprecated_entities {
            conn.execute(
                "UPDATE data_models SET status = 'deprecated' WHERE entity = ?1 AND status = 'active'",
                rusqlite::params![entity],
            ).ok();
        }

        // Add version record
        conn.execute(
            "INSERT OR IGNORE INTO version_records (version, document_name, release_date, breaking_changes_json)
             VALUES (?1, 'default', datetime(), ?2)",
            rusqlite::params![
                report.new_version,
                serde_json::to_string(&report.breaking_changes.iter().map(|b| &b.description).collect::<Vec<_>>()).unwrap_or_default(),
            ],
        ).ok();

        info!(version = %report.new_version, changes = report.breaking_changes.len(), "Version upgrade applied");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_diff_files() {
        let analyzer = crate::version_upgrade::VersionUpgradeAnalyzer::new(
            Arc::new(Database::in_memory().unwrap()), "/tmp"
        );
        let output = "A\tsrc/new.rs\nM\tsrc/modified.rs\nD\tsrc/deleted.rs\n";
        let files = analyzer.parse_diff_files(output);
        assert_eq!(files.len(), 3);
        assert_eq!(files[0], ("A".to_string(), "src/new.rs".to_string()));
        assert_eq!(files[1], ("M".to_string(), "src/modified.rs".to_string()));
        assert_eq!(files[2], ("D".to_string(), "src/deleted.rs".to_string()));
    }

    #[test]
    fn test_filter_by_status() {
        let analyzer = crate::version_upgrade::VersionUpgradeAnalyzer::new(
            Arc::new(Database::in_memory().unwrap()), "/tmp"
        );
        let files = vec![
            ("A".to_string(), "new.rs".to_string()),
            ("M".to_string(), "mod.rs".to_string()),
            ("A".to_string(), "another.rs".to_string()),
        ];
        let added = analyzer.filter_by_status(&files, "A");
        assert_eq!(added.len(), 2);
        let modified = analyzer.filter_by_status(&files, "M");
        assert_eq!(modified.len(), 1);
        let deleted = analyzer.filter_by_status(&files, "D");
        assert_eq!(deleted.len(), 0);
    }

    #[test]
    fn test_analyze_breaking_changes() {
        let analyzer = crate::version_upgrade::VersionUpgradeAnalyzer::new(
            Arc::new(Database::in_memory().unwrap()), "/tmp"
        );
        let diff = "\
-pub fn old_method(x: i32) -> String {
+pub fn new_method(x: i32, y: i32) -> Result<String> {
-    pub old_field: i32,
";
        let changes = analyzer.analyze_breaking_changes("src/test.rs", diff);
        assert!(!changes.is_empty(), "Should detect at least one breaking change");
        assert!(changes.iter().any(|c| matches!(c.change_type, ChangeType::MethodSignatureChanged)));
    }

    #[test]
    fn test_empty_analyze_no_changes() {
        let analyzer = crate::version_upgrade::VersionUpgradeAnalyzer::new(
            Arc::new(Database::in_memory().unwrap()), "/tmp"
        );
        let changes = analyzer.analyze_breaking_changes("test.rs", "");
        assert!(changes.is_empty());
    }

    #[test]
    fn test_apply_upgrade() {
        let db = Arc::new(Database::in_memory().unwrap());
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, version, created_at, updated_at)
                 VALUES ('default', '1.0', datetime(), datetime())"
            ).unwrap();
        }
        let analyzer = crate::version_upgrade::VersionUpgradeAnalyzer::new(
            db, "/tmp"
        );
        let report = VersionUpgradeReport {
            old_version: "1.0".to_string(),
            new_version: "2.0".to_string(),
            added_files: vec![],
            modified_files: vec![],
            deleted_files: vec![],
            breaking_changes: vec![],
            deprecated_entities: vec![],
            new_compat_layers: vec![],
            suggested_migrations: vec![],
        };
        analyzer.apply_upgrade(&report).unwrap();
    }
}