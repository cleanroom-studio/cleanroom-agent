//! Completeness validation — multi-layer verification of S.DEF analysis quality.

use std::collections::HashSet;
use std::sync::Arc;
use rusqlite::params;
use tracing::instrument;

use cleanroom_db::{Database, DbError, TaskRepository, TaskStatus};

/// Coverage score from 0.0 to 1.0.
#[derive(Debug, Clone)]
pub struct CoverageScore {
    pub file_coverage: f64,
    pub dependency_coverage: f64,
    pub interface_coverage: f64,
    pub entity_coverage: f64,
    pub cross_validation: f64,
    pub overall: f64,
}

/// Result of completeness verification for a single check.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub layer: &'static str,
    pub passed: bool,
    pub score: f64,
    pub details: Vec<String>,
    pub warnings: Vec<String>,
}

/// Full completeness report.
#[derive(Debug, Clone)]
pub struct CompletenessReport {
    pub document_name: String,
    pub results: Vec<VerificationResult>,
    pub overall_score: CoverageScore,
}

/// Completeness validator.
pub struct CompletenessValidator {
    db: Arc<Database>,
}

impl CompletenessValidator {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Run all five verification layers.
    #[instrument(skip(self))]
    pub fn validate(&self, document_name: &str) -> Result<CompletenessReport, DbError> {
        let layer1 = self.verify_file_coverage(document_name)?;
        let layer2 = self.verify_dependency_integrity(document_name)?;
        let layer3 = self.verify_interface_coverage(document_name)?;
        let layer4 = self.verify_entity_coverage(document_name)?;
        let layer5 = self.cross_validate(document_name)?;

        let results = vec![layer1, layer2, layer3, layer4, layer5];

        let mut scores = Vec::new();
        for r in &results { scores.push(r.score); }

        let overall = CoverageScore {
            file_coverage: scores[0],
            dependency_coverage: scores[1],
            interface_coverage: scores[2],
            entity_coverage: scores[3],
            cross_validation: scores[4],
            overall: scores.iter().sum::<f64>() / scores.len() as f64,
        };

        Ok(CompletenessReport {
            document_name: document_name.to_string(),
            results,
            overall_score: overall,
        })
    }

    /// Layer 1: File coverage — what % of tasks completed successfully.
    fn verify_file_coverage(&self, document_name: &str) -> Result<VerificationResult, DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());
        let all_tasks = repo.list(None, None, None)
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        // Filter tasks related to this document
        let doc_tasks: Vec<_> = all_tasks.iter().filter(|t| t.input_json.contains(document_name)).collect();
        let total = doc_tasks.len();
        let completed = doc_tasks.iter().filter(|t| t.status == TaskStatus::Completed).count();
        let failed = doc_tasks.iter().filter(|t| t.status == TaskStatus::Failed).count();
        let pending = doc_tasks.iter().filter(|t| t.status == TaskStatus::Pending).count();

        let score = if total > 0 { completed as f64 / total as f64 } else { 0.0 };
        let details = vec![format!("Total tasks: {}, Completed: {}, Failed: {}, Pending: {}", total, completed, failed, pending)];
        let mut warnings = Vec::new();

        if failed > 0 { warnings.push(format!("{} tasks failed analysis", failed)); }
        if pending > 0 { warnings.push(format!("{} tasks still pending", pending)); }

        Ok(VerificationResult {
            layer: "File Coverage",
            passed: score > 0.8 && failed == 0,
            score,
            details,
            warnings,
        })
    }

    /// Layer 2: Dependency integrity — check for isolated or missing entities.
    fn verify_dependency_integrity(&self, document_name: &str) -> Result<VerificationResult, DbError> {
        let conn = self.db.connection();
        let mut details = Vec::new();
        let mut warnings = Vec::new();

        // Check symbol registry → every symbol should resolve to an entity
        let sym_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM symbol_registry WHERE document_name = ?1",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        let model_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM data_models WHERE document_name = ?1",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        details.push(format!("Symbols registered: {}, Data models: {}", sym_count, model_count));

        // Check for dangling symbols (symbol with no matching entity)
        let dangling: i64 = conn.query_row(
            "SELECT COUNT(*) FROM symbol_registry s
             LEFT JOIN data_models d ON s.sdef_uri LIKE '%/' || d.entity
             WHERE s.document_name = ?1 AND d.entity IS NULL",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        if dangling > 0 {
            warnings.push(format!("{} symbols with no matching data model", dangling));
        }

        let score = if sym_count > 0 {
            1.0 - (dangling as f64 / sym_count as f64)
        } else {
            0.0
        };

        Ok(VerificationResult {
            layer: "Dependency Integrity",
            passed: dangling == 0 && model_count > 0,
            score,
            details,
            warnings,
        })
    }

    /// Layer 3: Interface coverage — every public interface has methods.
    fn verify_interface_coverage(&self, document_name: &str) -> Result<VerificationResult, DbError> {
        let conn = self.db.connection();
        let mut details = Vec::new();
        let mut warnings = Vec::new();

        // Count contracts with methods
        let contracts: i64 = conn.query_row(
            "SELECT COUNT(*) FROM contracts WHERE document_name = ?1",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        let methods: i64 = conn.query_row(
            "SELECT COUNT(*) FROM contract_methods cm
             JOIN contracts c ON cm.document_name = c.document_name AND cm.contract_name = c.name
             WHERE cm.document_name = ?1",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        details.push(format!("Contracts: {}, Methods: {}", contracts, methods));

        if contracts > 0 && methods == 0 {
            warnings.push("Contracts exist but no methods defined".to_string());
        }

        let score = if contracts > 0 {
            (methods as f64) / (contracts as f64 * 2.0).max(1.0)
        } else {
            1.0 // No contracts = no issue
        };

        Ok(VerificationResult {
            layer: "Interface Coverage",
            passed: contracts == 0 || methods > 0,
            score: score.min(1.0),
            details,
            warnings,
        })
    }

    /// Layer 4: Entity coverage — every data model has attributes with types.
    fn verify_entity_coverage(&self, document_name: &str) -> Result<VerificationResult, DbError> {
        let conn = self.db.connection();
        let mut details = Vec::new();
        let mut warnings = Vec::new();

        let models: i64 = conn.query_row(
            "SELECT COUNT(*) FROM data_models WHERE document_name = ?1",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        let attrs: i64 = conn.query_row(
            "SELECT COUNT(*) FROM data_attributes WHERE document_name = ?1",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        let typed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM data_attributes WHERE document_name = ?1 AND attr_type IS NOT NULL AND attr_type != ''",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        details.push(format!("Data models: {}, Attributes: {}, Typed: {}", models, attrs, typed));

        if models > 0 && attrs == 0 {
            warnings.push("Data models exist but no attributes defined".to_string());
        }
        if typed < attrs {
            warnings.push(format!("{} attributes missing type information", attrs - typed));
        }

        let score = if models > 0 {
            (attrs as f64) / (models as f64 * 3.0).max(1.0)
        } else {
            0.0
        };

        Ok(VerificationResult {
            layer: "Entity Coverage",
            passed: models > 0 && attrs > 0 && typed == attrs,
            score: score.min(1.0),
            details,
            warnings,
        })
    }

    /// Layer 5: Cross-validation — multiple data sources agree.
    fn cross_validate(&self, document_name: &str) -> Result<VerificationResult, DbError> {
        let conn = self.db.connection();
        let mut details = Vec::new();
        let mut warnings = Vec::new();

        // Check: every function_spec should have at least one param or a description
        let funcs: i64 = conn.query_row(
            "SELECT COUNT(*) FROM function_specs WHERE document_name = ?1",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        let funcs_with_logic: i64 = conn.query_row(
            "SELECT COUNT(*) FROM function_specs WHERE document_name = ?1 AND logic IS NOT NULL",
            params![document_name], |row| row.get(0),
        ).unwrap_or(0);

        details.push(format!("Functions: {}, With logic: {}", funcs, funcs_with_logic));

        if funcs > 0 && funcs_with_logic < funcs {
            warnings.push(format!("{} functions without logic/behavior description", funcs - funcs_with_logic));
        }

        // Check: function names should have symbols registered
        let sym_types: HashSet<String> = {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT symbol_type FROM symbol_registry WHERE document_name = ?1"
            ).map_err(|e| DbError::QueryFailed(e.to_string()))?;
            let mut rows = stmt.query(params![document_name])
                .map_err(|e| DbError::QueryFailed(e.to_string()))?;
            let mut set = HashSet::new();
            while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
                set.insert(row.get::<_, String>(0).map_err(|e| DbError::QueryFailed(e.to_string()))?);
            }
            set
        };

        details.push(format!("Symbol types registered: {:?}", sym_types));

        let has_functions = funcs > 0;
        let has_contracts = conn.query_row(
            "SELECT COUNT(*) FROM contracts WHERE document_name = ?1",
            params![document_name], |row| row.get::<_, i64>(0),
        ).unwrap_or(0) > 0;

        // Validation: if functions exist, they should show up somewhere
        if has_functions && !sym_types.contains("function") {
            warnings.push("Functions exist but no function symbols registered".to_string());
        }

        let score = if has_functions || has_contracts {
            let checks_passed = if sym_types.contains("class") || sym_types.contains("interface") { 1 } else { 0 }
                + if funcs_with_logic == funcs || funcs == 0 { 1 } else { 0 };
            checks_passed as f64 / 2.0
        } else {
            1.0
        };

        Ok(VerificationResult {
            layer: "Cross Validation",
            passed: warnings.is_empty(),
            score,
            details,
            warnings,
        })
    }
}

/// Format a completeness report as a human-readable string.
pub fn format_report(report: &CompletenessReport) -> String {
    let mut output = String::new();
    output.push_str(&format!("\n=== Completeness Report: {} ===\n", report.document_name));

    for result in &report.results {
        let icon = if result.passed { "✅" } else { "⚠️ " };
        output.push_str(&format!(
            "\n{} {} ({:.0}%)\n",
            icon, result.layer, result.score * 100.0
        ));
        for detail in &result.details {
            output.push_str(&format!("  • {}\n", detail));
        }
        for warn in &result.warnings {
            output.push_str(&format!("  ⚠  {}\n", warn));
        }
    }

    output.push_str(&format!(
        "\nOverall: {:.0}% (file={:.0}%, dep={:.0}%, iface={:.0}%, entity={:.0}%, cross={:.0}%)\n",
        report.overall_score.overall * 100.0,
        report.overall_score.file_coverage * 100.0,
        report.overall_score.dependency_coverage * 100.0,
        report.overall_score.interface_coverage * 100.0,
        report.overall_score.entity_coverage * 100.0,
        report.overall_score.cross_validation * 100.0,
    ));

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_with_data() -> Arc<Database> {
        let db = Arc::new(Database::in_memory().unwrap());
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, version, created_at, updated_at)
                 VALUES ('test-doc', '1.0', datetime(), datetime());
                 INSERT INTO data_models (entity, document_name, status)
                 VALUES ('User', 'test-doc', 'active');
                 INSERT INTO data_attributes (document_name, entity, name, attr_type)
                 VALUES ('test-doc', 'User', 'id', 'UUID');
                 INSERT INTO data_attributes (document_name, entity, name, attr_type)
                 VALUES ('test-doc', 'User', 'name', 'string');
                 INSERT INTO data_attributes (document_name, entity, name, attr_type)
                 VALUES ('test-doc', 'User', 'email', 'string');",
            ).unwrap();
        }
        db
    }

    #[test]
    fn test_empty_database() {
        let db = Arc::new(Database::in_memory().unwrap());
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, created_at, updated_at) VALUES ('empty', datetime(), datetime());"
            ).unwrap();
        }
        let validator = CompletenessValidator::new(db);
        let report = validator.validate("empty").unwrap();
        assert_eq!(report.results.len(), 5);
        // With no data, scores are low but no crashes
        for result in &report.results {
            assert!(result.score >= 0.0);
        }
    }

    #[test]
    fn test_with_data() {
        let db = setup_with_data();
        let validator = CompletenessValidator::new(db);
        let report = validator.validate("test-doc").unwrap();
        assert!(report.overall_score.entity_coverage > 0.0);
        assert_eq!(report.results.len(), 5);
    }

    #[test]
    fn test_format_report() {
        let db = setup_with_data();
        let validator = CompletenessValidator::new(db);
        let report = validator.validate("test-doc").unwrap();
        let formatted = format_report(&report);
        assert!(formatted.contains("test-doc"));
        assert!(formatted.contains("Overall:"));
        assert!(formatted.contains("Entity Coverage"));
    }
}