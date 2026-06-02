//! S.DEF Schema Validator.
//!
//! Validates `SoftwareDefinition` instances for:
//! - Required fields
//! - URI reference integrity (sdef:// URIs point to existing entities)
//! - Naming conventions
//! - Version consistency

use crate::SoftwareDefinition;
use std::collections::HashSet;

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    pub severity: Severity,
    pub field: String,
    pub message: String,
}

impl ValidationFinding {
    pub fn error(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { severity: Severity::Error, field: field.into(), message: message.into() }
    }
    pub fn warning(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { severity: Severity::Warning, field: field.into(), message: message.into() }
    }
    pub fn info(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { severity: Severity::Info, field: field.into(), message: message.into() }
    }
}

/// Result of a validation run.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub findings: Vec<ValidationFinding>,
}

impl ValidationResult {
    pub fn new(findings: Vec<ValidationFinding>) -> Self {
        let has_errors = findings.iter().any(|f| f.severity == Severity::Error);
        Self { valid: !has_errors, findings }
    }

    pub fn ok() -> Self {
        Self { valid: true, findings: vec![] }
    }
}

/// Validate an S.DEF document.
pub fn validate(sdef: &SoftwareDefinition) -> ValidationResult {
    let mut findings = Vec::new();

    // 1. Required fields
    if sdef.sdef_version.is_empty() {
        findings.push(ValidationFinding::error("sdef_version", "Schema version is required"));
    }
    if sdef.name.is_empty() {
        findings.push(ValidationFinding::error("name", "Software name is required"));
    }

    // 2. Check sdef_version format (date-based YYYY-MM-DD)
    if !sdef.sdef_version.is_empty() {
        let parts: Vec<&str> = sdef.sdef_version.split('-').collect();
        if parts.len() != 3 || parts[0].len() != 4 || parts[1].len() != 2 || parts[2].len() != 2 {
            findings.push(ValidationFinding::warning(
                "sdef_version",
                format!("Expected date format YYYY-MM-DD, got '{}'", sdef.sdef_version),
            ));
        }
    }

    // 3. URI reference integrity
    let all_uris = collect_uris(sdef);

    // Check data model relationships reference existing entities
    if let Some(models) = &sdef.data_models {
        let entity_names: HashSet<&str> = models.iter().map(|m| m.entity.as_str()).collect();
        for model in models {
            if let Some(rels) = &model.relationships {
                for rel in rels {
                    if !entity_names.contains(rel.target.as_str()) {
                        findings.push(ValidationFinding::warning(
                            format!("data_models.{}.relationships", model.entity),
                            format!("Relationship target '{}' does not match any data model entity", rel.target),
                        ));
                    }
                }
            }
        }
    }

    // 4. Contract implements cross-reference
    if let Some(contracts) = &sdef.contracts {
        if let Some(interfaces) = &contracts.interfaces {
            for iface in interfaces {
                if let Some(methods) = &iface.methods {
                    for method in methods {
                        if let Some(pre) = &method.preconditions {
                            for p in pre {
                                if p.starts_with("sdef://") && !all_uris.contains(p.as_str()) {
                                    findings.push(ValidationFinding::warning(
                                        format!("contracts.interfaces.{}.methods.{}.preconditions", iface.name, method.signature),
                                        format!("URI '{}' not found in document", p),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(classes) = &contracts.classes {
            let interface_names: HashSet<&str> = contracts.interfaces.as_ref()
                .map(|i| i.iter().map(|iface| iface.name.as_str()).collect())
                .unwrap_or_default();
            for class in classes {
                if let Some(implements) = &class.implements {
                    for impl_name in implements {
                        if !interface_names.contains(impl_name.as_str()) {
                            findings.push(ValidationFinding::warning(
                                format!("contracts.classes.{}.implements", class.name),
                                format!("Class implements '{}' which is not a defined interface", impl_name),
                            ));
                        }
                    }
                }
            }
        }
    }

    // 5. Behavior function references
    if let Some(behavior) = &sdef.behavior {
        if let Some(functions) = &behavior.functions {
            for func in functions {
                if let Some(inputs) = &func.inputs {
                    for input in inputs {
                        if input.param_type.starts_with("sdef://") && !all_uris.contains(input.param_type.as_str()) {
                            findings.push(ValidationFinding::warning(
                                format!("behavior.functions.{}.inputs", func.name),
                                format!("URI '{}' not found in document", input.param_type),
                            ));
                        }
                    }
                }
            }
        }
    }

    // 6. Empty document check
    if sdef.metadata.is_none()
        && sdef.system_boundary.is_none()
        && sdef.design_decisions.is_none()
        && sdef.data_models.is_none()
        && sdef.contracts.is_none()
        && sdef.behavior.is_none()
        && sdef.ui.is_none()
        && sdef.tests.is_none()
        && sdef.reconstruction_rules.is_none()
        && sdef.dependencies.is_none()
        && sdef.deployment.is_none()
    {
        findings.push(ValidationFinding::info(
            "root",
            "S.DEF document contains no domain-specific entities (all optional fields are empty)",
        ));
    }

    ValidationResult::new(findings)
}

/// Collect all defined URIs from the S.DEF document.
fn collect_uris(sdef: &SoftwareDefinition) -> HashSet<String> {
    let mut uris = HashSet::new();

    // Data model entities
    if let Some(models) = &sdef.data_models {
        for model in models {
            uris.insert(format!("sdef://{}", model.entity));
        }
    }

    // Contracts
    if let Some(contracts) = &sdef.contracts {
        if let Some(interfaces) = &contracts.interfaces {
            for iface in interfaces {
                uris.insert(format!("sdef://interface/{}", iface.name));
            }
        }
        if let Some(classes) = &contracts.classes {
            for class in classes {
                uris.insert(format!("sdef://class/{}", class.name));
            }
        }
        if let Some(enums) = &contracts.enums {
            for e in enums {
                uris.insert(format!("sdef://enum/{}", e.name));
            }
        }
    }

    // Functions
    if let Some(behavior) = &sdef.behavior {
        if let Some(functions) = &behavior.functions {
            for func in functions {
                uris.insert(format!("sdef://{}", func.name));
            }
        }
    }

    uris
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_empty_sdef_has_errors() {
        let sdef = SoftwareDefinition::default();
        let result = validate(&sdef);
        assert!(!result.valid, "Empty sdef should not be valid");
        assert!(result.findings.iter().any(|f| f.field == "sdef_version"));
        assert!(result.findings.iter().any(|f| f.field == "name"));
    }

    #[test]
    fn test_minimal_sdef_valid() {
        let sdef = SoftwareDefinition {
            sdef_version: "2026-05-27".to_string(),
            name: "test-project".to_string(),
            ..Default::default()
        };
        let result = validate(&sdef);
        assert!(result.valid);
    }

    #[test]
    fn test_invalid_date_format_warning() {
        let sdef = SoftwareDefinition {
            sdef_version: "2026/05/27".to_string(),
            name: "test".to_string(),
            ..Default::default()
        };
        let result = validate(&sdef);
        assert!(result.valid, "Date format should be a warning, not error");
        assert!(result.findings.iter().any(|f| f.field == "sdef_version"));
    }

    #[test]
    fn test_data_model_relationship_target_missing() {
        use crate::types::data_model::{DataModel, DataRelationship};
        let dm = DataModel {
            entity: "User".to_string(),
            status: None, version: None, deprecated: None,
            description: None, logical_model: None, attributes: None,
            validation_rules: None, physical_design: None, origin: None,
            relationships: Some(vec![DataRelationship {
                kind: "has_many".to_string(),
                target: "Post".to_string(),
                foreign_key: None,
                join_table: None,
                on_delete: None,
            }]),
        };
        let sdef = SoftwareDefinition {
            sdef_version: "2026-05-27".to_string(),
            name: "test".to_string(),
            data_models: Some(vec![dm]),
            ..Default::default()
        };
        let result = validate(&sdef);
        assert!(result.findings.iter().any(|f| f.message.contains("Post")));
    }

    #[test]
    fn test_class_implements_missing_interface() {
        use crate::types::contracts::{ClassContract, InterfaceContract, Contracts};
        let class = ClassContract {
            name: "UserServiceImpl".to_string(),
            status: None, version: None, deprecated: None,
            description: None,
            implements: Some(vec!["UserService".to_string()]),
            dependencies: None,
            methods: None,
            origin: None,
        };
        let sdef = SoftwareDefinition {
            sdef_version: "2026-05-27".to_string(),
            name: "test".to_string(),
            contracts: Some(Contracts {
                interfaces: Some(vec![InterfaceContract {
                    name: "OtherInterface".to_string(),
                    is_abstract: false, status: None, version: None,
                    deprecated: None, description: None,
                    methods: None, invariants: None, origin: None,
                }]),
                classes: Some(vec![class]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let result = validate(&sdef);
        assert!(result.findings.iter().any(|f| f.message.contains("UserService")));
    }

    #[test]
    fn test_uri_reference_check() {
        use crate::types::behavior::{Behavior, FunctionSpec, FunctionParam};
        let func = FunctionSpec {
            name: "createUser".to_string(),
            description: None,
            inputs: Some(vec![FunctionParam {
                name: "input".to_string(),
                param_type: "sdef://NonExistentType".to_string(),
                description: None,
            }]),
            outputs: None,
            logic: None,
            complexity: None,
            pure_function: false,
            edge_cases: None,
            origin: None,
        };
        let sdef = SoftwareDefinition {
            sdef_version: "2026-05-27".to_string(),
            name: "test".to_string(),
            behavior: Some(Behavior {
                functions: Some(vec![func]),
                flows: None,
                state_machines: None,
            }),
            ..Default::default()
        };
        let result = validate(&sdef);
        assert!(result.findings.iter().any(|f| f.message.contains("NonExistentType")));
    }
}
