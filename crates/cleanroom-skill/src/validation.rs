//! Frontmatter / path / directory validation. Output is compatible with the
//! agentskills.io `skills-ref` `validate` command.

use std::path::Path;

use crate::error::SkillResult;
use crate::parser::parse_skill_markdown;

#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub level: ValidationLevel,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel {
    Error,
    Warning,
}

#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub errors: Vec<ValidationIssue>,
    pub warnings: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
    pub fn issues(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.errors.iter().chain(self.warnings.iter())
    }
}

/// Validate a `SKILL.md` file at `path`.
pub fn validate_skill_dir(path: &Path) -> SkillResult<ValidationReport> {
    let mut report = ValidationReport::default();

    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            report.errors.push(ValidationIssue {
                level: ValidationLevel::Error,
                message: format!("cannot read {}: {e}", path.display()),
            });
            return Ok(report);
        }
    };

    match parse_skill_markdown(path, &content) {
        Ok(parsed) => {
            if parsed.name.is_empty() {
                report.errors.push(ValidationIssue {
                    level: ValidationLevel::Error,
                    message: "missing `name` field".into(),
                });
            }
            if parsed.description.is_empty() {
                report.errors.push(ValidationIssue {
                    level: ValidationLevel::Error,
                    message: "missing `description` field".into(),
                });
            } else if parsed.description.len() > 1024 {
                report.errors.push(ValidationIssue {
                    level: ValidationLevel::Error,
                    message: format!(
                        "description exceeds 1024 chars (got {})",
                        parsed.description.len()
                    ),
                });
            }
            if parsed.name.len() > 64 {
                report.errors.push(ValidationIssue {
                    level: ValidationLevel::Error,
                    message: format!("name exceeds 64 chars (got {})", parsed.name.len()),
                });
            }
            // Lenient warnings
            if !parsed.body.is_empty() && parsed.body.len() < 50 {
                report.warnings.push(ValidationIssue {
                    level: ValidationLevel::Warning,
                    message: "body is very short (< 50 chars); is the skill fully documented?"
                        .into(),
                });
            }
        }
        Err(e) => {
            report.errors.push(ValidationIssue {
                level: ValidationLevel::Error,
                message: format!("parse error: {e}"),
            });
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn valid_skill_reports_no_errors() {
        let tmp = tempdir().unwrap();
        let p = tmp.path().join("SKILL.md");
        fs::write(
            &p,
            "---\nname: ok\ndescription: A valid skill description\n---\n# body\nLong enough body content here.\n",
        )
        .unwrap();
        let r = validate_skill_dir(&p).unwrap();
        assert!(r.is_valid(), "errors: {:?}", r.errors);
    }

    #[test]
    fn missing_description_reports_error() {
        let tmp = tempdir().unwrap();
        let p = tmp.path().join("SKILL.md");
        fs::write(&p, "---\nname: ok\n---\nbody\n").unwrap();
        let r = validate_skill_dir(&p).unwrap();
        assert!(!r.is_valid());
    }
}
