//! Replication Protocol tool authorization (Action Surface enforcement).
//!
//! See `docs/21-skills-system.md` §6.3 + §8.3.

use std::path::Path;

use crate::error::{SkillError, SkillResult};
use crate::model::SkillDocument;
use glob::Pattern;

/// High-level coordinator — keeps the active skill in scope and exposes
/// `check()` + `check_path()` helpers.
#[derive(Debug, Default, Clone)]
pub struct ContextCoordinator {
    /// `None` means no active skill — caller is in "read-only explorer" mode.
    pub active_skill: Option<SkillDocument>,
}

impl ContextCoordinator {
    pub fn with_active_skill(skill: SkillDocument) -> Self {
        Self {
            active_skill: Some(skill),
        }
    }

    pub fn read_only() -> Self {
        Self { active_skill: None }
    }

    /// Check whether `tool_name` is allowed under the active skill.
    pub fn check(&self, tool_name: &str) -> SkillResult<()> {
        check_tool_authorization(self.active_skill.as_ref(), tool_name, None)
    }

    /// Check whether `tool_name` with the given `file_path` argument is allowed
    /// (also enforces `allowed-paths` for `fs.*` tools).
    pub fn check_path(&self, tool_name: &str, file_path: &str) -> SkillResult<()> {
        check_tool_authorization(self.active_skill.as_ref(), tool_name, Some(file_path))
    }
}

#[derive(Debug, Clone, Default)]
pub struct CoordinatorConfig {
    /// If true, tools not declared in `allowed-tools` are denied even when
    /// the list is empty (default: false — empty list = allow all).
    pub strict_mode: bool,
}

/// Core authorization function. Called by both MCP dispatch and the
/// `mcp_tool_bridge` proxy.
///
/// `file_path` is extracted from the tool's `path` argument. Pass `None` for
/// tools that don't take a path (e.g. `mcp.sdef.read_shard`).
pub fn check_tool_authorization(
    active_skill: Option<&SkillDocument>,
    tool_name: &str,
    file_path: Option<&str>,
) -> SkillResult<()> {
    // 1. No active skill → read-only tools allowed, write tools denied.
    let Some(skill) = active_skill else {
        if is_writing_tool(tool_name) {
            return Err(SkillError::NoActiveSkill(tool_name.to_string()));
        }
        return Ok(());
    };

    // 2. denied-tools always wins.
    if skill.denied_tools.iter().any(|t| t == tool_name) {
        return Err(SkillError::DeniedBySkill {
            skill: skill.name.clone(),
            tool: tool_name.to_string(),
        });
    }

    // 3. allowed-tools whitelist (empty = allow all, unless strict).
    if !skill.allowed_tools.is_empty() && !skill.allowed_tools.iter().any(|t| t == tool_name) {
        return Err(SkillError::NotInAllowedTools {
            skill: skill.name.clone(),
            tool: tool_name.to_string(),
        });
    }

    // 4. fs.* tools: enforce allowed-paths.
    if tool_name.starts_with("fs.") {
        if let Some(path) = file_path {
            if !is_path_allowed(path, &skill.allowed_paths) {
                return Err(SkillError::PathNotAllowed {
                    skill: skill.name.clone(),
                    path: path.to_string(),
                });
            }
        }
    }

    // 5. staging.* write tools: require staging.mode declared.
    if is_staging_write_tool(tool_name) && skill.staging.is_none() {
        return Err(SkillError::StagingNotConfigured {
            skill: skill.name.clone(),
        });
    }

    Ok(())
}

fn is_writing_tool(tool_name: &str) -> bool {
    // Tools that mutate state / filesystem / DB. Read-only tools bypass the
    // active-skill check.
    if tool_name.starts_with("staging.") && tool_name != "staging.read" && tool_name != "staging.diff"
    {
        return true;
    }
    if tool_name.starts_with("bash.") {
        return true;
    }
    // MCP write tools (most are write)
    let mcp_writes = [
        "mcp.task.complete_task",
        "mcp.task.fail_task",
        "mcp.task.create_task",
        "mcp.naming.register_custom_name",
        "mcp.symbol_registry.update",
        "mcp.sdef.upsert_data_model",
        "mcp.sdef.upsert_contract",
        "mcp.sdef.upsert_function_spec",
        "mcp.sdef.upsert_design_decision",
    ];
    mcp_writes.iter().any(|t| *t == tool_name)
}

fn is_staging_write_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "staging.write" | "staging.edit" | "staging.delete" | "staging.commit"
    )
}

/// Check whether a path matches at least one of the glob patterns.
pub fn is_path_allowed(path: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        // Empty = allow all paths (caller may want to restrict separately).
        return true;
    }
    for pat in patterns {
        if let Ok(p) = Pattern::new(pat) {
            if p.matches_path(Path::new(path)) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SkillDocument, SkillScope};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn fake_skill(name: &str, allowed_tools: Vec<String>, allowed_paths: Vec<String>) -> SkillDocument {
        SkillDocument {
            id: format!("{name}+fake"),
            name: name.to_string(),
            description: "d".into(),
            license: None,
            compatibility: None,
            tags: vec![],
            allowed_tools,
            denied_tools: vec![],
            allowed_paths,
            staging: None,
            output_schema: None,
            gates: vec![],
            divergence_spec: None,
            applies_to: vec![],
            token_budget: 4096,
            priority: "normal".into(),
            trigger: false,
            body: "b".into(),
            path: PathBuf::from("/x"),
            scope: SkillScope::Builtin,
            hash: "h".into(),
            last_modified: None,
            sdef_shard_uri: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn no_skill_denies_writes() {
        assert!(check_tool_authorization(None, "fs.read_file", None).is_ok());
        assert!(check_tool_authorization(None, "staging.write", None).is_err());
        assert!(check_tool_authorization(None, "bash.run", None).is_err());
        assert!(check_tool_authorization(None, "mcp.task.complete_task", None).is_err());
    }

    #[test]
    fn denied_tools_wins() {
        let s = fake_skill(
            "x",
            vec!["bash.run".into()],
            vec!["src/**".into()],
        );
        let mut s = s;
        s.denied_tools = vec!["bash.run".into()];
        let r = check_tool_authorization(Some(&s), "bash.run", None);
        assert!(matches!(r, Err(SkillError::DeniedBySkill { .. })));
    }

    #[test]
    fn allowed_tools_whitelist() {
        let s = fake_skill("x", vec!["fs.read_file".into()], vec![]);
        assert!(check_tool_authorization(Some(&s), "fs.read_file", None).is_ok());
        let r = check_tool_authorization(Some(&s), "staging.write", None);
        assert!(matches!(r, Err(SkillError::NotInAllowedTools { .. })));
    }

    #[test]
    fn fs_tool_path_check() {
        let s = fake_skill("x", vec!["fs.read_file".into()], vec!["src/**/*.rs".into()]);
        assert!(check_tool_authorization(Some(&s), "fs.read_file", Some("src/main.rs")).is_ok());
        let r = check_tool_authorization(Some(&s), "fs.read_file", Some("target/debug/foo"));
        assert!(matches!(r, Err(SkillError::PathNotAllowed { .. })));
    }

    #[test]
    fn staging_write_requires_staging_mode() {
        let s = fake_skill("x", vec!["staging.write".into()], vec![]);
        let r = check_tool_authorization(Some(&s), "staging.write", None);
        assert!(matches!(r, Err(SkillError::StagingNotConfigured { .. })));
    }
}
