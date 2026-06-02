//! `SKILL.md` parser — extracts YAML frontmatter + markdown body.
//!
//! The parser is intentionally lenient:
//! - Empty or absent frontmatter → fall back to filename as the name (per
//!   agentskills.io "convention files" pattern, see `parse_instruction_markdown`).
//! - Malformed YAML → caller can decide whether to skip or warn.
//! - Trailing whitespace / BOM tolerated.

use std::path::{Path, PathBuf};

use crate::error::{SkillError, SkillResult};
use crate::model::{ParsedSkill, SkillFrontmatter, SkillScope};

/// Parse a SKILL.md file's contents into a [`ParsedSkill`].
///
/// `path` is used for error messages and to infer the skill scope.
pub fn parse_skill_markdown(path: &Path, content: &str) -> SkillResult<ParsedSkill> {
    let (frontmatter, body) = split_frontmatter(content)?;

    let fm: SkillFrontmatter = if frontmatter.trim().is_empty() {
        SkillFrontmatter::default()
    } else {
        serde_yaml::from_str(&frontmatter).map_err(|e| SkillError::InvalidFrontmatter {
            path: path.to_path_buf(),
            message: format!("yaml parse: {e}"),
        })?
    };

    let scope = infer_scope_from_path(path);
    let name = if fm.name.is_empty() {
        // Lenient mode: fall back to the directory name.
        infer_name_from_path(path)
            .ok_or_else(|| SkillError::MissingField {
                path: path.to_path_buf(),
                field: "name".to_string(),
            })?
    } else {
        fm.name.clone()
    };

    if fm.description.is_empty() {
        return Err(SkillError::MissingField {
            path: path.to_path_buf(),
            field: "description".to_string(),
        });
    }

    // Validate name shape (per agentskills.io spec).
    validate_skill_name(&name).map_err(|msg| SkillError::InvalidFrontmatter {
        path: path.to_path_buf(),
        message: msg,
    })?;

    Ok(ParsedSkill {
        name,
        description: fm.description,
        license: fm.license,
        compatibility: fm.compatibility,
        metadata: fm.metadata,
        allowed_tools: fm.allowed_tools,
        x_cleanroom: fm.x_cleanroom,
        body: body.trim().to_string(),
        scope,
    })
}

/// Parse a "convention" markdown file (e.g. AGENTS.md, CLAUDE.md) into a
/// [`ParsedSkill`]. Frontmatter is entirely optional; the filename is used
/// as the skill name and a default scope tag is added.
pub fn parse_instruction_markdown(path: &Path, content: &str) -> SkillResult<ParsedSkill> {
    let (frontmatter, body) = split_frontmatter(content)?;

    let (name, fm, applies_to) = if frontmatter.trim().is_empty() {
        // No frontmatter — derive name from file stem, add convention tag.
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("instruction")
            .to_string();
        let stem = name.to_ascii_lowercase();
        (name, SkillFrontmatter::default(), vec![stem])
    } else {
        let fm: SkillFrontmatter = serde_yaml::from_str(&frontmatter).map_err(|e| {
            SkillError::InvalidFrontmatter {
                path: path.to_path_buf(),
                message: format!("yaml parse: {e}"),
            }
        })?;
        if fm.name.is_empty() {
            return Err(SkillError::MissingField {
                path: path.to_path_buf(),
                field: "name".to_string(),
            });
        }
        (fm.name.clone(), fm, vec![])
    };

    Ok(ParsedSkill {
        name,
        description: fm.description,
        license: fm.license,
        compatibility: fm.compatibility,
        metadata: fm.metadata,
        allowed_tools: fm.allowed_tools,
        x_cleanroom: fm.x_cleanroom,
        body: body.trim().to_string(),
        scope: infer_scope_from_path(path),
    })
    .map(|mut s| {
        if !applies_to.is_empty() {
            s.x_cleanroom.applies_to = applies_to;
        }
        s
    })
}

/// Split content into `(frontmatter, body)`. The frontmatter must be the first
/// thing in the file, delimited by `---` lines.
fn split_frontmatter(content: &str) -> SkillResult<(String, String)> {
    let content = content.trim_start_matches('\u{feff}'); // strip BOM

    let mut lines = content.split_inclusive('\n');

    let first = lines.next().unwrap_or("");
    if !first.trim_start().starts_with("---") {
        return Ok((String::new(), content.to_string()));
    }

    let mut yaml = String::new();
    let mut found_close = false;
    for line in lines {
        if line.trim_start().trim_end() == "---" {
            found_close = true;
            break;
        }
        yaml.push_str(line);
    }
    if !found_close {
        return Err(SkillError::InvalidFrontmatter {
            path: PathBuf::from("<content>"),
            message: "frontmatter opened with `---` but no closing `---` found".to_string(),
        });
    }

    // Body is whatever is left after the closing `---` line.
    let consumed = format!("{first}{yaml}---");
    let body_offset = consumed.len();
    let body = content[body_offset..].to_string();

    Ok((yaml, body))
}

/// Validate a skill name against the agentskills.io spec.
fn validate_skill_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 64 {
        return Err(format!("name must be 1-64 chars (got {})", name.len()));
    }
    if name.starts_with('-') || name.ends_with('-') {
        return Err("name must not start or end with `-`".to_string());
    }
    if name.contains("--") {
        return Err("name must not contain consecutive `--`".to_string());
    }
    for c in name.chars() {
        let is_lower = c.is_ascii_lowercase();
        let is_digit = c.is_ascii_digit();
        let is_dash = c == '-';
        if !(is_lower || is_digit || is_dash) {
            return Err(format!(
                "name contains invalid char `{c}` (only a-z, 0-9, - allowed)"
            ));
        }
    }
    Ok(())
}

fn infer_name_from_path(path: &Path) -> Option<String> {
    // Walk up to find the first directory name (the skill name).
    let p = path.to_path_buf();
    let mut cur: Option<&Path> = Some(p.as_path());
    let scope_markers = ["skills", ".cleanroom", ".agents"];
    while let Some(c) = cur {
        if let Some(name) = c.file_name().and_then(|s| s.to_str()) {
            if !scope_markers.contains(&name) && !name.ends_with(".md") {
                return Some(name.to_string());
            }
        }
        cur = c.parent();
    }
    path.file_stem().and_then(|s| s.to_str()).map(String::from)
}

fn infer_scope_from_path(path: &Path) -> SkillScope {
    let s = path.to_string_lossy();
    let p = std::path::Path::new(&*s);
    let components: Vec<String> = p
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();

    let is_user = components.iter().any(|c| c.starts_with(".local") || c == "$HOME")
        || s.contains("/.local/share/")
        || s.contains("/.config/")
        || s.starts_with("~/")
        || s.contains("/Users/");

    // Try to find the marker segment.
    for (i, c) in components.iter().enumerate() {
        if c == "skills" || c == ".cleanroom" {
            // The previous component tells us the project name; the segment
            // before the skill-name is the marker. Use a heuristic: if the
            // path is inside a project repo (has Cargo.toml / .git / src
            // ancestors) it's project scope; otherwise user scope.
            // For simplicity, treat the path containing /target/ or /src/
            // or /Cargo.toml as a strong project signal.
            let has_project_signal = components
                .iter()
                .any(|c| c == "src" || c == "target" || c == ".git" || c == "Cargo.toml");
            let _ = has_project_signal;
            let _ = i;
            // Distinguish .cleanroom vs .agents within the parent.
            let parent_is_cleanroom = components
                .iter()
                .take(components.len() - 1)
                .any(|c| c == ".cleanroom");
            let parent_is_agents = components
                .iter()
                .take(components.len() - 1)
                .any(|c| c == ".agents");
            return match (is_user, parent_is_cleanroom, parent_is_agents) {
                (true, true, _) => SkillScope::UserCleanroom,
                (true, _, true) => SkillScope::UserAgents,
                (false, true, _) => SkillScope::ProjectCleanroom,
                (false, _, true) => SkillScope::ProjectAgents,
                // If no marker is found but the path is `<crate>/skills/...`,
                // we treat it as Builtin (compiled into the binary).
                _ => SkillScope::Builtin,
            };
        }
    }
    // Default to Builtin for any other layout (e.g. `<crate>/skills/foo/SKILL.md`).
    SkillScope::Builtin
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parses_full_frontmatter() {
        let md = r#"---
name: rust-analysis
description: Analyze Rust source code.
license: MIT
allowed-tools:
  - fs.read_file
  - mcp.lsp_query
x-cleanroom:
  allowed-paths:
    - "src/**/*.rs"
  priority: high
  token-budget: 4096
---

# Rust Code Analysis

Workflow:
1. Read the source file
2. Query LSP for types
"#;
        let parsed =
            parse_skill_markdown(&PathBuf::from("/proj/.cleanroom/skills/rust-analysis/SKILL.md"), md)
                .expect("parse");
        assert_eq!(parsed.name, "rust-analysis");
        assert_eq!(parsed.description, "Analyze Rust source code.");
        assert_eq!(parsed.allowed_tools, vec!["fs.read_file", "mcp.lsp_query"]);
        assert_eq!(parsed.x_cleanroom.allowed_paths, vec!["src/**/*.rs"]);
        assert_eq!(parsed.x_cleanroom.priority.as_deref(), Some("high"));
        assert_eq!(parsed.x_cleanroom.token_budget, Some(4096));
        assert!(parsed.body.contains("Workflow"));
    }

    #[test]
    fn missing_description_rejected() {
        let md = "---\nname: x\n---\nbody\n";
        let err = parse_skill_markdown(&PathBuf::from("/x/SKILL.md"), md).unwrap_err();
        match err {
            SkillError::MissingField { field, .. } => assert_eq!(field, "description"),
            e => panic!("unexpected: {e:?}"),
        }
    }

    #[test]
    fn lenient_name_falls_back_to_directory() {
        let md = "---\ndescription: My skill\n---\nbody\n";
        let parsed = parse_skill_markdown(
            &PathBuf::from("/proj/.cleanroom/skills/named-thing/SKILL.md"),
            md,
        )
        .expect("parse");
        assert_eq!(parsed.name, "named-thing");
    }

    #[test]
    fn x_cleanroom_nested_block_parses() {
        let md = r#"---
name: gating-skill
description: d
x-cleanroom:
  staging:
    mode: git-worktree
    base: ".cleanroom/staging"
  gates:
    - name: compile
      type: compile
      blocking: true
  divergence-spec:
    required-categories:
      - library-substitution
    block-on-undisclosed: true
---

body
"#;
        let parsed = parse_skill_markdown(&PathBuf::from("/p/skills/gating-skill/SKILL.md"), md)
            .expect("parse");
        let staging = parsed.x_cleanroom.staging.expect("staging");
        assert_eq!(staging.mode, "git-worktree");
        assert_eq!(staging.base.as_deref(), Some(".cleanroom/staging"));
        assert_eq!(parsed.x_cleanroom.gates.len(), 1);
        assert_eq!(parsed.x_cleanroom.gates[0].name, "compile");
        assert!(parsed.x_cleanroom.gates[0].blocking);
        let div = parsed.x_cleanroom.divergence_spec.expect("div");
        assert_eq!(div.required_categories, vec!["library-substitution"]);
        assert!(div.block_on_undisclosed);
    }

    #[test]
    fn invalid_yaml_returns_error() {
        let md = "---\nname: [broken\n---\nbody\n";
        let err = parse_skill_markdown(&PathBuf::from("/p/skills/x/SKILL.md"), md).unwrap_err();
        assert!(matches!(err, SkillError::InvalidFrontmatter { .. }));
    }

    #[test]
    fn name_validation_rejects_uppercase() {
        assert!(validate_skill_name("Bad-Name").is_err());
        assert!(validate_skill_name("--leading").is_err());
        assert!(validate_skill_name("trailing-").is_err());
        assert!(validate_skill_name("a--b").is_err());
        assert!(validate_skill_name("good-name_1").is_err()); // underscore not allowed
        assert!(validate_skill_name("good-name-1").is_ok());
    }

    #[test]
    fn parses_instruction_markdown_with_no_frontmatter() {
        let md = "# Heading\nSome text\n";
        let parsed = parse_instruction_markdown(
            &PathBuf::from("/proj/AGENTS.md"),
            md,
        )
        .expect("parse");
        assert_eq!(parsed.name, "AGENTS");
        assert!(parsed.body.contains("Heading"));
    }

    #[test]
    fn split_frontmatter_handles_missing_close() {
        let md = "---\nname: x\nbody still inside";
        let err = split_frontmatter(md).unwrap_err();
        assert!(matches!(err, SkillError::InvalidFrontmatter { .. }));
    }
}
