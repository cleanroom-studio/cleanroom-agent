//! Tier 1 / Tier 2 prompt block construction.
//!
//! See `docs/21-skills-system.md` §5 for the progressive disclosure model.

use crate::model::{SelectionPolicy, SkillDocument, SkillIndex, SkillSummary};
use crate::select::select_skills;

/// Build the Tier 1 catalog block — a compact XML summary of every visible
/// skill. This is what gets injected into the system prompt at session start.
pub fn build_skill_catalog_block(index: &SkillIndex, task_type: Option<&str>) -> String {
    let mut out = String::from("<available_skills>\n");
    for skill in index.skills() {
        if let Some(tt) = task_type {
            if !skill.applies_to_task(tt) {
                continue;
            }
        }
        let summary = SkillSummary::from(skill);
        out.push_str("  <skill>\n");
        out.push_str(&format!("    <name>{}</name>\n", xml_escape(&summary.name)));
        out.push_str(&format!(
            "    <description>{}</description>\n",
            xml_escape(&summary.description)
        ));
        out.push_str(&format!("    <scope>{:?}</scope>\n", summary.scope));
        out.push_str(&format!("    <priority>{}</priority>\n", summary.priority));
        out.push_str(&format!(
            "    <token_budget>{}</token_budget>\n",
            summary.token_budget
        ));
        if !summary.allowed_tools.is_empty() {
            out.push_str("    <allowed_tools>\n");
            for t in &summary.allowed_tools {
                out.push_str(&format!("      <tool>{}</tool>\n", xml_escape(t)));
            }
            out.push_str("    </allowed_tools>\n");
        }
        if !summary.allowed_paths.is_empty() {
            out.push_str("    <allowed_paths>\n");
            for p in &summary.allowed_paths {
                out.push_str(&format!("      <path>{}</path>\n", xml_escape(p)));
            }
            out.push_str("    </allowed_paths>\n");
        }
        out.push_str("  </skill>\n");
    }
    out.push_str("</available_skills>\n\n");
    out.push_str("Use `skill_activate` MCP tool to load a skill's full instructions when relevant.\n");
    out.push_str(
        "When a skill references relative paths, resolve them against the skill's directory.\n",
    );
    out
}

/// Select the best skill for `query` and return its Tier 2 prompt block
/// (capped at `token_budget_chars`). Returns `None` if no skill passes the
/// policy's `min_score` threshold.
pub fn select_skill_prompt_block(
    index: &SkillIndex,
    query: &str,
    policy: &SelectionPolicy,
    token_budget_chars: usize,
) -> Option<(String, SkillSummary)> {
    let matches = select_skills(index, query, policy);
    matches
        .into_iter()
        .next()
        .map(|m| (m.skill.engineer_prompt_block(token_budget_chars), m.skill))
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

impl SkillSummary {
    /// Local helper: keep `engineer_prompt_block` callable on a summary so
    /// the catalog block can preview skills without loading the body.
    pub fn engineer_prompt_block(&self, _max_chars: usize) -> String {
        format!("[skill:{}]\n{}\n[/skill]", self.name, self.description)
    }
}

/// Backward-compatible free function form.
pub fn engineer_instruction(skill: &SkillDocument, max_chars: usize) -> String {
    skill.engineer_instruction(max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SkillDocument;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn skill(name: &str, desc: &str, applies_to: Vec<String>) -> SkillDocument {
        SkillDocument {
            id: format!("{name}+h"),
            name: name.to_string(),
            description: desc.to_string(),
            license: None,
            compatibility: None,
            tags: vec![],
            allowed_tools: vec!["fs.read_file".into()],
            denied_tools: vec![],
            allowed_paths: vec!["src/**/*.rs".into()],
            staging: None,
            output_schema: None,
            gates: vec![],
            divergence_spec: None,
            applies_to,
            token_budget: 4096,
            priority: "normal".into(),
            trigger: false,
            body: "body".into(),
            path: PathBuf::from("/x"),
            scope: crate::model::SkillScope::Builtin,
            hash: "h".into(),
            last_modified: None,
            sdef_shard_uri: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn catalog_block_includes_all_visible_skills() {
        let idx = SkillIndex::new(vec![
            skill("a", "alpha", vec![]),
            skill("b", "beta", vec![]),
        ]);
        let s = build_skill_catalog_block(&idx, None);
        assert!(s.contains("<name>a</name>"));
        assert!(s.contains("<name>b</name>"));
        assert!(s.contains("<available_skills>"));
    }

    #[test]
    fn catalog_filters_by_task_type() {
        let idx = SkillIndex::new(vec![
            skill("a", "alpha", vec!["LlmAnalyzeFile".into()]),
            skill("b", "beta", vec!["LlmGenerateCode".into()]),
        ]);
        let s = build_skill_catalog_block(&idx, Some("LlmAnalyzeFile"));
        assert!(s.contains("<name>a</name>"));
        assert!(!s.contains("<name>b</name>"));
    }

    #[test]
    fn prompt_block_uses_skill_template() {
        let idx = SkillIndex::new(vec![skill("rust-x", "R", vec![])]);
        let p = SelectionPolicy {
            top_k: 1,
            min_score: 0.0,
            ..Default::default()
        };
        let (block, summary) =
            select_skill_prompt_block(&idx, "rust", &p, 1000).expect("matched");
        assert!(block.contains("[skill:rust-x]"));
        assert!(block.contains("[/skill]"));
        assert_eq!(summary.name, "rust-x");
    }
}
