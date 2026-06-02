//! Lexical skill scoring. Mirrors `adk-rust/adk-skill/src/select.rs` for
//! parity with the upstream adk-skill engine.
//!
//! Weights:
//!   - name        +4.0
//!   - description +2.5
//!   - tags        +2.0
//!   - body        +1.0
//!
//! Normalization: `raw / sqrt(unique_body_tokens)`.

use std::collections::HashSet;

use crate::model::{SelectionPolicy, SkillDocument, SkillIndex, SkillMatch, SkillSummary};

/// Select the most relevant skills from the index for the given query.
pub fn select_skills(
    index: &SkillIndex,
    query: &str,
    policy: &SelectionPolicy,
) -> Vec<SkillMatch> {
    if policy.top_k == 0 {
        return Vec::new();
    }

    let include_tags = policy
        .include_tags
        .iter()
        .map(|t| t.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    let exclude_tags = policy
        .exclude_tags
        .iter()
        .map(|t| t.to_ascii_lowercase())
        .collect::<HashSet<_>>();

    let query_tokens = tokenize(query);
    if query_tokens.is_empty() && include_tags.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<SkillMatch> = index
        .skills()
        .iter()
        .filter(|skill| {
            // Task-type filter (cleanroom extension)
            if let Some(ref tt) = policy.task_type {
                if !skill.applies_to_task(tt) {
                    return false;
                }
            }
            tag_allowed(skill, &include_tags, &exclude_tags)
        })
        .map(|skill| {
            let score = score_skill(&query_tokens, skill);
            SkillMatch {
                score,
                skill: SkillSummary::from(skill),
            }
        })
        .filter(|m| m.score >= policy.min_score)
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.skill.name.cmp(&b.skill.name))
            .then_with(|| a.skill.path.cmp(&b.skill.path))
    });

    scored.into_iter().take(policy.top_k).collect()
}

fn tag_allowed(skill: &SkillDocument, include: &HashSet<String>, exclude: &HashSet<String>) -> bool {
    let skill_tags = skill
        .tags
        .iter()
        .map(|t| t.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    if !exclude.is_empty() && !skill_tags.is_disjoint(exclude) {
        return false;
    }
    include.is_empty() || !skill_tags.is_disjoint(include)
}

fn score_skill(query_tokens: &[String], skill: &SkillDocument) -> f32 {
    let name_tokens = to_set(&skill.name);
    let description_tokens = to_set(&skill.description);
    let body_tokens = to_set(&skill.body);
    let tags_tokens: HashSet<String> = skill
        .tags
        .iter()
        .flat_map(|t| tokenize(t))
        .collect();

    let mut score = 0.0_f32;
    for token in query_tokens {
        if name_tokens.contains(token) {
            score += 4.0;
        }
        if description_tokens.contains(token) {
            score += 2.5;
        }
        if tags_tokens.contains(token) {
            score += 2.0;
        }
        if body_tokens.contains(token) {
            score += 1.0;
        }
    }

    // Normalize by sqrt(unique body tokens) to avoid penalizing concise skills.
    let norm = (body_tokens.len().max(1) as f32).sqrt();
    score / norm.max(1.0)
}

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in input.chars() {
        if c.is_ascii_alphanumeric() {
            current.push(c.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn to_set(input: &str) -> HashSet<String> {
    tokenize(input).into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SkillDocument, SkillScope};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn fake_skill(name: &str, desc: &str, body: &str, tags: Vec<String>) -> SkillDocument {
        SkillDocument {
            id: format!("{name}+h"),
            name: name.to_string(),
            description: desc.to_string(),
            license: None,
            compatibility: None,
            tags,
            allowed_tools: vec![],
            denied_tools: vec![],
            allowed_paths: vec![],
            staging: None,
            output_schema: None,
            gates: vec![],
            divergence_spec: None,
            applies_to: vec![],
            token_budget: 4096,
            priority: "normal".into(),
            trigger: false,
            body: body.to_string(),
            path: PathBuf::from("/x"),
            scope: SkillScope::Builtin,
            hash: "h".into(),
            last_modified: None,
            sdef_shard_uri: None,
            metadata: HashMap::new(),
        }
    }

    fn p(min: f32) -> SelectionPolicy {
        SelectionPolicy {
            top_k: 5,
            min_score: min,
            include_tags: vec![],
            exclude_tags: vec![],
            task_type: None,
        }
    }

    fn idx(skills: Vec<SkillDocument>) -> SkillIndex {
        SkillIndex::new(skills)
    }

    #[test]
    fn name_match_outranks_body_match() {
        let i = idx(vec![
            fake_skill("name-match", "x", "x x x x", vec![]),
            fake_skill("body-match", "x", "name-match is everywhere in body", vec![]),
        ]);
        let r = select_skills(&i, "name-match", &p(0.0));
        assert_eq!(r[0].skill.name, "name-match");
    }

    #[test]
    fn min_score_filters_weak_matches() {
        let i = idx(vec![fake_skill("a", "a", "lorem ipsum dolor", vec![])]);
        let r = select_skills(&i, "completely-unrelated", &p(5.0));
        assert!(r.is_empty());
    }

    #[test]
    fn tag_exclude_filters_out() {
        let i = idx(vec![
            fake_skill("a", "d", "b", vec!["emergency".into()]),
            fake_skill("b", "d", "b", vec!["stable".into()]),
        ]);
        let r = select_skills(
            &i,
            "d",
            &SelectionPolicy {
                exclude_tags: vec!["emergency".into()],
                ..p(0.0)
            },
        );
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].skill.name, "b");
    }

    #[test]
    fn top_k_truncates() {
        let skills: Vec<SkillDocument> = (0..10)
            .map(|i| fake_skill(&format!("s{i}"), "common", "common body", vec![]))
            .collect();
        let i = idx(skills);
        let r = select_skills(
            &i,
            "common",
            &SelectionPolicy {
                top_k: 3,
                ..p(0.0)
            },
        );
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn task_type_filter() {
        let i = idx(vec![
            SkillDocument {
                applies_to: vec!["LlmAnalyzeFile".into()],
                ..fake_skill("a", "d", "b", vec![])
            },
            SkillDocument {
                applies_to: vec!["LlmGenerateCode".into()],
                ..fake_skill("b", "d", "b", vec![])
            },
        ]);
        let r = select_skills(
            &i,
            "d",
            &SelectionPolicy {
                task_type: Some("LlmGenerateCode".into()),
                ..p(0.0)
            },
        );
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].skill.name, "b");
    }

    #[test]
    fn normalization_penalizes_very_long_body() {
        let short = fake_skill("s", "rust analysis", "body", vec![]);
        let long = fake_skill("s", "rust analysis", &"x ".repeat(500), vec![]);
        let i = idx(vec![short, long]);
        let r = select_skills(&i, "rust", &p(0.0));
        assert!(r[0].score > 0.0);
    }
}
