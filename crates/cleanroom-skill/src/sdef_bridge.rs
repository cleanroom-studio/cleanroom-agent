//! `SkillDocument` <-> S.DEF `SoftwareMetadata.annotations` round-trip.
//!
//! S.DEF does not have a generic "Entity" type, so skills ride along
//! inside the existing `SoftwareMetadata.annotations` map under the key
//! `x-cleanroom-skill/<name>`. The serialized form is the full
//! `SkillDocument` JSON.
//!
//! See `PLAN2.md` §E.4 for the field mapping.

use serde_json::Value;
use sdef_core::SoftwareMetadata;

use crate::error::{SkillError, SkillResult};
use crate::model::SkillDocument;

/// Key prefix used inside `SoftwareMetadata.annotations` to namespace skills.
pub const ANNOTATION_KEY_PREFIX: &str = "x-cleanroom-skill/";

/// Convert a `SkillDocument` into a `(key, value)` pair suitable for
/// `SoftwareMetadata.annotations`. The `key` is `x-cleanroom-skill/<name>`.
pub fn skill_to_sdef_entity(skill: &SkillDocument) -> (String, Value) {
    let key = format!("{}{}", ANNOTATION_KEY_PREFIX, skill.name);
    let value = serde_json::to_value(skill).unwrap_or(Value::Null);
    (key, value)
}

/// Convert a `(key, value)` annotation pair back into a `SkillDocument`.
pub fn sdef_entity_to_skill(key: &str, value: &Value) -> SkillResult<SkillDocument> {
    if !key.starts_with(ANNOTATION_KEY_PREFIX) {
        return Err(SkillError::Sdef(format!(
            "annotation key `{key}` is not a skill (expected prefix `{ANNOTATION_KEY_PREFIX}`)"
        )));
    }
    serde_json::from_value::<SkillDocument>(value.clone())
        .map_err(|e| SkillError::Sdef(format!("decode skill `{key}`: {e}")))
}

/// Insert a skill into a `SoftwareMetadata` annotations map. Returns the
/// updated `SoftwareMetadata` (or `Some` if the caller wants to take
/// ownership). The map is initialized to an empty `HashMap` if absent.
pub fn put_skill(meta: &mut SoftwareMetadata, skill: &SkillDocument) {
    let (k, v) = skill_to_sdef_entity(skill);
    let map = meta
        .annotations
        .get_or_insert_with(Default::default);
    map.insert(k, v);
}

/// List all skills in a `SoftwareMetadata` annotations map.
pub fn list_skills(meta: &SoftwareMetadata) -> Vec<SkillDocument> {
    let Some(map) = &meta.annotations else {
        return Vec::new();
    };
    map.iter()
        .filter(|(k, _)| k.starts_with(ANNOTATION_KEY_PREFIX))
        .filter_map(|(_, v)| serde_json::from_value::<SkillDocument>(v.clone()).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SkillScope;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn fake_skill(name: &str) -> SkillDocument {
        SkillDocument {
            id: format!("{name}+h"),
            name: name.to_string(),
            description: "An analysis skill".into(),
            license: Some("MIT".into()),
            compatibility: None,
            tags: vec!["rust".into()],
            allowed_tools: vec!["fs.read_file".into(), "mcp.lsp_query".into()],
            denied_tools: vec!["bash.run".into()],
            allowed_paths: vec!["src/**/*.rs".into()],
            staging: None,
            output_schema: None,
            gates: vec![],
            divergence_spec: None,
            applies_to: vec!["LlmAnalyzeFile".into()],
            token_budget: 8192,
            priority: "high".into(),
            trigger: false,
            body: "## Workflow\n1. read\n2. write".into(),
            path: PathBuf::from("/proj/.cleanroom/skills/test/SKILL.md"),
            scope: SkillScope::ProjectCleanroom,
            hash: "abc123".into(),
            last_modified: Some(1717000000),
            sdef_shard_uri: Some("sdef://cleanroom/skills/test".into()),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn roundtrip_through_software_metadata() {
        let mut meta = SoftwareMetadata::default();
        let original = fake_skill("test");
        put_skill(&mut meta, &original);

        let skills = list_skills(&meta);
        assert_eq!(skills.len(), 1);
        let restored = &skills[0];

        assert_eq!(restored.name, original.name);
        assert_eq!(restored.description, original.description);
        assert_eq!(restored.body, original.body);
        assert_eq!(restored.allowed_tools, original.allowed_tools);
        assert_eq!(restored.denied_tools, original.denied_tools);
        assert_eq!(restored.allowed_paths, original.allowed_paths);
        assert_eq!(restored.applies_to, original.applies_to);
        assert_eq!(restored.token_budget, original.token_budget);
        assert_eq!(restored.priority, original.priority);
    }

    #[test]
    fn rejects_non_skill_keys() {
        let key = "not-a-skill";
        let v = serde_json::json!({});
        let r = sdef_entity_to_skill(key, &v);
        assert!(matches!(r, Err(SkillError::Sdef(_))));
    }

    #[test]
    fn list_skills_skips_non_skill_keys() {
        let mut meta = SoftwareMetadata::default();
        let map = meta.annotations.get_or_insert_with(Default::default);
        map.insert("some-other-thing".into(), serde_json::json!({}));
        put_skill(&mut meta, &fake_skill("a"));
        put_skill(&mut meta, &fake_skill("b"));
        let skills = list_skills(&meta);
        assert_eq!(skills.len(), 2);
    }
}
