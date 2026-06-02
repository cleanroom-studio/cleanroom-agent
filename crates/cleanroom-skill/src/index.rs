//! Build a [`SkillIndex`] from discovered SKILL.md files.
//!
//! See `docs/21-skills-system.md` §4.3 for the on-disk layout.

use std::fs;
use std::path::Path;

use crate::discovery::{builtin_skill_dir, discover_skill_files_with_extras};
use crate::error::SkillResult;
use crate::model::{last_modified_unix, make_skill_id, sha256_hex, SkillDocument, SkillIndex};
use crate::parser::parse_skill_markdown;

/// Load a complete [`SkillIndex`] by scanning the project root + built-in
/// skill directory. Returns an empty index if nothing is found.
pub fn load_skill_index(root: &Path) -> SkillResult<SkillIndex> {
    let mut extras = Vec::new();
    if let Some(builtin) = builtin_skill_dir() {
        extras.push(builtin);
    }
    // Also scan the user's home directory for personal skills (best-effort).
    if let Some(home) = dirs_home() {
        extras.push(home.join(".cleanroom").join("skills"));
        extras.push(home.join(".agents").join("skills"));
    }
    load_skill_index_with_extras(root, &extras)
}

/// Load a [`SkillIndex`] from an explicit list of search roots.
pub fn load_skill_index_with_extras(
    root: &Path,
    extra_dirs: &[std::path::PathBuf],
) -> SkillResult<SkillIndex> {
    let paths = discover_skill_files_with_extras(Some(root), extra_dirs);
    let mut index = SkillIndex::default();
    for path in paths {
        match load_one(&path) {
            Ok(doc) => index.push(doc),
            Err(e) => tracing::warn!(path = %path.display(), error = %e, "skip skill"),
        }
    }
    index.sort();
    Ok(index)
}

/// Load a [`SkillIndex`] from a single root (no extra dirs, no home scan).
/// Useful in tests where you want deterministic results.
pub fn load_skill_index_strict(root: &Path) -> SkillResult<SkillIndex> {
    load_skill_index_with_extras(root, &[])
}

/// Re-scan the filesystem and produce a fresh index (used by `skill_refresh`).
pub fn refresh_skill_index(prev: &SkillIndex, root: &Path) -> SkillResult<SkillIndex> {
    // For MVP, refresh = full reload. Future: incremental based on
    // `last_modified` deltas.
    let _ = prev;
    load_skill_index(root)
}

fn load_one(path: &std::path::Path) -> SkillResult<SkillDocument> {
    let bytes = fs::read(path)?;
    let content = std::str::from_utf8(&bytes).map_err(|e| crate::error::SkillError::Other(format!(
        "SKILL.md is not valid UTF-8: {e}"
    )))?;
    let parsed = parse_skill_markdown(path, content)?;
    let hash = sha256_hex(content.as_bytes());
    let id = make_skill_id(&parsed.name, &hash);
    let license = parsed.license.clone();
    let compatibility = parsed.compatibility.clone();
    let applies_to = parsed.x_cleanroom.applies_to.clone();
    let allowed_tools = effective_allowed_tools(&parsed);
    let denied_tools = parsed.x_cleanroom.denied_tools.clone();
    let allowed_paths = parsed.x_cleanroom.allowed_paths.clone();
    let staging = parsed.x_cleanroom.staging.clone();
    let output_schema = parsed.x_cleanroom.output_schema.clone();
    let gates = parsed.x_cleanroom.gates.clone();
    let divergence_spec = parsed.x_cleanroom.divergence_spec.clone();
    let token_budget = parsed.x_cleanroom.token_budget.unwrap_or(4096);
    let priority = parsed
        .x_cleanroom
        .priority
        .clone()
        .unwrap_or_else(|| "normal".to_string());
    let trigger = parsed.x_cleanroom.trigger.unwrap_or(false);
    let sdef_shard_uri = parsed.x_cleanroom.sdef_shard.clone();
    let metadata = parsed.metadata.clone();
    let body = parsed.body.clone();
    let name = parsed.name.clone();
    let tags = extract_tags(&parsed);

    Ok(SkillDocument {
        id,
        name,
        description: parsed.description.clone(),
        license,
        compatibility,
        tags,
        allowed_tools,
        denied_tools,
        allowed_paths,
        staging,
        output_schema,
        gates,
        divergence_spec,
        applies_to,
        token_budget,
        priority,
        trigger,
        body,
        path: path.to_path_buf(),
        scope: parsed.scope,
        hash,
        last_modified: last_modified_unix(path),
        sdef_shard_uri,
        metadata,
    })
}

fn effective_allowed_tools(parsed: &crate::model::ParsedSkill) -> Vec<String> {
    // x-cleanroom.allowed-tools takes precedence; fall back to the standard
    // `allowed-tools` field.
    if !parsed.x_cleanroom.allowed_tools.is_empty() {
        return parsed.x_cleanroom.allowed_tools.clone();
    }
    parsed.allowed_tools.clone()
}

fn extract_tags(parsed: &crate::model::ParsedSkill) -> Vec<String> {
    // Tags come from: 1) explicit `applies-to` (when it's a list of tags, not
    // task types) and 2) the `metadata` map's `tags` field. We treat
    // `applies-to` as tags conservatively.
    let mut tags: Vec<String> = parsed.x_cleanroom.applies_to.clone();
    if let Some(t) = parsed.metadata.get("tags") {
        for piece in t.split(',') {
            let trimmed = piece.trim();
            if !trimmed.is_empty() && !tags.iter().any(|x| x == trimmed) {
                tags.push(trimmed.to_string());
            }
        }
    }
    tags
}

fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn loads_valid_skill() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let dir = root.join(".cleanroom").join("skills").join("alpha");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            "---\nname: alpha\ndescription: descr\n---\nbody line",
        )
        .unwrap();

        let idx = load_skill_index_strict(root).unwrap();
        assert_eq!(idx.len(), 1);
        let s = &idx.skills()[0];
        assert_eq!(s.name, "alpha");
        assert!(s.id.starts_with("alpha+"));
    }

    #[test]
    fn skips_invalid_skill_files() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let dir = root.join(".cleanroom").join("skills").join("bad");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            "---\nname: bad\n---\nbody", // missing description
        )
        .unwrap();

        let idx = load_skill_index_strict(root).unwrap();
        assert_eq!(idx.len(), 0, "invalid skill should be skipped");
    }
}
