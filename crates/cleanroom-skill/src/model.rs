//! Core data types for cleanroom-skill: SkillFrontmatter, SkillDocument, SkillIndex,
//! SelectionPolicy, SkillMatch, SkillSummary.
//!
//! Field design follows the agentskills.io open format plus a `x-cleanroom.*`
//! extension namespace that carries Replication Protocol metadata
//! (allowed-paths, output-schema, gates, divergence-spec, etc.).
//! See [`docs/21-skills-system.md`](file://./docs/21-skills-system.md) §4 for the
//! full schema.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A fully-parsed `SKILL.md` frontmatter. The fields come in two layers:
///
/// 1. The standard agentskills.io fields (name / description / license /
///    compatibility / metadata / allowed-tools).
/// 2. The `x-cleanroom.*` extension namespace carrying Replication Protocol
///    metadata (allowed-paths, output-schema, gates, divergence-spec, staging).
///
/// `#[serde(default)]` on every field keeps the parser tolerant of skills
/// authored for other clients (OpenCode, Claude Code) that may omit most of
/// the metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SkillFrontmatter {
    // --- standard agentskills.io fields ---
    pub name: String,
    pub description: String,
    pub license: Option<String>,
    pub compatibility: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,

    // --- standard `allowed-tools` (string list, per spec) ---
    #[serde(rename = "allowed-tools")]
    pub allowed_tools: Vec<String>,

    // --- cleanroom extensions (x-cleanroom.*) ---
    #[serde(default, rename = "x-cleanroom")]
    pub x_cleanroom: XCleanroom,
}

/// The `x-cleanroom` extension block. Holds Replication Protocol metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct XCleanroom {
    /// Glob patterns for the `fs.*` read tools.
    #[serde(rename = "allowed-paths")]
    pub allowed_paths: Vec<String>,

    /// Detailed allowed-tools for cleanroom (supports `mcp.*` / `fs.*` /
    /// `staging.*` / `bash.*` namespaces). The standard `allowed-tools` field
    /// is interpreted as a flat list; this field is preferred when present.
    #[serde(rename = "allowed-tools")]
    pub allowed_tools: Vec<String>,

    /// Explicit deny list (overrides allowed-tools).
    #[serde(rename = "denied-tools")]
    pub denied_tools: Vec<String>,

    /// Task types this skill applies to (`LlmAnalyzeFile` / `LlmGenerateCode` / ...).
    #[serde(rename = "applies-to")]
    pub applies_to: Vec<String>,

    /// `sdef://` URI for the skill's portable S.DEF shard.
    #[serde(rename = "sdef-shard")]
    pub sdef_shard: Option<String>,

    /// Staging mode declaration (required if skill uses `staging.*` write tools).
    pub staging: Option<StagingConfig>,

    /// Required output structure (JSON Schema). Validated by `staging.write`.
    #[serde(rename = "output-schema")]
    pub output_schema: Option<serde_json::Value>,

    /// Verification gates the orchestrator must run before `staging.commit`.
    #[serde(default)]
    pub gates: Vec<VerificationGate>,

    /// Divergence reporting policy.
    #[serde(rename = "divergence-spec")]
    pub divergence_spec: Option<DivergenceSpec>,

    /// Token budget for the Tier 2 injected body.
    #[serde(rename = "token-budget")]
    pub token_budget: Option<u32>,

    /// Priority hint (used in Tier 1 catalog ordering).
    pub priority: Option<String>,

    /// If true, the skill is only activated by explicit `/name` invocation.
    pub trigger: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct StagingConfig {
    /// `git-worktree` | `tempdir`
    pub mode: String,
    /// Staging root directory.
    pub base: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VerificationGate {
    pub name: String,
    /// `schema` | `consistency` | `compile` | `test` | `lint` | `snapshot` | `roundtrip` | `lsp` | `custom`
    #[serde(rename = "type")]
    pub gate_type: String,
    /// Whether gate failure blocks staging.commit.
    #[serde(default = "default_blocking")]
    pub blocking: bool,
}

fn default_blocking() -> bool {
    true
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DivergenceSpec {
    /// Required divergence categories: `language-binding` / `paradigm-binding` /
    /// `library-substitution` / `version-compatibility`.
    #[serde(rename = "required-categories")]
    pub required_categories: Vec<String>,
    /// If true, the orchestrator blocks `staging.commit` when a required
    /// category is missing from the LLM's `divergences` array.
    #[serde(rename = "block-on-undisclosed")]
    pub block_on_undisclosed: bool,
}

/// Where a skill was discovered. Used to compute scope priority.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum SkillScope {
    /// Compiled into the binary at `$CARGO_MANIFEST_DIR/skills/`.
    Builtin,
    /// `<project>/.cleanroom/skills/<name>/SKILL.md` (highest project priority).
    ProjectCleanroom,
    /// `<project>/.agents/skills/<name>/SKILL.md` (cross-client convention).
    ProjectAgents,
    /// `~/.cleanroom/skills/<name>/SKILL.md` (user install).
    UserCleanroom,
    /// `~/.agents/skills/<name>/SKILL.md` (cross-client convention).
    UserAgents,
}

impl SkillScope {
    /// Returns a numeric priority (higher wins on name collision).
    pub fn priority(self) -> u8 {
        match self {
            SkillScope::Builtin => 100,
            SkillScope::ProjectCleanroom => 80,
            SkillScope::ProjectAgents => 60,
            SkillScope::UserCleanroom => 40,
            SkillScope::UserAgents => 20,
        }
    }
}

/// A parsed skill before it is assigned an id and indexed.
#[derive(Debug, Clone)]
pub struct ParsedSkill {
    pub name: String,
    pub description: String,
    pub license: Option<String>,
    pub compatibility: Option<String>,
    pub metadata: HashMap<String, String>,
    pub allowed_tools: Vec<String>,
    pub x_cleanroom: XCleanroom,
    pub body: String,
    pub scope: SkillScope,
}

/// A fully-indexed skill document with a content-based unique id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDocument {
    /// `name + 12 chars of content hash` (stable across reloads).
    pub id: String,
    pub name: String,
    pub description: String,
    pub license: Option<String>,
    pub compatibility: Option<String>,
    pub tags: Vec<String>,
    /// Effective allowed-tools (merged from standard + x-cleanroom).
    pub allowed_tools: Vec<String>,
    pub denied_tools: Vec<String>,
    pub allowed_paths: Vec<String>,
    pub staging: Option<StagingConfig>,
    pub output_schema: Option<serde_json::Value>,
    pub gates: Vec<VerificationGate>,
    pub divergence_spec: Option<DivergenceSpec>,
    pub applies_to: Vec<String>,
    pub token_budget: u32,
    /// One of `"low"` / `"normal"` / `"high"`.
    pub priority: String,
    pub trigger: bool,
    pub body: String,
    pub path: PathBuf,
    pub scope: SkillScope,
    pub hash: String,
    pub last_modified: Option<i64>,
    pub sdef_shard_uri: Option<String>,
    /// Arbitrary user metadata from frontmatter.
    pub metadata: HashMap<String, String>,
}

impl SkillDocument {
    /// Build the Tier 2 instruction block, capped at `max_chars` characters.
    pub fn engineer_instruction(&self, max_chars: usize) -> String {
        let mut body = self.body.clone();
        if body.chars().count() > max_chars {
            body = body.chars().take(max_chars).collect();
            body.push_str("\n[... truncated]");
        }

        let mut parts = Vec::new();
        parts.push(format!("[skill:{}]", self.name));
        parts.push(format!("# {}\n{}", self.name, self.description));

        if !self.allowed_tools.is_empty() {
            parts.push(format!(
                "You have access to the following tools: {}.",
                self.allowed_tools.join(", ")
            ));
        }
        if !self.allowed_paths.is_empty() {
            parts.push(format!(
                "Allowed file paths (fs.* tools): {}",
                self.allowed_paths.join(", ")
            ));
        }

        parts.push(format!("## Instructions\n{}", body));
        parts.push("[/skill]".to_string());

        parts.join("\n\n")
    }

    /// Build a lightweight prompt block (used for Tier 2 preselected injection).
    pub fn engineer_prompt_block(&self, max_chars: usize) -> String {
        let mut body = self.body.clone();
        if body.chars().count() > max_chars {
            body = body.chars().take(max_chars).collect();
        }
        format!("[skill:{}]\n{}\n[/skill]", self.name, body)
    }

    /// Returns true if the skill is applicable to the given `task_type`.
    pub fn applies_to_task(&self, task_type: &str) -> bool {
        self.applies_to.is_empty() || self.applies_to.iter().any(|t| t == task_type)
    }
}

/// A lightweight summary of a skill, excluding the heavy body.
#[derive(Debug, Clone, Serialize)]
pub struct SkillSummary {
    pub id: String,
    pub name: String,
    pub description: String,
    pub scope: SkillScope,
    pub priority: String,
    pub token_budget: u32,
    pub allowed_tools: Vec<String>,
    pub allowed_paths: Vec<String>,
    pub applies_to: Vec<String>,
    pub tags: Vec<String>,
    pub path: PathBuf,
    pub hash: String,
}

impl From<&SkillDocument> for SkillSummary {
    fn from(value: &SkillDocument) -> Self {
        Self {
            id: value.id.clone(),
            name: value.name.clone(),
            description: value.description.clone(),
            scope: value.scope,
            priority: value.priority.clone(),
            token_budget: value.token_budget,
            allowed_tools: value.allowed_tools.clone(),
            allowed_paths: value.allowed_paths.clone(),
            applies_to: value.applies_to.clone(),
            tags: value.tags.clone(),
            path: value.path.clone(),
            hash: value.hash.clone(),
        }
    }
}

/// A collection of indexed skills with deterministic ordering.
#[derive(Debug, Clone, Default)]
pub struct SkillIndex {
    skills: Vec<SkillDocument>,
}

impl SkillIndex {
    pub fn new(skills: Vec<SkillDocument>) -> Self {
        Self { skills }
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn skills(&self) -> &[SkillDocument] {
        &self.skills
    }

    pub fn into_skills(self) -> Vec<SkillDocument> {
        self.skills
    }

    pub fn summaries(&self) -> Vec<SkillSummary> {
        self.skills.iter().map(SkillSummary::from).collect()
    }

    pub fn find_by_name(&self, name: &str) -> Option<&SkillDocument> {
        self.skills.iter().find(|s| s.name == name)
    }

    pub fn find_by_id(&self, id: &str) -> Option<&SkillDocument> {
        self.skills.iter().find(|s| s.id == id)
    }

    /// Push a new skill (used by index loaders). Caller is responsible for
    /// sorting afterwards.
    pub fn push(&mut self, skill: SkillDocument) {
        self.skills.push(skill);
    }

    /// Sort by (priority desc, name asc) for deterministic iteration.
    pub fn sort(&mut self) {
        self.skills.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| a.path.cmp(&b.path))
        });
    }
}

/// Filter and scoring policy used by [`select_skills`].
#[derive(Debug, Clone)]
pub struct SelectionPolicy {
    pub top_k: usize,
    pub min_score: f32,
    pub include_tags: Vec<String>,
    pub exclude_tags: Vec<String>,
    /// Optional hard task_type filter — only skills with empty `applies_to`
    /// or matching this task_type are eligible.
    pub task_type: Option<String>,
}

impl Default for SelectionPolicy {
    fn default() -> Self {
        Self {
            top_k: 1,
            min_score: 1.0,
            include_tags: Vec::new(),
            exclude_tags: Vec::new(),
            task_type: None,
        }
    }
}

/// A ranked result.
#[derive(Debug, Clone, Serialize)]
pub struct SkillMatch {
    pub score: f32,
    pub skill: SkillSummary,
}

/// Build a stable id from name + content hash. Matches the
/// `name + first 12 hash chars` convention from `adk-skill`.
pub fn make_skill_id(name: &str, hash: &str) -> String {
    let short_hash = hash.chars().take(12).collect::<String>();
    format!("{name}+{short_hash}")
}

/// Compute SHA-256 of `bytes` and return lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

/// Get last-modified time (Unix seconds) of a path, if available.
pub fn last_modified_unix(path: &Path) -> Option<i64> {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_skill_id_uses_12_hash_chars() {
        let id = make_skill_id("rust-analysis", "abcdef1234567890abcdef");
        assert_eq!(id, "rust-analysis+abcdef123456");
    }

    #[test]
    fn skill_scope_priority_ordering() {
        let mut scopes = vec![
            SkillScope::UserAgents,
            SkillScope::Builtin,
            SkillScope::ProjectAgents,
            SkillScope::ProjectCleanroom,
            SkillScope::UserCleanroom,
        ];
        scopes.sort_by_key(|s| std::cmp::Reverse(s.priority()));
        assert_eq!(scopes[0], SkillScope::Builtin);
        assert_eq!(scopes[4], SkillScope::UserAgents);
    }

    #[test]
    fn engineer_instruction_caps_body_length() {
        let doc = SkillDocument {
            id: "x".into(),
            name: "test".into(),
            description: "d".into(),
            license: None,
            compatibility: None,
            tags: vec![],
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
            body: "x".repeat(1000),
            path: PathBuf::from("/x"),
            scope: SkillScope::Builtin,
            hash: "h".into(),
            last_modified: None,
            sdef_shard_uri: None,
            metadata: HashMap::new(),
        };
        let s = doc.engineer_instruction(50);
        // The body part must be truncated
        assert!(s.contains("[... truncated]"));
        // And the wrapping markers must be present
        assert!(s.contains("[skill:test]"));
        assert!(s.contains("[/skill]"));
    }
}
