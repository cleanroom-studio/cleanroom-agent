//! Skill management MCP tool parameters.
//!
//! These tools allow LLM agents to discover, activate, refresh, and validate
//! Skills (see `docs/21-skills-system.md`).
//!
//! Tools:
//! - `skill_list` — list available skills (Tier 1 catalog)
//! - `skill_activate` — load a skill's full body (Tier 2)
//! - `skill_refresh` — re-scan the filesystem for skill changes
//! - `skill_validate` — validate a single SKILL.md against the spec

use rmcp::schemars;
use serde::Deserialize;

/// Parameters for `skill_list`.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct SkillListParams {
    /// Filter by scope: `builtin` / `project-cleanroom` / `project-agents` /
    /// `user-cleanroom` / `user-agents`. None = all scopes.
    #[serde(default)]
    pub scope: Option<String>,

    /// If true, include the full body (heavy). Default: false.
    #[serde(default)]
    pub include_body: bool,

    /// Filter by `applies-to` task type.
    #[serde(default)]
    pub task_type: Option<String>,
}

/// Parameters for `skill_activate`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SkillActivateParams {
    /// The skill's name (must match the directory name).
    pub name: String,

    /// Optional token budget override (default: skill's `token-budget` field).
    #[serde(default)]
    pub token_budget: Option<u32>,
}

/// Parameters for `skill_refresh`.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct SkillRefreshParams {
    /// Optional root path to re-scan. Default: project root from env.
    #[serde(default)]
    pub path: Option<String>,
}

/// Parameters for `skill_validate`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SkillValidateParams {
    /// Absolute path to the SKILL.md to validate.
    pub path: String,
}
