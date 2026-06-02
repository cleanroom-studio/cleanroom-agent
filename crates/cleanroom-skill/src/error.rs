//! Error types for the cleanroom-skill crate.
//!
//! All public APIs return [`SkillResult<T>`]. The variants are deliberately
//! structured so a coordinator can branch on them (e.g. surface
//! `DeniedBySkill` differently from `PathNotAllowed`).

use std::path::PathBuf;
use thiserror::Error;

/// Crate-level error type. Used by every module in `cleanroom-skill`.
#[derive(Debug, Error)]
pub enum SkillError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("yaml parse error: {0}")]
    Yaml(String),

    #[error("invalid frontmatter in {path}: {message}")]
    InvalidFrontmatter { path: PathBuf, message: String },

    #[error("missing required field `{field}` in {path}")]
    MissingField { path: PathBuf, field: String },

    #[error("invalid skill root: {0}")]
    InvalidSkillsRoot(PathBuf),

    #[error("tool `{tool}` denied by active skill `{skill}`")]
    DeniedBySkill { skill: String, tool: String },

    #[error("tool `{tool}` not in skill `{skill}` allowed-tools")]
    NotInAllowedTools { skill: String, tool: String },

    #[error("path `{path}` not in skill `{skill}` allowed-paths")]
    PathNotAllowed { skill: String, path: String },

    #[error("staging mode not configured for skill `{skill}`")]
    StagingNotConfigured { skill: String },

    #[error("no active skill — tool `{0}` requires an active skill")]
    NoActiveSkill(String),

    #[error("staging error: {0}")]
    Staging(#[from] crate::staging_bridge::StagingBridgeError),

    #[error("sdef bridge error: {0}")]
    Sdef(String),

    #[error("validation failed: {0}")]
    Validation(String),

    #[error("other: {0}")]
    Other(String),
}

impl From<serde_yaml::Error> for SkillError {
    fn from(e: serde_yaml::Error) -> Self {
        SkillError::Yaml(e.to_string())
    }
}

impl From<rusqlite::Error> for SkillError {
    fn from(e: rusqlite::Error) -> Self {
        SkillError::Other(format!("sqlite: {e}"))
    }
}

pub type SkillResult<T> = Result<T, SkillError>;
