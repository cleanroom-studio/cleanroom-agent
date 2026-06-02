//! `cleanroom-skill` — Skills / Replication Protocol engine.
//!
//! Implements the agentskills.io open format plus a cleanroom extension block
//! (`x-cleanroom.*`) that carries the Replication Protocol metadata
//! (Context Contract, Action Surface, Output Schema, Verification Gates,
//! Divergence Spec).
//!
//! # Quick start
//!
//! ```rust,ignore
//! use cleanroom_skill::{load_skill_index, select_skills, SelectionPolicy};
//!
//! let index = load_skill_index("/path/to/project").unwrap();
//! let policy = SelectionPolicy { top_k: 1, min_score: 1.0, ..Default::default() };
//! let matches = select_skills(&index, "rust trait analysis", &policy);
//! for m in matches {
//!     println!("{}: {:.2}", m.skill.name, m.score);
//! }
//! ```
//!
//! # Architecture
//!
//! - [`model`] — Core types (`SkillFrontmatter`, `SkillDocument`, `SkillIndex`,
//!   `SelectionPolicy`, `SkillMatch`).
//! - [`parser`] — `SKILL.md` frontmatter + body parser (lenient + strict modes).
//! - [`discovery`] — Recursive filesystem scan for skill directories.
//! - [`index`] — Build a [`SkillIndex`] from discovered files.
//! - [`select`] — Lexical scoring (deterministic, single-threaded).
//! - [`injector`] — Tier 1 / Tier 2 prompt block construction.
//! - [`coordinator`] — Tool authorization (Replication Protocol Action Surface).
//! - [`validation`] — Frontmatter / name / path validation.
//! - [`db_cache`] — SQLite-backed skill cache.
//! - [`sdef_bridge`] — `<->` S.DEF `Entity` round-trip.
//! - [`error`] — `SkillError` and `SkillResult`.
//!
//! # See also
//!
//! - [`docs/21-skills-system.md`](file://./docs/21-skills-system.md) — design doc.
//! - [`docs/20-ui-generation.md`](file://./docs/20-ui-generation.md) — UI Replication
//!   Protocol that the Skills engine generalizes.
//! - `PLAN2.md` — implementation plan.

pub mod coordinator;
pub mod db_cache;
pub mod discovery;
pub mod error;
pub mod index;
pub mod injector;
pub mod model;
pub mod parser;
pub mod sdef_bridge;
pub mod select;
pub mod staging_bridge;
pub mod validation;

pub use coordinator::{check_tool_authorization, ContextCoordinator, CoordinatorConfig};
pub use discovery::{builtin_skill_dir, discover_skill_files, discover_skill_files_with_extras};
pub use error::{SkillError, SkillResult};
pub use index::{
    load_skill_index, load_skill_index_strict, load_skill_index_with_extras, refresh_skill_index,
};
pub use injector::{
    build_skill_catalog_block, engineer_instruction, select_skill_prompt_block,
};
pub use model::{
    make_skill_id, last_modified_unix, sha256_hex, ParsedSkill, SelectionPolicy, SkillDocument,
    SkillFrontmatter, SkillIndex, SkillMatch, SkillScope, SkillSummary, StagingConfig,
    VerificationGate, XCleanroom, DivergenceSpec,
};
pub use parser::{parse_instruction_markdown, parse_skill_markdown};
pub use select::select_skills;
pub use sdef_bridge::{sdef_entity_to_skill, skill_to_sdef_entity};
pub use validation::{validate_skill_dir, ValidationIssue, ValidationReport};
