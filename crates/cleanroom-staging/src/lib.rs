//! `cleanroom-staging` — Staging 隔离工作区抽象.
//!
//! LLM 通过 `staging.*` 工具把对源代码的修改写到隔离工作区(per-task),由
//! orchestrator 跑 verification gates 通过后再 `commit` 到目标源代码树。
//!
//! # Architecture
//!
//! - [`error::StagingError`] / [`error::StagingResult`] — 错误类型
//! - [`manifest::StagingEntry`] — 一次写操作的记录(可序列化到 SQLite)
//! - [`workspace::StagingWorkspace`] — backend trait
//! - [`workspace::StagingMode`] — `tempdir` / `git-worktree`
//! - [`workspace::CommitReport`] — commit 结果
//! - [`tempdir::TempDirBackend`] — MVP backend,基于 `tempfile::tempdir()`
//!
//! # See also
//!
//! - [`docs/21-skills-system.md`](file://./docs/21-skills-system.md) §6.5 (verification gates)
//! - [`docs/20-ui-generation.md`](file://./docs/20-ui-generation.md) §4 (staging 工具设计)
//! - `PLAN2.md` §5

pub mod error;
pub mod manifest;
pub mod tempdir;
pub mod workspace;

pub use error::{StagingError, StagingResult};
pub use manifest::{sha256_hex, StagingEntry, StagingOp};
pub use tempdir::TempDirBackend;
pub use workspace::{CommitReport, StagingMode, StagingWorkspace};
