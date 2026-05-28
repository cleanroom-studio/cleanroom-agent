//! cleanroom-db — SQLite database layer for Cleanroom Agent.

pub mod database;
pub mod error;
pub mod export_import;
pub mod migrations;
pub mod repositories;

pub use database::Database;
pub use error::{DbError, DbResult};

// Re-export common types
pub use repositories::{Task, TaskRepository, TaskStatus, TaskType};
pub use repositories::{Shard, ShardRepository, ShardStatus};
pub use repositories::{SymbolEntry, SymbolRepository, SymbolType};
pub use repositories::{Fingerprint, FingerprintRepository};