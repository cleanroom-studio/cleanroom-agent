//! cleanroom-db — SQLite database layer for Cleanroom Agent.

pub mod database;
pub mod embedded_schema;
pub mod error;
pub mod export_import;
pub mod migrations;
pub mod repositories;

pub use database::{Database, BackupConfig, verify_database_integrity, recover_from_backup};
pub use error::{DbError, DbResult};

// Re-export common types
pub use repositories::{Task, TaskRepository, TaskStatus, TaskType};
pub use repositories::{Shard, ShardRepository, ShardStatus};
pub use repositories::{SymbolEntry, SymbolRepository, SymbolType};
pub use repositories::{Fingerprint, FingerprintRepository};
pub use repositories::SdefRepository;
pub use repositories::{AgentMessage, AgentMessageRepository, MessageType};
pub use repositories::{TypeCacheEntry, TypeCacheRepository};
pub use repositories::{
    EvaluationRecord, EvaluationRepository, EvaluationSummary, EvaluationTrend,
};
pub use repositories::{
    LlmCallLog, LlmCallLogRepository, STATUS_ABORTED, STATUS_COMPLETED, STATUS_FAILED,
    STATUS_MAX_ITER, STATUS_REFUSED,
};