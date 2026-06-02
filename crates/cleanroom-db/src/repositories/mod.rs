//! Database repositories.

pub mod task_repository;
pub mod shard_repository;
pub mod sdef_repository;
pub mod symbol_repository;
pub mod fingerprint_repository;
pub mod checkpoint_repository;
pub mod audit_repository;
pub mod message_repository;
pub mod type_cache_repository;
pub mod evaluation_repository;
pub mod llm_call_log_repository;

pub use task_repository::{Task, TaskRepository, TaskStatus, TaskType};
pub use shard_repository::{Shard, ShardRepository, ShardStatus};
pub use sdef_repository::{
    Contract, DataAttribute, DataModel, DesignDecisionRecord, FunctionSpec, SdefDocument,
    SdefRepository, UiDocument, UiScreen,
};
pub use symbol_repository::{ResolutionResult, SymbolEntry, SymbolRepository, SymbolType};
pub use fingerprint_repository::{Fingerprint, FingerprintRepository};
pub use checkpoint_repository::{Checkpoint, CheckpointRepository};
pub use audit_repository::{AuditEntry, AuditRepository};
pub use message_repository::{AgentMessage, AgentMessageRepository, MessageType};
pub use type_cache_repository::{TypeCacheEntry, TypeCacheRepository};
pub use evaluation_repository::{
    EvaluationRecord, EvaluationRepository, EvaluationSummary, EvaluationTrend,
};
pub use llm_call_log_repository::{
    LlmCallLog, LlmCallLogRepository, STATUS_ABORTED, STATUS_COMPLETED, STATUS_FAILED,
    STATUS_MAX_ITER, STATUS_REFUSED,
};