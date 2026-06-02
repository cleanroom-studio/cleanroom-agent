//! cleanroom-agent — Agent core logic for Cleanroom.
//!
//! This crate provides the core agent system for the Cleanroom S.DEF (Software Definition Exchange Format)
//! intelligent agent. It handles the bidirectional flow between source code repositories and S.DEF
//! documents through a produce/consume pipeline.
//!
//! ## Architecture
//!
//! The agent system consists of two main pipelines:
//!
//! - **Producer Pipeline**: Analyzes code repositories and generates S.DEF documents
//!   - Repository scanning and file discovery
//!   - Module partitioning and dependency analysis
//!   - Tree-sitter based code parsing
//!   - IR (Intermediate Representation) to S.DEF mapping
//!   - Persistence to SQLite database
//!
//! - **Consumer Pipeline**: Generates code from S.DEF documents
//!   - S.DEF import and parsing
//!   - Language-specific code generation
//!   - Multi-language support (Rust, TypeScript, Python, C)
//!   - Verification and testing
//!
//! ## Key Components
//!
//! - [`CleanroomAgent`]: Top-level agent entry point with produce/consume/resume modes
//! - [`ProducerAgent`]: Analyzes code repositories and produces S.DEF
//! - [`ConsumerAgent`]: Generates code from S.DEF documents
//! - [`Orchestrator`]: Coordinates task execution and workflow management
//! - [`SdefMapper`]: Maps code analysis results to S.DEF entities
//! - [`DependencyGraph`]: Analyzes dependencies between entities
//! - [`ConsistencyService`]: Ensures S.DEF, DB, and code are in sync
//! - [`CompletenessValidator`]: Validates S.DEF analysis quality
//!
//! ## Usage
//!
//! ```no_run
//! use cleanroom_agent::{CleanroomAgent, AgentConfig, RunMode};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = AgentConfig::producer(PathBuf::from("./my-repo"));
//! let agent = CleanroomAgent::new(config)?;
//!
//! agent.run(RunMode::Produce {
//!     repo_path: PathBuf::from("./my-repo"),
//!     output_path: PathBuf::from("./output"),
//!     project_name: "my-project".to_string(),
//! }).await?;
//! # Ok(())
//! # }
//! ```

#![allow(missing_docs)]

pub mod agent;
pub mod orchestrator;
pub mod producer;
pub mod consumer;
pub mod llm_loop;
pub mod naming;
pub mod name_resolution;
pub mod consistency;
pub mod completeness;
pub mod compat_resolver;
pub mod incremental_analysis;
pub mod migration_gen;
pub mod version_upgrade;
pub mod absorb_human_changes;
pub mod consistency_checker;
pub mod code_merger;
pub mod repo_scanner;
pub mod module_partitioner;
pub mod dependency_graph;
pub mod ir_to_sdef;
pub mod producer_pipeline;
pub mod two_phase_commit;
pub mod scheduler;
pub mod test_extractor;
pub mod compat_detector;
pub mod design_decisions;
pub mod tree_sitter_parser;
pub mod lsp_analysis;

// Runtime control: pause/resume (docs/15 §10)
pub mod workflow_signal;

// LLM-driven producer: structured hints for the LLM (Phase 0.2)
pub mod auxiliary_hints;
pub use auxiliary_hints::{FileHints, HintLine, compute_hints, compute_hints_for_file};

// LLM-driven producer: S.DEF context loader (Phase 0.3)
pub mod sdef_context;
pub use sdef_context::{
    load_entity_with_attributes, load_function_and_dependents, load_module_subtree,
    load_shard_for_file, load_shard_for_task, DEFAULT_BUDGET_TOKENS,
};

// Interactive CLI modes (docs/15 §2-3)
pub mod interaction;

// Progress visualization (docs/15 §7)
pub mod progress_visualizer;

// Resilience: retry, recovery, degradation (docs/16)
pub mod retry;
pub mod recovery;
pub mod degradation;

// Multi-agent collaboration (docs/13)
pub mod collaboration;
pub mod reviewer;

// Evaluation & quality control (docs/14)
pub mod evaluation;

pub use agent::{CleanroomAgent, AgentConfig, RunMode};

// LLM agent loop (Phase 0.1) — wraps `cleanroom_meta_llm::chat::MetaProvider`.
// Phase 0.5 switched the basic-agent path to `cleanroom_meta_core::agent::MetaAgentBuilder`
// + `MetaBasicAgent` for tool-calling support; the public API is kept stable.
pub use llm_loop::{
    run_loop, run_loop_via_basic_agent, DefaultLlmAgent, LoopAgentOutput, LoopConfig,
    LoopContext, LoopError, LoopOutcome, LoopStats, UsageCapturingLlm, UsageCell,
};
pub mod mcp_tool_bridge;
pub use mcp_tool_bridge::{McpToolBridge, McpToolSpec, mcp_tool_catalog};
pub use naming::{DeterministicNames, Language, NameStyle, NamespaceMode};
pub use name_resolution::{NameResolutionService, ResolvedName};
pub use orchestrator::{Orchestrator, OrchestratorConfig, pid_file_path, port_file_path, read_port_file, write_port_file};
pub use producer::{ProducerAgent, ProducerConfig};
pub use consumer::{ConsumerAgent, ConsumerConfig, CompatibilityMode, Fidelity, llm_regenerate_file};
pub use repo_scanner::{scan_repository, group_by_language, ScanConfig, SourceFile};
pub use module_partitioner::{partition_files, PartitionConfig, Module, ModuleType};
pub use dependency_graph::{DependencyGraph, DepNode, DepNodeType, DepEdge, DepEdgeKind};
pub use ir_to_sdef::{SdefMapper, MapperConfig, IrEntity, IrAttribute, IrMethod, IrParam};
pub use producer_pipeline::{run_analysis_pipeline, PipelineResult, DepInfo};
pub use compat_resolver::{CompatibilityResolver, CompatibilityMode as ResolverMode, InclusionFilter};
pub use completeness::{CompletenessValidator, CompletenessReport, VerificationResult, CoverageScore, format_report};
pub use incremental_analysis::{IncrementalAnalyzer, IncrementalDiff};
pub use migration_gen::{MigrationGenerator, MigrationCode};
pub use version_upgrade::{VersionUpgradeAnalyzer, VersionUpgradeReport, BreakingChange, ChangeType, SuggestedMigration};
pub use absorb_human_changes::{HumanChangeAbsorber, AbsorbResult, HumanChange, ChangeType as AbsorbChangeType};
pub use consistency_checker::{ConsistencyChecker, ConsistencyCheckerConfig};
pub use consistency::{ConsistencyService, CheckLevel, FixStrategy, Inconsistency};
pub use code_merger::{CodeMerger, MergeConfig, MergeResult, CodeFragment, MergeConflict};
pub use scheduler::{Scheduler, TaskPlan, ProgressSummary};
pub use test_extractor::{extract_tests, build_test_contract, persist_test_contract, ExtractionResult};
pub use compat_detector::{CompatDetector, DetectionResult, CompatPattern, CompatCategory, build_compat_module};
pub use design_decisions::{infer_decisions, persist_decisions, InferenceResult};

// Multi-agent collaboration re-exports
pub use collaboration::messages::{MessageSender, MessagePoller};
pub use collaboration::conflict_detector::{ConflictDetector, Conflict, Resolution};
pub use collaboration::health_monitor::HealthMonitor;
pub use reviewer::{ReviewerAgent, ReviewerConfig, ReviewReport, reviewer_loop};
pub use lsp_analysis::{
    analyze_file_with_lsp_fallback, lookup_cached_type, has_cached_types,
    EnhancedFileAnalysis, EnrichedSymbol, AnalysisSource, LspAnalysisOptions,
};
pub use evaluation::{
    BenchmarkProject, BenchmarkSuite, ExpectedStats,
    EvaluationRunner, EvalConfig, ContinuousEval,
    EvaluationReport, ProjectEvalResult, EvaluationSummaryReport,
    AnalysisQualityReport, CoverageMetrics, AccuracyMetrics, EfficiencyMetrics,
    GenerationQualityReport, RoundtripFidelity, CodeQualityMetrics,
    OperationalMetrics,
};
pub use workflow_signal::{WorkflowSignal, GLOBAL_SIGNAL};
pub use interaction::{InteractionMode, InteractiveContext, ReviewItem, UserDecision, present_for_review, prompt_user};
pub use progress_visualizer::ProgressVisualizer;
pub use retry::{RetryConfig, retry_with_backoff, retry_sync_with_backoff};
pub use recovery::{RecoveryReport, recover_on_startup};
pub use degradation::{DegradationMode, Operation};