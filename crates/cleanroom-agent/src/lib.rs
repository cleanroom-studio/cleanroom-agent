//! cleanroom-agent — Agent core logic for Cleanroom.

#![warn(missing_docs)]

pub mod orchestrator;
pub mod producer;
pub mod consumer;
pub mod naming;
pub mod consistency;
pub mod repo_scanner;
pub mod module_partitioner;
pub mod dependency_graph;
pub mod ir_to_sdef;
pub mod producer_pipeline;
pub mod two_phase_commit;

pub use orchestrator::{Orchestrator, OrchestratorConfig};
pub use producer::{ProducerAgent, ProducerConfig};
pub use consumer::{ConsumerAgent, ConsumerConfig, CompatibilityMode, Fidelity};
pub use repo_scanner::{scan_repository, group_by_language, ScanConfig, SourceFile};
pub use module_partitioner::{partition_files, PartitionConfig, Module, ModuleType};
pub use dependency_graph::{DependencyGraph, DepNode, DepNodeType, DepEdge, DepEdgeKind};
pub use ir_to_sdef::{SdefMapper, MapperConfig, IrEntity, IrAttribute, IrMethod, IrParam};
pub use producer_pipeline::{run_analysis_pipeline, PipelineResult, DepInfo};