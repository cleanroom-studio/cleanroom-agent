//! cleanroom-agent — Agent core logic for Cleanroom.

#![warn(missing_docs)]

pub mod orchestrator;
pub mod producer;
pub mod consumer;
pub mod naming;
pub mod consistency;

pub use orchestrator::{Orchestrator, OrchestratorConfig};
pub use producer::{ProducerAgent, ProducerConfig};
pub use consumer::{ConsumerAgent, ConsumerConfig, CompatibilityMode, Fidelity};