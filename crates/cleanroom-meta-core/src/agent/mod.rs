// Runtime-independent modules (available on all platforms)
mod config;
pub mod error;
pub mod memory;
mod output;
mod protocol;
pub mod task;

pub mod prebuilt;

// Exports for all platforms
pub use config::MetaConfig;
pub use error::MetaResultError;
pub use output::MetaOutputT;
pub use protocol::MetaProtocol;
mod base;
mod builder;
mod context;
mod executor;
// mod runnable;
mod actor;
pub(crate) mod constants;
mod direct;
mod hooks;
mod state;

pub use actor::MetaActorAgent;
#[cfg(not(target_arch = "wasm32"))]
pub use actor::MetaActorAgentHandle;
pub use base::{MetaDeriveT, MetaBaseAgent};
pub use builder::MetaAgentBuilder;
pub use context::{Context, MetaContextError};
pub use direct::{MetaDirectAgent, MetaDirectAgentHandle};
pub use executor::{
    MetaExecutor, ExecutorConfig, TurnResult, event_helper::EventHelper,
    memory_helper::MemoryHelper, tool_processor::ToolProcessor, turn_engine,
};
pub use hooks::{MetaHooks, MetaHookOutcome};
