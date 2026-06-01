use crate::agent::{MetaResultError, MetaContextError};

use crate::agent::error::{AgentBuildError, RunnableAgentError};
#[cfg(not(target_arch = "wasm32"))]
use crate::{environment::EnvironmentError, runtime::RuntimeError};
use cleanroom_meta_llm::error::MetaError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[cfg(not(target_arch = "wasm32"))]
    #[error(transparent)]
    EnvironmentError(#[from] EnvironmentError),
    #[cfg(not(target_arch = "wasm32"))]
    #[error(transparent)]
    RuntimeError(#[from] RuntimeError),
    #[error(transparent)]
    AgentBuildError(#[from] AgentBuildError),
    #[error(transparent)]
    RunnableAgentError(#[from] RunnableAgentError),
    #[error(transparent)]
    MetaError(#[from] MetaError),
    #[error(transparent)]
    MetaResultError(#[from] MetaResultError),
    #[error(transparent)]
    MetaContextError(#[from] MetaContextError),
    #[error("Custom Error: {0}")]
    CustomError(String),
}
