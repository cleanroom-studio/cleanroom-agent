//! Task management MCP tool parameters and handlers.

use rmcp::schemars;
use serde::{Deserialize, Serialize};

/// Task creation parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateTaskParams {
    pub task_type: String,
    pub input: serde_json::Value,
    #[serde(default = "default_priority")]
    pub priority: i32,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

fn default_priority() -> i32 { 5 }

/// Task claim parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ClaimTaskParams {
    pub agent_id: String,
}

/// Task progress update parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct UpdateProgressParams {
    pub task_id: String,
    pub progress: f64,
    pub message: Option<String>,
}

/// Task completion parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CompleteTaskParams {
    pub task_id: String,
    pub output: serde_json::Value,
}

/// Task failure parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FailTaskParams {
    pub task_id: String,
    pub error_message: String,
}

/// Heartbeat parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HeartbeatParams {
    pub task_id: String,
    pub agent_id: String,
}

/// Task listing parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListTasksParams {
    pub status: Option<String>,
    pub task_type: Option<String>,
    pub assigned_to: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize { 20 }

/// Result of a task listing or get operation.
#[derive(Debug, Serialize)]
pub struct TaskResult {
    pub task_id: String,
    pub task_type: String,
    pub status: String,
    pub priority: i32,
    pub input: serde_json::Value,
    pub output: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub assigned_to: Option<String>,
    pub progress: f64,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub retry_count: i32,
    pub max_retries: i32,
    pub last_heartbeat: Option<String>,
    pub dependencies: Vec<String>,
    pub version: i32,
}
