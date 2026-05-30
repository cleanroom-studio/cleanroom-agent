//! Producer Agent — analyzes code repositories and produces S.DEF documents.
//!
//! The Producer Agent is responsible for the "produce" phase of the Cleanroom
//! agent pipeline. It takes a source code repository and generates a complete
//! S.DEF (Software Definition Exchange Format) document describing the codebase.
//!
//! # Pipeline
//!
//! The producer uses the full analysis pipeline via [`run_analysis_pipeline`]:
//! 1. Repository scanning via [`scan_repository`](crate::repo_scanner::scan_repository)
//! 2. Module partitioning via [`partition_files`](crate::module_partitioner::partition_files)
//! 3. Dependency graph construction via [`DependencyGraph`]
//! 4. IR to S.DEF mapping via [`SdefMapper`]
//! 5. Persistence to database
//!
//! # Task Processing
//!
//! The agent claims and processes tasks from the database task queue.
//! Each task may represent a different phase of repository analysis.

use std::path::Path;
use std::sync::Arc;

use cleanroom_db::{Database, DbError, Task, TaskRepository, TaskType};
use tracing::{info, instrument};

use crate::producer_pipeline::{run_analysis_pipeline, run_analysis_pipeline_with_lsp};

/// Producer configuration.
///
/// Contains settings for the producer agent's behavior during code analysis.
#[derive(Debug, Clone)]
pub struct ProducerConfig {
    /// List of programming languages the producer should recognize
    pub languages: Vec<String>,
    /// Whether to enable LSP (Language Server Protocol) for enhanced analysis
    pub lsp_enabled: bool,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            languages: vec![
                "typescript".to_string(),
                "javascript".to_string(),
                "python".to_string(),
                "rust".to_string(),
                "c".to_string(),
                "cpp".to_string(),
                "go".to_string(),
                "java".to_string(),
            ],
            lsp_enabled: false, // Disabled by default; enable with --lsp flag
        }
    }
}

/// Producer Agent — analyzes code repositories and produces S.DEF documents.
///
/// The Producer Agent claims tasks from the database queue and executes
/// repository analysis via the producer pipeline. Each successful analysis
/// results in a complete S.DEF document stored in the database.
///
/// # Task Types
///
/// The producer handles the following task types:
/// - [`TaskType::RepoAnalyze`]: Full repository analysis
#[allow(dead_code)]
pub struct ProducerAgent {
    /// Producer configuration settings
    pub(crate) config: ProducerConfig,
    /// Database connection for task persistence
    db: Arc<Database>,
    /// Unique agent identifier for task claiming
    agent_id: String,
}

impl ProducerAgent {
    /// Create a new producer agent.
    pub fn new(config: ProducerConfig, db: Arc<Database>) -> Self {
        let agent_id = format!("producer-{}", uuid::Uuid::new_v4());
        Self {
            config,
            db,
            agent_id,
        }
    }

    /// Get agent ID.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Claim and process a task.
    #[instrument(skip(self))]
    pub async fn process_next_task(&self) -> Result<Option<Task>, DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());

        if let Some(task) = repo.claim(&self.agent_id)? {
            info!(task_id = %task.task_id, task_type = ?task.task_type, "Processing task");

            match task.task_type {
                TaskType::RepoAnalyze => self.analyze_repo(&task).await?,
                _ => {
                    repo.complete(&task.task_id, "{}")?;
                }
            }

            return Ok(Some(task));
        }

        Ok(None)
    }

    /// Full repository analysis using the integrated pipeline.
    async fn analyze_repo(&self, task: &Task) -> Result<(), DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());
        
        // Parse task input
        let input: serde_json::Value = serde_json::from_str(&task.input_json)
            .unwrap_or_else(|_| serde_json::json!({}));
        
        let repo_path = input.get("repo_path")
            .and_then(|v| v.as_str())
            .map(Path::new)
            .unwrap_or_else(|| Path::new("."));
        
        let project_name = input.get("project_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unnamed");
        
        info!(path = %repo_path.display(), project = %project_name, "Starting pipeline");

        // Update progress to 10%
        repo.update_progress(&task.task_id, 0.1)?;

        // Run the full pipeline (with LSP if enabled)
        let result = if self.config.lsp_enabled {
            run_analysis_pipeline_with_lsp(
                self.db.clone(),
                repo_path,
                project_name,
                "0.1.0",
                None,
                true,
            ).await?
        } else {
            run_analysis_pipeline(
                self.db.clone(),
                repo_path,
                project_name,
                "0.1.0",
                None,
            ).await?
        };

        // Update progress to 90%
        repo.update_progress(&task.task_id, 0.9)?;

        // Create output summary
        let output = crate::producer_pipeline::result_to_json(&result);

        // Complete task
        repo.complete(&task.task_id, &serde_json::to_string(&output).unwrap_or_default())?;

        info!(
            files = result.file_count,
            modules = result.module_count,
            data_models = result.sdef.data_models.as_ref().map(|v| v.len()).unwrap_or(0),
            "Repository analysis complete"
        );

        Ok(())
    }

    /// Send heartbeat for current task.
    pub async fn heartbeat(&self, task_id: &str) -> Result<(), DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());
        repo.heartbeat(task_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_process_next_task_no_tasks() {
        let db = Arc::new(Database::in_memory().unwrap());
        let agent = ProducerAgent::new(ProducerConfig::default(), db);
        let result = agent.process_next_task().await.unwrap();
        assert!(result.is_none(), "No tasks should be available");
    }
}