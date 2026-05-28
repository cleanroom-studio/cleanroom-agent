//! Producer Agent — analyzes code repositories.

use std::path::Path;
use std::sync::Arc;

use cleanroom_db::{Database, DbError, Task, TaskRepository, TaskType};
use tracing::{info, instrument};

use crate::producer_pipeline::run_analysis_pipeline;

/// Producer configuration.
#[derive(Debug, Clone)]
pub struct ProducerConfig {
    /// Supported languages.
    pub languages: Vec<String>,
    /// LSP server configurations.
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
                "go".to_string(),
            ],
            lsp_enabled: true,
        }
    }
}

/// Producer Agent — analyzes code repositories.
pub struct ProducerAgent {
    /// Configuration.
    config: ProducerConfig,
    /// Database connection.
    db: Arc<Database>,
    /// Agent ID.
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

        // Run the full pipeline
        let result = run_analysis_pipeline(
            self.db.clone(),
            repo_path,
            project_name,
            "0.1.0",
            None,
        ).await?;

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