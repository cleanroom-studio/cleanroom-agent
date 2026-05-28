//! Orchestrator — coordinates task execution.

use std::path::PathBuf;
use std::sync::Arc;

use cleanroom_db::{Database, TaskRepository, TaskType};
use tracing::{info, instrument};

/// Orchestrator configuration.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Source repository path.
    pub repo_path: PathBuf,
    /// Output directory for S.DEF.
    pub output_path: PathBuf,
    /// Database path.
    pub db_path: PathBuf,
    /// Checkpoint interval in seconds.
    pub checkpoint_interval_secs: u64,
    /// Idle timeout for agents.
    pub agent_idle_timeout_secs: u64,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            repo_path: PathBuf::from("."),
            output_path: PathBuf::from("./output"),
            db_path: PathBuf::from("state.db"),
            checkpoint_interval_secs: 600, // 10 minutes
            agent_idle_timeout_secs: 300, // 5 minutes
        }
    }
}

/// Orchestrator — coordinates task execution.
pub struct Orchestrator {
    /// Configuration.
    config: OrchestratorConfig,
    /// Database connection.
    db: Arc<Database>,
}

impl Orchestrator {
    /// Create a new orchestrator.
    pub fn new(config: OrchestratorConfig) -> Result<Self, cleanroom_db::DbError> {
        let db = Database::open(&config.db_path)?;
        Ok(Self {
            config,
            db: Arc::new(db),
        })
    }

    /// Get database.
    pub fn db(&self) -> &Arc<Database> {
        &self.db
    }

    /// Get configuration.
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Create initial tasks for repository analysis.
    #[instrument(skip(self))]
    pub async fn create_initial_tasks(&self) -> Result<(), cleanroom_db::DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());

        // Create the main repo analyze task
        let task = cleanroom_db::Task {
            task_id: uuid::Uuid::new_v4().to_string(),
            task_type: TaskType::RepoAnalyze,
            status: cleanroom_db::TaskStatus::Pending,
            priority: 10,
            input_json: serde_json::json!({
                "repo_path": self.config.repo_path.to_string_lossy(),
            }).to_string(),
            output_json: None,
            error_message: None,
            assigned_to: None,
            progress: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 3,
            last_heartbeat: None,
            dependencies_json: "[]".to_string(),
            version: 1,
        };

        repo.create(&task)?;
        info!(task_id = %task.task_id, "Created initial task");
        Ok(())
    }

    /// Start the workflow.
    #[instrument(skip(self))]
    pub async fn start_workflow(&self) -> Result<(), cleanroom_db::DbError> {
        self.create_initial_tasks().await?;
        info!("Workflow started");
        Ok(())
    }
}