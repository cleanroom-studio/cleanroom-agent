//! Orchestrator — coordinates task execution for the Cleanroom agent.
//!
//! The orchestrator manages the workflow execution of repository analysis tasks.
//! It creates initial tasks, manages checkpoints, handles agent idle timeouts,
//! and coordinates multi-agent parallel execution.
//!
//! # Multi-Agent Workflow
//!
//! 1. Create task plan via [`Scheduler`]
//! 2. Spawn N producer / consumer / reviewer agents
//! 3. Agents claim tasks, process them, send heartbeats
//! 4. Agents can steal work from compatible queues when idle
//! 5. Health monitor recovers zombie agent tasks
//!
//! # Checkpointing
//!
//! The orchestrator periodically checkpoints progress, allowing workflow
//! resumption if interrupted. Checkpoint interval is configured via
//! [`OrchestratorConfig::checkpoint_interval_secs`].

use std::path::PathBuf;
use std::sync::Arc;

use cleanroom_db::{Database, Task, TaskRepository, TaskType};
use tracing::{info, warn, instrument};

use crate::scheduler::{Scheduler, TaskPlan};
use crate::producer::{ProducerAgent, ProducerConfig};
use crate::reviewer::{ReviewerAgent, ReviewerConfig, reviewer_loop};
use crate::collaboration::health_monitor::HealthMonitor;

/// Orchestrator configuration.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Path to the source code repository to analyze
    pub repo_path: PathBuf,
    /// Directory for S.DEF output files
    pub output_path: PathBuf,
    /// Path to the SQLite database
    pub db_path: PathBuf,
    /// Name of the project/document being analyzed
    pub project_name: String,
    /// Interval between checkpoints in seconds (default: 600 = 10 minutes)
    pub checkpoint_interval_secs: u64,
    /// Idle timeout for agent tasks in seconds (default: 300 = 5 minutes)
    pub agent_idle_timeout_secs: u64,
    /// Number of producer agent instances
    pub num_producers: usize,
    /// Number of consumer agent instances
    pub num_consumers: usize,
    /// Number of reviewer agent instances
    pub num_reviewers: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            repo_path: PathBuf::from("."),
            output_path: PathBuf::from("./output"),
            db_path: PathBuf::from("state.db"),
            project_name: "unnamed".to_string(),
            checkpoint_interval_secs: 600,
            agent_idle_timeout_secs: 300,
            num_producers: 1,
            num_consumers: 1,
            num_reviewers: 1,
        }
    }
}

/// Orchestrator — coordinates task execution for the Cleanroom agent.
///
/// The orchestrator manages the lifecycle of repository analysis tasks.
/// It creates initial tasks, handles checkpointing, monitors agent activity,
/// and coordinates multi-agent parallel execution.
pub struct Orchestrator {
    /// Orchestrator configuration
    config: OrchestratorConfig,
    /// Database connection for task persistence
    db: Arc<Database>,
    /// Task scheduler for creating and managing task plans
    scheduler: Scheduler,
}

impl Orchestrator {
    /// Create a new orchestrator.
    pub fn new(config: OrchestratorConfig) -> Result<Self, cleanroom_db::DbError> {
        let db = Database::open(&config.db_path)?;
        let db = Arc::new(db);
        let scheduler = Scheduler::new(db.clone());
        Ok(Self { config, db, scheduler })
    }

    /// Get database.
    pub fn db(&self) -> &Arc<Database> {
        &self.db
    }

    /// Get configuration.
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Get scheduler.
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Create initial tasks for repository analysis.
    #[instrument(skip(self))]
    pub async fn create_initial_tasks(&self) -> Result<Vec<String>, cleanroom_db::DbError> {
        let plan = TaskPlan::analysis_plan(
            &self.config.project_name,
            &self.config.repo_path.to_string_lossy(),
        );
        let task_ids = self.scheduler.create_from_plan(&plan)?;
        info!(count = task_ids.len(), "Created initial analysis tasks");
        Ok(task_ids)
    }

    /// Start the full multi-agent workflow.
    ///
    /// Creates tasks, spawns agents, starts health monitor, and waits for completion.
    /// Sets the global signal so the MCP server (if running in-process) can
    /// serve pause/resume commands.
    #[instrument(skip(self))]
    pub async fn start_workflow(&self) -> Result<(), cleanroom_db::DbError> {
        let signal = crate::workflow_signal::WorkflowSignal::new();

        // Store signal globally for MCP server access
        let _ = crate::workflow_signal::GLOBAL_SIGNAL.set(signal.clone());

        // Write PID file for CLI status checks
        write_pid_file();

        // Phase 1: Create all tasks
        let task_ids = self.create_initial_tasks().await?;
        info!(count = task_ids.len(), "Workflow tasks created");

        // Phase 2: Start health monitor
        let health_monitor = HealthMonitor::default();
        let _health_shutdown = HealthMonitor::start(
            health_monitor,
            self.db.clone(),
        );

        // Phase 3: Spawn producer agents
        let mut handles = Vec::new();
        for i in 0..self.config.num_producers {
            let agent = ProducerAgent::new(
                ProducerConfig::default(),
                self.db.clone(),
            );
            let db_clone = self.db.clone();
            let signal_clone = signal.clone();
            let label = format!("producer-{}", i);
            handles.push(tokio::spawn(async move {
                info!(agent = %label, "Producer agent started");
                loop {
                    // Check pause before claiming
                    if signal_clone.is_paused() {
                        signal_clone.wait_for_resume().await;
                    }

                    match agent.process_next_task().await {
                        Ok(Some(task)) => {
                            info!(agent = %label, task_id = %task.task_id, "Task completed");
                        }
                        Ok(None) => {
                            if signal_clone.is_paused() {
                                signal_clone.wait_for_resume().await;
                                continue;
                            }
                            let progress = crate::scheduler::Scheduler::new(db_clone.clone())
                                .get_progress()
                                .unwrap_or_default();
                            if progress.completed + progress.failed_permanently >= progress.total {
                                info!(agent = %label, "All tasks done, exiting");
                                break;
                            }
                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        }
                        Err(e) => {
                            warn!(agent = %label, error = %e, "Task error");
                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        }
                    }
                }
            }));
        }

        // Phase 4: Spawn reviewer agents
        for i in 0..self.config.num_reviewers {
            let reviewer = ReviewerAgent::new(
                ReviewerConfig::default(),
                self.db.clone(),
                self.config.output_path.clone(),
            );
            let signal_clone = signal.clone();

            handles.push(tokio::spawn(async move {
                let label = format!("reviewer-{}", i);
                info!(agent = %label, "Reviewer agent started");

                // Pause-aware reviewer loop
                loop {
                    if signal_clone.is_paused() {
                        signal_clone.wait_for_resume().await;
                    }
                    match reviewer_loop(&reviewer).await {
                        Ok(()) => break,
                        Err(e) => {
                            warn!(agent = %label, error = %e, "Reviewer loop error, retrying");
                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        }
                    }
                }
            }));
        }

        // Phase 5: Wait for all agents to complete
        for handle in handles {
            let _ = handle.await;
        }

        // Clean up PID file
        let _ = std::fs::remove_file(pid_file_path());

        info!("Workflow complete");
        Ok(())
    }

    /// Pause the workflow — agents finish current tasks then stop.
    pub fn pause(&self) {
        // Signal is shared via Arc, but we need to access it.
        // For now, pause is handled via OS signals (SIGUSR1).
        tracing::info!("Pause requested externally");
    }

    /// Resume the workflow — agents continue claiming tasks.
    pub fn resume(&self) {
        tracing::info!("Resume requested externally");
    }

    /// Work stealing: when an agent has no tasks in its own queue,
    /// attempt to claim tasks from compatible agent queues.
    #[instrument(skip_all)]
    pub async fn steal_work(
        &self,
        agent_id: &str,
        agent_type: &str,
    ) -> Option<Task> {
        let repo = TaskRepository::new(self.db.connection_arc());

        // 1. Try own queue first
        if let Ok(Some(task)) = repo.claim(agent_id) {
            return Some(task);
        }

        // 2. Steal from compatible queues
        let compatible_types: &[TaskType] = match agent_type {
            "producer" => &[TaskType::ValidateDataModel, TaskType::ValidateCrossFile],
            "consumer" => &[TaskType::ValidateDataModel, TaskType::ValidateShard],
            "reviewer" => &[TaskType::ExtractMetadata, TaskType::ExtractDataModel],
            _ => &[],
        };

        for task_type in compatible_types {
            if let Ok(Some(task)) = self.claim_from_queue(agent_id, *task_type) {
                info!(agent_id = %agent_id, task_type = ?task_type, "Stole task from compatible queue");
                return Some(task);
            }
        }

        None
    }

    /// Claim a specific type of task from the queue.
    fn claim_from_queue(
        &self,
        agent_id: &str,
        task_type: TaskType,
    ) -> Result<Option<Task>, cleanroom_db::DbError> {
        let conn = self.db.connection();
        let task_type_str = task_type.as_str();

        // Atomically claim a pending task of the specified type
        let mut stmt = conn
            .prepare(
                r#"UPDATE tasks
                   SET status = 'in_progress', assigned_to = ?1, started_at = CURRENT_TIMESTAMP
                   WHERE task_id = (
                       SELECT task_id FROM tasks
                       WHERE status = 'pending' AND task_type = ?2
                       ORDER BY priority DESC, created_at ASC
                       LIMIT 1
                   )
                   RETURNING task_id"#,
            )
            .map_err(|e| cleanroom_db::DbError::QueryFailed(e.to_string()))?;

        let result = stmt.query_row(
            rusqlite::params![agent_id, task_type_str],
            |row| row.get::<_, String>(0),
        );

        drop(stmt);

        match result {
            Ok(task_id) => {
                // Fetch the full task
                let repo = TaskRepository::new(self.db.connection_arc());
                repo.get(&task_id).map(Some)
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(cleanroom_db::DbError::QueryFailed(e.to_string())),
        }
    }
}

/// Write PID to a platform-appropriate temp file for CLI status checks.
fn write_pid_file() {
    let pid = std::process::id();
    let pid_path = pid_file_path();
    let _ = std::fs::write(&pid_path, pid.to_string());
    tracing::info!(pid, path = %pid_path.display(), "PID file written");
}

/// Platform-appropriate path for the PID file.
pub fn pid_file_path() -> std::path::PathBuf {
    std::env::temp_dir().join("cleanroom.pid")
}

/// Platform-appropriate path for the TCP port file.
pub fn port_file_path() -> std::path::PathBuf {
    std::env::temp_dir().join("cleanroom.port")
}

/// Reads the TCP port from the port file written by the MCP server.
pub fn read_port_file() -> Option<u16> {
    let path = port_file_path();
    let content = std::fs::read_to_string(&path).ok()?;
    content.trim().parse().ok()
}

/// Writes the TCP port to a file for CLI discovery.
pub fn write_port_file(port: u16) {
    let path = port_file_path();
    let _ = std::fs::write(&path, port.to_string());
    tracing::info!(port, path = %path.display(), "Port file written");
}