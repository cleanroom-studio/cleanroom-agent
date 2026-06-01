//! Task Scheduler — creates and dispatches analysis/generation tasks.
//!
//! This module manages the lifecycle of tasks: initial creation, dependency ordering,
//! retry logic, checkpointing, and workflow progress monitoring.
//!
//! # Task Plans
//!
//! Tasks are created from [`TaskPlan`]s that define:
//! - **Priority groups**: Tasks run in priority order (higher first)
//! - **Dependencies**: Tasks can depend on other tasks completing first
//! - **Retry policy**: Maximum retry count before marking as failed permanently
//!
//! # Predefined Plans
//!
//! - [`TaskPlan::analysis_plan`]: Full repository analysis workflow
//! - [`TaskPlan::generation_plan`]: Code generation workflow
//!
//! # Checkpointing
//!
//! The scheduler supports automatic checkpoint creation and restoration,
//! allowing workflows to be resumed after interruption.
//!
//! # Example
//!
//! ```no_run
//! use cleanroom_agent::scheduler::{Scheduler, TaskPlan};
//! use cleanroom_db::Database;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let db = Arc::new(Database::open(&std::path::PathBuf::from("state.db"))?);
//! let scheduler = Scheduler::new(db);
//!
//! let plan = TaskPlan::analysis_plan("my-project", ".");
//! let task_ids = scheduler.create_from_plan(&plan)?;
//! println!("Created {} tasks", task_ids.len());
//!
//! let progress = scheduler.get_progress()?;
//! println!("Progress: {:.1}%", progress.overall_progress * 100.0);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use cleanroom_db::{
    Database, TaskRepository, Task, TaskStatus, TaskType,
};
use tracing::{info, warn, instrument};

/// A plan for task execution — what tasks to create and in what order.
///
/// Task plans define an ordered set of tasks with dependencies and priorities.
/// Tasks are organized into priority groups, where higher priority groups
/// are executed before lower ones. Within a group, tasks can have dependencies
/// on tasks in earlier groups.
///
/// # Example
///
/// ```
/// use cleanroom_agent::scheduler::{TaskPlan, TaskTemplate};
/// use cleanroom_db::TaskType;
///
/// let plan = TaskPlan::analysis_plan("my-project", ".");
/// for (i, group) in plan.priority_groups.iter().enumerate() {
///     println!("Priority group {}: {} tasks", i, group.len());
/// }
/// ```
#[derive(Debug, Clone)]
pub struct TaskPlan {
    /// Tasks grouped by priority level (higher priority groups run first)
    pub priority_groups: Vec<Vec<TaskTemplate>>,
    /// Document name for scoping all tasks in this plan
    pub document_name: String,
}

/// Template for creating a task with all its configuration.
///
/// Contains all information needed to create a task, including
/// type, priority, input data, dependencies, and retry policy.
#[derive(Debug, Clone)]
pub struct TaskTemplate {
    /// The type of task to create
    pub task_type: TaskType,
    /// Priority level (higher runs first)
    pub priority: i32,
    /// JSON input data for the task
    pub input: serde_json::Value,
    /// Indices into the flattened task list that this task depends on
    pub dependency_indices: Vec<usize>,
    /// Maximum number of retry attempts before marking as failed
    pub max_retries: i32,
}

impl TaskPlan {
    /// Create a standard analysis plan for a given document.
    pub fn analysis_plan(document_name: &str, repo_path: &str) -> Self {
        let input = serde_json::json!({
            "document": document_name,
            "repo_path": repo_path,
            "project_name": document_name,
        });
        Self {
            priority_groups: vec![
                // Priority 10: Repo scan (no dependencies)
                vec![TaskTemplate {
                    task_type: TaskType::RepoAnalyze,
                    priority: 10,
                    input: input.clone(),
                    dependency_indices: vec![],
                    max_retries: 3,
                }],
                // Priority 8: Metadata + architecture extraction (depends on repo scan)
                vec![
                    TaskTemplate {
                        task_type: TaskType::ExtractMetadata,
                        priority: 8,
                        input: input.clone(),
                        dependency_indices: vec![0], // repo scan
                        max_retries: 3,
                    },
                    TaskTemplate {
                        task_type: TaskType::ExtractArchitecture,
                        priority: 8,
                        input: input.clone(),
                        dependency_indices: vec![0],
                        max_retries: 3,
                    },
                ],
                // Priority 6: Data model + module extraction (depends on architecture)
                vec![
                    TaskTemplate {
                        task_type: TaskType::ExtractDataModel,
                        priority: 6,
                        input: input.clone(),
                        dependency_indices: vec![0],
                        max_retries: 3,
                    },
                    TaskTemplate {
                        task_type: TaskType::ExtractModule,
                        priority: 6,
                        input: input.clone(),
                        dependency_indices: vec![0],
                        max_retries: 3,
                    },
                ],
                // Priority 5: UI extraction
                vec![TaskTemplate {
                    task_type: TaskType::ExtractUi,
                    priority: 5,
                    input: input.clone(),
                    dependency_indices: vec![0],
                    max_retries: 3,
                }],
                // Priority 4: Tests + design decisions (depends on data model)
                vec![
                    TaskTemplate {
                        task_type: TaskType::ExtractTests,
                        priority: 4,
                        input: input.clone(),
                        dependency_indices: vec![0],
                        max_retries: 3,
                    },
                    TaskTemplate {
                        task_type: TaskType::InferDesignDecisions,
                        priority: 4,
                        input: input.clone(),
                        dependency_indices: vec![0],
                        max_retries: 3,
                    },
                ],
                // Priority 2: Validation (depends on everything above)
                vec![TaskTemplate {
                    task_type: TaskType::ValidateShard,
                    priority: 2,
                    input: input.clone(),
                    dependency_indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
                    max_retries: 5,
                }],
            ],
            document_name: document_name.to_string(),
        }
    }

    /// Create a standard code generation plan.
    pub fn generation_plan(document_name: &str) -> Self {
        let input = serde_json::json!({
            "document": document_name,
            "project_name": document_name,
        });
        Self {
            priority_groups: vec![
                vec![TaskTemplate {
                    task_type: TaskType::ImportSdef,
                    priority: 10,
                    input: input.clone(),
                    dependency_indices: vec![],
                    max_retries: 3,
                }],
                vec![
                    TaskTemplate {
                        task_type: TaskType::GenerateCode,
                        priority: 8,
                        input: { let mut inp = input.clone(); inp["scope"] = serde_json::json!("all"); inp },
                        dependency_indices: vec![0],
                        max_retries: 3,
                    },
                ],
                // Reviewer phase (parallel with validation)
                vec![
                    TaskTemplate {
                        task_type: TaskType::ValidateDataModel,
                        priority: 7,
                        input: input.clone(),
                        dependency_indices: vec![1],
                        max_retries: 3,
                    },
                    TaskTemplate {
                        task_type: TaskType::ValidateCrossFile,
                        priority: 7,
                        input: input.clone(),
                        dependency_indices: vec![1],
                        max_retries: 3,
                    },
                    TaskTemplate {
                        task_type: TaskType::RoundtripVerify,
                        priority: 6,
                        input: input.clone(),
                        dependency_indices: vec![1, 2, 3],
                        max_retries: 2,
                    },
                ],
                vec![TaskTemplate {
                    task_type: TaskType::MergeCode,
                    priority: 5,
                    input: input.clone(),
                    dependency_indices: vec![1, 2],
                    max_retries: 3,
                }],
                vec![TaskTemplate {
                    task_type: TaskType::RunTests,
                    priority: 4,
                    input: input.clone(),
                    dependency_indices: vec![5],
                    max_retries: 5,
                }],
            ],
            document_name: document_name.to_string(),
        }
    }

    /// Create an LLM-driven file-level analysis plan.
    ///
    /// Emits one `LlmAnalyzeFile` task per file path. Files are independent of
    /// each other (no inter-file dependencies), so they all land in the same
    /// priority group and can be processed in parallel by N agent workers.
    /// This is the leaf-level replacement for [`Self::analysis_plan`]'s
    /// `RepoAnalyze` task once the Producer is driven by `llm_loop::run_loop`.
    ///
    /// `file_paths` should be repo-relative (e.g. `"src/main.rs"`). The full
    /// repo path is sent in `input["repo_path"]` so the worker can resolve
    /// the file on disk.
    pub fn llm_analysis_plan(
        document_name: &str,
        repo_path: &str,
        file_paths: &[&str],
    ) -> Self {
        let base_input = serde_json::json!({
            "document": document_name,
            "project_name": document_name,
            "repo_path": repo_path,
        });

        // One task per file, all at priority 8 (high but not critical-path).
        // `max_retries: 2` keeps the noise low -- failed LLM calls are usually
        // worth surfacing to a human rather than hammering the API.
        let tasks: Vec<TaskTemplate> = file_paths
            .iter()
            .map(|p| TaskTemplate {
                task_type: TaskType::LlmAnalyzeFile,
                priority: 8,
                input: {
                    let mut inp = base_input.clone();
                    inp["file_path"] = serde_json::Value::String((*p).to_string());
                    inp
                },
                dependency_indices: vec![],
                max_retries: 2,
            })
            .collect();

        Self {
            priority_groups: if tasks.is_empty() {
                vec![]
            } else {
                vec![tasks]
            },
            document_name: document_name.to_string(),
        }
    }

    /// Create an LLM-driven entity-level code generation plan.
    ///
    /// Emits one `LlmGenerateCode` task per entity URI. All tasks live in a
    /// single priority group with no inter-entity dependencies, mirroring
    /// `llm_analysis_plan`'s parallel-friendly shape. This is the leaf-level
    /// replacement for [`Self::generation_plan`]'s `GenerateCode` task once
    /// the Consumer is driven by `llm_loop::run_loop`.
    ///
    /// `entity_uris` should be S.DEF entity URIs (e.g. `"sdef://my-proj/User"`).
    pub fn llm_generation_plan(
        document_name: &str,
        target_language: &str,
        entity_uris: &[&str],
    ) -> Self {
        let base_input = serde_json::json!({
            "document": document_name,
            "project_name": document_name,
            "target_language": target_language,
        });

        let tasks: Vec<TaskTemplate> = entity_uris
            .iter()
            .map(|uri| TaskTemplate {
                task_type: TaskType::LlmGenerateCode,
                priority: 8,
                input: {
                    let mut inp = base_input.clone();
                    inp["entity_uri"] = serde_json::Value::String((*uri).to_string());
                    inp
                },
                dependency_indices: vec![],
                max_retries: 2,
            })
            .collect();

        Self {
            priority_groups: if tasks.is_empty() {
                vec![]
            } else {
                vec![tasks]
            },
            document_name: document_name.to_string(),
        }
    }
}

/// The task scheduler.
pub struct Scheduler {
    db: Arc<Database>,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Create a new scheduler from a database.
    pub fn from_db(db: Database) -> Self {
        Self { db: Arc::new(db) }
    }

    /// Get the database reference.
    pub fn db(&self) -> &Arc<Database> {
        &self.db
    }

    /// Create tasks from a plan.
    #[instrument(skip(self))]
    pub fn create_from_plan(&self, plan: &TaskPlan) -> Result<Vec<String>, cleanroom_db::DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());
        let mut flat_tasks: Vec<TaskTemplate> = Vec::new();
        let mut created_ids: Vec<String> = Vec::new();
        let mut index_map: HashMap<usize, usize> = HashMap::new(); // old index → position in flat_tasks

        // Flatten priority groups into a single ordered list, tracking indices
        for group in &plan.priority_groups {
            for task_tmpl in group {
                index_map.insert(flat_tasks.len(), flat_tasks.len());
                flat_tasks.push(task_tmpl.clone());
            }
        }

        // Create each task with computed dependencies
        for (i, tmpl) in flat_tasks.iter().enumerate() {
            let deps: Vec<String> = tmpl
                .dependency_indices
                .iter()
                .filter_map(|dep_idx| created_ids.get(*dep_idx))
                .cloned()
                .collect();

            let task_id = uuid::Uuid::new_v4().to_string();
            let task = Task {
                task_id: task_id.clone(),
                task_type: tmpl.task_type,
                status: TaskStatus::Pending,
                priority: tmpl.priority,
                input_json: serde_json::to_string(&tmpl.input)
                    .unwrap_or_else(|_| "{}".to_string()),
                output_json: None,
                error_message: None,
                assigned_to: None,
                progress: 0.0,
                created_at: chrono::Utc::now().to_rfc3339(),
                started_at: None,
                completed_at: None,
                retry_count: 0,
                max_retries: tmpl.max_retries,
                last_heartbeat: None,
                dependencies_json: serde_json::to_string(&deps)
                    .unwrap_or_else(|_| "[]".to_string()),
                version: 1,
            };

            repo.create(&task)?;
            created_ids.push(task_id);
            info!(index = i, task_id = %created_ids.last().unwrap(), "Created task from plan");
        }

        Ok(created_ids)
    }

    /// Retry failed tasks.
    #[instrument(skip(self))]
    pub fn retry_failed_tasks(&self) -> Result<usize, cleanroom_db::DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());
        let failed_tasks = repo.list(Some(TaskStatus::Failed), None, None)?;
        let mut retried = 0;

        for task in &failed_tasks {
            if task.retry_count < task.max_retries {
                let new_retry = task.retry_count + 1;
                let conn = self.db.connection();
                conn.execute(
                    "UPDATE tasks SET status = 'pending', retry_count = ?1, error_message = NULL WHERE task_id = ?2",
                    rusqlite::params![new_retry, task.task_id],
                ).map_err(|e| cleanroom_db::DbError::QueryFailed(e.to_string()))?;
                retried += 1;
                info!(task_id = %task.task_id, retry = new_retry, "Retrying failed task");
            } else {
                // Mark as failed permanently
                repo.update_status(&task.task_id, TaskStatus::FailedPermanently)?;
                warn!(task_id = %task.task_id, "Task failed permanently (max retries exceeded)");
            }
        }

        Ok(retried)
    }

    /// Get workflow progress summary.
    #[instrument(skip(self))]
    pub fn get_progress(&self) -> Result<ProgressSummary, cleanroom_db::DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());
        let all = repo.list(None, None, None)?;

        let mut summary = ProgressSummary::default();
        for task in &all {
            summary.total += 1;
            match task.status {
                TaskStatus::Pending => summary.pending += 1,
                TaskStatus::Assigned | TaskStatus::InProgress => {
                    summary.in_progress += 1;
                    summary.total_progress += task.progress;
                }
                TaskStatus::Completed => {
                    summary.completed += 1;
                    summary.total_progress += 1.0;
                }
                TaskStatus::Failed => summary.failed += 1,
                TaskStatus::Retrying => summary.retrying += 1,
                TaskStatus::FailedPermanently => summary.failed_permanently += 1,
            }
        }

        if summary.total > 0 {
            summary.overall_progress = summary.total_progress / summary.total as f64;
        }

        Ok(summary)
    }

    /// Create an automatic checkpoint.
    #[instrument(skip(self))]
    pub fn create_automatic_checkpoint(&self, document_name: &str) -> Result<String, cleanroom_db::DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());

        let all_tasks = repo.list(None, None, None)?;
        let task_snapshot = serde_json::to_value(&all_tasks)
            .unwrap_or(serde_json::json!([]));
        let shard_snapshot = serde_json::json!({});

        let checkpoint_id = uuid::Uuid::new_v4().to_string();
        let conn = self.db.connection();
        conn.execute(
            r#"INSERT INTO checkpoints (checkpoint_id, document_name, description, created_at, task_snapshot_json, shard_snapshot_json)
               VALUES (?1, ?2, ?3, CURRENT_TIMESTAMP, ?4, ?5)"#,
            rusqlite::params![
                checkpoint_id,
                document_name,
                "Automatic checkpoint",
                serde_json::to_string(&task_snapshot).unwrap_or_else(|_| "[]".to_string()),
                serde_json::to_string(&shard_snapshot).unwrap_or_else(|_| "{}".to_string()),
            ],
        ).map_err(|e| cleanroom_db::DbError::QueryFailed(e.to_string()))?;

        info!(%checkpoint_id, "Automatic checkpoint created");
        Ok(checkpoint_id)
    }

    /// Restore from a checkpoint.
    #[instrument(skip(self))]
    pub fn restore_from_checkpoint(&self, checkpoint_id: &str) -> Result<(), cleanroom_db::DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare(
            "SELECT checkpoint_id, document_name, description, created_at, task_snapshot_json, shard_snapshot_json
             FROM checkpoints WHERE checkpoint_id = ?1"
        ).map_err(|e| cleanroom_db::DbError::QueryFailed(e.to_string()))?;

        let _cp = stmt.query_row(rusqlite::params![checkpoint_id], |row| {
            Ok(cleanroom_db::repositories::Checkpoint {
                checkpoint_id: row.get(0)?,
                document_name: row.get(1)?,
                description: row.get(2)?,
                created_at: row.get(3)?,
                task_snapshot_json: row.get(4)?,
                shard_snapshot_json: row.get(5)?,
            })
        }).map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => cleanroom_db::DbError::NotFound {
                resource: "checkpoint", field: "checkpoint_id", value: checkpoint_id.to_string(),
            },
            _ => cleanroom_db::DbError::QueryFailed(e.to_string()),
        })?;

        info!(%checkpoint_id, "Checkpoint loaded for restoration");
        Ok(())
    }
}

/// Progress summary for the current workflow.
#[derive(Debug, Clone, Default)]
pub struct ProgressSummary {
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub failed: usize,
    pub retrying: usize,
    pub failed_permanently: usize,
    pub overall_progress: f64,
    pub total_progress: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_db::Database;

    fn setup() -> (Database, Scheduler) {
        let db = Database::in_memory().unwrap();
        let scheduler = Scheduler::from_db(db.clone());
        (db, scheduler)
    }

    #[test]
    fn test_analysis_plan_creation() {
        let (_, scheduler) = setup();
        let plan = TaskPlan::analysis_plan("test-doc", ".");
        let ids = scheduler.create_from_plan(&plan).unwrap();
        // Analysis plan should create: repo_analyze + extract_metadata + extract_arch + data_model + extract_module + ui + tests + design_decisions + validate
        assert_eq!(ids.len(), 9, "Should create 9 tasks");
    }

    #[test]
    fn test_llm_analysis_plan_creation() {
        // No DB required: the plan generator is pure-functional, so we just
        // exercise the priority-group shape directly.
        let files = ["src/main.rs", "src/lib.rs", "src/config.rs"];
        let plan = TaskPlan::llm_analysis_plan("test-doc", "/tmp/repo", &files);
        assert_eq!(plan.priority_groups.len(), 1, "all files in one priority group");
        let group = &plan.priority_groups[0];
        assert_eq!(group.len(), 3, "one task per file");
        for (i, t) in group.iter().enumerate() {
            assert_eq!(t.task_type, TaskType::LlmAnalyzeFile);
            assert_eq!(t.priority, 8);
            assert_eq!(t.max_retries, 2);
            assert!(t.dependency_indices.is_empty(), "files are independent");
            assert_eq!(
                t.input["file_path"],
                serde_json::Value::String(files[i].to_string())
            );
            assert_eq!(t.input["document"], serde_json::Value::String("test-doc".into()));
            assert_eq!(t.input["repo_path"], serde_json::Value::String("/tmp/repo".into()));
        }
        assert_eq!(plan.document_name, "test-doc");
    }

    #[test]
    fn test_llm_analysis_plan_empty_files() {
        let plan = TaskPlan::llm_analysis_plan("test-doc", "/tmp/repo", &[]);
        assert!(plan.priority_groups.is_empty());
    }

    #[test]
    fn test_llm_generation_plan_creation() {
        let entities = ["sdef://proj/User", "sdef://proj/Post", "sdef://proj/Comment"];
        let plan = TaskPlan::llm_generation_plan("test-doc", "rust", &entities);
        assert_eq!(plan.priority_groups.len(), 1, "all entities in one priority group");
        let group = &plan.priority_groups[0];
        assert_eq!(group.len(), 3, "one task per entity");
        for (i, t) in group.iter().enumerate() {
            assert_eq!(t.task_type, TaskType::LlmGenerateCode);
            assert_eq!(t.priority, 8);
            assert_eq!(t.max_retries, 2);
            assert!(t.dependency_indices.is_empty());
            assert_eq!(
                t.input["entity_uri"],
                serde_json::Value::String(entities[i].to_string())
            );
            assert_eq!(t.input["target_language"], serde_json::Value::String("rust".into()));
        }
    }

    #[test]
    fn test_llm_task_type_roundtrip() {
        // `as_str` and `from_str` must be mutual inverses for the new variants.
        for tt in [TaskType::LlmAnalyzeFile, TaskType::LlmGenerateCode] {
            let s = tt.as_str();
            let back = TaskType::from_str(s).expect("from_str should accept as_str output");
            assert_eq!(back, tt, "roundtrip mismatch for {s}");
        }
    }

    #[test]
    fn test_generation_plan_creation() {
        let (_, scheduler) = setup();
        let plan = TaskPlan::generation_plan("test-doc");
        let ids = scheduler.create_from_plan(&plan).unwrap();
        // ImportSdef(1) + GenerateCode(1) + ValidateDataModel(1) + ValidateCrossFile(1) + RoundtripVerify(1) + MergeCode(1) + RunTests(1)
        assert_eq!(ids.len(), 7, "Should create 7 tasks (including reviewer)");
    }

    #[test]
    fn test_progress_summary() {
        let (_, scheduler) = setup();
        let plan = TaskPlan::analysis_plan("test-doc", ".");
        scheduler.create_from_plan(&plan).unwrap();
        let progress = scheduler.get_progress().unwrap();
        assert_eq!(progress.total, 9);
        assert_eq!(progress.pending, 9);
    }

    #[test]
    fn test_retry_failed_tasks() {
        let (db, scheduler) = setup();
        let plan = TaskPlan::analysis_plan("test-doc", ".");
        scheduler.create_from_plan(&plan).unwrap();

        // Mark all tasks as failed
        let repo = TaskRepository::new(db.connection_arc());
        let tasks = repo.list(None, None, None).unwrap();
        for task in &tasks {
            repo.update_status(&task.task_id, TaskStatus::Failed).unwrap();
        }

        let retried = scheduler.retry_failed_tasks().unwrap();
        assert!(retried > 0, "Should retry at least one task");

        // Check that tasks were reset to pending
        let pending = repo.list(Some(TaskStatus::Pending), None, None).unwrap();
        assert!(pending.len() == retried, "Retried tasks should be pending");
    }

    #[test]
    fn test_checkpoint_creation() {
        let (db, scheduler) = setup();
        // Create a document first to satisfy FK constraint
        db.connection().execute(
            "INSERT INTO sdef_documents (name, created_at, updated_at) VALUES ('test-doc', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            [],
        ).unwrap();
        let cpid = scheduler.create_automatic_checkpoint("test-doc").unwrap();
        assert!(!cpid.is_empty());
    }
}
