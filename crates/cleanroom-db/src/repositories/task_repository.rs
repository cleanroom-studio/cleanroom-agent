//! Task repository for task CRUD operations.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::instrument;

use crate::error::{DbError, DbResult};

/// Task status enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Assigned,
    InProgress,
    Completed,
    Failed,
    Retrying,
    FailedPermanently,
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Assigned => "assigned",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Retrying => "retrying",
            Self::FailedPermanently => "failed_permanently",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "assigned" => Some(Self::Assigned),
            "in_progress" => Some(Self::InProgress),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "retrying" => Some(Self::Retrying),
            "failed_permanently" => Some(Self::FailedPermanently),
            _ => None,
        }
    }
}

/// Task type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    RepoAnalyze,
    ExtractMetadata,
    ExtractArchitecture,
    ExtractDataModel,
    ExtractModule,
    ExtractUi,
    ExtractTests,
    InferDesignDecisions,
    ValidateShard,
    GenerateCode,
    RunTests,
    MergeCode,
    ImportSdef,
    ExportSdef,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RepoAnalyze => "REPO_ANALYZE",
            Self::ExtractMetadata => "EXTRACT_METADATA",
            Self::ExtractArchitecture => "EXTRACT_ARCHITECTURE",
            Self::ExtractDataModel => "EXTRACT_DATA_MODEL",
            Self::ExtractModule => "EXTRACT_MODULE",
            Self::ExtractUi => "EXTRACT_UI",
            Self::ExtractTests => "EXTRACT_TESTS",
            Self::InferDesignDecisions => "INFER_DESIGN_DECISIONS",
            Self::ValidateShard => "VALIDATE_SHARD",
            Self::GenerateCode => "GENERATE_CODE",
            Self::RunTests => "RUN_TESTS",
            Self::MergeCode => "MERGE_CODE",
            Self::ImportSdef => "IMPORT_SDEF",
            Self::ExportSdef => "EXPORT_SDEF",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "REPO_ANALYZE" => Some(Self::RepoAnalyze),
            "EXTRACT_METADATA" => Some(Self::ExtractMetadata),
            "EXTRACT_ARCHITECTURE" => Some(Self::ExtractArchitecture),
            "EXTRACT_DATA_MODEL" => Some(Self::ExtractDataModel),
            "EXTRACT_MODULE" => Some(Self::ExtractModule),
            "EXTRACT_UI" => Some(Self::ExtractUi),
            "EXTRACT_TESTS" => Some(Self::ExtractTests),
            "INFER_DESIGN_DECISIONS" => Some(Self::InferDesignDecisions),
            "VALIDATE_SHARD" => Some(Self::ValidateShard),
            "GENERATE_CODE" => Some(Self::GenerateCode),
            "RUN_TESTS" => Some(Self::RunTests),
            "MERGE_CODE" => Some(Self::MergeCode),
            "IMPORT_SDEF" => Some(Self::ImportSdef),
            "EXPORT_SDEF" => Some(Self::ExportSdef),
            _ => None,
        }
    }
}

/// Task model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub task_id: String,
    pub task_type: TaskType,
    pub status: TaskStatus,
    pub priority: i32,
    pub input_json: String,
    pub output_json: Option<String>,
    pub error_message: Option<String>,
    pub assigned_to: Option<String>,
    pub progress: f64,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub retry_count: i32,
    pub max_retries: i32,
    pub last_heartbeat: Option<String>,
    pub dependencies_json: String,
    pub version: i32,
}

/// Task repository.
pub struct TaskRepository {
    conn: Arc<Mutex<Connection>>,
}

impl TaskRepository {
    /// Create a new task repository from Arc connection.
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Create a new task.
    #[instrument(skip_all)]
    pub fn create(&self, task: &Task) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO tasks (
                task_id, task_type, status, priority, input_json, output_json,
                error_message, assigned_to, progress, started_at, completed_at,
                retry_count, max_retries, last_heartbeat, dependencies_json, version
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)"#,
            params![
                task.task_id,
                task.task_type.as_str(),
                task.status.as_str(),
                task.priority,
                task.input_json,
                task.output_json,
                task.error_message,
                task.assigned_to,
                task.progress,
                task.started_at,
                task.completed_at,
                task.retry_count,
                task.max_retries,
                task.last_heartbeat,
                task.dependencies_json,
                task.version,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Get a task by ID.
    #[instrument(skip_all)]
    pub fn get(&self, task_id: &str) -> DbResult<Task> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT task_id, task_type, status, priority, input_json, output_json,
                   error_message, assigned_to, progress, created_at, started_at,
                   completed_at, retry_count, max_retries, last_heartbeat,
                   dependencies_json, version FROM tasks WHERE task_id = ?1"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![task_id], |row| {
            let task_type_str: String = row.get(1)?;
            let status_str: String = row.get(2)?;
            Ok(Task {
                task_id: row.get(0)?,
                task_type: TaskType::from_str(&task_type_str).unwrap_or(TaskType::RepoAnalyze),
                status: TaskStatus::from_str(&status_str).unwrap_or(TaskStatus::Pending),
                priority: row.get(3)?,
                input_json: row.get(4)?,
                output_json: row.get(5)?,
                error_message: row.get(6)?,
                assigned_to: row.get(7)?,
                progress: row.get(8)?,
                created_at: row.get(9)?,
                started_at: row.get(10)?,
                completed_at: row.get(11)?,
                retry_count: row.get(12)?,
                max_retries: row.get(13)?,
                last_heartbeat: row.get(14)?,
                dependencies_json: row.get(15)?,
                version: row.get(16)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "task",
                field: "task_id",
                value: task_id.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    /// Update task status.
    #[instrument(skip_all)]
    pub fn update_status(&self, task_id: &str, status: TaskStatus) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute(
                "UPDATE tasks SET status = ?1 WHERE task_id = ?2",
                params![status.as_str(), task_id],
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        if rows == 0 {
            return Err(DbError::NotFound {
                resource: "task",
                field: "task_id",
                value: task_id.to_string(),
            });
        }
        Ok(())
    }

    /// Update task progress.
    #[instrument(skip_all)]
    pub fn update_progress(&self, task_id: &str, progress: f64) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute(
                "UPDATE tasks SET progress = ?1 WHERE task_id = ?2",
                params![progress, task_id],
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        if rows == 0 {
            return Err(DbError::NotFound {
                resource: "task",
                field: "task_id",
                value: task_id.to_string(),
            });
        }
        Ok(())
    }

    /// Claim a task atomically (set status to in_progress and assigned_to).
    #[instrument(skip_all)]
    pub fn claim(&self, agent_id: &str) -> DbResult<Option<Task>> {
        let conn = self.conn.lock().unwrap();

        // Try to claim a pending task
        let mut stmt = conn
            .prepare(
                r#"UPDATE tasks
                   SET status = 'in_progress', assigned_to = ?1, started_at = CURRENT_TIMESTAMP
                   WHERE task_id = (
                       SELECT task_id FROM tasks
                       WHERE status = 'pending'
                       ORDER BY priority DESC, created_at ASC
                       LIMIT 1
                   )
                   RETURNING task_id"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let result = stmt.query_row(params![agent_id], |row| row.get::<_, String>(0));

        drop(stmt);

        match result {
            Ok(task_id) => {
                // Fetch the full task
                let mut stmt = conn
                    .prepare(
                        r#"SELECT task_id, task_type, status, priority, input_json, output_json,
                           error_message, assigned_to, progress, created_at, started_at,
                           completed_at, retry_count, max_retries, last_heartbeat,
                           dependencies_json, version FROM tasks WHERE task_id = ?1"#,
                    )
                    .map_err(|e| DbError::QueryFailed(e.to_string()))?;

                let task = stmt
                    .query_row(params![task_id], |row| {
                        let task_type_str: String = row.get(1)?;
                        let status_str: String = row.get(2)?;
                        Ok(Task {
                            task_id: row.get(0)?,
                            task_type: TaskType::from_str(&task_type_str)
                                .unwrap_or(TaskType::RepoAnalyze),
                            status: TaskStatus::from_str(&status_str)
                                .unwrap_or(TaskStatus::Pending),
                            priority: row.get(3)?,
                            input_json: row.get(4)?,
                            output_json: row.get(5)?,
                            error_message: row.get(6)?,
                            assigned_to: row.get(7)?,
                            progress: row.get(8)?,
                            created_at: row.get(9)?,
                            started_at: row.get(10)?,
                            completed_at: row.get(11)?,
                            retry_count: row.get(12)?,
                            max_retries: row.get(13)?,
                            last_heartbeat: row.get(14)?,
                            dependencies_json: row.get(15)?,
                            version: row.get(16)?,
                        })
                    })
                    .map_err(|e| DbError::QueryFailed(e.to_string()))?;

                Ok(Some(task))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(DbError::QueryFailed(e.to_string())),
        }
    }

    /// Update heartbeat.
    #[instrument(skip_all)]
    pub fn heartbeat(&self, task_id: &str) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE tasks SET last_heartbeat = CURRENT_TIMESTAMP WHERE task_id = ?1",
            params![task_id],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Complete a task with output.
    #[instrument(skip_all)]
    pub fn complete(&self, task_id: &str, output_json: &str) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"UPDATE tasks
               SET status = 'completed', output_json = ?1, completed_at = CURRENT_TIMESTAMP
               WHERE task_id = ?2"#,
            params![output_json, task_id],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// List tasks with optional filters.
    #[instrument(skip_all)]
    pub fn list(
        &self,
        status: Option<TaskStatus>,
        task_type: Option<TaskType>,
        assigned_to: Option<&str>,
    ) -> DbResult<Vec<Task>> {
        let conn = self.conn.lock().unwrap();

        let mut query = String::from(
            r#"SELECT task_id, task_type, status, priority, input_json, output_json,
               error_message, assigned_to, progress, created_at, started_at,
               completed_at, retry_count, max_retries, last_heartbeat,
               dependencies_json, version FROM tasks WHERE 1=1"#,
        );

        if status.is_some() {
            query.push_str(" AND status = ?");
        }
        if task_type.is_some() {
            query.push_str(" AND task_type = ?");
        }
        if assigned_to.is_some() {
            query.push_str(" AND assigned_to = ?");
        }

        query.push_str(" ORDER BY priority DESC, created_at ASC");

        let mut stmt = conn
            .prepare(&query)
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let status_str = status.map(|s| s.as_str().to_string());
        let task_type_str = task_type.map(|t| t.as_str().to_string());

        let tasks = stmt
            .query_map(
                rusqlite::params_from_iter(
                    status_str
                        .as_ref()
                        .into_iter()
                        .chain(task_type_str.as_ref()),
                ),
                |row| {
                    let task_type_str: String = row.get(1)?;
                    let status_str: String = row.get(2)?;
                    Ok(Task {
                        task_id: row.get(0)?,
                        task_type: TaskType::from_str(&task_type_str)
                            .unwrap_or(TaskType::RepoAnalyze),
                        status: TaskStatus::from_str(&status_str)
                            .unwrap_or(TaskStatus::Pending),
                        priority: row.get(3)?,
                        input_json: row.get(4)?,
                        output_json: row.get(5)?,
                        error_message: row.get(6)?,
                        assigned_to: row.get(7)?,
                        progress: row.get(8)?,
                        created_at: row.get(9)?,
                        started_at: row.get(10)?,
                        completed_at: row.get(11)?,
                        retry_count: row.get(12)?,
                        max_retries: row.get(13)?,
                        last_heartbeat: row.get(14)?,
                        dependencies_json: row.get(15)?,
                        version: row.get(16)?,
                    })
                },
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(tasks)
    }

    /// Delete a task.
    #[instrument(skip_all)]
    pub fn delete(&self, task_id: &str) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute("DELETE FROM tasks WHERE task_id = ?1", params![task_id])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        if rows == 0 {
            return Err(DbError::NotFound {
                resource: "task",
                field: "task_id",
                value: task_id.to_string(),
            });
        }
        Ok(())
    }
}