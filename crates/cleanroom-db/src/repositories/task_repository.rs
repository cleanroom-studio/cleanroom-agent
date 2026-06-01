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
    // Reviewer task types (docs/13 §2.2)
    ValidateDataModel,
    ValidateCrossFile,
    RoundtripVerify,
    // LLM-driven task types (Phase 0.4). `LlmAnalyzeFile` is the
    // leaf-level replacement for `RepoAnalyze` once the Producer pipeline
    // is driven by `llm_loop::run_loop`; `LlmGenerateCode` replaces
    // `GenerateCode` for the Consumer pipeline. Both have a corresponding
    // CHECK-constraint entry in `migrations/007_llm_task_types.sql`.
    LlmAnalyzeFile,
    LlmGenerateCode,
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
            Self::ValidateDataModel => "VALIDATE_DATA_MODEL",
            Self::ValidateCrossFile => "VALIDATE_CROSS_FILE",
            Self::RoundtripVerify => "ROUNDTRIP_VERIFY",
            Self::LlmAnalyzeFile => "LLM_ANALYZE_FILE",
            Self::LlmGenerateCode => "LLM_GENERATE_CODE",
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
            "VALIDATE_DATA_MODEL" => Some(Self::ValidateDataModel),
            "VALIDATE_CROSS_FILE" => Some(Self::ValidateCrossFile),
            "ROUNDTRIP_VERIFY" => Some(Self::RoundtripVerify),
            "LLM_ANALYZE_FILE" => Some(Self::LlmAnalyzeFile),
            "LLM_GENERATE_CODE" => Some(Self::LlmGenerateCode),
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

    /// Update specific fields of a pending task (priority, input, dependencies, max_retries).
    ///
    /// Only tasks in `pending` status can be modified. Returns an error if the
    /// task is in_progress, completed, or failed_permanently.
    #[instrument(skip_all)]
    pub fn update_fields(
        &self,
        task_id: &str,
        priority: Option<i32>,
        input_json: Option<&str>,
        dependencies_json: Option<&str>,
        max_retries: Option<i32>,
    ) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();

        // Verify task exists and is in pending state
        let status: String = conn
            .query_row(
                "SELECT status FROM tasks WHERE task_id = ?1",
                params![task_id],
                |row| row.get(0),
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        if status != "pending" {
            return Err(DbError::ConstraintViolation(format!(
                "Task {} is in status '{}', only 'pending' tasks can be modified",
                task_id, status
            )));
        }

        // Build dynamic UPDATE
        let mut sets = Vec::new();
        let mut param_idx = 1;
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(p) = priority {
            sets.push(format!("priority = ?{}", param_idx));
            param_values.push(Box::new(p));
            param_idx += 1;
        }
        if let Some(input) = input_json {
            sets.push(format!("input_json = ?{}", param_idx));
            param_values.push(Box::new(input.to_string()));
            param_idx += 1;
        }
        if let Some(deps) = dependencies_json {
            sets.push(format!("dependencies_json = ?{}", param_idx));
            param_values.push(Box::new(deps.to_string()));
            param_idx += 1;
        }
        if let Some(retries) = max_retries {
            sets.push(format!("max_retries = ?{}", param_idx));
            param_values.push(Box::new(retries));
            param_idx += 1;
        }
        // bump version on any change
        sets.push("version = version + 1".to_string());

        if sets.is_empty() {
            return Ok(()); // nothing to update
        }

        let query = format!(
            "UPDATE tasks SET {} WHERE task_id = ?{}",
            sets.join(", "),
            param_idx
        );
        param_values.push(Box::new(task_id.to_string()));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        conn.execute(&query, rusqlite::params_from_iter(param_refs))
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        tracing::info!(%task_id, "Updated task fields");
        Ok(())
    }

    /// Remove a dependency from all tasks that depend on the given task_id.
    #[instrument(skip_all)]
    pub fn cascade_remove_dependency(&self, removed_task_id: &str) -> DbResult<usize> {
        let conn = self.conn.lock().unwrap();
        let mut cascaded = 0;

        // Find all tasks that list `removed_task_id` as a dependency
        let mut stmt = conn
            .prepare("SELECT task_id, dependencies_json FROM tasks WHERE dependencies_json LIKE ?1")
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let pattern = format!("%{}%", removed_task_id);
        let rows: Vec<(String, String)> = stmt
            .query_map(params![pattern], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        drop(stmt);

        for (tid, deps_str) in &rows {
            if let Ok(mut deps) = serde_json::from_str::<Vec<String>>(deps_str) {
                let new_deps: Vec<String> =
                    deps.into_iter().filter(|d| d != removed_task_id).collect();
                let new_json = serde_json::to_string(&new_deps).unwrap_or_default();
                conn.execute(
                    "UPDATE tasks SET dependencies_json = ?1 WHERE task_id = ?2",
                    params![new_json, tid],
                )
                .map_err(|e| DbError::QueryFailed(e.to_string()))?;
                cascaded += 1;
            }
        }

        Ok(cascaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Database;
    use uuid::Uuid;

    fn create_test_task(task_type: TaskType) -> Task {
        Task {
            task_id: Uuid::new_v4().to_string(),
            task_type,
            status: TaskStatus::Pending,
            priority: 5,
            input_json: r#"{"test": true}"#.to_string(),
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
        }
    }

    fn setup() -> (Database, TaskRepository) {
        let db = Database::in_memory().unwrap();
        let repo = TaskRepository::new(db.connection_arc());
        (db, repo)
    }

    #[test]
    fn test_create_and_get_task() {
        let (_, repo) = setup();
        let task = create_test_task(TaskType::RepoAnalyze);
        repo.create(&task).unwrap();

        let fetched = repo.get(&task.task_id).unwrap();
        assert_eq!(fetched.task_id, task.task_id);
        assert_eq!(fetched.task_type, TaskType::RepoAnalyze);
        assert_eq!(fetched.status, TaskStatus::Pending);
        assert_eq!(fetched.priority, 5);
        assert_eq!(fetched.input_json, r#"{"test": true}"#);
    }

    #[test]
    fn test_get_non_existent_task() {
        let (_, repo) = setup();
        let result = repo.get("non-existent-id");
        assert!(result.is_err());
        match result {
            Err(DbError::NotFound { resource, field: _, value }) => {
                assert_eq!(resource, "task");
                assert_eq!(value, "non-existent-id");
            }
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn test_update_status() {
        let (_, repo) = setup();
        let task = create_test_task(TaskType::ExtractDataModel);
        repo.create(&task).unwrap();

        repo.update_status(&task.task_id, TaskStatus::InProgress).unwrap();
        let fetched = repo.get(&task.task_id).unwrap();
        assert_eq!(fetched.status, TaskStatus::InProgress);
    }

    #[test]
    fn test_update_progress() {
        let (_, repo) = setup();
        let task = create_test_task(TaskType::ExtractArchitecture);
        repo.create(&task).unwrap();

        repo.update_progress(&task.task_id, 0.5).unwrap();
        let fetched = repo.get(&task.task_id).unwrap();
        assert!((fetched.progress - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_complete_task() {
        let (_, repo) = setup();
        let task = create_test_task(TaskType::ExtractModule);
        repo.create(&task).unwrap();

        repo.complete(&task.task_id, r#"{"done": true}"#).unwrap();
        let fetched = repo.get(&task.task_id).unwrap();
        assert_eq!(fetched.status, TaskStatus::Completed);
        assert_eq!(fetched.output_json.unwrap(), r#"{"done": true}"#);
    }

    #[test]
    fn test_delete_task() {
        let (_, repo) = setup();
        let task = create_test_task(TaskType::ImportSdef);
        repo.create(&task).unwrap();

        repo.delete(&task.task_id).unwrap();
        let result = repo.get(&task.task_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_heartbeat() {
        let (_, repo) = setup();
        let task = create_test_task(TaskType::ExportSdef);
        repo.create(&task).unwrap();

        repo.heartbeat(&task.task_id).unwrap();
        let fetched = repo.get(&task.task_id).unwrap();
        assert!(fetched.last_heartbeat.is_some());
    }

    #[test]
    fn test_claim_task_atomic() {
        let (_, repo) = setup();
        let task = create_test_task(TaskType::RepoAnalyze);
        repo.create(&task).unwrap();

        let claimed = repo.claim("agent-1").unwrap();
        assert!(claimed.is_some());
        assert_eq!(claimed.unwrap().task_id, task.task_id);

        // Second claim should return None (no pending tasks)
        let second_claim = repo.claim("agent-2").unwrap();
        assert!(second_claim.is_none());
    }

    #[test]
    fn test_list_tasks_with_filters() {
        let (_, repo) = setup();

        let task1 = create_test_task(TaskType::RepoAnalyze);
        repo.create(&task1).unwrap();

        let mut task2 = create_test_task(TaskType::ExtractDataModel);
        task2.task_id = Uuid::new_v4().to_string();
        task2.status = TaskStatus::Completed;
        repo.create(&task2).unwrap();

        // List all
        let all = repo.list(None, None, None).unwrap();
        assert_eq!(all.len(), 2);

        // Filter by status
        let completed = repo.list(Some(TaskStatus::Completed), None, None).unwrap();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].task_type, TaskType::ExtractDataModel);
    }

    #[test]
    fn test_task_status_transitions() {
        let (_, repo) = setup();
        let task = create_test_task(TaskType::RepoAnalyze);
        repo.create(&task).unwrap();

        // pending -> in_progress (valid)
        repo.update_status(&task.task_id, TaskStatus::InProgress).unwrap();

        // in_progress -> completed (valid)
        repo.update_status(&task.task_id, TaskStatus::Completed).unwrap();

        // completed -> anything (invalid, trigger should block)
        let result = repo.update_status(&task.task_id, TaskStatus::Pending);
        // Should fail because trigger blocks completed -> other status
        assert!(result.is_err());
    }

    #[test]
    fn test_progress_cannot_decrease() {
        let (_, repo) = setup();
        let task = create_test_task(TaskType::RepoAnalyze);
        repo.create(&task).unwrap();

        repo.update_progress(&task.task_id, 0.5).unwrap();

        // Progress decreases should be blocked by trigger
        let result = repo.update_progress(&task.task_id, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn test_claim_task_priority_order() {
        let (_, repo) = setup();

        let mut low_priority = create_test_task(TaskType::RepoAnalyze);
        low_priority.priority = 1;
        repo.create(&low_priority).unwrap();

        let mut high_priority = create_test_task(TaskType::ExtractDataModel);
        high_priority.task_id = Uuid::new_v4().to_string();
        high_priority.priority = 10;
        repo.create(&high_priority).unwrap();

        // Should claim highest priority first
        let claimed = repo.claim("agent-1").unwrap().unwrap();
        assert_eq!(claimed.priority, 10);
        assert_eq!(claimed.task_type, TaskType::ExtractDataModel);
    }
}