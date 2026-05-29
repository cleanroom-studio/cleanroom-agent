//! Consumer Agent — generates code from S.DEF.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::io::Write;

use tracing::info;
use rusqlite::params;

use cleanroom_db::{Database, DbError, Task, TaskRepository, TaskType};

pub mod code_generator;
use code_generator::{create_generator, GeneratedCode};

/// Compatibility mode for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityMode {
    Full,
    Mixed,
    Clean,
    Custom,
}

impl Default for CompatibilityMode {
    fn default() -> Self { Self::Mixed }
}

/// Fidelity level for reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fidelity {
    High,
    Medium,
    Low,
}

impl Default for Fidelity {
    fn default() -> Self { Self::Medium }
}

/// Consumer configuration.
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    pub language: String,
    pub framework: Option<String>,
    pub compatibility_mode: CompatibilityMode,
    pub fidelity: Fidelity,
    pub output_path: PathBuf,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            language: "typescript".to_string(),
            framework: None,
            compatibility_mode: CompatibilityMode::default(),
            fidelity: Fidelity::default(),
            output_path: PathBuf::from("./generated"),
        }
    }
}

/// Consumer Agent — generates code from S.DEF.
pub struct ConsumerAgent {
    config: ConsumerConfig,
    db: Arc<Database>,
    agent_id: String,
}

impl ConsumerAgent {
    pub fn new(config: ConsumerConfig, db: Arc<Database>) -> Self {
        let agent_id = format!("consumer-{}", uuid::Uuid::new_v4());
        Self { config, db, agent_id }
    }

    pub fn agent_id(&self) -> &str { &self.agent_id }

    /// Generate code from S.DEF stored in the database.
    pub async fn generate_code(&self) -> Result<(), DbError> {
        info!(language = %self.config.language, output = %self.config.output_path.display(), "Starting code generation");

        // 1. Create the code generator
        let generator = match create_generator(&self.config.language) {
            Some(g) => g,
            None => {
                return Err(DbError::QueryFailed(format!(
                    "Unsupported language: {}. Supported: rust, typescript, python", self.config.language
                )));
            }
        };

        // 2. Ensure output directory exists
        fs::create_dir_all(&self.config.output_path)
            .map_err(|e| DbError::QueryFailed(format!("Failed to create output dir: {}", e)))?;

        // 3. Read documents and data models from DB
        let mut total_files = 0;
        let documents = self.read_documents()?;
        info!(count = documents.len(), "Documents found");

        for doc_name in &documents {
            let models = self.read_data_models(doc_name)?;
            info!(document = %doc_name, models = models.len(), "Generating code");

            // 4. Generate code for each data model
            for model in &models {
                let files = generator.generate_data_model(model);
                for file in files {
                    self.write_code_file(&file)?;
                    total_files += 1;
                }
            }

            // 5. Generate code for contracts (interfaces)
            let contracts = self.read_contracts(doc_name)?;
            for contract in &contracts {
                let files = generator.generate_interface(contract);
                for file in files {
                    self.write_code_file(&file)?;
                    total_files += 1;
                }
            }
        }

        info!(files = total_files, language = %self.config.language, "Code generation complete");
        Ok(())
    }

    /// Read document names from the database.
    fn read_documents(&self) -> Result<Vec<String>, DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare("SELECT name FROM sdef_documents ORDER BY name")
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        let mut rows = stmt.query([])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        let mut names = Vec::new();
        while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
            names.push(row.get::<_, String>(0).map_err(|e| DbError::QueryFailed(e.to_string()))?);
        }
        drop(rows);
        drop(stmt);
        drop(conn);
        Ok(names)
    }

    /// Read data models from the database.
    fn read_data_models(&self, document_name: &str) -> Result<Vec<sdef_core::DataModel>, DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare(
            "SELECT entity, description, version, logical_model FROM data_models WHERE document_name = ?1"
        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut rows = stmt.query(params![document_name])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut entities = Vec::new();
        while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
            entities.push((
                row.get::<_, String>(0).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                row.get::<_, Option<String>>(1).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                row.get::<_, Option<String>>(2).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                row.get::<_, Option<String>>(3).map_err(|e| DbError::QueryFailed(e.to_string()))?,
            ));
        }
        drop(rows);
        drop(stmt);
        drop(conn);

        let mut models = Vec::new();
        for (entity, description, version, logical_model) in entities {
            let attrs = self.read_attributes(document_name, &entity)?;
            models.push(sdef_core::DataModel {
                entity,
                status: None,
                version,
                deprecated: None,
                description,
                logical_model,
                attributes: if attrs.is_empty() { None } else { Some(attrs) },
                relationships: None,
                validation_rules: None,
                physical_design: None,
            });
        }
        Ok(models)
    }

    /// Read attributes for a data model.
    fn read_attributes(&self, document_name: &str, entity: &str) -> Result<Vec<sdef_core::DataAttribute>, DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare(
            "SELECT name, attr_type, format, description, required, identity, generated, unique_flag, internal, deprecated
             FROM data_attributes WHERE document_name = ?1 AND entity = ?2"
        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut rows = stmt.query(params![document_name, entity])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut attrs = Vec::new();
        while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
            attrs.push(sdef_core::DataAttribute {
                name: row.get(0).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                attr_type: row.get(1).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                format: row.get(2).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                description: row.get(3).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                required: row.get(4).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                identity: row.get(5).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                generated: row.get(6).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                unique: row.get(7).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                internal: row.get(8).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                deprecated: row.get(9).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                default: None,
                compatibility: None,
                constraints: None,
            });
        }
        drop(rows);
        drop(stmt);
        drop(conn);
        Ok(attrs)
    }

    /// Read interface contracts from the database.
    fn read_contracts(&self, document_name: &str) -> Result<Vec<sdef_core::InterfaceContract>, DbError> {
        let conn = self.db.connection();
        let mut stmt = conn.prepare(
            "SELECT name, description, is_abstract FROM contracts
             WHERE document_name = ?1 AND contract_type = 'interface'"
        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut rows = stmt.query(params![document_name])
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let mut contracts = Vec::new();
        while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
            contracts.push(sdef_core::InterfaceContract {
                name: row.get(0).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                is_abstract: row.get::<_, bool>(2).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                status: Some("active".to_string()),
                version: None,
                deprecated: None,
                description: row.get(1).map_err(|e| DbError::QueryFailed(e.to_string()))?,
                methods: None,
                invariants: None,
            });
        }
        drop(rows);
        drop(stmt);
        drop(conn);
        Ok(contracts)
    }

    /// Write a generated code file to disk.
    fn write_code_file(&self, code: &GeneratedCode) -> Result<(), DbError> {
        let file_path = self.config.output_path.join(&code.file_path);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| DbError::QueryFailed(format!("Failed to create dir: {}", e)))?;
        }
        let mut file = fs::File::create(&file_path)
            .map_err(|e| DbError::QueryFailed(format!("Failed to create file: {}", e)))?;
        file.write_all(code.content.as_bytes())
            .map_err(|e| DbError::QueryFailed(format!("Failed to write file: {}", e)))?;
        info!(path = %file_path.display(), "Generated file");
        Ok(())
    }

    /// Process a generation task.
    pub async fn process_next_task(&self) -> Result<Option<Task>, DbError> {
        let repo = TaskRepository::new(self.db.connection_arc());
        if let Some(task) = repo.claim(&self.agent_id)? {
            info!(task_id = %task.task_id, task_type = ?task.task_type, "Processing task");
            match task.task_type {
                TaskType::GenerateCode => self.generate_code().await?,
                TaskType::MergeCode => self.merge_code(&task).await?,
                TaskType::RunTests => self.run_tests(&task).await?,
                _ => { repo.complete(&task.task_id, "{}")?; }
            }
            return Ok(Some(task));
        }
        Ok(None)
    }

    async fn merge_code(&self, _task: &Task) -> Result<(), DbError> {
        Ok(())
    }

    async fn run_tests(&self, _task: &Task) -> Result<(), DbError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db() -> Arc<Database> {
        let db = Arc::new(Database::in_memory().unwrap());
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, version, description, created_at, updated_at)
                 VALUES ('test-proj', '0.1.0', 'A test', datetime(), datetime());
                 INSERT INTO data_models (entity, document_name, status, description)
                 VALUES ('User', 'test-proj', 'active', 'A system user');
                 INSERT INTO data_attributes (document_name, entity, name, attr_type, description, required, identity, generated, unique_flag)
                 VALUES ('test-proj', 'User', 'id', 'UUID', 'Primary key', 1, 1, 1, 1);
                 INSERT INTO data_attributes (document_name, entity, name, attr_type, description, required)
                 VALUES ('test-proj', 'User', 'email', 'string', 'Email address', 1);"
            ).unwrap();
        }
        db
    }

    #[tokio::test]
    async fn test_generate_code_typescript() {
        let db = setup_db();
        let tmpdir = std::env::temp_dir().join("cleanroom_test_consumer_ts");
        let _ = std::fs::remove_dir_all(&tmpdir);

        let config = ConsumerConfig {
            language: "typescript".to_string(),
            output_path: tmpdir.clone(),
            ..ConsumerConfig::default()
        };
        let agent = ConsumerAgent::new(config, db);
        agent.generate_code().await.unwrap();

        // Check that files were generated
        let entries = std::fs::read_dir(&tmpdir).unwrap();
        let count = entries.count();
        assert!(count > 0, "Should generate at least one file");

        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[tokio::test]
    async fn test_generate_code_rust() {
        let db = setup_db();
        let tmpdir = std::env::temp_dir().join("cleanroom_test_consumer_rs");
        let _ = std::fs::remove_dir_all(&tmpdir);

        let config = ConsumerConfig {
            language: "rust".to_string(),
            output_path: tmpdir.clone(),
            ..ConsumerConfig::default()
        };
        let agent = ConsumerAgent::new(config, db);
        agent.generate_code().await.unwrap();

        let entries = std::fs::read_dir(&tmpdir).unwrap();
        let count = entries.count();
        assert!(count > 0, "Should generate at least one file");

        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[tokio::test]
    async fn test_unsupported_language() {
        let db = setup_db();
        let config = ConsumerConfig {
            language: "brainfuck".to_string(),
            ..ConsumerConfig::default()
        };
        let agent = ConsumerAgent::new(config, db);
        let result = agent.generate_code().await;
        assert!(result.is_err());
    }
}