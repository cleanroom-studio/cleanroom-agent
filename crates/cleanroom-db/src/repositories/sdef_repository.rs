//! S.DEF repository for S.DEF entity operations.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::instrument;

use crate::error::{DbError, DbResult};

/// S.DEF Document model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdefDocument {
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Data Model entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataModel {
    pub entity: String,
    pub document_name: String,
    pub status: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub logical_model: Option<String>,
}

/// Data Attribute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAttribute {
    pub id: Option<i64>,
    pub document_name: String,
    pub entity: String,
    pub name: String,
    pub attr_type: String,
    pub format: Option<String>,
    pub description: Option<String>,
    pub required: bool,
    pub identity: bool,
    pub generated: bool,
    pub unique_flag: bool,
    pub internal: bool,
    pub deprecated: bool,
    pub default_value: Option<String>,
    pub constraints_json: Option<String>,
}

/// Contract entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    pub name: String,
    pub document_name: String,
    pub contract_type: String,
    pub status: String,
    pub version: Option<String>,
    pub is_abstract: bool,
    pub description: Option<String>,
    pub implements_json: Option<String>,
    pub dependencies_json: Option<String>,
    pub invariants_json: Option<String>,
    pub http_method: Option<String>,
    pub api_path: Option<String>,
    pub auth: Option<String>,
    pub rate_limit: Option<String>,
}

/// Function specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSpec {
    pub id: Option<i64>,
    pub document_name: String,
    pub name: String,
    pub description: Option<String>,
    pub logic: Option<String>,
    pub complexity: Option<String>,
    pub pure_function: bool,
}

/// Design-decision record — corresponds to a row in `design_decisions`.
/// Used by the Phase 0.5 LLM writer path (`llm_sdef_parser`) to persist
/// design rationales extracted from `LlmAnalyzeFile` output.
///
/// Note: `id` is `TEXT NOT NULL` (not autoincrement INTEGER), so the
/// caller must always supply a stable id (the writer generates
/// `dd-<uuid>` if absent). `rationale` is also `NOT NULL` in the schema,
/// so the writer uses an empty string when the LLM omits it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignDecisionRecord {
    pub id: String,
    pub document_name: String,
    pub topic: String,
    pub decision: String,
    pub rationale: String,
    pub context: Option<String>,
    pub alternatives_json: Option<String>,
    pub consequences_json: Option<String>,
}

/// UI Document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiDocument {
    pub document_name: String,
    pub pen_version: Option<String>,
    pub raw_content_json: String,
}

/// UI Screen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiScreen {
    pub id: String,
    pub document_name: String,
    pub name: String,
    pub route: Option<String>,
    pub purpose: Option<String>,
    pub layout_description: Option<String>,
}

/// S.DEF repository.
pub struct SdefRepository {
    conn: Arc<Mutex<Connection>>,
}

impl SdefRepository {
    /// Create a new S.DEF repository from an owned connection.
    /// Wraps the connection in an `Arc<Mutex<...>>` so the same handle
    /// can be shared across threads (Phase 0.5 LLM writer path needs this).
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    /// Create a new S.DEF repository from a shared `Arc<Mutex<Connection>>`.
    /// Use this when you already have a `Database::connection_arc()` and
    /// don't want to open a new connection just to write a few rows.
    pub fn new_with_arc(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    // ============ Document Operations ============

    /// Create or update a document.
    #[instrument(skip_all)]
    pub fn upsert_document(&self, doc: &SdefDocument) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO sdef_documents (name, version, description, created_at, updated_at)
               VALUES (?1, ?2, ?3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
               ON CONFLICT(name) DO UPDATE SET
                   version = ?2,
                   description = ?3,
                   updated_at = CURRENT_TIMESTAMP"#,
            params![doc.name, doc.version, doc.description],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Get a document by name.
    #[instrument(skip_all)]
    pub fn get_document(&self, name: &str) -> DbResult<SdefDocument> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT name, version, description, created_at, updated_at FROM sdef_documents WHERE name = ?1",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![name], |row| {
            Ok(SdefDocument {
                name: row.get(0)?,
                version: row.get(1)?,
                description: row.get(2)?,
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "document",
                field: "name",
                value: name.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    /// List all documents.
    #[instrument(skip_all)]
    pub fn list_documents(&self) -> DbResult<Vec<SdefDocument>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT name, version, description, created_at, updated_at FROM sdef_documents ORDER BY name",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let docs = stmt
            .query_map([], |row| {
                Ok(SdefDocument {
                    name: row.get(0)?,
                    version: row.get(1)?,
                    description: row.get(2)?,
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(docs)
    }

    // ============ Data Model Operations ============

    /// Create a data model.
    #[instrument(skip_all)]
    pub fn create_data_model(&self, model: &DataModel) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO data_models (entity, document_name, status, version, description, logical_model)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6)"#,
            params![
                model.entity,
                model.document_name,
                model.status,
                model.version,
                model.description,
                model.logical_model,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Get data model with attributes.
    #[instrument(skip_all)]
    pub fn get_data_model(&self, document_name: &str, entity: &str) -> DbResult<(DataModel, Vec<DataAttribute>)> {
        let conn = self.conn.lock().unwrap();

        let mut model_stmt = conn
            .prepare(
                "SELECT entity, document_name, status, version, description, logical_model
                 FROM data_models WHERE document_name = ?1 AND entity = ?2",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let model = model_stmt
            .query_row(params![document_name, entity], |row| {
                Ok(DataModel {
                    entity: row.get(0)?,
                    document_name: row.get(1)?,
                    status: row.get(2)?,
                    version: row.get(3)?,
                    description: row.get(4)?,
                    logical_model: row.get(5)?,
                })
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                    resource: "data_model",
                    field: "entity",
                    value: entity.to_string(),
                },
                _ => DbError::QueryFailed(e.to_string()),
            })?;

        drop(model_stmt);

        let mut attr_stmt = conn
            .prepare(
                "SELECT id, document_name, entity, name, attr_type, format, description,
                        required, identity, generated, unique_flag, internal, deprecated,
                        default_value, constraints_json
                 FROM data_attributes WHERE document_name = ?1 AND entity = ?2",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let attributes = attr_stmt
            .query_map(params![document_name, entity], |row| {
                Ok(DataAttribute {
                    id: row.get(0)?,
                    document_name: row.get(1)?,
                    entity: row.get(2)?,
                    name: row.get(3)?,
                    attr_type: row.get(4)?,
                    format: row.get(5)?,
                    description: row.get(6)?,
                    required: row.get(7)?,
                    identity: row.get(8)?,
                    generated: row.get(9)?,
                    unique_flag: row.get(10)?,
                    internal: row.get(11)?,
                    deprecated: row.get(12)?,
                    default_value: row.get(13)?,
                    constraints_json: row.get(14)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok((model, attributes))
    }

    /// Create a data attribute.
    #[instrument(skip_all)]
    pub fn create_data_attribute(&self, attr: &DataAttribute) -> DbResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO data_attributes (
                document_name, entity, name, attr_type, format, description,
                required, identity, generated, unique_flag, internal, deprecated,
                default_value, constraints_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)"#,
            params![
                attr.document_name,
                attr.entity,
                attr.name,
                attr.attr_type,
                attr.format,
                attr.description,
                attr.required,
                attr.identity,
                attr.generated,
                attr.unique_flag,
                attr.internal,
                attr.deprecated,
                attr.default_value,
                attr.constraints_json,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        Ok(conn.last_insert_rowid())
    }

    // ============ Contract Operations ============

    /// Create a contract.
    #[instrument(skip_all)]
    pub fn create_contract(&self, contract: &Contract) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO contracts (
                name, document_name, contract_type, status, version, is_abstract,
                description, implements_json, dependencies_json, invariants_json,
                http_method, api_path, auth, rate_limit
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)"#,
            params![
                contract.name,
                contract.document_name,
                contract.contract_type,
                contract.status,
                contract.version,
                contract.is_abstract,
                contract.description,
                contract.implements_json,
                contract.dependencies_json,
                contract.invariants_json,
                contract.http_method,
                contract.api_path,
                contract.auth,
                contract.rate_limit,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Get a contract by name.
    #[instrument(skip_all)]
    pub fn get_contract(&self, document_name: &str, name: &str) -> DbResult<Contract> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT name, document_name, contract_type, status, version, is_abstract,
                   description, implements_json, dependencies_json, invariants_json,
                   http_method, api_path, auth, rate_limit
                   FROM contracts WHERE document_name = ?1 AND name = ?2"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![document_name, name], |row| {
            Ok(Contract {
                name: row.get(0)?,
                document_name: row.get(1)?,
                contract_type: row.get(2)?,
                status: row.get(3)?,
                version: row.get(4)?,
                is_abstract: row.get(5)?,
                description: row.get(6)?,
                implements_json: row.get(7)?,
                dependencies_json: row.get(8)?,
                invariants_json: row.get(9)?,
                http_method: row.get(10)?,
                api_path: row.get(11)?,
                auth: row.get(12)?,
                rate_limit: row.get(13)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "contract",
                field: "name",
                value: name.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    // ============ UI Operations ============

    /// Create a UI document.
    #[instrument(skip_all)]
    pub fn create_ui_document(&self, doc: &UiDocument) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO ui_documents (document_name, pen_version, raw_content_json) VALUES (?1, ?2, ?3)",
            params![doc.document_name, doc.pen_version, doc.raw_content_json],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Get UI document.
    #[instrument(skip_all)]
    pub fn get_ui_document(&self, document_name: &str) -> DbResult<UiDocument> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT document_name, pen_version, raw_content_json FROM ui_documents WHERE document_name = ?1",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![document_name], |row| {
            Ok(UiDocument {
                document_name: row.get(0)?,
                pen_version: row.get(1)?,
                raw_content_json: row.get(2)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "ui_document",
                field: "document_name",
                value: document_name.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    // ============ FTS Search ============

    /// Full-text search across S.DEF documents.
    #[instrument(skip_all)]
    pub fn search(&self, query: &str) -> DbResult<Vec<SdefDocument>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"SELECT d.name, d.version, d.description, d.created_at, d.updated_at
                   FROM sdef_documents d
                   JOIN sdef_fts f ON d.rowid = f.rowid
                   WHERE sdef_fts MATCH ?1"#,
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let docs = stmt
            .query_map(params![query], |row| {
                Ok(SdefDocument {
                    name: row.get(0)?,
                    version: row.get(1)?,
                    description: row.get(2)?,
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(docs)
    }

    // ============ Function Spec Operations ============

    /// Get a function spec by name.
    #[instrument(skip_all)]
    pub fn get_function_spec(&self, document_name: &str, name: &str) -> DbResult<FunctionSpec> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, document_name, name, description, logic, complexity, pure_function
                 FROM function_specs WHERE document_name = ?1 AND name = ?2",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![document_name, name], |row| {
            Ok(FunctionSpec {
                id: row.get(0)?,
                document_name: row.get(1)?,
                name: row.get(2)?,
                description: row.get(3)?,
                logic: row.get(4)?,
                complexity: row.get(5)?,
                pure_function: row.get(6)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "function_spec",
                field: "name",
                value: name.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    /// List function specs for a document.
    #[instrument(skip_all)]
    pub fn list_function_specs(&self, document_name: &str) -> DbResult<Vec<FunctionSpec>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, document_name, name, description, logic, complexity, pure_function
                 FROM function_specs WHERE document_name = ?1 ORDER BY name",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let funcs = stmt
            .query_map(params![document_name], |row| {
                Ok(FunctionSpec {
                    id: row.get(0)?,
                    document_name: row.get(1)?,
                    name: row.get(2)?,
                    description: row.get(3)?,
                    logic: row.get(4)?,
                    complexity: row.get(5)?,
                    pure_function: row.get(6)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(funcs)
    }

    /// Create a function spec (Phase 0.5 writer path).
    /// Returns the new row id. `id` on the input is ignored.
    #[instrument(skip_all)]
    pub fn create_function_spec(&self, spec: &FunctionSpec) -> DbResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO function_specs (document_name, name, description, logic, complexity, pure_function)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6)"#,
            params![
                spec.document_name,
                spec.name,
                spec.description,
                spec.logic,
                spec.complexity,
                spec.pure_function,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(conn.last_insert_rowid())
    }

    // ============ Design Decision Operations ============

    /// Create a design-decision row (Phase 0.5 writer path).
    /// `id` and `rationale` are NOT NULL in the schema; caller must
    /// supply both (the writer generates `dd-<uuid>` and an empty
    /// rationale fallback).
    #[instrument(skip_all)]
    pub fn create_design_decision(&self, dd: &DesignDecisionRecord) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO design_decisions
               (id, document_name, topic, decision, rationale, context, alternatives_json, consequences_json)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
            params![
                dd.id,
                dd.document_name,
                dd.topic,
                dd.decision,
                dd.rationale,
                dd.context,
                dd.alternatives_json,
                dd.consequences_json,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    // ============ UI Screen Operations ============

    /// Get a UI screen by ID.
    #[instrument(skip_all)]
    pub fn get_ui_screen(&self, document_name: &str, screen_id: &str) -> DbResult<UiScreen> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, document_name, name, route, purpose, layout_description
                 FROM ui_screens WHERE document_name = ?1 AND id = ?2",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        stmt.query_row(params![document_name, screen_id], |row| {
            Ok(UiScreen {
                id: row.get(0)?,
                document_name: row.get(1)?,
                name: row.get(2)?,
                route: row.get(3)?,
                purpose: row.get(4)?,
                layout_description: row.get(5)?,
            })
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => DbError::NotFound {
                resource: "ui_screen",
                field: "id",
                value: screen_id.to_string(),
            },
            _ => DbError::QueryFailed(e.to_string()),
        })
    }

    /// List UI screens for a document.
    #[instrument(skip_all)]
    pub fn list_ui_screens(&self, document_name: &str) -> DbResult<Vec<UiScreen>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, document_name, name, route, purpose, layout_description
                 FROM ui_screens WHERE document_name = ?1 ORDER BY name",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let screens = stmt
            .query_map(params![document_name], |row| {
                Ok(UiScreen {
                    id: row.get(0)?,
                    document_name: row.get(1)?,
                    name: row.get(2)?,
                    route: row.get(3)?,
                    purpose: row.get(4)?,
                    layout_description: row.get(5)?,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(screens)
    }
}