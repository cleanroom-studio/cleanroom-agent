//! S.DEF Exporter - Convert database to S.DEF format.

use rusqlite::{params, Connection};
use std::sync::Mutex;
use tracing::instrument;

use crate::error::{DbError, DbResult};
use crate::repositories::*;
use sdef_core::SoftwareDefinition;

/// S.DEF Exporter.
pub struct SdefExporter {
    conn: Mutex<Connection>,
}

impl SdefExporter {
    /// Create a new exporter.
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Mutex::new(conn),
        }
    }

    /// Export a complete S.DEF document from database.
    #[instrument(skip_all)]
    pub fn export(&self, document_name: &str) -> DbResult<SoftwareDefinition> {
        let conn = self.conn.lock().unwrap();

        // Get document metadata
        let mut doc_stmt = conn
            .prepare("SELECT name, version, description, created_at, updated_at FROM sdef_documents WHERE name = ?1")
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let doc: SdefDocument = doc_stmt
            .query_row(params![document_name], |row| {
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
                    value: document_name.to_string(),
                },
                _ => DbError::QueryFailed(e.to_string()),
            })?;

        drop(doc_stmt);

        // Build SoftwareDefinition
        let mut sdef = SoftwareDefinition::default();
        sdef.name = doc.name.clone();
        sdef.version = doc.version.clone();
        sdef.description = doc.description.clone();

        // Metadata
        sdef.metadata = Some(sdef_core::SoftwareMetadata {
            authors: None,
            license: None,
            homepage: None,
            repository: None,
            category: None,
            tags: None,
            target_platforms: None,
            compatibility_policy: None,
            annotations: None,
        });

        // Data models
        sdef.data_models = Some(self.export_data_models(&conn, document_name)?);

        // Design decisions
        sdef.design_decisions = Some(self.export_design_decisions(&conn, document_name)?);

        // Behavior (functions)
        sdef.behavior = Some(sdef_core::Behavior {
            functions: Some(self.export_functions(&conn, document_name)?),
            flows: None,
            state_machines: None,
        });

        Ok(sdef)
    }

    fn export_data_models(
        &self,
        conn: &Connection,
        document_name: &str,
    ) -> DbResult<Vec<sdef_core::DataModel>> {
        let mut stmt = conn
            .prepare("SELECT entity, status, version, description, logical_model FROM data_models WHERE document_name = ?1")
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let models: Vec<(String, String, Option<String>, Option<String>, Option<String>)> = stmt
            .query_map(params![document_name], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        let mut entities = Vec::new();
        for (entity, status, version, description, logical_model) in models {
            let mut attr_stmt = conn
                .prepare(
                    "SELECT name, attr_type, format, description, required, identity, generated,
                            unique_flag, internal, deprecated, default_value, constraints_json
                     FROM data_attributes WHERE document_name = ?1 AND entity = ?2",
                )
                .map_err(|e| DbError::QueryFailed(e.to_string()))?;

            let attrs: Vec<DataAttribute> = attr_stmt
                .query_map(params![document_name, &entity], |row| {
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

            drop(attr_stmt);

            entities.push(sdef_core::DataModel {
                entity,
                status: if status == "deprecated" || status == "legacy" {
                    Some(status)
                } else {
                    None
                },
                version,
                deprecated: None,
                description,
                logical_model,
                attributes: if attrs.is_empty() {
                    None
                } else {
                    Some(
                        attrs
                            .into_iter()
                            .map(|a| sdef_core::DataAttribute {
                                name: a.name,
                                attr_type: a.attr_type,
                                format: a.format,
                                description: a.description,
                                required: a.required,
                                default: a.default_value.and_then(|v| {
                                    serde_json::from_str(&v).ok()
                                }),
                                identity: a.identity,
                                generated: a.generated,
                                unique: a.unique_flag,
                                internal: a.internal,
                                deprecated: a.deprecated,
                                compatibility: None,
                                constraints: a.constraints_json.and_then(|c| {
                                    serde_json::from_str(&c).ok()
                                }),
                                origin: None,
                            })
                            .collect(),
                    )
                },
                relationships: None,
                validation_rules: None,
                physical_design: None,
                origin: None,
            });
        }

        Ok(entities)
    }

    fn export_functions(
        &self,
        conn: &Connection,
        document_name: &str,
    ) -> DbResult<Vec<sdef_core::FunctionSpec>> {
        let mut stmt = conn
            .prepare(
                "SELECT name, description, logic, complexity, pure_function FROM function_specs WHERE document_name = ?1",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let functions = stmt
            .query_map(params![document_name], |row| {
                Ok(sdef_core::FunctionSpec {
                    name: row.get(0)?,
                    description: row.get(1)?,
                    inputs: None,
                    outputs: None,
                    logic: row.get(2)?,
                    complexity: row.get(3)?,
                    pure_function: row.get(4)?,
                    edge_cases: None,
                    origin: None,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(functions)
    }

    fn export_design_decisions(
        &self,
        conn: &Connection,
        document_name: &str,
    ) -> DbResult<Vec<sdef_core::DesignDecision>> {
        let mut stmt = conn
            .prepare(
                "SELECT id, topic, decision, rationale, context, alternatives_json, consequences_json
                 FROM design_decisions WHERE document_name = ?1",
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let decisions = stmt
            .query_map(params![document_name], |row| {
                Ok(sdef_core::DesignDecision {
                    id: row.get(0)?,
                    topic: row.get(1)?,
                    decision: row.get(2)?,
                    rationale: row.get(3)?,
                    context: row.get(4)?,
                    alternatives: row
                        .get::<_, Option<String>>(5)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    consequences: row
                        .get::<_, Option<String>>(6)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    constraints: None,
                })
            })
            .map_err(|e| DbError::QueryFailed(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(decisions)
    }

    /// Export a single shard content hash.
    #[instrument(skip_all)]
    pub fn export_shard(&self, sdef_uri: &str) -> DbResult<Option<String>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn
            .prepare("SELECT content_hash FROM shards WHERE sdef_uri = ?1")
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let result = stmt
            .query_row(params![sdef_uri], |row| row.get(0))
            .ok();

        Ok(result)
    }
}