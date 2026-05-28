//! S.DEF Importer - Convert S.DEF format to database.

use rusqlite::{params, Connection};
use std::sync::Mutex;
use tracing::instrument;

use crate::error::{DbError, DbResult};
use sdef_core::SoftwareDefinition;

/// S.DEF Importer.
pub struct SdefImporter {
    conn: Mutex<Connection>,
}

impl SdefImporter {
    /// Create a new importer.
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Mutex::new(conn),
        }
    }

    /// Import a complete S.DEF document into database.
    #[instrument(skip_all)]
    pub fn import(&self, sdef: &SoftwareDefinition) -> DbResult<String> {
        let conn = self.conn.lock().unwrap();

        // Get document name from metadata
        let doc_name = sdef.name.clone();

        // Create document record
        conn.execute(
            r#"INSERT INTO sdef_documents (name, version, description)
               VALUES (?1, ?2, ?3)
               ON CONFLICT(name) DO UPDATE SET version = ?2, description = ?3"#,
            params![doc_name, sdef.version, sdef.description],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        // Import data models
        if let Some(data_models) = &sdef.data_models {
            for model in data_models {
                let status = model
                    .status
                    .clone()
                    .unwrap_or_else(|| "active".to_string());

                conn.execute(
                    "INSERT INTO data_models (entity, document_name, status, version, description, logical_model)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        model.entity,
                        doc_name,
                        status,
                        model.version,
                        model.description,
                        model.logical_model,
                    ],
                )
                .map_err(|e| DbError::QueryFailed(e.to_string()))?;

                // Import attributes
                if let Some(attributes) = &model.attributes {
                    for attr in attributes {
                        let default_value = attr.default.as_ref().map(|v| {
                            serde_json::to_string(v).unwrap_or_default()
                        });
                        let constraints = attr.constraints.as_ref().map(|c| {
                            serde_json::to_string(c).unwrap_or_default()
                        });

                        conn.execute(
                            r#"INSERT INTO data_attributes (
                                document_name, entity, name, attr_type, format, description,
                                required, identity, generated, unique_flag, internal, deprecated,
                                default_value, constraints_json
                            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)"#,
                            params![
                                doc_name,
                                model.entity,
                                attr.name,
                                attr.attr_type,
                                attr.format,
                                attr.description,
                                attr.required,
                                attr.identity,
                                attr.generated,
                                attr.unique,
                                attr.internal,
                                attr.deprecated,
                                default_value,
                                constraints,
                            ],
                        )
                        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
                    }
                }
            }
        }

        // Import design decisions
        if let Some(decisions) = &sdef.design_decisions {
            for decision in decisions {
                conn.execute(
                    "INSERT INTO design_decisions (id, document_name, topic, decision, rationale, context, alternatives_json, consequences_json)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        decision.id,
                        doc_name,
                        decision.topic,
                        decision.decision,
                        decision.rationale,
                        decision.context,
                        serde_json::to_string(&decision.alternatives).ok(),
                        serde_json::to_string(&decision.consequences).ok(),
                    ],
                )
                .map_err(|e| DbError::QueryFailed(e.to_string()))?;
            }
        }

        // Import behavior (functions)
        if let Some(behavior) = &sdef.behavior {
            if let Some(functions) = &behavior.functions {
                for func in functions {
                    conn.execute(
                        "INSERT INTO function_specs (document_name, name, description, logic, complexity, pure_function)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![
                            doc_name,
                            func.name,
                            func.description,
                            func.logic,
                            func.complexity,
                            func.pure_function,
                        ],
                    )
                    .map_err(|e| DbError::QueryFailed(e.to_string()))?;
                }
            }
        }

        drop(conn);
        Ok(doc_name)
    }

    /// Import a single shard.
    #[instrument(skip_all)]
    pub fn import_shard(
        &self,
        shard_id: &str,
        document_name: &str,
        sdef_uri: &str,
        section_type: &str,
        _content_json: &str,
    ) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO shards (
                shard_id, document_name, sdef_uri, section_type, status
            ) VALUES (?1, ?2, ?3, ?4, 'generated')
               ON CONFLICT(shard_id) DO UPDATE SET
                   sdef_uri = ?3,
                   section_type = ?4"#,
            params![shard_id, document_name, sdef_uri, section_type],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }
}