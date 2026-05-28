//! Symbol registry repository for naming service operations.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::instrument;

use crate::error::{DbError, DbResult};

/// Symbol type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolType {
    Class,
    Interface,
    Function,
    Variable,
    Constant,
    Enum,
    Type,
}

impl SymbolType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Class => "class",
            Self::Interface => "interface",
            Self::Function => "function",
            Self::Variable => "variable",
            Self::Constant => "constant",
            Self::Enum => "enum",
            Self::Type => "type",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "class" => Some(Self::Class),
            "interface" => Some(Self::Interface),
            "function" => Some(Self::Function),
            "variable" => Some(Self::Variable),
            "constant" => Some(Self::Constant),
            "enum" => Some(Self::Enum),
            "type" => Some(Self::Type),
            _ => None,
        }
    }
}

/// Symbol registry entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEntry {
    pub id: Option<i64>,
    pub document_name: String,
    pub sdef_uri: String,
    pub language: String,
    pub symbol_type: SymbolType,
    pub concrete_name: String,
    pub is_user_defined: bool,
    pub created_at: Option<String>,
}

/// Resolution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionResult {
    pub sdef_uri: String,
    pub concrete_name: String,
    pub is_user_defined: bool,
}

/// Symbol repository.
pub struct SymbolRepository {
    conn: Arc<Mutex<Connection>>,
}

impl SymbolRepository {
    /// Create a new symbol repository.
    pub fn new(conn: Connection) -> Self {
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    /// Create from an existing Arc-wrapped connection.
    pub fn from_arc(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Register a symbol atomically.
    /// Returns the concrete name (either newly registered or existing).
    #[instrument(skip_all)]
    pub fn register(&self, entry: &SymbolEntry) -> DbResult<String> {
        let conn = self.conn.lock().unwrap();

        // Check if already exists
        let existing: Option<String> = conn
            .query_row(
                r#"SELECT concrete_name FROM symbol_registry
                   WHERE document_name = ?1 AND sdef_uri = ?2
                     AND language = ?3 AND symbol_type = ?4"#,
                params![
                    entry.document_name,
                    entry.sdef_uri,
                    entry.language,
                    entry.symbol_type.as_str(),
                ],
                |row| row.get(0),
            )
            .ok();

        if let Some(name) = existing {
            return Ok(name);
        }

        // Try to insert
        match conn.execute(
            r#"INSERT INTO symbol_registry (
                document_name, sdef_uri, language, symbol_type, concrete_name, is_user_defined
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"#,
            params![
                entry.document_name,
                entry.sdef_uri,
                entry.language,
                entry.symbol_type.as_str(),
                entry.concrete_name,
                entry.is_user_defined,
            ],
        ) {
            Ok(_) => Ok(entry.concrete_name.clone()),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(entry.concrete_name.clone()),
            Err(e) if e.to_string().contains("UNIQUE constraint failed") => {
                // Name conflict - generate suffix
                let base_name = &entry.concrete_name;
                for i in 1..100 {
                    let new_name = format!("{}_{}", base_name, i);
                    match conn.execute(
                        r#"INSERT INTO symbol_registry (
                            document_name, sdef_uri, language, symbol_type, concrete_name, is_user_defined
                        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"#,
                        params![
                            entry.document_name,
                            entry.sdef_uri,
                            entry.language,
                            entry.symbol_type.as_str(),
                            new_name,
                            entry.is_user_defined,
                        ],
                    ) {
                        Ok(_) => return Ok(new_name),
                        Err(_) => continue,
                    }
                }
                Err(DbError::ConstraintViolation("Failed to resolve name conflict".to_string()))
            }
            Err(e) => Err(DbError::QueryFailed(e.to_string())),
        }
    }

    /// Resolve a name: get concrete name from S.DEF URI.
    #[instrument(skip_all)]
    pub fn resolve(&self, document_name: &str, sdef_uri: &str, language: &str) -> DbResult<Option<String>> {
        let conn = self.conn.lock().unwrap();

        let result = conn
            .query_row(
                r#"SELECT concrete_name FROM symbol_registry
                   WHERE document_name = ?1 AND sdef_uri = ?2 AND language = ?3"#,
                params![document_name, sdef_uri, language],
                |row| row.get(0),
            )
            .ok();

        Ok(result)
    }

    /// Batch resolve names.
    #[instrument(skip_all)]
    pub fn batch_resolve(
        &self,
        document_name: &str,
        uris: &[(&str, SymbolType)],
        language: &str,
    ) -> DbResult<Vec<ResolutionResult>> {
        let conn = self.conn.lock().unwrap();
        let mut results = Vec::new();

        for (uri, symbol_type) in uris {
            if let Ok(name) = conn.query_row(
                r#"SELECT concrete_name, is_user_defined FROM symbol_registry
                   WHERE document_name = ?1 AND sdef_uri = ?2 AND language = ?3 AND symbol_type = ?4"#,
                params![document_name, uri, language, symbol_type.as_str()],
                |row| {
                    Ok(ResolutionResult {
                        sdef_uri: uri.to_string(),
                        concrete_name: row.get(0)?,
                        is_user_defined: row.get(1)?,
                    })
                },
            ) {
                results.push(name);
            }
        }

        Ok(results)
    }

    /// Register a custom (user-defined) name.
    #[instrument(skip_all)]
    pub fn register_custom(&self, entry: &SymbolEntry) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();

        // Check for duplicate concrete_name
        let exists = conn
            .query_row(
                r#"SELECT 1 FROM symbol_registry
                   WHERE document_name = ?1 AND language = ?2 AND concrete_name = ?3"#,
                params![entry.document_name, entry.language, entry.concrete_name],
                |row| row.get::<_, i32>(0),
            )
            .ok();

        if exists.is_some() {
            return Err(DbError::ConstraintViolation(format!(
                "Name '{}' already exists in {} for {}",
                entry.concrete_name, entry.language, entry.document_name
            )));
        }

        conn.execute(
            r#"INSERT INTO symbol_registry (
                document_name, sdef_uri, language, symbol_type, concrete_name, is_user_defined
            ) VALUES (?1, ?2, ?3, ?4, ?5, TRUE)"#,
            params![
                entry.document_name,
                entry.sdef_uri,
                entry.language,
                entry.symbol_type.as_str(),
                entry.concrete_name,
            ],
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        Ok(())
    }

    /// List symbols by language and optional type filter.
    #[instrument(skip_all)]
    pub fn list(
        &self,
        document_name: &str,
        language: &str,
        symbol_type: Option<SymbolType>,
    ) -> DbResult<Vec<SymbolEntry>> {
        let conn = self.conn.lock().unwrap();

        let query = match symbol_type {
            Some(_) => {
                "SELECT id, document_name, sdef_uri, language, symbol_type, concrete_name, is_user_defined, created_at
                 FROM symbol_registry
                 WHERE document_name = ?1 AND language = ?2 AND symbol_type = ?3
                 ORDER BY concrete_name"
            }
            None => {
                "SELECT id, document_name, sdef_uri, language, symbol_type, concrete_name, is_user_defined, created_at
                 FROM symbol_registry
                 WHERE document_name = ?1 AND language = ?2
                 ORDER BY concrete_name"
            }
        };

        let mut stmt = conn
            .prepare(query)
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

        let entries = match symbol_type {
            Some(st) => stmt
                .query_map(params![document_name, language, st.as_str()], |row| {
                    Ok(SymbolEntry {
                        id: row.get(0)?,
                        document_name: row.get(1)?,
                        sdef_uri: row.get(2)?,
                        language: row.get(3)?,
                        symbol_type: SymbolType::from_str(&row.get::<_, String>(4)?)
                            .unwrap_or(SymbolType::Variable),
                        concrete_name: row.get(5)?,
                        is_user_defined: row.get(6)?,
                        created_at: row.get(7)?,
                    })
                })
                .map_err(|e| DbError::QueryFailed(e.to_string()))?
                .filter_map(|r| r.ok())
                .collect(),
            None => stmt
                .query_map(params![document_name, language], |row| {
                    Ok(SymbolEntry {
                        id: row.get(0)?,
                        document_name: row.get(1)?,
                        sdef_uri: row.get(2)?,
                        language: row.get(3)?,
                        symbol_type: SymbolType::from_str(&row.get::<_, String>(4)?)
                            .unwrap_or(SymbolType::Variable),
                        concrete_name: row.get(5)?,
                        is_user_defined: row.get(6)?,
                        created_at: row.get(7)?,
                    })
                })
                .map_err(|e| DbError::QueryFailed(e.to_string()))?
                .filter_map(|r| r.ok())
                .collect(),
        };

        Ok(entries)
    }
}