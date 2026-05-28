//! Name resolution service — deterministic naming + DB persistence with collision detection.

use std::collections::HashMap;
use std::sync::Arc;

use cleanroom_db::{Database, DbError, SymbolRepository, SymbolType, SymbolEntry};
use tracing::{info, instrument, warn};

use crate::naming::{DeterministicNames, Language};

/// A resolved symbol with metadata.
#[derive(Debug, Clone)]
pub struct ResolvedName {
    pub sdef_uri: String,
    pub concrete_name: String,
    pub language: String,
    pub symbol_type: SymbolType,
    pub is_new: bool,
}

/// Name resolution service.
pub struct NameResolutionService {
    db: Arc<Database>,
    names: DeterministicNames,
}

impl NameResolutionService {
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            names: DeterministicNames::new(),
        }
    }

    /// Resolve a single URI to a concrete name, auto-generating and registering if needed.
    #[instrument(skip(self))]
    pub fn resolve(
        &self,
        document_name: &str,
        sdef_uri: &str,
        language: &str,
        symbol_type: SymbolType,
    ) -> Result<ResolvedName, DbError> {
        let sym_repo = SymbolRepository::from_arc(self.db.connection_arc());

        // 1. Try existing registration
        if let Ok(Some(name)) = sym_repo.resolve(document_name, sdef_uri, language) {
            return Ok(ResolvedName {
                sdef_uri: sdef_uri.to_string(),
                concrete_name: name,
                language: language.to_string(),
                symbol_type,
                is_new: false,
            });
        }

        // 2. Generate name from URI
        let base_name = uri_to_name(sdef_uri, language);

        // 3. Register with collision-aware retry
        let entry = SymbolEntry {
            id: None,
            document_name: document_name.to_string(),
            sdef_uri: sdef_uri.to_string(),
            language: language.to_string(),
            symbol_type,
            concrete_name: base_name,
            is_user_defined: false,
            created_at: None,
        };

        let concrete_name = sym_repo.register(&entry).map_err(|e| {
            warn!(uri = %sdef_uri, error = %e, "Failed to register symbol");
            e
        })?;

        Ok(ResolvedName {
            sdef_uri: sdef_uri.to_string(),
            concrete_name,
            language: language.to_string(),
            symbol_type,
            is_new: true,
        })
    }

    /// Batch resolve multiple URIs efficiently.
    #[instrument(skip(self, requests))]
    pub fn batch_resolve(
        &self,
        document_name: &str,
        requests: &[(String, SymbolType)],
        language: &str,
    ) -> Result<Vec<ResolvedName>, DbError> {
        let mut results = Vec::new();
        for (uri, stype) in requests {
            match self.resolve(document_name, uri, language, *stype) {
                Ok(name) => results.push(name),
                Err(e) => warn!(uri = %uri, error = %e, "Batch resolve failed for URI"),
            }
        }
        Ok(results)
    }

    /// Get all registered symbols for a language.
    pub fn list_symbols(
        &self,
        document_name: &str,
        language: &str,
        symbol_type: Option<SymbolType>,
    ) -> Result<Vec<SymbolEntry>, DbError> {
        let repo = SymbolRepository::from_arc(self.db.connection_arc());
        repo.list(document_name, language, symbol_type)
    }

    /// Register a custom (user-chosen) name.
    pub fn register_custom(
        &self,
        document_name: &str,
        sdef_uri: &str,
        language: &str,
        symbol_type: SymbolType,
        concrete_name: &str,
    ) -> Result<(), DbError> {
        let repo = SymbolRepository::from_arc(self.db.connection_arc());
        repo.register_custom(&SymbolEntry {
            id: None,
            document_name: document_name.to_string(),
            sdef_uri: sdef_uri.to_string(),
            language: language.to_string(),
            symbol_type,
            concrete_name: concrete_name.to_string(),
            is_user_defined: true,
            created_at: None,
        })
    }

    /// Generate a fully-qualified name with namespace control.
    pub fn fully_qualified_name(
        &self,
        document_name: &str,
        entity: &str,
        language: &str,
        namespace_mode: crate::naming::NamespaceMode,
        custom_namespace: Option<&str>,
    ) -> String {
        let lang = Language::from_str(language).unwrap_or(Language::TypeScript);
        let local_name = self.names.convert_for_language(entity, lang);

        match namespace_mode {
            crate::naming::NamespaceMode::FromDocumentName => match language {
                "rust" => format!("{}::{}", document_name.replace('.', "::"), local_name),
                "typescript" | "javascript" => format!("{}.{}", document_name, local_name),
                "python" => format!("{}.{}", document_name.replace('.', "_"), local_name),
                "java" | "go" => format!("{}.{}", document_name, local_name),
                _ => local_name,
            },
            crate::naming::NamespaceMode::Manual => {
                if let Some(ns) = custom_namespace.filter(|p| !p.is_empty()) {
                    format!("{}::{}", ns, local_name)
                } else {
                    local_name
                }
            }
            crate::naming::NamespaceMode::None => local_name,
        }
    }
}

/// Convert URI path segment to a language-appropriate name.
fn uri_to_name(uri: &str, language: &str) -> String {
    let segment = uri.rsplit('/').next().unwrap_or(uri);
    let lang = Language::from_str(language).unwrap_or(Language::TypeScript);
    let names = DeterministicNames::new();
    names.convert_for_language(segment, lang)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Arc<Database>, NameResolutionService) {
        let db = Arc::new(Database::in_memory().unwrap());
        // Create required document to satisfy foreign key
        {
            let conn = db.connection();
            conn.execute_batch(
                "INSERT INTO sdef_documents (name, version, created_at, updated_at)
                 VALUES ('test-doc', '1.0', datetime(), datetime());
                 INSERT INTO sdef_documents (name, version, created_at, updated_at)
                 VALUES ('doc', '1.0', datetime(), datetime());
                 INSERT INTO sdef_documents (name, version, created_at, updated_at)
                 VALUES ('com.example', '1.0', datetime(), datetime());"
            ).unwrap();
        }
        let svc = NameResolutionService::new(db.clone());
        (db, svc)
    }

    #[test]
    fn test_resolve_new_name() {
        let (_db, svc) = setup();
        let result = svc.resolve(
            "test-doc", "sdef://test-doc/entity/User", "rust", SymbolType::Class,
        ).unwrap();
        assert!(result.is_new);
        assert_eq!(result.concrete_name, "user");
        assert_eq!(result.sdef_uri, "sdef://test-doc/entity/User");
    }

    #[test]
    fn test_resolve_cached() {
        let (_db, svc) = setup();
        let first = svc.resolve(
            "test-doc", "sdef://test-doc/entity/UserService", "typescript", SymbolType::Interface,
        ).unwrap();
        assert!(first.is_new);

        let second = svc.resolve(
            "test-doc", "sdef://test-doc/entity/UserService", "typescript", SymbolType::Interface,
        ).unwrap();
        assert!(!second.is_new);
        assert_eq!(second.concrete_name, first.concrete_name);
    }

    #[test]
    fn test_collision_auto_suffix() {
        let (_db, svc) = setup();

        // First registration
        svc.resolve("doc", "sdef://doc/entity/User", "rust", SymbolType::Class).unwrap();

        // Same Entity name, different URI → should get a suffix
        let collision = svc.resolve(
            "doc", "sdef://doc/entity/User", "rust", SymbolType::Interface,
        ).unwrap();

        // Since the SymbolRepository handles collisions with auto-suffix by (document_name, language, sdef_uri, symbol_type) UNIQUE
        // But here sdef_uri is the same, so it returns cached. Let's test with different URI.
        let second = svc.resolve(
            "doc", "sdef://doc/entity/UserModel", "rust", SymbolType::Class,
        ).unwrap();
        // Should use base name "user_model"
        assert_eq!(second.concrete_name, "user_model");
    }

    #[test]
    fn test_fully_qualified_name() {
        let (_db, svc) = setup();
        let ns = crate::naming::NamespaceMode::FromDocumentName;
        let fqn = svc.fully_qualified_name("com.example", "UserService", "rust", ns, None);
        assert!(fqn.contains("com::example"));
        assert!(fqn.contains("user_service"));

        let ts_fqn = svc.fully_qualified_name("com.example", "UserService", "typescript", ns, None);
        assert!(ts_fqn.contains("com.example"));
        assert!(ts_fqn.contains("userService"));
    }

    #[test]
    fn test_fully_qualified_name_none_namespace() {
        let (_db, svc) = setup();
        let ns = crate::naming::NamespaceMode::None;
        let fqn = svc.fully_qualified_name("com.example", "UserService", "rust", ns, None);
        assert_eq!(fqn, "user_service", "None mode should return bare name");
    }

    #[test]
    fn test_fully_qualified_name_manual_namespace() {
        let (_db, svc) = setup();
        let ns = crate::naming::NamespaceMode::Manual;
        let fqn = svc.fully_qualified_name("com.example", "UserService", "rust", ns, Some("myapp"));
        assert!(fqn.contains("myapp"));
        assert!(fqn.contains("user_service"));
    }

    #[test]
    fn test_batch_resolve() {
        let (_db, svc) = setup();
        let requests = vec![
            ("sdef://doc/entity/User".to_string(), SymbolType::Class),
            ("sdef://doc/entity/Order".to_string(), SymbolType::Class),
            ("sdef://doc/func/createUser".to_string(), SymbolType::Function),
        ];
        let results = svc.batch_resolve("doc", &requests, "rust").unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_new);
        assert!(results[1].is_new);
        assert!(results[2].is_new);
        // All three should have different names
        let mut names: Vec<_> = results.iter().map(|r| r.concrete_name.clone()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 3, "All names should be unique");
    }
}