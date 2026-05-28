//! IR → S.DEF mapper — converts code analysis results into S.DEF entities.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use sdef_core::*;
use tracing::{info, instrument};
use chrono::Utc;

use crate::module_partitioner::{Module, ModuleType};
use crate::naming::{DeterministicNames, Language as Lang};
use crate::repo_scanner::SourceFile;
use cleanroom_db::{Database, DbError, SymbolRepository, SymbolType, SymbolEntry};
use cleanroom_db::repositories::sdef_repository as sdef_repo;

/// Configuration for the S.DEF mapper.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Document name (project name).
    pub document_name: String,
    /// Software version.
    pub version: String,
    /// Project description.
    pub description: Option<String>,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            document_name: "unnamed".to_string(),
            version: "0.1.0".to_string(),
            description: None,
        }
    }
}

/// Intermediate representation of a code entity.
#[derive(Debug, Clone)]
pub enum IrEntity {
    /// A data model / entity.
    DataModel {
        name: String,
        description: Option<String>,
        attributes: Vec<IrAttribute>,
    },
    /// An interface / trait.
    Interface {
        name: String,
        description: Option<String>,
        methods: Vec<IrMethod>,
    },
    /// A function.
    Function {
        name: String,
        description: Option<String>,
        inputs: Vec<IrParam>,
        outputs: Vec<IrParam>,
    },
}

/// Attribute in IR.
#[derive(Debug, Clone)]
pub struct IrAttribute {
    pub name: String,
    pub attr_type: String,
    pub description: Option<String>,
    pub required: bool,
}

/// Method in IR.
#[derive(Debug, Clone)]
pub struct IrMethod {
    pub name: String,
    pub params: Vec<IrParam>,
}

/// Parameter in IR.
#[derive(Debug, Clone)]
pub struct IrParam {
    pub name: String,
    pub param_type: String,
    pub description: Option<String>,
}

/// Maps code analysis results into S.DEF entities.
pub struct SdefMapper {
    config: MapperConfig,
    names: DeterministicNames,
    db: Arc<Database>,
}

impl SdefMapper {
    /// Create a new mapper.
    pub fn new(config: MapperConfig, db: Arc<Database>) -> Self {
        Self {
            config,
            names: DeterministicNames::new(),
            db,
        }
    }

    /// Run the full mapping pipeline: scan → partition → analyze → persist.
    #[instrument(skip(self, files))]
    pub async fn map_all(
        &self,
        files: Vec<SourceFile>,
        modules: Vec<Module>,
    ) -> Result<(SoftwareDefinition, Vec<Vec<String>>), DbError> {
        let mut all_entities: Vec<IrEntity> = Vec::new();
        let mut cycle_chain: Vec<Vec<String>> = Vec::new();

        for module in &modules {
            let entities = self.analyze_module(module);
            all_entities.extend(entities);
        }

        for file in &files {
            let entities = self.analyze_file(file);
            all_entities.extend(entities);
        }

        // Build SoftwareDefinition
        let sdef = self.build_sdef(&all_entities, &modules, &files)?;

        // Persist
        self.persist_to_db(&sdef, &files, &all_entities, &mut cycle_chain)?;

        Ok((sdef, cycle_chain))
    }

    fn analyze_module(&self, module: &Module) -> Vec<IrEntity> {
        let mut entities = Vec::new();

        let mut attrs = vec![
            IrAttribute {
                name: "id".to_string(),
                attr_type: "UUID".to_string(),
                description: Some("Auto-generated primary key".to_string()),
                required: true,
            },
            IrAttribute {
                name: "created_at".to_string(),
                attr_type: "timestamp".to_string(),
                description: Some("Creation timestamp".to_string()),
                required: false,
            },
        ];

        for file in &module.files {
            let stem = file.path.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            if !stem.is_empty() && !stem.starts_with("index") && !stem.starts_with("mod") {
                attrs.push(IrAttribute {
                    name: format!("{}_ref", stem),
                    attr_type: "string".to_string(),
                    description: Some(format!("Reference to {}", stem)),
                    required: false,
                });
            }
        }

        entities.push(IrEntity::DataModel {
            name: module.name.clone(),
            description: Some(format!(
                "Module '{}' with {} file(s) in {:?}",
                module.name, module.files.len(), module.module_type
            )),
            attributes: attrs,
        });

        entities.extend(module.files.iter().filter_map(|file| {
            let stem = file.path.file_stem()?.to_str()?;
            let lang = file.language.as_deref().unwrap_or("unknown");

            match lang {
                "rust" | "go" | "csharp" | "java" => {
                    Some(IrEntity::DataModel {
                        name: to_pascal_case(stem),
                        description: Some(format!("Entity from {}", stem)),
                        attributes: vec![
                            IrAttribute { name: "id".to_string(), attr_type: "UUID".to_string(), description: Some("Primary identifier".to_string()), required: true },
                        ],
                    })
                }
                "typescript" | "javascript" => {
                    Some(IrEntity::Interface {
                        name: stem.trim_end_matches(".d").to_string(),
                        description: Some(format!("Interface from {}", stem)),
                        methods: vec![],
                    })
                }
                "python" => {
                    Some(IrEntity::Function {
                        name: to_snake_case(stem),
                        description: Some(format!("Function from module {}", stem)),
                        inputs: vec![],
                        outputs: vec![],
                    })
                }
                _ => None,
            }
        }));

        entities
    }

    fn analyze_file(&self, file: &SourceFile) -> Vec<IrEntity> {
        let mut entities = Vec::new();
        let stem = match file.path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => return entities,
        };

        // Try tree-sitter parser first if we can read the file
        if let Ok(content) = std::fs::read_to_string(&file.path) {
            if let Some(lang) = &file.language {
                if let Some(analysis) = crate::tree_sitter_parser::parse_file_with_ts(
                    &file.path,
                    &content,
                    lang,
                ) {
                    info!("tree-sitter parsed '{}': {} entities, {} imports",
                        file.relative_path.display(),
                        analysis.entities.len(),
                        analysis.imports.len());
                    return analysis.entities;
                }
            }
        }

        // Fallback: use the existing regex-based analysis
        if stem == stem.to_lowercase() && !stem.contains('.') {
            entities.push(IrEntity::DataModel {
                name: to_pascal_case(stem),
                description: Some(format!("Data entity from {}", file.relative_path.display())),
                attributes: vec![
                    IrAttribute { name: "id".to_string(), attr_type: "UUID".to_string(), description: Some("Primary identifier".to_string()), required: true },
                ],
            });

            entities.push(IrEntity::Function {
                name: format!("get_{}", stem),
                description: Some(format!("Retrieve {}", stem)),
                inputs: vec![IrParam { name: "id".to_string(), param_type: "UUID".to_string(), description: Some("Entity identifier".to_string()) }],
                outputs: vec![IrParam { name: "result".to_string(), param_type: to_pascal_case(stem), description: Some("Found entity".to_string()) }],
            });
        }

        entities
    }

    fn build_sdef(
        &self,
        entities: &[IrEntity],
        modules: &[Module],
        files: &[SourceFile],
    ) -> Result<SoftwareDefinition, DbError> {
        let mut sdef = SoftwareDefinition::default();
        sdef.sdef_version = CURRENT_SCHEMA_VERSION.to_string();
        sdef.name = self.config.document_name.clone();
        sdef.version = Some(self.config.version.clone());
        sdef.description = self.config.description.clone();

        // Metadata
        sdef.metadata = Some(SoftwareMetadata {
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

        // Design decisions
        let mut decisions = Vec::new();
        for (i, module) in modules.iter().enumerate() {
            decisions.push(DesignDecision {
                id: format!("dd-{:04}", i + 1),
                topic: format!("Module structure: {}", module.name),
                decision: format!("Organize as {}", module_type_str(&module.module_type)),
                rationale: format!("Found {} file(s) with languages: {:?}",
                    module.files.len(), module.languages),
                context: Some(format!("Root: {:?}", module.root_path)),
                alternatives: None,
                consequences: None,
                constraints: None,
            });
        }
        sdef.design_decisions = Some(decisions);

        // Data models
        let data_models: Vec<DataModel> = entities.iter().filter_map(|e| {
            if let IrEntity::DataModel { name, description, attributes } = e {
                let sdef_attrs = if attributes.is_empty() { None } else {
                    Some(attributes.iter().map(|a| DataAttribute {
                        name: a.name.clone(),
                        attr_type: a.attr_type.clone(),
                        format: None,
                        description: a.description.clone(),
                        required: a.required,
                        default: None,
                        identity: a.name == "id",
                        generated: a.name == "id",
                        unique: a.name == "id",
                        internal: false,
                        deprecated: false,
                        compatibility: None,
                        constraints: None,
                    }).collect())
                };

                Some(DataModel {
                    entity: name.clone(),
                    status: None,
                    version: Some(self.config.version.clone()),
                    deprecated: None,
                    description: description.clone(),
                    logical_model: None,
                    attributes: sdef_attrs,
                    relationships: None,
                    validation_rules: None,
                    physical_design: None,
                })
            } else { None }
        }).collect();
        if !data_models.is_empty() {
            sdef.data_models = Some(data_models);
        }

        // Contracts (interfaces)
        let interfaces: Vec<InterfaceContract> = entities.iter().filter_map(|e| {
            if let IrEntity::Interface { name, description, methods } = e {
                Some(InterfaceContract {
                    name: to_pascal_case(name),
                    is_abstract: false,
                    status: None,
                    version: Some(self.config.version.clone()),
                    deprecated: None,
                    description: description.clone(),
                    methods: if methods.is_empty() { None } else {
                        Some(methods.iter().map(|m| ContractMethod {
                            signature: format!("{}({})", m.name, m.params.iter()
                                .map(|p| format!("{}: {}", p.name, p.param_type))
                                .collect::<Vec<_>>().join(", ")),
                            status: Some("active".to_string()),
                            deprecated: None,
                            behavior: Some(format!("Method {}", m.name)),
                            preconditions: None,
                            postconditions: None,
                            errors: None,
                        }).collect())
                    },
                    invariants: None,
                })
            } else { None }
        }).collect();

        if !interfaces.is_empty() {
            sdef.contracts = Some(Contracts {
                interfaces: Some(interfaces),
                classes: None,
                enums: None,
                apis: None,
                compatibility_modules: None,
                data_migrations: None,
            });
        }

        // Behavior (functions)
        let functions: Vec<FunctionSpec> = entities.iter().filter_map(|e| {
            if let IrEntity::Function { name, description, inputs, outputs } = e {
                Some(FunctionSpec {
                    name: name.clone(),
                    description: description.clone(),
                    inputs: if inputs.is_empty() { None } else {
                        Some(inputs.iter().map(|p| FunctionParam {
                            name: p.name.clone(),
                            param_type: p.param_type.clone(),
                            description: p.description.clone(),
                        }).collect())
                    },
                    outputs: if outputs.is_empty() { None } else {
                        Some(outputs.iter().map(|p| FunctionParam {
                            name: p.name.clone(),
                            param_type: p.param_type.clone(),
                            description: p.description.clone(),
                        }).collect())
                    },
                    logic: None,
                    complexity: None,
                    pure_function: false,
                    edge_cases: None,
                })
            } else { None }
        }).collect();

        if !functions.is_empty() {
            sdef.behavior = Some(Behavior {
                functions: Some(functions),
                flows: None,
                state_machines: None,
            });
        }

        // Architecture
        if !modules.is_empty() {
            let arch_layers: Vec<ArchitectureLayer> = modules.iter().map(|m| {
                ArchitectureLayer {
                    name: m.name.clone(),
                    components: if m.files.is_empty() { None } else {
                        Some(m.files.iter().map(|f| f.relative_path.to_string_lossy().to_string()).collect())
                    },
                }
            }).collect();

            sdef.architecture = Some(Architecture {
                style: Some("layered".to_string()),
                rationale: Some("Automatic module detection from source structure".to_string()),
                layers: if arch_layers.is_empty() { None } else { Some(arch_layers) },
                modules: None,
                communication: None,
                cross_cutting_concerns: None,
            });
        }

        info!(
            data_models = %sdef.data_models.as_ref().map(|v| v.len()).unwrap_or(0),
            interfaces = %sdef.contracts.as_ref().and_then(|c| c.interfaces.as_ref()).map(|v| v.len()).unwrap_or(0),
            functions = %sdef.behavior.as_ref().and_then(|b| b.functions.as_ref()).map(|v| v.len()).unwrap_or(0),
            "Built SoftwareDefinition"
        );

        Ok(sdef)
    }

    fn persist_to_db(
        &self,
        sdef: &SoftwareDefinition,
        files: &[SourceFile],
        _entities: &[IrEntity],
        _cycle_chain: &mut Vec<Vec<String>>,
    ) -> Result<(), DbError> {
        use rusqlite::params;
        let conn = self.db.connection();

        // 1. Save document
        conn.execute(
            "INSERT INTO sdef_documents (name, version, description, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(name) DO UPDATE SET version = ?2, description = ?3, updated_at = ?5",
            params![
                self.config.document_name,
                self.config.version,
                self.config.description,
                Utc::now().to_rfc3339(),
                Utc::now().to_rfc3339(),
            ],
        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

        // 2. Save data models
        if let Some(models) = &sdef.data_models {
            for model in models {
                conn.execute(
                    "INSERT OR IGNORE INTO data_models (entity, document_name, status, version, description)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        model.entity, self.config.document_name, "active",
                        model.version, model.description,
                    ],
                ).map_err(|e| DbError::QueryFailed(e.to_string()))?;

                if let Some(attrs) = &model.attributes {
                    for attr in attrs {
                        conn.execute(
                            "INSERT INTO data_attributes (document_name, entity, name, attr_type, description, required, identity, generated, unique_flag, internal, deprecated)
                             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                            params![
                                self.config.document_name, model.entity,
                                attr.name, attr.attr_type, attr.description,
                                attr.required, attr.identity, attr.generated,
                                attr.unique, attr.internal, attr.deprecated,
                            ],
                        ).map_err(|e| DbError::QueryFailed(e.to_string()))?;
                    }
                }
            }
        }

        // 3. Register symbols directly
        if let Some(models) = &sdef.data_models {
            for model in models {
                for file in files {
                    let lang = match file.language.as_deref() {
                        Some("rust") => "rust",
                        Some("typescript") | Some("javascript") => "typescript",
                        Some("python") => "python",
                        Some("go") => "go",
                        Some("java") => "java",
                        _ => continue,
                    };
                    let name = self.names.convert_for_language(
                        &model.entity,
                        Lang::from_str(lang).unwrap_or(Lang::Rust),
                    );
                    let uri = format!("sdef://{}/entity/{}", self.config.document_name, model.entity);
                    // Use INSERT OR IGNORE to handle conflicts
                    conn.execute(
                        "INSERT OR IGNORE INTO symbol_registry (document_name, sdef_uri, language, symbol_type, concrete_name, is_user_defined)
                         VALUES (?1, ?2, ?3, ?4, ?5, 0)",
                        params![self.config.document_name, uri, lang, "class", name],
                    ).ok();
                }
            }
        }

        Ok(())
    }
}

fn to_pascal_case(s: &str) -> String {
    s.split(|c: char| c == '_' || c == '-')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect(),
                None => String::new(),
            }
        })
        .collect()
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 { result.push('_'); }
        result.push(c.to_ascii_lowercase());
    }
    result
}

fn module_type_str(mt: &ModuleType) -> &'static str {
    match mt {
        ModuleType::CargoCrate => "Cargo crate",
        ModuleType::NpmPackage => "npm package",
        ModuleType::PythonPackage => "Python package",
        ModuleType::GoModule => "Go module",
        ModuleType::Directory => "directory",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_file(name: &str, lang: &str) -> SourceFile {
        SourceFile {
            path: std::path::PathBuf::from(name),
            language: Some(lang.to_string()),
            size_bytes: 100,
            relative_path: std::path::PathBuf::from(name),
        }
    }

    fn sample_module(name: &str, file_names: &[&str], mt: ModuleType) -> Module {
        let files: Vec<SourceFile> = file_names.iter()
            .map(|f| sample_file(f, "rust"))
            .collect();
        Module {
            name: name.to_string(),
            files,
            module_type: mt,
            root_path: std::path::PathBuf::from(name),
            languages: vec!["rust".to_string()],
        }
    }

    #[test]
    fn test_analyze_module_creates_data_model() {
        let db = Arc::new(cleanroom_db::Database::in_memory().unwrap());
        let mapper = SdefMapper::new(MapperConfig::default(), db);
        let module = sample_module("users", &["user.rs", "address.rs"], ModuleType::CargoCrate);
        let entities = mapper.analyze_module(&module);
        assert!(!entities.is_empty());
        let has_data_model = entities.iter().any(|e| matches!(e, IrEntity::DataModel { .. }));
        assert!(has_data_model);
    }

    #[test]
    fn test_analyze_file_creates_entities() {
        let db = Arc::new(cleanroom_db::Database::in_memory().unwrap());
        let mapper = SdefMapper::new(MapperConfig::default(), db);
        let file = sample_file("user.rs", "rust");
        let entities = mapper.analyze_file(&file);
        assert_eq!(entities.len(), 2);
        assert!(matches!(entities[0], IrEntity::DataModel { .. }));
        assert!(matches!(entities[1], IrEntity::Function { .. }));
    }

    #[test]
    fn test_build_sdef_with_entities() {
        let db = Arc::new(cleanroom_db::Database::in_memory().unwrap());
        let mapper = SdefMapper::new(MapperConfig {
            document_name: "test-project".to_string(),
            version: "1.0.0".to_string(),
            description: Some("A test".to_string()),
        }, db);

        let entities = vec![
            IrEntity::DataModel {
                name: "User".to_string(),
                description: Some("A user entity".to_string()),
                attributes: vec![IrAttribute {
                    name: "name".to_string(), attr_type: "string".to_string(),
                    description: Some("Full name".to_string()), required: true,
                }],
            },
            IrEntity::Function {
                name: "create_user".to_string(), description: Some("Create a user".to_string()),
                inputs: vec![], outputs: vec![],
            },
        ];
        let modules = vec![sample_module("core", &["user.rs"], ModuleType::Directory)];
        let files = vec![sample_file("user.rs", "rust")];

        let sdef = mapper.build_sdef(&entities, &modules, &files).unwrap();
        assert_eq!(sdef.name, "test-project");
        assert!(sdef.data_models.is_some());
        assert!(sdef.behavior.is_some());
        assert!(sdef.design_decisions.is_some());
        assert!(sdef.architecture.is_some());
    }

    #[test]
    fn test_build_sdef_empty() {
        let db = Arc::new(cleanroom_db::Database::in_memory().unwrap());
        let mapper = SdefMapper::new(MapperConfig::default(), db);
        let sdef = mapper.build_sdef(&[], &[], &[]).unwrap();
        assert_eq!(sdef.name, "unnamed");
        assert!(sdef.data_models.is_none());
        assert!(sdef.behavior.is_none());
    }
}