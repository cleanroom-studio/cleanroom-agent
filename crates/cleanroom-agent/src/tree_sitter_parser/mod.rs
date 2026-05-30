//! tree-sitter based source code parser.
//!
//! Parses source files into an Intermediate Representation (IR) using
//! tree-sitter grammars. Falls back gracefully to regex-based analysis
//! when tree-sitter grammar files are unavailable.
//!
//! Since the `tree-sitter` Rust crate does not support loading grammars
//! from `.so` files, this module provides a registry-based approach:
//!
//! - Use `register_grammar()` to supply a tree-sitter `Language` for a language.
//! - If no language is registered, falls back to regex-based extraction.
//!
//! Language-specific CST walking utilities are in sub-modules:
//! - `rust_parser` — struct/enum/trait/function extraction
//! - `typescript_parser` — class/interface/type/function/enum extraction
//! - `python_parser` — class/function extraction

pub mod rust_parser;
pub mod typescript_parser;
pub mod python_parser;
pub mod c_parser;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use tracing::{info, warn};
use tree_sitter::Language;

use crate::ir_to_sdef::{IrEntity, IrAttribute, IrMethod, IrParam};

/// A tree-sitter language loaded into the parser.
#[derive(Clone)]
pub struct LoadedGrammar {
    /// Language name (e.g. "rust", "typescript").
    pub language: String,
    /// Tree-sitter Language (static reference).
    pub ts_language: Language,
}


/// Get the global grammar registry using OnceLock (safe, no unsafe needed).
fn grammar_registry() -> &'static Mutex<HashMap<String, LoadedGrammar>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, LoadedGrammar>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register a tree-sitter Language for a language identifier.
///
/// Typical usage (requires the `tree-sitter-{lang}` crate):
///
/// ```ignore
/// let lang = tree_sitter::Language::new(tree_sitter_rust::LANGUAGE);
/// crate::tree_sitter_parser::register_grammar("rust", lang);
/// ```
pub fn register_grammar(language: &str, ts_language: Language) {
    let mut registry = grammar_registry().lock().unwrap();
    registry.insert(
        language.to_string(),
        LoadedGrammar {
            language: language.to_string(),
            ts_language,
        },
    );
    info!(language = %language, "tree-sitter grammar registered");
}

/// Check if a grammar is currently registered.
pub fn has_grammar(language: &str) -> bool {
    let registry = grammar_registry().lock().unwrap();
    registry.contains_key(language)
}

/// Initialize built-in tree-sitter grammars.
///
/// Registers static grammars for 16+ languages.
/// Call this once during application startup.
pub fn init_builtin_grammars() {
    register_grammar("rust", Language::new(tree_sitter_rust::LANGUAGE));
    register_grammar("typescript", Language::new(tree_sitter_typescript::LANGUAGE_TYPESCRIPT));
    register_grammar("javascript", Language::new(tree_sitter_javascript::LANGUAGE));
    register_grammar("python", Language::new(tree_sitter_python::LANGUAGE));
    register_grammar("c", Language::new(tree_sitter_c::LANGUAGE));
    register_grammar("cpp", Language::new(tree_sitter_cpp::LANGUAGE));
    register_grammar("go", Language::new(tree_sitter_go::LANGUAGE));
    register_grammar("java", Language::new(tree_sitter_java::LANGUAGE));
    register_grammar("csharp", Language::new(tree_sitter_c_sharp::LANGUAGE));
    register_grammar("swift", Language::new(tree_sitter_swift::LANGUAGE));
    register_grammar("kotlin", Language::new(tree_sitter_kotlin_ng::LANGUAGE));
    register_grammar("php", Language::new(tree_sitter_php::LANGUAGE_PHP));
    register_grammar("ruby", Language::new(tree_sitter_ruby::LANGUAGE));
    register_grammar("lua", Language::new(tree_sitter_lua::LANGUAGE));
    register_grammar("shell", Language::new(tree_sitter_bash::LANGUAGE));
    register_grammar("json", Language::new(tree_sitter_json::LANGUAGE));
    register_grammar("toml", Language::new(tree_sitter_toml_ng::LANGUAGE));
    register_grammar("yaml", Language::new(tree_sitter_yaml::LANGUAGE));

    info!(
        "Initialized {} built-in tree-sitter grammars",
        registered_languages().len()
    );
}

/// List all registered grammar languages.
pub fn registered_languages() -> Vec<String> {
    let registry = grammar_registry().lock().unwrap();
    registry.keys().cloned().collect()
}

/// File analysis result from tree-sitter parsing.
#[derive(Debug, Clone)]
pub struct TsFileAnalysis {
    /// File path.
    pub file_path: String,
    /// Language.
    pub language: String,
    /// Extracted IR entities.
    pub entities: Vec<IrEntity>,
    /// Imports found in the file.
    pub imports: Vec<String>,
    /// Whether tree-sitter was actually used (vs fallback).
    pub used_tree_sitter: bool,
}

/// Language-specific grammar descriptor for CST node type names.
#[derive(Debug, Clone)]
pub struct GrammarDescriptor {
    /// Language name for display.
    pub language: &'static str,
    /// File extensions this grammar handles.
    pub extensions: &'static [&'static str],
    /// Node type names that represent top-level definitions.
    pub top_level_nodes: &'static [&'static str],
}

/// Known descriptors for supported languages.
pub fn known_grammars() -> Vec<GrammarDescriptor> {
    vec![
        GrammarDescriptor {
            language: "rust",
            extensions: &["rs"],
            top_level_nodes: &[
                "struct_item", "enum_item", "trait_item",
                "impl_item", "function_item", "type_item",
                "const_item", "static_item", "mod_item",
            ],
        },
        GrammarDescriptor {
            language: "typescript",
            extensions: &["ts", "tsx"],
            top_level_nodes: &[
                "interface_declaration", "type_alias_declaration",
                "class_declaration", "enum_declaration",
                "function_declaration", "method_signature",
                "abstract_method_signature",
            ],
        },
        GrammarDescriptor {
            language: "javascript",
            extensions: &["js", "jsx", "mjs", "cjs"],
            top_level_nodes: &[
                "class_declaration", "function_declaration",
                "arrow_function",
            ],
        },
        GrammarDescriptor {
            language: "python",
            extensions: &["py", "pyi"],
            top_level_nodes: &[
                "class_definition", "function_definition",
                "decorated_definition",
            ],
        },
        GrammarDescriptor {
            language: "c",
            extensions: &["c", "h"],
            top_level_nodes: &[
                "struct_specifier", "enum_specifier",
                "function_definition", "type_definition",
            ],
        },
        GrammarDescriptor {
            language: "cpp",
            extensions: &["cpp", "cc", "cxx", "c++", "hpp", "hh", "hxx", "h++"],
            top_level_nodes: &[
                "class_specifier", "struct_specifier", "enum_specifier",
                "function_definition", "template_declaration", "namespace_definition",
            ],
        },
        GrammarDescriptor {
            language: "go",
            extensions: &["go"],
            top_level_nodes: &[
                "type_declaration", "function_declaration",
                "method_declaration", "import_declaration",
            ],
        },
        GrammarDescriptor {
            language: "java",
            extensions: &["java"],
            top_level_nodes: &[
                "class_declaration", "interface_declaration",
                "enum_declaration", "method_declaration",
            ],
        },
        GrammarDescriptor {
            language: "csharp",
            extensions: &["cs"],
            top_level_nodes: &[
                "class_declaration", "interface_declaration",
                "struct_declaration", "enum_declaration",
                "method_declaration", "namespace_declaration",
            ],
        },
        GrammarDescriptor {
            language: "swift",
            extensions: &["swift"],
            top_level_nodes: &[
                "class_declaration", "struct_declaration",
                "enum_declaration", "protocol_declaration",
                "function_declaration", "extension_declaration",
            ],
        },
        GrammarDescriptor {
            language: "kotlin",
            extensions: &["kt", "kts"],
            top_level_nodes: &[
                "class_declaration", "object_declaration",
                "function_declaration", "interface_declaration",
            ],
        },
        GrammarDescriptor {
            language: "php",
            extensions: &["php"],
            top_level_nodes: &[
                "class_declaration", "interface_declaration",
                "trait_declaration", "enum_declaration",
                "function_definition", "method_declaration",
            ],
        },
        GrammarDescriptor {
            language: "ruby",
            extensions: &["rb", "rake", "gemspec", "ru"],
            top_level_nodes: &[
                "class", "module", "method", "singleton_method",
            ],
        },
        GrammarDescriptor {
            language: "lua",
            extensions: &["lua"],
            top_level_nodes: &[
                "function_declaration", "local_function",
                "variable_declaration",
            ],
        },
        GrammarDescriptor {
            language: "shell",
            extensions: &["sh", "bash", "zsh", "ksh"],
            top_level_nodes: &[
                "function_definition", "command",
                "variable_assignment",
            ],
        },
        GrammarDescriptor {
            language: "json",
            extensions: &["json"],
            top_level_nodes: &["document"],
        },
        GrammarDescriptor {
            language: "toml",
            extensions: &["toml"],
            top_level_nodes: &["document"],
        },
        GrammarDescriptor {
            language: "yaml",
            extensions: &["yaml", "yml"],
            top_level_nodes: &["document"],
        },
    ]
}

/// Parse a file using tree-sitter if the grammar is registered.
pub fn parse_file_with_ts(
    file_path: &Path,
    content: &str,
    language: &str,
) -> Option<TsFileAnalysis> {
    let grammar = {
        let registry = grammar_registry().lock().ok()?;
        registry.get(language).cloned()
    };

    let loaded = match grammar {
        Some(g) => g,
        None => {
            warn!(language = %language, "tree-sitter grammar not registered; falling back to regex");
            return None;
        }
    };

    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&loaded.ts_language).ok()?;

    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    let descriptors = known_grammars();
    let desc = descriptors.iter().find(|d| d.language == language)?;

    let entities = extract_entities(&root, content, desc, file_path);
    let imports = extract_imports(&root, content, desc);

    Some(TsFileAnalysis {
        file_path: file_path.to_string_lossy().to_string(),
        language: language.to_string(),
        entities,
        imports,
        used_tree_sitter: true,
    })
}

/// Extract entities from the CST by walking known top-level nodes.
fn extract_entities(
    root: &tree_sitter::Node,
    source: &str,
    desc: &GrammarDescriptor,
    _file_path: &Path,
) -> Vec<IrEntity> {
    let mut entities = Vec::new();
    let mut cursor = root.walk();

    // Visit top-level definitions
    for node in root.children(&mut cursor) {
        for &top_kind in desc.top_level_nodes {
            if node.kind() == top_kind {
                if let Some(entity) = extract_top_level_node(&node, source, top_kind) {
                    entities.push(entity);
                }
                break;
            }
        }
    }

    entities
}

/// Extract a single top-level definition node to an IR entity.
fn extract_top_level_node(
    node: &tree_sitter::Node,
    source: &str,
    kind: &str,
) -> Option<IrEntity> {
    // C-specific node kinds use their own sub-parsers that handle name extraction internally
    match kind {
        "struct_specifier" => c_parser::extract_c_struct(node, source),
        "enum_specifier" => c_parser::extract_c_enum(node, source),
        "function_definition" => c_parser::extract_c_function(node, source),
        "type_definition" => {
            // typedef struct { ... } Name; or typedef enum { ... } Name;
            // Look for the type (struct_specifier or enum_specifier) inside
            if let Some(inner) = node.child_by_field_name("type") {
                let entity = match inner.kind() {
                    "struct_specifier" => c_parser::extract_c_struct(&inner, source),
                    "enum_specifier" => c_parser::extract_c_enum(&inner, source),
                    _ => None,
                };
                if entity.is_some() {
                    return entity;
                }
                // Fallback: handle anonymous struct/enum with typedef name in declarator
                if let Some(decl) = node.child_by_field_name("declarator") {
                    if let Ok(decl_name) = decl.utf8_text(source.as_bytes()) {
                        let name = decl_name.trim().to_string();
                        if !name.is_empty() {
                            // Use the dedicated C struct field extractor for anonymous structs
                            match inner.kind() {
                                "struct_specifier" => {
                                    let attrs = c_parser::extract_c_struct_fields(&inner, source);
                                    return Some(IrEntity::DataModel {
                                        name,
                                        description: None,
                                        attributes: attrs,
                                    });
                                }
                                _ => {}
                            }
                        }
                    }
                }
                None
            } else {
                None
            }
        }
        _ => {
            // Non-C languages: all have a "name" field on top-level nodes
            let name_node = node.child_by_field_name("name")?;
            let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();
            let description = extract_doc_comment(node, source);

            match kind {
                // Rust struct -> DataModel
                "struct_item" | "class_declaration" => {
                    let attrs = extract_field_attributes(node, source);
                    Some(IrEntity::DataModel { name, description, attributes: attrs })
                }
                // Rust trait / TypeScript interface -> Interface
                "trait_item" | "interface_declaration" | "type_alias_declaration" => {
                    let methods = extract_methods(node, source);
                    Some(IrEntity::Interface { name, description, methods })
                }
                // Function -> Function (non-C languages)
                "function_item" | "function_declaration" => {
                    let (inputs, outputs) = extract_function_params(node, source);
                    Some(IrEntity::Function { name, description, inputs, outputs })
                }
                // Rust enum -> DataModel (with variant attributes)
                "enum_item" | "enum_declaration" => {
                    let attrs = extract_enum_variants(node, source);
                    Some(IrEntity::DataModel { name, description, attributes: attrs })
                }
                _ => None,
            }
        }
    }
}

/// Extract doc comment that precedes a node.
fn extract_doc_comment(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut comments = Vec::new();
    let mut prev = node.prev_sibling();
    while let Some(sib) = prev {
        if sib.kind().contains("comment") {
            if let Ok(text) = sib.utf8_text(source.as_bytes()) {
                let cleaned = text.lines()
                    .map(|l| l.trim_start_matches("//").trim_start_matches("///")
                        .trim_start_matches('#').trim_start_matches('"')
                        .trim())
                    .collect::<Vec<_>>()
                    .join(" ");
                if !cleaned.is_empty() {
                    comments.push(cleaned);
                }
            }
        } else {
            break;
        }
        prev = sib.prev_sibling();
    }
    comments.reverse();
    if comments.is_empty() { None } else { Some(comments.join("\n")) }
}

/// Extract field attributes from a struct/class body.
fn extract_field_attributes(node: &tree_sitter::Node, source: &str) -> Vec<IrAttribute> {
    let mut attrs = Vec::new();
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        // Rust: field_declaration nodes
        // TypeScript: property_signature, public_field_definition
        match child.kind() {
            "field_declaration" | "property_signature"
            | "public_field_definition" | "required_field" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                        let ty = child.child_by_field_name("type")
                            .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                            .unwrap_or("unknown")
                            .to_string();
                        let desc = extract_doc_comment(&child, source);
                        attrs.push(IrAttribute {
                            name: name.to_string(),
                            attr_type: ty,
                            description: desc,
                            required: true,
                        });
                    }
                }
            }
            _ => {}
        }
    }
    attrs
}

/// Extract methods from a trait/interface body.
fn extract_methods(node: &tree_sitter::Node, source: &str) -> Vec<IrMethod> {
    let mut methods = Vec::new();
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_item" | "function_declaration" | "method_signature"
            | "abstract_method_signature" | "method_definition" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                        let params = extract_params_from_node(&child, source);
                        methods.push(IrMethod {
                            name: name.to_string(),
                            params,
                        });
                    }
                }
            }
            _ => {}
        }
    }
    methods
}

/// Extract parameters from a function/method node.
fn extract_params_from_node(node: &tree_sitter::Node, source: &str) -> Vec<IrParam> {
    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.children(&mut cursor) {
            if child.kind() == "parameter" || child.kind() == "required_parameter"
                || child.kind() == "optional_parameter" || child.kind() == "pattern" {
                let name = child.child_by_field_name("name")
                    .or_else(|| child.child_by_field_name("pattern"))
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .unwrap_or("_")
                    .to_string();
                let ty = child.child_by_field_name("type")
                    .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                    .unwrap_or("any")
                    .to_string();
                params.push(IrParam {
                    name,
                    param_type: ty,
                    description: None,
                });
            }
        }
    }
    params
}

/// Extract function parameters as input/output pairs.
fn extract_function_params(node: &tree_sitter::Node, source: &str) -> (Vec<IrParam>, Vec<IrParam>) {
    let params = extract_params_from_node(node, source);
    let return_type = node.child_by_field_name("return_type")
        .and_then(|r| r.utf8_text(source.as_bytes()).ok())
        .unwrap_or("void")
        .to_string();

    let inputs = params;
    let outputs = if return_type != "void" && return_type != "()" {
        vec![IrParam {
            name: "result".to_string(),
            param_type: return_type,
            description: None,
        }]
    } else {
        vec![]
    };

    (inputs, outputs)
}

/// Extract enum variants as attributes.
fn extract_enum_variants(node: &tree_sitter::Node, source: &str) -> Vec<IrAttribute> {
    let mut attrs = Vec::new();
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        if child.kind() == "enum_variant" || child.kind() == "enum_value" {
            if let Some(name_node) = child.child_by_field_name("name") {
                if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                    attrs.push(IrAttribute {
                        name: name.to_string(),
                        attr_type: "enum_variant".to_string(),
                        description: extract_doc_comment(&child, source),
                        required: false,
                    });
                }
            }
        }
    }
    attrs
}

/// Extract import statements.
fn extract_imports(root: &tree_sitter::Node, source: &str, _desc: &GrammarDescriptor) -> Vec<String> {
    let mut imports = Vec::new();
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        let kind = node.kind();
        let is_import = kind.contains("use_declaration")
            || kind.contains("import")
            || kind.contains("module")
            || kind == "expression_statement"; // fallback

        if is_import {
            if let Ok(text) = node.utf8_text(source.as_bytes()) {
                let line = text.lines().next().unwrap_or(text).trim().to_string();
                if !line.is_empty() && line.len() < 200 {
                    imports.push(line);
                }
            }
        }
    }
    imports
}

/// Parse a file, trying tree-sitter first, falling back to regex.
pub fn analyze_file_with_ts_fallback(
    file_path: &Path,
    content: &str,
    language: &str,
) -> Option<TsFileAnalysis> {
    // Try tree-sitter first
    if let Some(analysis) = parse_file_with_ts(file_path, content, language) {
        if !analysis.entities.is_empty() || !analysis.imports.is_empty() {
            return Some(analysis);
        }
    }

    // Fallback: use regex-based extraction
    Some(analyze_file_regex(file_path, content, language))
}

/// Minimal regex-based file analysis fallback.
fn analyze_file_regex(
    file_path: &Path,
    content: &str,
    language: &str,
) -> TsFileAnalysis {
    let mut entities = Vec::new();

    // Extract struct/class definitions
    let struct_re = regex::Regex::new(
        r"(?m)^\s*(public\s+|pub\s+)?(struct|class|interface|trait|type)\s+(\w+)"
    ).ok();
    let func_re = regex::Regex::new(
        r"(?m)^\s*(public\s+|pub\s+)?(fn|function|def|func)\s+(\w+)\s*\(([^)]*)\)"
    ).ok();
    let import_re = regex::Regex::new(
        r"(?m)^\s*(use|import|from|require)\s+([^;]+)"
    ).ok();
    let attr_re = regex::Regex::new(
        r"(?m)^\s*(pub\s+)?(\w+)\s*:\s*(\w+)"
    ).ok();

    // Structs / classes
    if let Some(ref re) = struct_re {
        for cap in re.captures_iter(content) {
            let name = cap[3].to_string();
            let kind_str = &cap[2];
            match kind_str {
                "struct" | "class" => {
                    let attrs = extract_attrs_regex(content, &name, attr_re.as_ref());
                    entities.push(IrEntity::DataModel {
                        name: name.clone(),
                        description: None,
                        attributes: attrs,
                    });
                }
                "interface" | "trait" => {
                    entities.push(IrEntity::Interface {
                        name: name.clone(),
                        description: None,
                        methods: vec![],
                    });
                }
                _ => {}
            }
        }
    }

    // Functions
    if let Some(ref re) = func_re {
        for cap in re.captures_iter(content) {
            let name = cap[3].to_string();
            let params_str = &cap[4];
            let inputs: Vec<IrParam> = params_str.split(',')
                .filter_map(|p| {
                    let parts: Vec<&str> = p.split(':').collect();
                    if parts.len() >= 2 {
                        Some(IrParam {
                            name: parts[0].trim().to_string(),
                            param_type: parts[1].trim().to_string(),
                            description: None,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            entities.push(IrEntity::Function {
                name,
                description: None,
                inputs,
                outputs: vec![],
            });
        }
    }

    // Imports
    let mut imports = Vec::new();
    if let Some(ref re) = import_re {
        for cap in re.captures_iter(content) {
            imports.push(cap[0].to_string());
        }
    }

    TsFileAnalysis {
        file_path: file_path.to_string_lossy().to_string(),
        language: language.to_string(),
        entities,
        imports,
        used_tree_sitter: false,
    }
}

/// Extract attributes from a struct body using regex.
fn extract_attrs_regex(
    content: &str,
    struct_name: &str,
    attr_re: Option<&regex::Regex>,
) -> Vec<IrAttribute> {
    let mut attrs = Vec::new();

    // Find the struct body
    let pattern = format!(r"(?s)(struct|class)\s+{}\s*\{{(.+?)\}}", regex::escape(struct_name));
    if let Ok(re) = regex::Regex::new(&pattern) {
        if let Some(cap) = re.captures(content) {
            let body = &cap[2];
            if let Some(ref attr_re) = attr_re {
                for attr_cap in attr_re.captures_iter(body) {
                    let name = attr_cap[2].to_string();
                    let ty = attr_cap[3].to_string();
                    attrs.push(IrAttribute {
                        name,
                        attr_type: ty,
                        description: None,
                        required: true,
                    });
                }
            }
        }
    }
    attrs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_fallback_rust() {
        let content = r#"
pub struct User {
    pub id: u64,
    pub name: String,
}

fn create_user(name: String) -> User {
    User { id: 1, name }
}
"#;
        let analysis = analyze_file_regex(
            Path::new("test.rs"),
            content,
            "rust",
        );
        assert!(!analysis.entities.is_empty(), "Should extract entities");
        assert!(!analysis.used_tree_sitter, "Should note fallback");
    }

    #[test]
    fn test_regex_fallback_typescript() {
        let content = r#"
export interface User {
    id: number;
    name: string;
}

function greet(name: string): string {
    return "Hello " + name;
}
"#;
        let analysis = analyze_file_regex(
            Path::new("test.ts"),
            content,
            "typescript",
        );
        assert!(!analysis.entities.is_empty(), "Should extract entities");
    }

    #[test]
    fn test_regex_fallback_python() {
        let content = r#"
class User:
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

def create_user(name: str) -> User:
    return User(1, name)
"#;
        let analysis = analyze_file_regex(
            Path::new("test.py"),
            content,
            "python",
        );
        assert!(!analysis.entities.is_empty(), "Should extract entities");
    }

    #[test]
    fn test_known_grammars_defined() {
        let grammars = known_grammars();
        assert!(!grammars.is_empty(), "Should have grammars defined");
        let rust = grammars.iter().find(|g| g.language == "rust");
        assert!(rust.is_some(), "Rust grammar should be defined");
        assert!(rust.unwrap().top_level_nodes.contains(&"struct_item"));
    }
}
