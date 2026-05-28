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

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use tracing::{info, warn};
use tree_sitter::Language;

use crate::ir_to_sdef::{IrEntity, IrAttribute, IrMethod, IrParam};

/// A tree-sitter language loaded into the parser.
pub struct LoadedGrammar {
    /// Language name (e.g. "rust", "typescript").
    pub language: String,
    /// Tree-sitter Language (static reference).
    pub ts_language: Language,
}


/// Get the global grammar registry.
fn grammar_registry() -> &'static Mutex<HashMap<String, LoadedGrammar>> {
    // Use Box::leak to create a static reference from a heap allocation.
    // This is safe because the registry lives for the entire program lifetime.
    static mut REGISTRY: Option<&Mutex<HashMap<String, LoadedGrammar>>> = None;

    // SAFETY: Single-threaded init at first access; after that read-only.
    unsafe {
        REGISTRY.get_or_insert_with(|| {
            Box::leak(Box::new(Mutex::new(HashMap::new())))
        })
    }
}

/// Register a tree-sitter Language for a language identifier.
///
/// Typical usage (requires the `tree-sitter-{lang}` crate):
///
/// ```no_run
/// let lang = tree_sitter_rust::language();
/// tree_sitter_parser::register_grammar("rust", lang);
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
        // Function -> Function
        "function_item" | "function_declaration" | "function_definition" => {
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
    fn test_grammar_search_paths() {
        let paths = grammar_search_paths();
        assert!(!paths.is_empty(), "Should have at least some search paths");
        assert!(paths.iter().any(|p| p.to_string_lossy().contains("grammars")));
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
