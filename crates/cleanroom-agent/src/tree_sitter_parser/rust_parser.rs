//! Rust-specific tree-sitter parser helpers.
//!
//! Provides Rust-specific utilities for extracting entities from
//! tree-sitter CST nodes, including type resolution and visibility analysis.

use tree_sitter::Node;

use crate::ir_to_sdef::{IrEntity, IrAttribute, IrMethod, IrParam};

/// Extract all public items from a Rust module.
pub fn extract_public_items(
    root: &Node,
    source: &str,
) -> Vec<IrEntity> {
    let mut entities = Vec::new();
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        // Check visibility
        let is_pub = node.kind() == "function_item"
            || node.kind() == "struct_item"
            || node.kind() == "enum_item"
            || node.kind() == "trait_item"
            || has_pub_visibility(&node, source);

        if !is_pub {
            continue;
        }

        match node.kind() {
            "struct_item" => {
                if let Some(entity) = extract_rust_struct(&node, source) {
                    entities.push(entity);
                }
            }
            "enum_item" => {
                if let Some(entity) = extract_rust_enum(&node, source) {
                    entities.push(entity);
                }
            }
            "trait_item" => {
                if let Some(entity) = extract_rust_trait(&node, source) {
                    entities.push(entity);
                }
            }
            "function_item" => {
                if let Some(entity) = extract_rust_function(&node, source) {
                    entities.push(entity);
                }
            }
            _ => {}
        }
    }

    entities
}

/// Check if a node has `pub` visibility.
fn has_pub_visibility(node: &Node, source: &str) -> bool {
    if let Some(vis) = node.child_by_field_name("visibility") {
        if let Ok(text) = vis.utf8_text(source.as_bytes()) {
            return text.contains("pub");
        }
    }
    false
}

/// Extract a Rust struct as DataModel.
fn extract_rust_struct(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let mut attrs = Vec::new();
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "field_declaration" {
                if let Some(field_name) = child.child_by_field_name("name") {
                    if let Ok(fname) = field_name.utf8_text(source.as_bytes()) {
                        let ty = child.child_by_field_name("type")
                            .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                            .unwrap_or("unknown");
                        attrs.push(IrAttribute {
                            name: fname.to_string(),
                            attr_type: ty.to_string(),
                            description: None,
                            required: true,
                        });
                    }
                }
            }
        }
    }

    Some(IrEntity::DataModel {
        name,
        description: None,
        attributes: attrs,
    })
}

/// Extract a Rust enum.
fn extract_rust_enum(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let mut attrs = Vec::new();
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "enum_variant" {
                if let Some(variant_name) = child.child_by_field_name("name") {
                    if let Ok(vname) = variant_name.utf8_text(source.as_bytes()) {
                        attrs.push(IrAttribute {
                            name: vname.to_string(),
                            attr_type: "enum_variant".to_string(),
                            description: None,
                            required: false,
                        });
                    }
                }
            }
        }
    }

    Some(IrEntity::DataModel {
        name,
        description: None,
        attributes: attrs,
    })
}

/// Extract a Rust trait as Interface.
fn extract_rust_trait(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let mut methods = Vec::new();
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "function_item" {
                if let Some(method_name) = child.child_by_field_name("name") {
                    if let Ok(mname) = method_name.utf8_text(source.as_bytes()) {
                        let params = extract_params(&child, source);
                        methods.push(IrMethod {
                            name: mname.to_string(),
                            params,
                        });
                    }
                }
            }
        }
    }

    Some(IrEntity::Interface {
        name,
        description: None,
        methods,
    })
}

/// Extract a Rust function.
fn extract_rust_function(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let inputs = extract_params(node, source);
    let return_type = node.child_by_field_name("return_type")
        .and_then(|r| r.utf8_text(source.as_bytes()).ok())
        .unwrap_or("()");

    let outputs = if return_type != "()" {
        vec![IrParam {
            name: "result".to_string(),
            param_type: return_type.to_string(),
            description: None,
        }]
    } else {
        vec![]
    };

    Some(IrEntity::Function {
        name,
        description: None,
        inputs,
        outputs,
    })
}

/// Extract params from a function node.
fn extract_params(node: &Node, source: &str) -> Vec<IrParam> {
    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.children(&mut cursor) {
            if child.kind() == "parameter" {
                let name = child.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .unwrap_or("_");
                let ty = child.child_by_field_name("type")
                    .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                    .unwrap_or("unknown");
                params.push(IrParam {
                    name: name.to_string(),
                    param_type: ty.to_string(),
                    description: None,
                });
            }
        }
    }
    params
}
