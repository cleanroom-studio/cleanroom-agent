//! Python-specific tree-sitter parser helpers.

use tree_sitter::Node;

use crate::ir_to_sdef::{IrEntity, IrAttribute, IrParam};

/// Extract top-level Python definitions as IR entities.
pub fn extract_python_definitions(
    root: &Node,
    source: &str,
) -> Vec<IrEntity> {
    let mut entities = Vec::new();
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        // Handle decorated definitions
        let actual = if node.kind() == "decorated_definition" {
            node.child_by_field_name("definition").unwrap_or(node)
        } else {
            node
        };

        match actual.kind() {
            "class_definition" => {
                if let Some(entity) = extract_python_class(&actual, source) {
                    entities.push(entity);
                }
            }
            "function_definition" => {
                if let Some(entity) = extract_python_function(&actual, source) {
                    entities.push(entity);
                }
            }
            _ => {}
        }
    }

    entities
}

/// Extract a Python class as DataModel.
fn extract_python_class(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let mut attrs = Vec::new();
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();

        // Look for __init__ method to extract parameters
        for child in body.children(&mut cursor) {
            if child.kind() == "function_definition" {
                if let Some(method_name) = child.child_by_field_name("name") {
                    if let Ok(mname) = method_name.utf8_text(source.as_bytes()) {
                        if mname == "__init__" {
                            // Extract constructor params
                            if let Some(params_node) = child.child_by_field_name("parameters") {
                                let mut param_cursor = params_node.walk();
                                for param in params_node.children(&mut param_cursor) {
                                    if param.kind() == "identifier" {
                                        if let Ok(pname) = param.utf8_text(source.as_bytes()) {
                                            if pname != "self" {
                                                let ty = param.parent()
                                                    .and_then(|p| p.child_by_field_name("type"))
                                                    .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                                                    .unwrap_or("Any");
                                                attrs.push(IrAttribute {
                                                    name: pname.to_string(),
                                                    attr_type: ty.to_string(),
                                                    description: None,
                                                    required: true,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Also look for class variable assignments
            if child.kind() == "expression_statement" {
                if let Some(left) = child.child_by_field_name("left") {
                    if let Ok(text) = left.utf8_text(source.as_bytes()) {
                        let name = text.trim();
                        if !name.starts_with('_') && !name.starts_with("self.") {
                            attrs.push(IrAttribute {
                                name: name.to_string(),
                                attr_type: "unknown".to_string(),
                                description: None,
                                required: false,
                            });
                        }
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

/// Extract a Python function.
fn extract_python_function(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let inputs = extract_python_params(node, source);
    let return_type = node.child_by_field_name("return_type")
        .and_then(|r| r.utf8_text(source.as_bytes()).ok())
        .unwrap_or("None");

    let outputs = if return_type != "None" {
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

/// Extract Python function parameters (skip self/cls).
fn extract_python_params(node: &Node, source: &str) -> Vec<IrParam> {
    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.children(&mut cursor) {
            if child.kind() == "identifier" {
                if let Ok(name) = child.utf8_text(source.as_bytes()) {
                    if name != "self" && name != "cls" {
                        let ty = child.parent()
                            .and_then(|p| p.child_by_field_name("type"))
                            .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                            .unwrap_or("Any");
                        params.push(IrParam {
                            name: name.to_string(),
                            param_type: ty.to_string(),
                            description: None,
                        });
                    }
                }
            }
            if child.kind() == "typed_parameter" {
                let name = child.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .unwrap_or("_");
                let ty = child.child_by_field_name("type")
                    .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                    .unwrap_or("Any");
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
