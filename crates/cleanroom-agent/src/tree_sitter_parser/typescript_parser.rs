//! TypeScript/JavaScript-specific tree-sitter parser helpers.

use tree_sitter::Node;

use crate::ir_to_sdef::{IrEntity, IrAttribute, IrMethod, IrParam};

/// Extract exported declarations as IR entities.
pub fn extract_exported_declarations(
    root: &Node,
    source: &str,
) -> Vec<IrEntity> {
    let mut entities = Vec::new();
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        let actual_node = unwrap_export(node);

        match actual_node.kind() {
            "class_declaration" => {
                if let Some(entity) = extract_ts_class(&actual_node, source) {
                    entities.push(entity);
                }
            }
            "interface_declaration" => {
                if let Some(entity) = extract_ts_interface(&actual_node, source) {
                    entities.push(entity);
                }
            }
            "type_alias_declaration" => {
                if let Some(entity) = extract_ts_type_alias(&actual_node, source) {
                    entities.push(entity);
                }
            }
            "function_declaration" => {
                if let Some(entity) = extract_ts_function(&actual_node, source) {
                    entities.push(entity);
                }
            }
            "enum_declaration" => {
                if let Some(entity) = extract_ts_enum(&actual_node, source) {
                    entities.push(entity);
                }
            }
            _ => {}
        }
    }

    entities
}

/// Unwrap `export` statement to get the inner declaration.
fn unwrap_export<'a>(node: Node<'a>) -> Node<'a> {
    if node.kind() == "export_statement" {
        node.child_by_field_name("declaration")
            .unwrap_or(node)
    } else {
        node
    }
}

/// Extract a TS class as DataModel.
fn extract_ts_class(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let mut attrs = Vec::new();
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "public_field_definition" {
                if let Some(prop) = child.child_by_field_name("name") {
                    if let Ok(pname) = prop.utf8_text(source.as_bytes()) {
                        let ty = child.child_by_field_name("type")
                            .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                            .unwrap_or("any");
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

    Some(IrEntity::DataModel {
        name,
        description: None,
        attributes: attrs,
    })
}

/// Extract a TS interface.
fn extract_ts_interface(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let mut methods = Vec::new();
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "method_signature" {
                if let Some(method_name) = child.child_by_field_name("name") {
                    if let Ok(mname) = method_name.utf8_text(source.as_bytes()) {
                        let params = extract_ts_params(&child, source);
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

/// Extract a TS type alias.
fn extract_ts_type_alias(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    Some(IrEntity::Interface {
        name,
        description: None,
        methods: vec![],
    })
}

/// Extract a TS function.
fn extract_ts_function(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let inputs = extract_ts_params(node, source);
    let return_type = node.child_by_field_name("return_type")
        .and_then(|r| r.utf8_text(source.as_bytes()).ok())
        .unwrap_or("void");

    let outputs = if return_type != "void" {
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

/// Extract a TS enum as DataModel.
fn extract_ts_enum(node: &Node, source: &str) -> Option<IrEntity> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

    let mut attrs = Vec::new();
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "enum_assignment" {
                if let Some(enum_name) = child.child_by_field_name("name") {
                    if let Ok(ename) = enum_name.utf8_text(source.as_bytes()) {
                        attrs.push(IrAttribute {
                            name: ename.to_string(),
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

/// Extract TypeScript function parameters.
fn extract_ts_params(node: &Node, source: &str) -> Vec<IrParam> {
    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.children(&mut cursor) {
            if child.kind() == "required_parameter" || child.kind() == "optional_parameter" {
                let name = child.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .unwrap_or("_");
                let ty = child.child_by_field_name("type")
                    .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                    .unwrap_or("any");
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
