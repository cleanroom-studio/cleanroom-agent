//! Rust code generator.
//!
//! Transforms S.DEF (Software Definition Exchange Format) entities into
//! Rust source code, including structs, traits, and functions.
//!
//! # Generated Code
//!
//! - Data models become `pub struct` with serde derives
//! - Interfaces become `pub trait`
//! - Functions become `pub fn` with typed parameters
//! - Field names use snake_case
//! - Types are mapped from S.DEF types to Rust types
//!
//! # Type Mapping
//!
//! | S.DEF Type | Rust Type |
//! |------------|-----------|
//! | UUID       | uuid::Uuid |
//! | timestamp  | chrono::DateTime<chrono::Utc> |
//! | string     | String |
//! | integer   | i32 |
//! | int64      | i64 |
//! | boolean    | bool |
//! | json       | serde_json::Value |

use super::{CodeGenerator, GeneratedCode};
use sdef_core::{DataModel, InterfaceContract, ClassContract, FunctionSpec};

/// Rust language code generator.
///
/// Implements the [`CodeGenerator`] trait to produce Rust source code
/// from S.DEF entities. Generates structs with serde derives, traits,
/// and free functions with proper type mappings.
pub struct RustGenerator;

impl CodeGenerator for RustGenerator {
    fn generate_data_model(&self, model: &DataModel) -> Vec<GeneratedCode> {
        let mut output = String::new();
        output.push_str("use crate::*;\n\n");

        // Deduplicate attributes by name to avoid duplicate field definitions
        let deduped_attrs: Vec<&sdef_core::DataAttribute> = model.attributes.as_ref()
            .map(|attrs| {
                let mut seen = std::collections::HashSet::new();
                attrs.iter().filter(|a| seen.insert(a.name.as_str())).collect()
            })
            .unwrap_or_default();

        // Genereate #[derive] BEFORE struct definition (required by Rust)
        if !deduped_attrs.is_empty() {
            output.push_str("#[derive(serde::Serialize, serde::Deserialize)]\n");
        }

        // Generate struct
        if let Some(desc) = &model.description {
            output.push_str("/// ");
            output.push_str(desc);
            output.push('\n');
        }
        output.push_str(&format!("pub struct {} {{\n", to_pascal_case(&model.entity)));
        
        // Generate fields
        for attr in &deduped_attrs {
            if let Some(desc) = &attr.description {
                output.push_str("    /// ");
                output.push_str(desc);
                output.push('\n');
            }
            let ty = rust_type(&attr.attr_type, attr.required);
            output.push_str(&format!(
                "    pub {}: {},\n",
                to_snake_case(&attr.name),
                ty,
            ));
        }
        
        output.push_str("}\n");
        
        // Generate impl block
        output.push_str(&format!("\nimpl {} {{\n", to_pascal_case(&model.entity)));
        output.push_str("    /// Create a new instance.\n");
        output.push_str("    pub fn new(");

        // Constructor params (skip generated/internal + auto-fields)
        let ctor_params: Vec<String> = deduped_attrs.iter()
            .filter(|a| !a.generated && !a.internal
                && a.name != "id"
                && a.name != "created_at")
            .map(|attr| {
                format!("{}: {}", to_snake_case(&attr.name), rust_type(&attr.attr_type, attr.required))
            })
            .collect();
        output.push_str(&ctor_params.join(", "));
        output.push_str(") -> Self {\n");
        output.push_str(&format!("        Self {{\n"));
        // Constructor body — non-generated, non-internal fields + defaults for auto-fields
        for attr in deduped_attrs.iter().filter(|a| !a.generated && !a.internal) {
            if attr.name == "id" {
                output.push_str("            id: uuid::Uuid::new_v4(),\n");
            } else if attr.name == "created_at" {
                output.push_str("            created_at: Some(chrono::Utc::now()),\n");
            } else {
                output.push_str(&format!(
                    "            {}: {},\n",
                    to_snake_case(&attr.name),
                    to_snake_case(&attr.name)
                ));
            }
        }
        output.push_str("        }\n    }\n}\n");
        
        vec![GeneratedCode {
            file_path: format!("{}.rs", to_snake_case(&model.entity)),
            content: output,
            language: "rust".to_string(),
        }]
    }
    
    fn generate_interface(&self, interface: &InterfaceContract) -> Vec<GeneratedCode> {
        let mut output = String::new();
        
        if let Some(desc) = &interface.description {
            output.push_str("/// ");
            output.push_str(desc);
            output.push('\n');
        }
        output.push_str(&format!("pub trait {} {{\n", to_pascal_case(&interface.name)));
        
        if let Some(methods) = &interface.methods {
            for method in methods {
                if let Some(behavior) = &method.behavior {
                    output.push_str("    /// ");
                    output.push_str(behavior);
                    output.push('\n');
                }
                // Extract method name from signature
                let method_name = method.signature.split('(').next().unwrap_or(&method.signature);
                let params = method.signature.split('(').nth(1)
                    .map(|s| s.trim_end_matches(')').to_string())
                    .unwrap_or_default();
                output.push_str(&format!("    fn {}({});\n", to_snake_case(method_name), params));
            }
        }
        
        output.push_str("}\n");
        
        vec![GeneratedCode {
            file_path: format!("{}.rs", to_snake_case(&interface.name)),
            content: output,
            language: "rust".to_string(),
        }]
    }
    
    fn generate_class(&self, class: &ClassContract) -> Vec<GeneratedCode> {
        let mut output = String::new();
        
        // Derives
        let mut derives: Vec<String> = vec!["Debug".to_string(), "Clone".to_string()];
        if let Some(implements) = &class.implements {
            if !implements.is_empty() {
                derives.extend(implements.iter().cloned());
            }
        }
        output.push_str(&format!("#[derive({})]\n", derives.join(", ")));
        
        if let Some(desc) = &class.description {
            output.push_str("/// ");
            output.push_str(desc);
            output.push('\n');
        }
        output.push_str(&format!("pub struct {} {{\n", to_pascal_case(&class.name)));
        output.push_str("    // TODO: Add fields from data model\n");
        output.push_str("}\n\n");
        
        // Implement interfaces
        if let Some(implements) = &class.implements {
            for impl_trait in implements {
                output.push_str(&format!(
                    "impl {} for {} {{\n",
                    impl_trait,
                    to_pascal_case(&class.name)
                ));
                output.push_str("    // TODO: Implement trait methods\n}\n\n");
            }
        }
        
        // Impl block
        output.push_str(&format!("impl {} {{\n", to_pascal_case(&class.name)));
        output.push_str("    pub fn new() -> Self {\n");
        output.push_str("        Self { }\n    }\n}\n");
        
        vec![GeneratedCode {
            file_path: format!("{}.rs", to_snake_case(&class.name)),
            content: output,
            language: "rust".to_string(),
        }]
    }
    
    fn generate_function(&self, func: &FunctionSpec) -> GeneratedCode {
        let mut output = String::new();
        
        if let Some(desc) = &func.description {
            output.push_str("/// ");
            output.push_str(desc);
            output.push('\n');
        }
        
        let fn_name = to_snake_case(&func.name);
        
        // Generate parameters
        let params = if let Some(inputs) = &func.inputs {
            inputs.iter()
                .map(|p| format!("{}: {}", to_snake_case(&p.name), rust_type(&p.param_type, false)))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            String::new()
        };
        
        // Generate return type
        let ret_type = if let Some(outputs) = &func.outputs {
            if outputs.len() == 1 {
                rust_type(&outputs[0].param_type, false)
            } else if outputs.len() > 1 {
                let types: Vec<String> = outputs.iter()
                    .map(|o| rust_type(&o.param_type, false))
                    .collect();
                format!("({})", types.join(", "))
            } else {
                "()".to_string()
            }
        } else {
            "()".to_string()
        };
        
        output.push_str(&format!("pub fn {}({}) -> {} {{\n", fn_name, params, ret_type));
        
        if let Some(logic) = &func.logic {
            output.push_str("    // ");
            output.push_str(logic);
            output.push('\n');
        }
        
        output.push_str("    todo!()\n}\n");
        
        GeneratedCode {
            file_path: format!("{}.rs", fn_name),
            content: output,
            language: "rust".to_string(),
        }
    }
    
    fn file_extension(&self) -> &str {
        "rs"
    }
    
    fn language_id(&self) -> &str {
        "rust"
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
    let mut words: Vec<String> = Vec::new();

    // First split on non-alphanumeric characters (underscores, hyphens, etc.)
    for segment in s.split(|c: char| !c.is_alphanumeric()).filter(|p| !p.is_empty()) {
        let chars: Vec<char> = segment.chars().collect();
        let mut current = String::new();

        for i in 0..chars.len() {
            let c = chars[i];
            let next = chars.get(i + 1);

            if current.is_empty() {
                current.push(c);
                continue;
            }

            // CamelCase boundary: uppercase followed by lowercase starts a new word
            if c.is_uppercase() {
                if let Some(&n) = next {
                    if n.is_lowercase() {
                        // Check if current is an uppercase acronym run (e.g., "HTTP" + "Server")
                        if current.chars().all(|ch| ch.is_uppercase() || ch.is_digit(10)) {
                            if current.len() > 1 {
                                words.push(current.clone());
                                current.clear();
                            } else {
                                // Single uppercase char = part of next word (e.g., "getUser")
                                words.push(current.clone());
                                current.clear();
                            }
                        } else {
                            words.push(current.clone());
                            current.clear();
                        }
                    }
                }
            }

            current.push(c);
        }

        if !current.is_empty() {
            words.push(current);
        }
    }

    words.iter()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join("_")
}

fn rust_type(sdef_type: &str, required: bool) -> String {
    let lower = sdef_type.to_lowercase();
    let trimmed = lower.trim();

    let base = if trimmed.ends_with('*') && !trimmed.contains('(') {
        // Pointer types: strip trailing asterisk
        let inner = trimmed.trim_end_matches('*').trim();
        let inner_rust = rust_type(inner, true);
        if inner_rust == "String" || inner_rust == "i8" {
            "String".to_string()
        } else if inner_rust.starts_with('[') {
            // Array type: Vec already handled
            format!("Vec<{}>", inner_rust.trim_start_matches('[').trim_end_matches(']'))
        } else {
            format!("Option<Box<{}>>", inner_rust)
        }
    } else {
        let s = match trimmed {
            // String types
            "string" | "text" | "varchar" | "sds" => "String",
            "char" => "i8",
            "unsigned char" => "u8",
            // Signed integers (standard C)
            "integer" | "int" | "int32" => "i32",
            "int8" | "signed char" => "i8",
            "int16" | "short" | "short int" => "i16",
            "int64" | "bigint" | "long" | "long int" | "long long" | "long long int" => "i64",
            // C99 stdint.h fixed-width types (lowercase)
            "int8_t" => "i8",
            "int16_t" => "i16",
            "int32_t" => "i32",
            "int64_t" => "i64",
            // Unsigned integers (standard C)
            "uint" | "unsigned" | "unsigned int" => "u32",
            "uint8" => "u8",
            "uint16" | "unsigned short" | "unsigned short int" => "u16",
            "uint32" => "u32",
            "uint64" | "unsigned long" | "unsigned long int" | "u64" | "unsigned long long" | "unsigned long long int" => "u64",
            "size_t" | "usize" => "usize",
            "ssize_t" => "isize",
            // C99 stdint.h unsigned fixed-width types (lowercase)
            "uint8_t" => "u8",
            "uint16_t" => "u16",
            "uint32_t" => "u32",
            "uint64_t" => "u64",
            // Floating point
            "float" => "f32",
            "double" | "long double" | "decimal" => "f64",
            // Boolean
            "boolean" | "bool" | "_bool" | "boolean_t" => "bool",
            // UUID
            "uuid" => "uuid::Uuid",
            // Time
            "timestamp" | "datetime" => "chrono::DateTime<chrono::Utc>",
            "date" => "chrono::NaiveDate",
            "time_t" => "i64",
            "mstime_t" | "monotime" => "u64",
            "lu_byte" => "u8",
            // Complex types
            "json" | "jsonb" => "serde_json::Value",
            "bytes" | "bytea" => "Vec<u8>",
            "void" => "()",
            "any" => "serde_json::Value",
            // Everything else: treat as a named type reference (PascalCase)
            other => {
                let pascal = to_pascal_case(other);
                return pascal;
            }
        };
        s.to_string()
    };

    if required {
        base
    } else {
        format!("Option<{}>", base)
    }
}