//! C code generator — generates .h and .c files from S.DEF models.

use super::{CodeGenerator, GeneratedCode};
use sdef_core::{DataModel, InterfaceContract, ClassContract, FunctionSpec};

/// C language code generator.
pub struct CGenerator;

impl CodeGenerator for CGenerator {
    fn generate_data_model(&self, model: &DataModel) -> Vec<GeneratedCode> {
        let mut files = Vec::new();

        // Header file
        let mut header = String::new();
        let guard = format!("{}_H", model.entity.to_uppercase());

        header.push_str(&format!("#ifndef {}\n#define {}\n\n", guard, guard));

        if let Some(desc) = &model.description {
            for line in desc.lines() {
                header.push_str(&format!(" * {}\n", line));
            }
        }

        header.push_str(&format!("typedef struct {{\n",));

        if let Some(attrs) = &model.attributes {
            for attr in attrs {
                let c_type = c_type_name(&attr.attr_type);
                let c_name = to_c_identifier(&attr.name);
                header.push_str(&format!("    {} {};\n", c_type, c_name));
            }
        }

        header.push_str(&format!("}} {};\n\n", to_pascal_case(&model.entity)));

        // Constructor / destructor declarations
        header.push_str(&format!(
            "{} *{}_new();\n",
            to_pascal_case(&model.entity),
            to_snake_case(&model.entity)
        ));
        header.push_str(&format!(
            "void {}_free({} *ptr);\n",
            to_snake_case(&model.entity),
            to_pascal_case(&model.entity)
        ));

        header.push_str(&format!("#endif /* {} */\n", guard));

        let hdr_fname = format!("{}.h", to_snake_case(&model.entity));
        files.push(GeneratedCode {
            file_path: hdr_fname,
            content: header,
            language: "c".to_string(),
        });

        // Implementation file
        let mut impl_code = String::new();
        impl_code.push_str(&format!("#include \"{}.h\"\n", to_snake_case(&model.entity)));
        impl_code.push_str("#include <stdlib.h>\n\n");

        impl_code.push_str(&format!(
            "{} *{}_new() {{\n",
            to_pascal_case(&model.entity),
            to_snake_case(&model.entity)
        ));
        impl_code.push_str(&format!(
            "    {} *ptr = malloc(sizeof({}));\n",
            to_pascal_case(&model.entity),
            to_pascal_case(&model.entity)
        ));
        if let Some(attrs) = &model.attributes {
            for attr in attrs {
                let c_name = to_c_identifier(&attr.name);
                if attr.attr_type.to_lowercase() == "string" || attr.attr_type == "char*" || attr.attr_type == "char *" {
                    impl_code.push_str(&format!("    ptr->{} = NULL;\n", c_name));
                } else {
                    impl_code.push_str(&format!("    ptr->{} = 0;\n", c_name));
                }
            }
        }
        impl_code.push_str("    return ptr;\n}\n\n");

        impl_code.push_str(&format!(
            "void {}_free({} *ptr) {{\n",
            to_snake_case(&model.entity),
            to_pascal_case(&model.entity)
        ));
        impl_code.push_str("    if (ptr) {\n");
        if let Some(attrs) = &model.attributes {
            for attr in attrs {
                let c_name = to_c_identifier(&attr.name);
                if attr.attr_type.to_lowercase() == "string" || attr.attr_type == "char*" || attr.attr_type == "char *" {
                    impl_code.push_str(&format!("        free(ptr->{});\n", c_name));
                }
            }
        }
        impl_code.push_str("        free(ptr);\n    }\n}\n");

        let impl_fname = format!("{}.c", to_snake_case(&model.entity));
        files.push(GeneratedCode {
            file_path: impl_fname,
            content: impl_code,
            language: "c".to_string(),
        });

        files
    }

    fn generate_interface(&self, interface: &InterfaceContract) -> Vec<GeneratedCode> {
        let mut header = String::new();
        let guard = format!("{}_H", interface.name.to_uppercase());

        header.push_str(&format!("#ifndef {}\n#define {}\n\n", guard, guard));

        if let Some(desc) = &interface.description {
            for line in desc.lines() {
                header.push_str(&format!(" * {}\n", line));
            }
        }

        // Generate function pointer declarations for interface methods
        if let Some(methods) = &interface.methods {
            for method in methods {
                let ret_type = "int"; // default
                header.push_str(&format!(
                    "{} (*{})(void *ctx);\n",
                    ret_type,
                    to_snake_case(&method.signature.split('(').next().unwrap_or(&method.signature)),
                ));
            }
        }

        header.push_str(&format!("\n#endif /* {} */\n", guard));

        vec![GeneratedCode {
            file_path: format!("{}.h", to_snake_case(&interface.name)),
            content: header,
            language: "c".to_string(),
        }]
    }

    fn generate_class(&self, _class: &ClassContract) -> Vec<GeneratedCode> {
        vec![]
    }

    fn generate_function(&self, func: &FunctionSpec) -> GeneratedCode {
        let mut code = String::new();

        if let Some(desc) = &func.description {
            for line in desc.lines() {
                code.push_str(&format!("// {}\n", line));
            }
        }

        // Return type
        let ret_type = func.outputs.as_ref()
            .and_then(|o| o.first())
            .map(|o| c_type_name(&o.param_type))
            .unwrap_or("void");

        let params_str = func.inputs.as_ref()
            .map(|inputs| {
                inputs.iter()
                    .map(|p| format!("{} {}", c_type_name(&p.param_type), to_c_identifier(&p.name)))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        code.push_str(&format!(
            "{} {}({}) {{\n",
            ret_type,
            to_snake_case(&func.name),
            params_str,
        ));

        if ret_type != "void" {
            code.push_str("    // TODO: implement\n");
            code.push_str("    return 0;\n");
        } else {
            code.push_str("    // TODO: implement\n");
        }
        code.push_str("}\n");

        GeneratedCode {
            file_path: format!("{}.c", to_snake_case(&func.name)),
            content: code,
            language: "c".to_string(),
        }
    }

    fn file_extension(&self) -> &str {
        "c"
    }

    fn language_id(&self) -> &str {
        "c"
    }
}

// ============ Helpers ============

fn c_type_name(sdef_type: &str) -> &str {
    match sdef_type.to_lowercase().as_str() {
        "uuid" | "string" => "char*",
        "int" | "integer" | "i32" => "int",
        "i64" | "long" => "long long",
        "f32" | "float" => "float",
        "f64" | "double" => "double",
        "bool" | "boolean" => "int",
        "void" => "void",
        _ => sdef_type, // use as-is (e.g., "dict*", "sds")
    }
}

fn to_c_identifier(name: &str) -> String {
    let mut result = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_alphanumeric() || ch == '_' {
            result.push(ch);
        } else if ch == '-' || ch == ' ' {
            result.push('_');
        } else if i > 0 && ch.is_uppercase() {
            // camelCase/PascalCase → snake_case
            result.push('_');
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push(ch);
        }
    }
    result.to_lowercase()
}

fn to_snake_case(name: &str) -> String {
    let mut result = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
            result.push(ch.to_ascii_lowercase());
        } else if ch.is_alphanumeric() || ch == '_' {
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push('_');
        }
    }
    result
}

fn to_pascal_case(name: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    for ch in name.chars() {
        if ch == '_' || ch == '-' || ch == ' ' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(name: &str) -> DataModel {
        DataModel {
            entity: name.to_string(),
            status: None,
            version: None,
            deprecated: None,
            description: Some("A test model".to_string()),
            logical_model: None,
            attributes: Some(vec![
                sdef_core::DataAttribute {
                    name: "id".to_string(),
                    attr_type: "UUID".to_string(),
                    format: None,
                    description: Some("Primary key".to_string()),
                    required: true,
                    default: None,
                    identity: true,
                    generated: true,
                    unique: true,
                    internal: false,
                    deprecated: false,
                    compatibility: None,
                    constraints: None,
                    origin: None,
                },
                sdef_core::DataAttribute {
                    name: "userName".to_string(),
                    attr_type: "string".to_string(),
                    format: None,
                    description: Some("Username".to_string()),
                    required: true,
                    default: None,
                    identity: false,
                    generated: false,
                    unique: false,
                    internal: false,
                    deprecated: false,
                    compatibility: None,
                    constraints: None,
                    origin: None,
                },
            ]),
            relationships: None,
            validation_rules: None,
            physical_design: None,
            origin: None,
        }
    }

    #[test]
    fn test_generate_c_struct() {
        let gen = CGenerator;
        let model = make_model("User");
        let files = gen.generate_data_model(&model);
        assert!(files.len() >= 2, "Should produce .h and .c files, got {} files: {:?}", files.len(), files.iter().map(|f| &f.file_path).collect::<Vec<_>>());

        let header = files.iter().find(|f| f.file_path.ends_with(".h")).unwrap();
        eprintln!("=== HEADER ===\n{}", header.content);
        assert!(header.content.contains("typedef struct"));
        assert!(header.content.contains("char*") || header.content.contains("char *"), "Should contain char* type");
        assert!(header.content.contains("User_new") || header.content.contains("user_new"), "Should contain constructor");
        assert!(header.content.contains("User_free") || header.content.contains("user_free"), "Should contain destructor");

        let impl_file = files.iter().find(|f| f.file_path.ends_with(".c")).unwrap();
        assert!(impl_file.content.contains("_new()"), "Should contain _new() constructor");
        assert!(impl_file.content.contains("_free("), "Should contain _free() destructor");
    }

    #[test]
    fn test_generate_function() {
        let gen = CGenerator;
        let func = FunctionSpec {
            name: "calculateSum".to_string(),
            description: Some("Calculate sum of two ints".to_string()),
            inputs: Some(vec![
                sdef_core::FunctionParam {
                    name: "a".to_string(),
                    param_type: "int".to_string(),
                    description: None,
                },
                sdef_core::FunctionParam {
                    name: "b".to_string(),
                    param_type: "int".to_string(),
                    description: None,
                },
            ]),
            outputs: Some(vec![
                sdef_core::FunctionParam {
                    name: "result".to_string(),
                    param_type: "int".to_string(),
                    description: None,
                },
            ]),
            logic: None,
            complexity: None,
            pure_function: false,
            edge_cases: None,
            origin: None,
        };
        let code = gen.generate_function(&func);
        assert!(code.content.contains("int calculate_sum(int a, int b)"));
    }
}
