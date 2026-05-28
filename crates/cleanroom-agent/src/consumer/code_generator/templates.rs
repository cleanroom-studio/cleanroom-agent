//! Code generation templates for various output patterns.
//!
//! Templates are embedded at compile time using `include_str!`
//! and used by language-specific code generators.

/// Template for a Rust data model (struct).
pub const RUST_STRUCT_TEMPLATE: &str = r#"
{description_doc}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {name} {
{fields}
}
"#;

/// Template for a Rust data model field.
pub const RUST_FIELD_TEMPLATE: &str = r#"
    {description_doc}
    pub {field_name}: {field_type},
"#;

/// Template for a Rust trait (interface).
pub const RUST_TRAIT_TEMPLATE: &str = r#"
{description_doc}
pub trait {name} {
{methods}
}
"#;

/// Template for a Rust impl block.
pub const RUST_IMPL_TEMPLATE: &str = r#"
impl {name} {
{methods}
}
"#;

/// Template for a TypeScript interface.
pub const TS_INTERFACE_TEMPLATE: &str = r#"
{description_doc}
export interface {name} {
{fields}
}
"#;

/// Template for a TypeScript class.
pub const TS_CLASS_TEMPLATE: &str = r#"
{description_doc}
export class {name} {
{fields}
{constructor}

{methods}
}
"#;

/// Template for a Python dataclass.
pub const PYTHON_DATACLASS_TEMPLATE: &str = r#"
{description_doc}
@dataclass
class {name}:
{fields}
"#;

/// Template for a Python function.
pub const PYTHON_FUNCTION_TEMPLATE: &str = r#"
{description_doc}
def {name}({params}) -> {return_type}:
{body}
"#;

/// Template for a Rust function.
pub const RUST_FUNCTION_TEMPLATE: &str = r#"
{description_doc}
pub fn {name}({params}) -> {return_type} {
{body}
}
"#;

/// Template for a TypeScript function.
pub const TS_FUNCTION_TEMPLATE: &str = r#"
{description_doc}
export function {name}({params}): {return_type} {
{body}
}
"#;

/// Render a description into a doc comment.
pub fn description_to_doc_comment(language: &str, text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    match language {
        "rust" => format!("/// {}", text),
        "typescript" | "javascript" => format!("/** {} */", text),
        "python" => format!("\"\"\"{}\"\"\"", text),
        _ => format!("// {}", text),
    }
}
