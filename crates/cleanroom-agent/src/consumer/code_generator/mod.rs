//! Code generators for different target languages.

pub mod rust_gen;
pub mod typescript_gen;
pub mod python_gen;
pub mod templates;

use sdef_core::{DataModel, InterfaceContract, ClassContract, FunctionSpec};

/// Output of code generation.
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    /// File path relative to output root.
    pub file_path: String,
    /// Generated source code content.
    pub content: String,
    /// Language identifier.
    pub language: String,
}

/// Language-specific code generator trait.
pub trait CodeGenerator {
    /// Generate code from a data model.
    fn generate_data_model(&self, model: &DataModel) -> Vec<GeneratedCode>;
    
    /// Generate code from an interface contract.
    fn generate_interface(&self, interface: &InterfaceContract) -> Vec<GeneratedCode>;
    
    /// Generate code from a class contract.
    fn generate_class(&self, class: &ClassContract) -> Vec<GeneratedCode>;
    
    /// Generate code from a function spec.
    fn generate_function(&self, func: &FunctionSpec) -> GeneratedCode;
    
    /// Get the file extension for this language.
    fn file_extension(&self) -> &str;
    
    /// Get the language identifier.
    fn language_id(&self) -> &str;
}

/// Create a code generator for the specified language.
pub fn create_generator(language: &str) -> Option<Box<dyn CodeGenerator + Send + Sync>> {
    match language.to_lowercase().as_str() {
        "rust" => Some(Box::new(rust_gen::RustGenerator)),
        "typescript" | "ts" => Some(Box::new(typescript_gen::TypeScriptGenerator)),
        "javascript" | "js" => Some(Box::new(typescript_gen::TypeScriptGenerator)),
        "python" | "py" => Some(Box::new(python_gen::PythonGenerator)),
        _ => None,
    }
}