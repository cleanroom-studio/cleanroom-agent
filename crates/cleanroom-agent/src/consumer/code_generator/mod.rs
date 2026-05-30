//! Code generators for different target languages.
//!
//! This module provides language-specific code generators that transform
//! S.DEF (Software Definition Exchange Format) entities into source code.
//!
//! # Architecture
//!
//! Each language generator implements the [`CodeGenerator`] trait:
//! - [`RustGenerator`] for Rust code
//! - [`TypeScriptGenerator`] for TypeScript/JavaScript code
//! - [`PythonGenerator`] for Python code
//! - [`CGenerator`] for C code
//!
//! # Usage
//!
//! ```ignore
//! use cleanroom_agent::consumer::code_generator::{create_generator, CodeGenerator};
//! use sdef_core::{DataModel, DataAttribute};
//!
//! // Create a generator for the target language
//! let generator = create_generator("rust").expect("Rust is supported");
//!
//! // Create a sample data model
//! let model = DataModel {
//!     entity: "User".to_string(),
//!     description: Some("Represents a system user".to_string()),
//!     ..Default::default()
//! };
//!
//! // Generate code
//! let files = generator.generate_data_model(&model);
//! for file in files {
//!     println!("Generated: {}", file.file_path);
//! }
//! ```

pub mod rust_gen;
pub mod typescript_gen;
pub mod python_gen;
pub mod c_gen;
pub mod templates;

use sdef_core::{DataModel, InterfaceContract, ClassContract, FunctionSpec};

/// Output of code generation.
///
/// Contains the generated source code and metadata about the output.
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    /// File path relative to the output root directory
    pub file_path: String,
    /// Generated source code content
    pub content: String,
    /// Programming language identifier (rust, typescript, python, c)
    pub language: String,
}

/// Language-specific code generator trait.
///
/// Implement this trait to add support for a new programming language.
/// Each generator transforms S.DEF entities into language-appropriate code.
///
/// # Example
///
/// ```no_run
/// use cleanroom_agent::consumer::code_generator::{GeneratedCode, CodeGenerator};
/// use sdef_core::{DataModel, InterfaceContract, ClassContract, FunctionSpec};
///
/// struct MyGenerator;
///
/// impl CodeGenerator for MyGenerator {
///     fn generate_data_model(&self, model: &DataModel) -> Vec<GeneratedCode> {
///         // Generate data model code in MyLanguage
///         vec![]
///     }
///
///     fn generate_interface(&self, interface: &InterfaceContract) -> Vec<GeneratedCode> {
///         // Generate interface code in MyLanguage
///         vec![]
///     }
///
///     fn generate_class(&self, class: &ClassContract) -> Vec<GeneratedCode> {
///         // Generate class code in MyLanguage
///         vec![]
///     }
///
///     fn generate_function(&self, func: &FunctionSpec) -> GeneratedCode {
///         // Generate function code in MyLanguage
///         todo!()
///     }
///
///     fn file_extension(&self) -> &str {
///         "mylang"
///     }
///
///     fn language_id(&self) -> &str {
///         "mylanguage"
///     }
/// }
/// ```
pub trait CodeGenerator {
    /// Generate code from a data model entity.
    ///
    /// Transforms a S.DEF [`DataModel`] into source code appropriate
    /// for the target language (e.g., struct, class, dataclass).
    ///
    /// # Arguments
    /// * `model` - The data model entity to generate code from
    ///
    /// # Returns
    /// A vector of [`GeneratedCode`] files to be written
    fn generate_data_model(&self, model: &DataModel) -> Vec<GeneratedCode>;
    
    /// Generate code from an interface contract.
    ///
    /// Transforms a S.DEF [`InterfaceContract`] into source code appropriate
    /// for the target language (e.g., trait, interface, abstract class).
    ///
    /// # Arguments
    /// * `interface` - The interface contract to generate code from
    ///
    /// # Returns
    /// A vector of [`GeneratedCode`] files to be written
    fn generate_interface(&self, interface: &InterfaceContract) -> Vec<GeneratedCode>;
    
    /// Generate code from a class contract.
    ///
    /// Transforms a S.DEF [`ClassContract`] into source code appropriate
    /// for the target language.
    ///
    /// # Arguments
    /// * `class` - The class contract to generate code from
    ///
    /// # Returns
    /// A vector of [`GeneratedCode`] files to be written
    fn generate_class(&self, class: &ClassContract) -> Vec<GeneratedCode>;
    
    /// Generate code from a function specification.
    ///
    /// Transforms a S.DEF [`FunctionSpec`] into source code appropriate
    /// for the target language.
    ///
    /// # Arguments
    /// * `func` - The function specification to generate code from
    ///
    /// # Returns
    /// A [`GeneratedCode`] file to be written
    fn generate_function(&self, func: &FunctionSpec) -> GeneratedCode;
    
    /// Get the file extension for files generated in this language.
    ///
    /// # Returns
    /// File extension without the dot (e.g., "rs", "ts", "py")
    fn file_extension(&self) -> &str;
    
    /// Get the language identifier string.
    ///
    /// # Returns
    /// Language identifier (e.g., "rust", "typescript", "python", "c")
    fn language_id(&self) -> &str;
}

/// Create a code generator for the specified language.
///
/// Returns a boxed generator instance that can generate code in the requested
/// programming language. If the language is not supported, returns `None`.
///
/// # Arguments
/// * `language` - Programming language identifier (case-insensitive)
///   - "rust": Rust code generator
///   - "typescript", "ts", "javascript", "js": TypeScript code generator
///   - "python", "py": Python code generator
///   - "c", "c++", "cpp", "h": C code generator
///
/// # Returns
/// `Some(Box<dyn CodeGenerator + Send + Sync>)` if language is supported,
/// `None` otherwise.
///
/// # Example
///
/// ```no_run
/// use cleanroom_agent::consumer::code_generator::create_generator;
///
/// let rust_gen = create_generator("rust");
/// assert!(rust_gen.is_some());
///
/// let unknown = create_generator("brainfuck");
/// assert!(unknown.is_none());
/// ```
pub fn create_generator(language: &str) -> Option<Box<dyn CodeGenerator + Send + Sync>> {
    match language.to_lowercase().as_str() {
        "rust" => Some(Box::new(rust_gen::RustGenerator)),
        "typescript" | "ts" => Some(Box::new(typescript_gen::TypeScriptGenerator)),
        "javascript" | "js" => Some(Box::new(typescript_gen::TypeScriptGenerator)),
        "python" | "py" => Some(Box::new(python_gen::PythonGenerator)),
        "c" | "c++" | "cpp" | "h" => Some(Box::new(c_gen::CGenerator)),
        _ => None,
    }
}