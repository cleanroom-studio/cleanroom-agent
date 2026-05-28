//! Language detection utilities.

use std::collections::HashMap;

/// Known file extensions mapped to languages.
const LANGUAGE_EXTENSIONS: &[(&str, &str)] = &[
    // Rust
    ("rs", "rust"),
    // TypeScript/JavaScript
    ("ts", "typescript"),
    ("tsx", "typescript"),
    ("js", "javascript"),
    ("jsx", "javascript"),
    ("mjs", "javascript"),
    ("cjs", "javascript"),
    // Python
    ("py", "python"),
    ("pyi", "python"),
    // Go
    ("go", "go"),
    // Java
    ("java", "java"),
    // C/C++
    ("c", "c"),
    ("cc", "c++"),
    ("cpp", "c++"),
    ("cxx", "c++"),
    ("h", "c"),
    ("hpp", "c++"),
    // C#
    ("cs", "csharp"),
    // Ruby
    ("rb", "ruby"),
    // PHP
    ("php", "php"),
    // Swift
    ("swift", "swift"),
    // Kotlin
    ("kt", "kotlin"),
    ("kts", "kotlin"),
    // Scala
    ("scala", "scala"),
    ("sc", "scala"),
    // Shell
    ("sh", "shell"),
    ("bash", "shell"),
    ("zsh", "shell"),
    // SQL
    ("sql", "sql"),
    // HTML
    ("html", "html"),
    ("htm", "html"),
    // CSS
    ("css", "css"),
    ("scss", "scss"),
    ("sass", "sass"),
    ("less", "less"),
    // Vue
    ("vue", "vue"),
    // Svelte
    ("svelte", "svelte"),
    // YAML
    ("yaml", "yaml"),
    ("yml", "yaml"),
    // JSON
    ("json", "json"),
    // TOML
    ("toml", "toml"),
    // Markdown
    ("md", "markdown"),
    ("mdx", "markdown"),
    // XML
    ("xml", "xml"),
    // Docker
    ("dockerfile", "dockerfile"),
];

lazy_static::lazy_static! {
    static ref EXTENSION_MAP: HashMap<&'static str, &'static str> = {
        let mut map = HashMap::new();
        for (ext, lang) in LANGUAGE_EXTENSIONS {
            map.insert(*ext, *lang);
        }
        map
    };
}

/// Detect language from file extension.
pub fn detect_language(file_path: &str) -> Option<&'static str> {
    let extension = std::path::Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    extension.and_then(|ext| EXTENSION_MAP.get(ext.as_str()).copied())
}

/// Get all supported languages.
pub fn supported_languages() -> Vec<&'static str> {
    let mut languages: Vec<&str> = EXTENSION_MAP.values().copied().collect();
    languages.sort();
    languages.dedup();
    languages
}

/// Check if a language is supported.
pub fn is_language_supported(language: &str) -> bool {
    supported_languages().contains(&language)
}