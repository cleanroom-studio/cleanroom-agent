//! Naming Service — deterministic symbol naming with namespace support.

use std::collections::HashMap;

/// Namespace mode for fully-qualified name generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NamespaceMode {
    /// Prefix with the document/module name (default).
    FromDocumentName,
    /// User-specified custom namespace prefix.
    Manual,
    /// No namespace prefix — use bare names.
    None,
}

impl NamespaceMode {
    /// Parse from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "from_document_name" | "from-document-name" | "document" => Self::FromDocumentName,
            "manual" => Self::Manual,
            "none" => Self::None,
            _ => Self::FromDocumentName,
        }
    }

    /// Apply namespace to a name given a document name and optional custom prefix.
    pub fn apply(&self, name: &str, document_name: &str, custom_namespace: Option<&str>) -> String {
        match self {
            Self::FromDocumentName => {
                // Replace dots/slashes with language-appropriate separators
                let ns = document_name.replace('.', "::").replace('/', "::");
                format!("{}::{}", ns, name)
            }
            Self::Manual => {
                if let Some(prefix) = custom_namespace.filter(|p| !p.is_empty()) {
                    format!("{}::{}", prefix, name)
                } else {
                    name.to_string()
                }
            }
            Self::None => name.to_string(),
        }
    }
}

impl Default for NamespaceMode {
    fn default() -> Self {
        Self::FromDocumentName
    }
}

/// Name style for a programming language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameStyle {
    PascalCase,
    CamelCase,
    SnakeCase,
    UpperSnakeCase,
}

/// A supported programming language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Java,
    Go,
    CSharp,
}

impl Language {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rust" => Some(Self::Rust),
            "python" => Some(Self::Python),
            "typescript" | "ts" => Some(Self::TypeScript),
            "javascript" | "js" => Some(Self::JavaScript),
            "java" => Some(Self::Java),
            "go" => Some(Self::Go),
            "csharp" | "c#" => Some(Self::CSharp),
            _ => None,
        }
    }

    pub fn name_style(&self) -> NameStyle {
        match self {
            Self::Rust => NameStyle::SnakeCase,
            Self::Python => NameStyle::SnakeCase,
            Self::TypeScript | Self::JavaScript => NameStyle::CamelCase,
            Self::Java => NameStyle::PascalCase,
            Self::Go => NameStyle::PascalCase,
            Self::CSharp => NameStyle::PascalCase,
        }
    }

    #[allow(dead_code)]
    pub fn module_keyword(&self) -> Option<&'static str> {
        match self {
            Self::Rust => Some("mod"),
            Self::Python => Some(""),
            _ => None,
        }
    }
}

/// Deterministic name generator.
pub struct DeterministicNames {
    /// Language-specific rules.
    rules: HashMap<Language, NameStyle>,
}

impl DeterministicNames {
    /// Create a new name generator.
    pub fn new() -> Self {
        let mut rules = HashMap::new();
        rules.insert(Language::Rust, NameStyle::SnakeCase);
        rules.insert(Language::Python, NameStyle::SnakeCase);
        rules.insert(Language::TypeScript, NameStyle::CamelCase);
        rules.insert(Language::JavaScript, NameStyle::CamelCase);
        rules.insert(Language::Java, NameStyle::PascalCase);
        rules.insert(Language::Go, NameStyle::PascalCase);
        rules.insert(Language::CSharp, NameStyle::PascalCase);

        Self { rules }
    }

    /// Convert a name to the specified style.
    pub fn convert(&self, name: &str, style: NameStyle) -> String {
        let words = Self::split_words(name);
        match style {
            NameStyle::PascalCase => words
                .iter()
                .map(|w| Self::capitalize(w))
                .collect::<Vec<_>>()
                .join(""),
            NameStyle::CamelCase => {
                let mut result = Vec::new();
                for (i, w) in words.iter().enumerate() {
                    if i == 0 {
                        result.push(w.to_lowercase());
                    } else {
                        result.push(Self::capitalize(w));
                    }
                }
                result.join("")
            }
            NameStyle::SnakeCase => words
                .iter()
                .map(|w| w.to_lowercase())
                .collect::<Vec<_>>()
                .join("_"),
            NameStyle::UpperSnakeCase => words
                .iter()
                .map(|w| w.to_uppercase())
                .collect::<Vec<_>>()
                .join("_"),
        }
    }

    /// Convert a name for a specific language.
    pub fn convert_for_language(&self, name: &str, language: Language) -> String {
        let style = self.rules.get(&language).copied().unwrap_or(NameStyle::CamelCase);
        self.convert(name, style)
    }

    fn split_words(name: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current = String::new();

        for c in name.chars() {
            if c.is_uppercase() && !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            if c.is_alphanumeric() {
                current.push(c);
            } else if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
        }

        if !current.is_empty() {
            words.push(current);
        }

        words
    }

    fn capitalize(s: &str) -> String {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().chain(chars).collect(),
        }
    }
}

impl Default for DeterministicNames {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> DeterministicNames {
        DeterministicNames::new()
    }

    #[test]
    fn test_snake_case_basic() {
        let names = setup();
        assert_eq!(names.convert("UserName", NameStyle::SnakeCase), "user_name");
        assert_eq!(names.convert("userName", NameStyle::SnakeCase), "user_name");
        assert_eq!(names.convert("simple", NameStyle::SnakeCase), "simple");
    }

    #[test]
    fn test_camel_case_basic() {
        let names = setup();
        assert_eq!(names.convert("user_name", NameStyle::CamelCase), "userName");
        assert_eq!(names.convert("UserName", NameStyle::CamelCase), "userName");
    }

    #[test]
    fn test_pascal_case_basic() {
        let names = setup();
        assert_eq!(names.convert("user_name", NameStyle::PascalCase), "UserName");
        assert_eq!(names.convert("simple", NameStyle::PascalCase), "Simple");
    }

    #[test]
    fn test_upper_snake_case_basic() {
        let names = setup();
        assert_eq!(names.convert("userName", NameStyle::UpperSnakeCase), "USER_NAME");
        assert_eq!(names.convert("user_name", NameStyle::UpperSnakeCase), "USER_NAME");
    }

    #[test]
    fn test_rust_naming() {
        let names = setup();
        assert_eq!(names.convert_for_language("UserName", Language::Rust), "user_name");
        assert_eq!(names.convert_for_language("MyClass", Language::Rust), "my_class");
    }

    #[test]
    fn test_typescript_naming() {
        let names = setup();
        assert_eq!(names.convert_for_language("user_name", Language::TypeScript), "userName");
    }

    #[test]
    fn test_go_naming() {
        let names = setup();
        assert_eq!(names.convert_for_language("user_name", Language::Go), "UserName");
    }

    #[test]
    fn test_python_naming() {
        let names = setup();
        assert_eq!(names.convert_for_language("UserName", Language::Python), "user_name");
    }

    #[test]
    fn test_all_languages_have_styles() {
        let names = setup();
        for lang in &[Language::Rust, Language::Python, Language::TypeScript,
                      Language::JavaScript, Language::Java, Language::Go, Language::CSharp] {
            let result = names.convert_for_language("TestName", *lang);
            assert!(!result.is_empty(), "Language {:?} produced empty result", lang);
        }
    }

    #[test]
    fn test_from_str() {
        assert_eq!(Language::from_str("rust"), Some(Language::Rust));
        assert_eq!(Language::from_str("typescript"), Some(Language::TypeScript));
        assert_eq!(Language::from_str("ts"), Some(Language::TypeScript));
        assert_eq!(Language::from_str("python"), Some(Language::Python));
        assert_eq!(Language::from_str("unknown"), None);
    }

    #[test]
    fn test_empty_name() {
        let names = setup();
        assert_eq!(names.convert("", NameStyle::SnakeCase), "");
        assert_eq!(names.convert("", NameStyle::PascalCase), "");
    }
}