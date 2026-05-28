//! Naming Service — deterministic symbol naming.

use std::collections::HashMap;

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