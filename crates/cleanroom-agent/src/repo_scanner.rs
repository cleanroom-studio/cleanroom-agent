//! Repository scanner — traverses file trees and discovers source files.
//!
//! This module provides functionality for scanning directories and discovering
//! source code files. It supports filtering by file type, excluding common
//! directories like node_modules and target, and detecting programming languages
//! based on file extensions.
//!
//! # Supported Languages
//!
//! Files are detected based on extension mapping:
//! - `.rs` → Rust
//! - `.ts`, `.tsx` → TypeScript
//! - `.js`, `.jsx`, `.mjs`, `.cjs` → JavaScript
//! - `.py`, `.pyi` → Python
//! - `.go` → Go
//! - `.java` → Java
//! - `.c`, `.h` → C
//! - `.cpp`, `.cc`, `.cxx`, `.hpp` → C++
//!
//! # Excluded Patterns
//!
//! By default, the following are excluded:
//! - `node_modules/`, `target/`, `.git/`, `vendor/`
//! - `*.lock` files
//! - `.siefignore` for custom exclusions
//! - Hidden files (starting with `.`)

use std::path::{Path, PathBuf};
use tracing::{info, instrument};

/// A discovered source file in the repository.
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// Detected language based on extension.
    pub language: Option<String>,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Relative path from repo root.
    pub relative_path: PathBuf,
}

/// Scanner configuration.
#[derive(Debug, Clone)]
pub struct ScanConfig {
    /// Root directory to scan.
    pub root: PathBuf,
    /// Additional exclude patterns beyond .gitignore.
    pub exclude_patterns: Vec<String>,
    /// Whether to include dotfiles.
    pub include_dotfiles: bool,
    /// Maximum file size to include (bytes).
    pub max_file_size: u64,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            exclude_patterns: vec![
                "node_modules".to_string(),
                "target".to_string(),
                ".git".to_string(),
                "vendor".to_string(),
                ".siefignore".to_string(),
                "*.lock".to_string(),
                "tests/fixtures".to_string(),
            ],
            include_dotfiles: false,
            max_file_size: 1_000_000, // 1MB
        }
    }
}

/// File extension → language mapping.
fn detect_language(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?.to_lowercase();

    // Handle special filename-based detection
    let filename = path.file_name()?.to_str()?.to_lowercase();
    if filename == "dockerfile" {
        return Some("dockerfile".to_string());
    }

    let lang = match ext.as_str() {
        "rs" => "rust",
        "ts" | "tsx" | "mts" | "cts" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "py" | "pyi" | "pyx" => "python",
        "go" => "go",
        "java" => "java",
        "kt" | "kts" => "kotlin",
        "swift" => "swift",
        "rb" | "rake" | "gemspec" | "ru" => "ruby",
        "php" => "php",
        "cs" => "csharp",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "c++" | "hpp" | "hh" | "hxx" | "h++" => "cpp",
        "lua" => "lua",
        "dart" => "dart",
        "vue" => "vue",
        "svelte" => "svelte",
        "astro" => "astro",
        "scala" | "sc" => "scala",
        "clj" | "cljs" | "cljc" | "edn" => "clojure",
        "ex" | "exs" => "elixir",
        "gleam" => "gleam",
        "elm" => "elm",
        "hs" | "lhs" => "haskell",
        "jl" => "julia",
        "ml" | "mli" => "ocaml",
        "fs" | "fsi" | "fsx" | "fsscript" => "fsharp",
        "nix" => "nix",
        "zig" | "zon" => "zig",
        "tf" | "tfvars" => "terraform",
        "typ" | "typc" => "typst",
        "prisma" => "prisma",
        "sh" | "bash" | "zsh" | "ksh" => "shell",
        "yaml" | "yml" => "yaml",
        "json" | "jsonc" => "json",
        "toml" => "toml",
        "xml" | "xsl" | "xsd" => "xml",
        "md" | "mdx" => "markdown",
        "sql" => "sql",
        "html" | "htm" => "html",
        "css" | "scss" | "sass" | "less" => "css",
        "proto" | "protobuf" => "protobuf",
        _ => return None,
    };
    Some(lang.to_string())
}

/// Check if a path matches any exclude pattern.
fn is_excluded(path: &Path, config: &ScanConfig) -> bool {
    let path_str = path.to_string_lossy();
    
    for pattern in &config.exclude_patterns {
        let normalized = pattern.replace('*', "");
        if path_str.contains(&normalized) {
            return true;
        }
        // Check glob-like patterns e.g. "*.lock"
        if pattern.starts_with("*.") && pattern.len() > 2 {
            if path_str.ends_with(&pattern[1..]) {
                return true;
            }
        }
    }
    
    // Skip dotfiles if not included
    if !config.include_dotfiles {
        if let Some(name) = path.file_name() {
            if let Some(s) = name.to_str() {
                if s.starts_with('.') && s != ".siefignore" {
                    return true;
                }
            }
        }
    }
    
    false
}

/// Walk the repository and discover all source files.
#[instrument(skip(config))]
pub fn scan_repository(config: &ScanConfig) -> Vec<SourceFile> {
    let mut files = Vec::new();
    walk_directory(&config.root, config, &mut files, PathBuf::new());
    info!(count = files.len(), "Repository scan complete");
    files
}

fn walk_directory(dir: &Path, config: &ScanConfig, files: &mut Vec<SourceFile>, relative_base: PathBuf) {
    if is_excluded(dir, config) {
        return;
    }
    
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        
        // Skip excluded
        if is_excluded(&path, config) {
            continue;
        }
        
        let relative_path = relative_base.join(&name);

        if path.is_dir() {
            walk_directory(&path, config, files, relative_path);
        } else if path.is_file() && entry.metadata().map_or(false, |m| m.len() > 0) {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            
            if size > config.max_file_size {
                continue;
            }
            
            // Only include files with recognized extensions
            if detect_language(&path).is_none() {
                continue;
            }
            
            let lang = detect_language(&path);
            files.push(SourceFile {
                path: path.clone(),
                language: lang,
                size_bytes: size,
                relative_path,
            });
        }
    }
}

/// Get a summary of files grouped by language.
pub fn group_by_language(files: &[SourceFile]) -> Vec<(String, usize, u64)> {
    let mut by_lang: std::collections::HashMap<String, (usize, u64)> = std::collections::HashMap::new();
    
    for file in files {
        let lang = file.language.as_deref().unwrap_or("unknown").to_string();
        let entry = by_lang.entry(lang).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += file.size_bytes;
    }
    
    let mut result: Vec<(String, usize, u64)> = by_lang
        .into_iter()
        .map(|(lang, (count, size))| (lang, count, size))
        .collect();
    result.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by file count descending
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(Path::new("main.rs")), Some("rust".to_string()));
        assert_eq!(detect_language(Path::new("app.ts")), Some("typescript".to_string()));
        assert_eq!(detect_language(Path::new("app.py")), Some("python".to_string()));
        assert_eq!(detect_language(Path::new("main.go")), Some("go".to_string()));
        assert_eq!(detect_language(Path::new("unknown.xyz")), None);
    }

    #[test]
    fn test_is_excluded() {
        let config = ScanConfig::default();
        assert!(is_excluded(Path::new("node_modules/foo/bar.js"), &config));
        assert!(is_excluded(Path::new(".git/config"), &config));
        assert!(is_excluded(Path::new("Cargo.lock"), &config));
        assert!(!is_excluded(Path::new("src/main.rs"), &config));
    }

    #[test]
    fn test_group_by_language_empty() {
        let result = group_by_language(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_group_by_language() {
        let files = vec![
            SourceFile {
                path: PathBuf::from("main.rs"),
                language: Some("rust".to_string()),
                size_bytes: 100,
                relative_path: PathBuf::from("main.rs"),
            },
            SourceFile {
                path: PathBuf::from("lib.rs"),
                language: Some("rust".to_string()),
                size_bytes: 200,
                relative_path: PathBuf::from("lib.rs"),
            },
            SourceFile {
                path: PathBuf::from("app.ts"),
                language: Some("typescript".to_string()),
                size_bytes: 150,
                relative_path: PathBuf::from("app.ts"),
            },
        ];
        
        let grouped = group_by_language(&files);
        assert_eq!(grouped.len(), 2);
        let rust_entry = grouped.iter().find(|(l, _, _)| l == "rust").unwrap();
        assert_eq!(rust_entry.1, 2); // 2 files
        assert_eq!(rust_entry.2, 300); // 300 bytes
    }
}