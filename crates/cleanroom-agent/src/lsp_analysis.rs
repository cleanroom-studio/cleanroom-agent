//! LSP-enhanced analysis — LSP-first, tree-sitter-fallback type resolution.
//!
//! This module provides a unified analysis interface that:
//! 1. Tries LSP analysis first (highest quality type information)
//! 2. Falls back to tree-sitter when LSP is unavailable (no server installed)
//! 3. Caches all results in `type_cache` for cross-agent reuse
//!
//! # Architecture
//!
//! ```text
//! analyze_file_with_lsp_fallback()
//!   │
//!   ├─ check type_cache ──→ hit ──→ return cached
//!   │
//!   ├─ try LspServerPool.analyze_file() ──→ success ──→ cache & return
//!   │
//!   └─ fallback: tree-sitter parse ──→ return (no cache? optional)
//! ```text

use std::path::Path;
use std::sync::Arc;

use cleanroom_db::{Database, DbError, TypeCacheRepository};
use cleanroom_lsp::{LspServerPool, FileAnalysis, DocumentSymbol};
use tracing::{info, warn};

/// Result of a file analysis — whether from LSP or tree-sitter.
#[derive(Debug, Clone)]
pub struct EnhancedFileAnalysis {
    /// File path (relative to repo root if possible).
    pub file_path: String,
    /// Programming language.
    pub language: String,
    /// Symbols extracted (structs, traits, functions, etc.).
    pub symbols: Vec<EnrichedSymbol>,
    /// Source of the analysis results.
    pub source: AnalysisSource,
}

/// Where the analysis results came from.
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisSource {
    /// Results came from cached type_cache lookup.
    TypeCache,
    /// Results came from an LSP server (highest quality).
    Lsp,
    /// Results came from tree-sitter parsing (fallback).
    TreeSitter,
}

/// A symbol enriched with type information.
#[derive(Debug, Clone)]
pub struct EnrichedSymbol {
    /// Symbol name (class name, function name, etc.).
    pub name: String,
    /// Resolved type info (if available from LSP — empty for tree-sitter).
    pub resolved_type: Option<String>,
    /// LSP symbol kind string representation.
    pub kind: String,
    /// File this symbol belongs to.
    pub file_path: String,
}

/// Options for LSP-enhanced analysis.
#[derive(Debug, Clone)]
pub struct LspAnalysisOptions {
    /// Whether to attempt LSP analysis.
    pub lsp_enabled: bool,
    /// Whether to cache results in type_cache.
    pub cache_results: bool,
}

impl Default for LspAnalysisOptions {
    fn default() -> Self {
        Self {
            lsp_enabled: true,
            cache_results: true,
        }
    }
}

// ─── Main Analysis Entry Point ──────────────────────────────────────

/// Analyze a single file using LSP (if available) with tree-sitter fallback.
///
/// # Priority
///
/// 1. TypeCache hit → return immediately
/// 2. LSP analysis → if server available and connected
/// 3. Tree-sitter → if LSP unavailable or fails
///
/// # Parameters
///
/// - `pool`: Optional LSP server pool. `None` means skip LSP entirely.
/// - `file_path`: Absolute path to the source file.
/// - `language`: Language identifier (e.g., "rust", "typescript").
/// - `repo_root`: Repository root for relative path resolution.
/// - `db`: Database for type_cache operations.
/// - `options`: Analysis configuration.
pub async fn analyze_file_with_lsp_fallback(
    pool: Option<&Arc<LspServerPool>>,
    file_path: &Path,
    language: &str,
    repo_root: &Path,
    db: Option<&Arc<Database>>,
    options: &LspAnalysisOptions,
) -> Result<EnhancedFileAnalysis, DbError> {
    let file_path_str = file_path.to_string_lossy().to_string();
    let relative_path = file_path
        .strip_prefix(repo_root)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| file_path_str.clone());

    // Stage 1: Check type_cache
    if let Some(db) = db {
        if options.cache_results {
            let cache_repo = TypeCacheRepository::new(db.connection_arc());
            if let Ok(Some(entry)) = cache_repo.lookup(&relative_path, language) {
                info!(
                    path = %relative_path,
                    cached_type = %entry.resolved_type,
                    "TypeCache hit"
                );
                return Ok(EnhancedFileAnalysis {
                    file_path: relative_path.clone(),
                    language: language.to_string(),
                    symbols: vec![EnrichedSymbol {
                        name: entry.entity_uri.clone(),
                        resolved_type: Some(entry.resolved_type),
                        kind: "cached".to_string(),
                        file_path: relative_path.clone(),
                    }],
                    source: AnalysisSource::TypeCache,
                });
            }
        }
    }

    // Stage 2: Try LSP analysis
    if options.lsp_enabled {
        if let Some(pool) = pool {
            match try_lsp_analysis(pool, file_path, language, &relative_path, repo_root, db, options).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!(
                        path = %relative_path,
                        error = %e,
                        "LSP analysis failed, falling back to tree-sitter"
                    );
                }
            }
        }
    }

    // Stage 3: Fallback to tree-sitter
    info!(path = %relative_path, "Using tree-sitter fallback");
    let result = analyze_with_treesitter(file_path, language, &relative_path);
    Ok(result)
}

// ─── LSP Analysis ───────────────────────────────────────────────────

async fn try_lsp_analysis(
    pool: &Arc<LspServerPool>,
    file_path: &Path,
    language: &str,
    relative_path: &str,
    _repo_root: &Path,
    db: Option<&Arc<Database>>,
    options: &LspAnalysisOptions,
) -> Result<EnhancedFileAnalysis, DbError> {
    // Get or start LSP server for this language
    let handle = pool.get_server(language)
        .await
        .map_err(|e| DbError::QueryFailed(format!("LSP server unavailable for {}: {}", language, e)))?;

    if !handle.is_connected() {
        return Err(DbError::QueryFailed(format!(
            "LSP server for {} is not connected (language server may not be installed)",
            language
        )));
    }

    // Run LSP analysis
    let file_analysis = handle.analyze_file(
        &file_path.to_string_lossy(),
        language,
    ).map_err(|e| DbError::QueryFailed(format!("LSP analysis failed for {}: {}", language, e)))?;

    // Convert LSP results to enriched symbols
    let symbols = extract_enriched_symbols(&file_analysis, relative_path);

    // Cache type info for cross-agent reuse
    if options.cache_results {
        if let Some(db) = db {
            cache_analysis_results(db, relative_path, language, &symbols)?;
        }
    }

    info!(
        path = %relative_path,
        symbol_count = symbols.len(),
        "LSP analysis complete"
    );

    Ok(EnhancedFileAnalysis {
        file_path: relative_path.to_string(),
        language: language.to_string(),
        symbols,
        source: AnalysisSource::Lsp,
    })
}

/// Extract enriched symbols from LSP FileAnalysis output.
fn extract_enriched_symbols(analysis: &FileAnalysis, file_path: &str) -> Vec<EnrichedSymbol> {
    let mut symbols = Vec::new();

    for symbol in &analysis.symbols {
        let resolved_type = extract_type_from_symbol(symbol);
        let kind = symbol.detail.clone().unwrap_or_else(|| "unknown".to_string());

        symbols.push(EnrichedSymbol {
            name: symbol.name.clone(),
            resolved_type,
            kind,
            file_path: file_path.to_string(),
        });

        // Recurse into children (e.g., class methods)
        for child in &symbol.children {
            symbols.push(EnrichedSymbol {
                name: format!("{}::{}", symbol.name, child.name),
                resolved_type: extract_type_from_symbol(child),
                kind: child.detail.clone().unwrap_or_else(|| "unknown".to_string()),
                file_path: file_path.to_string(),
            });
        }
    }

    symbols
}

/// Extract type info from an LSP DocumentSymbol's detail field.
///
/// Many LSP servers encode the type in the detail string:
/// - rust-analyzer: "pub struct User" → type = "struct"
/// - typescript-language-server: "interface User" → type = "interface"
fn extract_type_from_symbol(symbol: &DocumentSymbol) -> Option<String> {
    symbol.detail.as_ref().map(|d| {
        // Normalize: split into words and find type keywords
        let lower = d.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        for word in &words {
            match *word {
                "struct" => return "struct".to_string(),
                "interface" | "interface:" => return "interface".to_string(),
                "class" | "classes" => return "class".to_string(),
                "fn" | "func" | "function" => return "function".to_string(),
                "enum" => return "enum".to_string(),
                "trait" => return "trait".to_string(),
                _ => continue,
            }
        }

        // Fallback: check for compound words like "UserService" after "class"
        if words.iter().any(|w| *w == "class") {
            return "class".to_string();
        }
        if words.iter().any(|w| *w == "fn") {
            return "function".to_string();
        }

        d.clone()
    })
}

// ─── Tree-Sitter Fallback ──────────────────────────────────────────

fn analyze_with_treesitter(
    file_path: &Path,
    language: &str,
    relative_path: &str,
) -> EnhancedFileAnalysis {
    // Use the existing tree-sitter parser infrastructure
    let content = std::fs::read_to_string(file_path)
        .unwrap_or_default();

    let symbols = parse_symbols_with_treesitter(&content, language, relative_path);

    EnhancedFileAnalysis {
        file_path: relative_path.to_string(),
        language: language.to_string(),
        symbols,
        source: AnalysisSource::TreeSitter,
    }
}

/// Use tree-sitter and regex to extract basic symbols from source code.
///
/// This is a lightweight fallback that extracts symbol names without
/// full type resolution. First tries tree-sitter if grammar is available,
/// then falls back to regex.
fn parse_symbols_with_treesitter(
    _content: &str,
    language: &str,
    file_path: &str,
) -> Vec<EnrichedSymbol> {
    // Try tree-sitter parser first
    if let Ok(entities) = crate::tree_sitter_parser::parse_entities_from_source(_content, language) {
        return entities
            .into_iter()
            .map(|(name, kind)| EnrichedSymbol {
                name,
                resolved_type: None,
                kind,
                file_path: file_path.to_string(),
            })
            .collect();
    }

    // Ultimate fallback: regex-based extraction
    fallback_regex_parse(_content, file_path)
}

/// Last-resort regex-based symbol extraction when tree-sitter fails.
fn fallback_regex_parse(content: &str, file_path: &str) -> Vec<EnrichedSymbol> {
    let mut symbols = Vec::new();

    // Match common patterns: struct/class/fn/interface/enum declarations
    let patterns: &[(&str, &str)] = &[
        (r"(?m)^\s*(?:pub\s+)?struct\s+(\w+)", "struct"),
        (r"(?m)^\s*(?:pub\s+)?enum\s+(\w+)", "enum"),
        (r"(?m)^\s*(?:pub\s+)?trait\s+(\w+)", "trait"),
        (r"(?m)^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", "function"),
        (r"(?m)^\s*(?:export\s+)?class\s+(\w+)", "class"),
        (r"(?m)^\s*(?:export\s+)?interface\s+(\w+)", "interface"),
        (r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
        (r"(?m)^\s*class\s+(\w+)\s*:", "class"),
        (r"(?m)^\s*def\s+(\w+)\s*\(", "function"),
    ];

    for (pattern, kind) in patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            for cap in re.captures_iter(content) {
                if let Some(name) = cap.get(1) {
                    symbols.push(EnrichedSymbol {
                        name: name.as_str().to_string(),
                        resolved_type: None,
                        kind: kind.to_string(),
                        file_path: file_path.to_string(),
                    });
                }
            }
        }
    }

    symbols
}

// ─── Type Cache Integration ────────────────────────────────────────

/// Cache analysis results for cross-agent reuse via the type_cache table.
fn cache_analysis_results(
    db: &Arc<Database>,
    relative_path: &str,
    language: &str,
    symbols: &[EnrichedSymbol],
) -> Result<(), DbError> {
    let cache_repo = TypeCacheRepository::new(db.connection_arc());

    for symbol in symbols {
        let entry = cleanroom_db::TypeCacheEntry {
            entity_uri: format!("{}#{}", relative_path, symbol.name),
            language: language.to_string(),
            resolved_type: symbol
                .resolved_type
                .clone()
                .unwrap_or_else(|| symbol.kind.clone()),
            source_file: Some(relative_path.to_string()),
            from_lsp: true, // LSP-sourced type info is more reliable
            cached_at: chrono::Utc::now().to_rfc3339(),
        };

        cache_repo.cache(&entry)?;
    }

    info!(
        path = %relative_path,
        count = symbols.len(),
        "Cached analysis results in type_cache"
    );

    Ok(())
}

/// Look up cached type information for a specific entity.
///
/// Used by Consumer Agents to resolve types without re-running analysis.
pub fn lookup_cached_type(
    db: &Arc<Database>,
    entity_uri: &str,
    language: &str,
) -> Result<Option<cleanroom_db::TypeCacheEntry>, DbError> {
    let cache_repo = TypeCacheRepository::new(db.connection_arc());
    cache_repo.lookup(entity_uri, language)
}

/// Check if the type_cache has been populated for a language.
pub fn has_cached_types(
    db: &Arc<Database>,
    language: &str,
) -> Result<bool, DbError> {
    let cache_repo = TypeCacheRepository::new(db.connection_arc());
    let count = cache_repo.clear_by_language(language);
    match count {
        Ok(n) => Ok(n > 0),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_regex_parse_rust() {
        let content = r#"
pub struct User {
    pub id: u64,
    pub name: String,
}

pub enum Status {
    Active,
    Inactive,
}

pub trait Repository {
    fn find_by_id(&self, id: u64) -> Option<User>;
}

pub async fn create_user(name: String) -> Result<User, Error> {
    todo!()
}
"#;

        let symbols = fallback_regex_parse(content, "src/models.rs");
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"User"), "Should detect struct User");
        assert!(names.contains(&"Status"), "Should detect enum Status");
        assert!(names.contains(&"Repository"), "Should detect trait Repository");
        assert!(names.contains(&"create_user"), "Should detect fn create_user");
    }

    #[test]
    fn test_fallback_regex_parse_typescript() {
        let content = r#"
export interface User {
    id: number;
    name: string;
}

export class UserService {
    async getUser(id: number): Promise<User> {
        return { id, name: "" };
    }
}

export function validateUser(user: User): boolean {
    return user.id > 0;
}
"#;

        let symbols = fallback_regex_parse(content, "src/user.ts");
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"User"), "Should detect interface User");
        assert!(names.contains(&"UserService"), "Should detect class UserService");
        assert!(names.contains(&"validateUser"), "Should detect function validateUser");
        // Note: class methods (like getUser) are not detectable by regex alone;
        // they require full LSP or tree-sitter analysis for accurate extraction.
    }

    #[test]
    fn test_extract_type_from_symbol() {
        use cleanroom_lsp::{DocumentSymbol, SymbolKind};

        let struct_sym = DocumentSymbol {
            name: "User".to_string(),
            kind: SymbolKind::STRUCT,
            range: None,
            children: vec![],
            detail: Some("pub struct User".to_string()),
        };
        assert_eq!(
            extract_type_from_symbol(&struct_sym),
            Some("struct".to_string())
        );

        let func_sym = DocumentSymbol {
            name: "create_user".to_string(),
            kind: SymbolKind::FUNCTION,
            range: None,
            children: vec![],
            detail: Some("pub fn create_user".to_string()),
        };
        assert_eq!(
            extract_type_from_symbol(&func_sym),
            Some("function".to_string())
        );
    }
}
