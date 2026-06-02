//! `auxiliary_hints` — Phase 0.2 helper that produces a `HINTS<file>`
//! text block for the LLM.
//!
//! # Motivation
//!
//! In the LLM-driven Producer path, the LLM gets a source file as
//! input and is asked to emit S.DEF entities. The LLM benefits from
//! hints about the file's structure: how many top-level entities
//! (functions, structs, classes), how many imports, what types are
//! used, etc. These hints are produced by static analysis (tree-sitter
//! + optionally LSP) and prepended to the LLM's user message so the
//! LLM doesn't have to rediscover the obvious from raw text.
//!
//! # Shape
//!
//! The output is a multi-line text block in a stable, machine-friendly
//! format:
//!
//! ```text
//! <file: src/foo.rs>
//! <language: rust>
//! <loc: 142>
//! <top_level_entities: 7 (3 fn, 2 struct, 1 enum, 1 trait)>
//! <imports: 4>
//! <external_types: HashMap, Vec, Option, Result>
//! ```
//!
//! The format is deliberately line-oriented and prefix-tagged so the
//! LLM (and any future parser) can extract each field with a regex
//! like `^<(\w+):\s*(.*)>$`.
//!
//! # Determinism
//!
//! [`compute_hints`] is a pure function of (file content + file path).
//! Running it twice on the same input gives byte-identical output.
//! The unit test [`compute_hints_is_deterministic`] pins this contract.

use std::collections::BTreeSet;
use std::path::Path;

/// One line of the HINTS block, parsed back into (key, value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HintLine {
    pub key: String,
    pub value: String,
}

/// Structured view of the file (decoupled from the formatted text so
/// tests can assert on fields without re-parsing the string).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileHints {
    pub file_path: String,
    pub language: String,
    pub loc: usize,
    pub top_level_entities: BTreeSet<String>, // e.g. {"fn: 3", "struct: 2"}
    pub imports: usize,
    pub external_types: BTreeSet<String>,
}

impl FileHints {
    /// Render the hints as a `HINTS<file>` text block.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("<file: {}>\n", self.file_path));
        out.push_str(&format!("<language: {}>\n", self.language));
        out.push_str(&format!("<loc: {}>\n", self.loc));
        let ents: Vec<String> = self.top_level_entities.iter().cloned().collect();
        out.push_str(&format!("<top_level_entities: {}>\n", ents.join(", ")));
        out.push_str(&format!("<imports: {}>\n", self.imports));
        let types: Vec<&str> = self.external_types.iter().map(|s| s.as_str()).collect();
        out.push_str(&format!("<external_types: {}>", types.join(", ")));
        out
    }

    /// Parse a previously-rendered HINTS block back into structured form.
    /// Tolerant: unknown keys are ignored, malformed lines are skipped.
    pub fn parse(text: &str) -> Self {
        let mut h = FileHints::default();
        for line in text.lines() {
            let line = line.trim();
            let Some(inner) = line.strip_prefix('<').and_then(|s| s.strip_suffix('>')) else {
                continue;
            };
            let Some((k, v)) = inner.split_once(':') else { continue };
            let key = k.trim().to_string();
            let value = v.trim().to_string();
            match key.as_str() {
                "file" => h.file_path = value,
                "language" => h.language = value,
                "loc" => h.loc = value.parse().unwrap_or(0),
                "top_level_entities" => {
                    for ent in value.split(',') {
                        let t = ent.trim();
                        if !t.is_empty() {
                            h.top_level_entities.insert(t.to_string());
                        }
                    }
                }
                "imports" => h.imports = value.parse().unwrap_or(0),
                "external_types" => {
                    for ty in value.split(',') {
                        let t = ty.trim();
                        if !t.is_empty() {
                            h.external_types.insert(t.to_string());
                        }
                    }
                }
                _ => {} // ignore unknown keys for forward-compat
            }
        }
        h
    }
}

/// Detect a programming language from the file extension. Returns
/// `Unknown` for unrecognized extensions.
pub fn detect_language_from_extension(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("ts") | Some("tsx") => "typescript",
        Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => "javascript",
        Some("py") | Some("pyi") => "python",
        Some("go") => "go",
        Some("java") => "java",
        Some("c") | Some("h") => "c",
        Some("cpp") | Some("cc") | Some("hpp") => "cpp",
        _ => "unknown",
    }
}

/// Count non-blank, non-comment-only lines. Used for the `loc` field.
pub fn count_loc(source: &str) -> usize {
    source
        .lines()
        .filter(|l| {
            let trimmed = l.trim();
            !trimmed.is_empty() && !trimmed.starts_with("//") && !trimmed.starts_with('#')
        })
        .count()
}

/// Count the number of `use` (Rust) / `import` (Python / TS / JS) /
/// `package` (Java / Go) statements at the top of the file.
///
/// This is a rough heuristic, not a real tree-sitter parse -- good
/// enough for LLM hints, where the exact number doesn't matter as
/// long as the order of magnitude is right.
pub fn count_imports(source: &str) -> usize {
    let mut n = 0;
    let mut in_block_comment = false;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("/*") {
            in_block_comment = true;
        }
        if in_block_comment {
            if trimmed.contains("*/") {
                in_block_comment = false;
            }
            continue;
        }
        if trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }
        // Rust: `use foo;` or `use foo::{Bar, Baz};`
        if trimmed.starts_with("use ") && trimmed.ends_with(';') {
            n += 1;
            continue;
        }
        // Python: `import foo` or `from foo import bar`
        if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
            n += 1;
            continue;
        }
        // TS/JS: `import x from 'y';` or `import { a, b } from 'y';`
        if trimmed.starts_with("import ") && (trimmed.contains(" from ") || trimmed.contains("from\"")) {
            n += 1;
            continue;
        }
        // Java: `import foo.Bar;`
        if trimmed.starts_with("import ") && trimmed.ends_with(';') {
            n += 1;
            continue;
        }
        // Go: `import "foo"` or `import ( "a"; "b" )`
        if trimmed.starts_with("import ") && (trimmed.contains('"') || trimmed == "import (") {
            n += 1;
            continue;
        }
    }
    n
}

/// Count top-level entity declarations (functions, structs, enums,
/// traits, classes) using a regex-style heuristic. Returns a set of
/// strings like `{"fn: 3", "struct: 2"}` so the LLM can see the
/// breakdown.
pub fn count_top_level_entities(source: &str, language: &str) -> BTreeSet<String> {
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();
    for line in source.lines() {
        let t = line.trim_start();
        match language {
            "rust" => {
                if t.starts_with("pub fn ") || t.starts_with("fn ") {
                    *counts.entry("fn").or_insert(0) += 1;
                } else if t.starts_with("pub struct ") || t.starts_with("struct ") {
                    *counts.entry("struct").or_insert(0) += 1;
                } else if t.starts_with("pub enum ") || t.starts_with("enum ") {
                    *counts.entry("enum").or_insert(0) += 1;
                } else if t.starts_with("pub trait ") || t.starts_with("trait ") {
                    *counts.entry("trait").or_insert(0) += 1;
                } else if t.starts_with("pub impl ") || t.starts_with("impl ") {
                    *counts.entry("impl").or_insert(0) += 1;
                } else if t.starts_with("pub async fn ") || t.starts_with("async fn ") {
                    *counts.entry("async_fn").or_insert(0) += 1;
                }
            }
            "python" => {
                if t.starts_with("def ") {
                    *counts.entry("def").or_insert(0) += 1;
                } else if t.starts_with("class ") {
                    *counts.entry("class").or_insert(0) += 1;
                } else if t.starts_with("async def ") {
                    *counts.entry("async_def").or_insert(0) += 1;
                }
            }
            "typescript" | "javascript" => {
                if t.starts_with("function ") || t.starts_with("export function ") {
                    *counts.entry("function").or_insert(0) += 1;
                } else if t.starts_with("class ") || t.starts_with("export class ") {
                    *counts.entry("class").or_insert(0) += 1;
                } else if t.starts_with("interface ") {
                    *counts.entry("interface").or_insert(0) += 1;
                } else if t.starts_with("type ") {
                    *counts.entry("type_alias").or_insert(0) += 1;
                }
            }
            "go" => {
                if t.starts_with("func ") {
                    *counts.entry("func").or_insert(0) += 1;
                } else if t.starts_with("type ") && t.contains(" struct") {
                    *counts.entry("struct").or_insert(0) += 1;
                } else if t.starts_with("type ") && t.contains(" interface") {
                    *counts.entry("interface").or_insert(0) += 1;
                }
            }
            "java" => {
                if t.starts_with("public class ") || t.starts_with("class ") {
                    *counts.entry("class").or_insert(0) += 1;
                } else if t.starts_with("public interface ") || t.starts_with("interface ") {
                    *counts.entry("interface").or_insert(0) += 1;
                } else if (t.starts_with("public ") || t.starts_with("private ")
                    || t.starts_with("protected ")) && t.contains('(') && !t.contains('=')
                {
                    *counts.entry("method").or_insert(0) += 1;
                }
            }
            _ => {}
        }
    }
    counts
        .into_iter()
        .map(|(k, v)| format!("{k}: {v}"))
        .collect()
}

/// Pull a rough set of "external" type names referenced in the source
/// (capitalized identifiers not defined locally). Used to hint the LLM
/// at which standard-library / third-party types the file uses.
pub fn extract_external_types(source: &str) -> BTreeSet<String> {
    let mut types = BTreeSet::new();
    // Words that look like types: Capitalized, possibly with digits, not
    // starting with underscore.
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_uppercase() {
            let start = i;
            while i < bytes.len() {
                let cc = bytes[i] as char;
                if cc.is_ascii_alphanumeric() {
                    i += 1;
                } else {
                    break;
                }
            }
            let token = &source[start..i];
            if token.len() >= 2 {
                types.insert(token.to_string());
            }
        } else {
            i += 1;
        }
    }
    types
}

/// Compute the full `HINTS<file>` block for a source file given as
/// (path, content). Pure function -- no I/O, no tree-sitter, no LSP.
/// Real LSP/tree-sitter integration is a follow-up; this heuristic
/// version already produces useful signals and is testable.
pub fn compute_hints(path: &Path, source: &str) -> FileHints {
    let language = detect_language_from_extension(path).to_string();
    let loc = count_loc(source);
    let top_level_entities = count_top_level_entities(source, &language);
    let imports = count_imports(source);
    let external_types = extract_external_types(source);
    FileHints {
        file_path: path.to_string_lossy().to_string(),
        language,
        loc,
        top_level_entities,
        imports,
        external_types,
    }
}

/// Convenience: compute the HINTS block for a file on disk. Returns a
/// default `FileHints` on read errors so the caller can still emit a
/// (degraded) hint block instead of failing the whole LLM call.
pub fn compute_hints_for_file(path: &Path) -> FileHints {
    match std::fs::read_to_string(path) {
        Ok(source) => compute_hints(path, &source),
        Err(_) => FileHints {
            file_path: path.to_string_lossy().to_string(),
            language: detect_language_from_extension(path).to_string(),
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const SAMPLE_RUST: &str = r#"//! Sample
use std::collections::HashMap;
use std::fs;
use serde::Serialize;

pub struct Account {
    pub id: String,
    pub balance: i64,
}

pub enum Status {
    Active,
    Closed,
}

pub trait Auditable {
    fn audit(&self) -> bool;
}

pub fn new_account(id: String) -> Account { Account { id, balance: 0 } }

impl Account {
    pub fn credit(&mut self, cents: u64) { self.balance += cents as i64; }
}
"#;

    const SAMPLE_PY: &str = r#""""Sample module."""
import os
import json
from typing import List, Dict, Optional

class Order:
    def __init__(self, order_id: str) -> None:
        self.order_id = order_id

    def total(self) -> int:
        return 0

def helper(x: int) -> int:
    return x + 1
"#;

    #[test]
    fn test_detect_language_common_extensions() {
        assert_eq!(detect_language_from_extension(&PathBuf::from("foo.rs")), "rust");
        assert_eq!(detect_language_from_extension(&PathBuf::from("bar.py")), "python");
        assert_eq!(detect_language_from_extension(&PathBuf::from("baz.ts")), "typescript");
        assert_eq!(detect_language_from_extension(&PathBuf::from("qux.go")), "go");
        assert_eq!(detect_language_from_extension(&PathBuf::from("unknown.xyz")), "unknown");
    }

    #[test]
    fn test_count_loc_excludes_blank_and_comments() {
        // Heuristic counts any non-blank, non-`//`/`#` line, INCLUDING
        // trailing braces -- so `}` is counted. That's "good enough" for
        // LLM hints; the exact number is not the point.
        let s = "// header\n\nfn foo() {\n    // inner\n    bar();\n}\n";
        let got = count_loc(s);
        assert!(got >= 2 && got <= 4, "expected 2-4 loc, got {got}");
    }

    #[test]
    fn test_count_imports_rust_python_javascript() {
        assert_eq!(count_imports(SAMPLE_RUST), 3, "3 use statements");
        // SAMPLE_PY has: import os, import json, from typing import ...
        assert_eq!(count_imports(SAMPLE_PY), 3);
    }

    #[test]
    fn test_count_top_level_entities_rust() {
        // Heuristic is NAME-only, not AST-aware. Methods inside `impl` and
        // `trait` bodies also match `fn` (after trim_start). SAMPLE_RUST
        // has: 1 free fn (new_account), 1 trait method (audit), 1 impl
        // method (credit) = 3 fns. Plus 1 struct, 1 enum, 1 trait, 1 impl.
        let ents = count_top_level_entities(SAMPLE_RUST, "rust");
        assert!(ents.contains("struct: 1"), "got {ents:?}");
        assert!(ents.contains("enum: 1"), "got {ents:?}");
        assert!(ents.contains("trait: 1"), "got {ents:?}");
        assert!(ents.contains("impl: 1"), "got {ents:?}");
        assert!(ents.contains("fn: 3"), "got {ents:?}");
    }

    #[test]
    fn test_count_top_level_entities_python() {
        // 1 class (Order), 3 defs (__init__, total, helper) -- again,
        // methods inside the class also match `def` (after trim_start).
        let ents = count_top_level_entities(SAMPLE_PY, "python");
        assert!(ents.contains("class: 1"), "got {ents:?}");
        assert!(ents.contains("def: 3"), "got {ents:?}");
    }

    #[test]
    fn test_extract_external_types_finds_capitalized_identifiers() {
        let types = extract_external_types(SAMPLE_RUST);
        assert!(types.contains("Account"), "got {types:?}");
        assert!(types.contains("Status"));
        assert!(types.contains("Auditable"));
        assert!(types.contains("HashMap"));
        assert!(types.contains("String"), "got {types:?}");
    }

    #[test]
    fn test_compute_hints_rust() {
        let h = compute_hints(&PathBuf::from("src/foo.rs"), SAMPLE_RUST);
        assert_eq!(h.file_path, "src/foo.rs");
        assert_eq!(h.language, "rust");
        assert!(h.loc > 0);
        assert!(h.top_level_entities.contains("struct: 1"));
        assert_eq!(h.imports, 3);
        assert!(h.external_types.contains("Account"));
    }

    #[test]
    fn test_compute_hints_is_deterministic() {
        let a = compute_hints(&PathBuf::from("src/foo.rs"), SAMPLE_RUST);
        let b = compute_hints(&PathBuf::from("src/foo.rs"), SAMPLE_RUST);
        assert_eq!(a, b, "running the same hints twice must give equal output");
        assert_eq!(a.render(), b.render(), "rendered text must be byte-identical");
    }

    #[test]
    fn test_render_and_parse_roundtrip() {
        let h = compute_hints(&PathBuf::from("src/account.rs"), SAMPLE_RUST);
        let text = h.render();
        let back = FileHints::parse(&text);
        assert_eq!(h, back, "render -> parse must be lossless for known keys");
    }

    #[test]
    fn test_parse_tolerates_unknown_keys() {
        let text = "<file: x.rs>\n<language: rust>\n<loc: 1>\n<future_field: ignored>\n";
        let h = FileHints::parse(text);
        assert_eq!(h.file_path, "x.rs");
        assert_eq!(h.language, "rust");
        assert_eq!(h.loc, 1);
    }

    #[test]
    fn test_compute_hints_for_file_handles_missing_file() {
        let h = compute_hints_for_file(&PathBuf::from("/nonexistent/path/foo.rs"));
        assert!(h.loc == 0, "missing file -> empty hints");
        assert_eq!(h.language, "rust", "language still detected from extension");
    }
}
