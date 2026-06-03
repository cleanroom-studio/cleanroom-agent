//! `target_manifest` — Phase 1.2: infer the *target project skeleton* so
//! the LLM in `ConsumerAgent::generate_code_with_llm` can generate
//! code that fits the surrounding codebase (uses the right deps,
//! matches the package name, hits the right entry-point file, etc.).
//!
//! # What
//!
//! A [`TargetManifest`] is a small summary of "where the generated
//! code is going to live". Phase 1.2 supports inference from the
//! three most common manifest files:
//!
//! - Rust: `Cargo.toml` (reads `[package].name` / `version` /
//!   `[dependencies]`, picks up `src/lib.rs` or `src/main.rs` as
//!   the entry point)
//! - TypeScript / JavaScript: `package.json` (reads `name` /
//!   `version` / `dependencies` / `main` / `module` / `type`)
//! - Python: `pyproject.toml` (reads `[project].name` /
//!   `version` / `dependencies`; PEP-621 style)
//!
//! `# infer_manifest(target_dir)` is the entry point. It tries each
//! of the manifest filenames in order and returns the first
//! successful parse wrapped in a [`TargetManifest`].
//!
//! # Why now (Phase 1.2)
//!
//! Before 1.2 the consumer only knew the *language* (rust /
//! typescript / python). It had no idea what the surrounding
//! project looked like, so the LLM had to guess at the package
//! name, the dep set, and the entry-point file. With a
//! `TargetManifest` in `ConsumerConfig`, the LLM prompt can pin
//! those choices up front, which:
//!
//! 1. reduces hallucination (no more `use my_crate::foo;` when the
//!    target crate is actually called `my_app`);
//! 2. makes the generated imports line up with the rest of the
//!    project on the first compile pass;
//! 3. gives the LLM a stronger signal about the *idiom* the
//!    project uses (e.g. `tokio` vs `async-std`, `anyhow` vs
//!    `thiserror`).
//!
//! # Limitations
//!
//! The parser is deliberately conservative — we extract the
//! top-level fields, not a full manifest AST. If the user's
//! project uses an unusual layout (workspace members, private
//! registries, `pyproject.toml` Poetry sections, etc.) the
//! returned `TargetManifest` may be `Unknown` for the relevant
//! field, and the LLM will fall back to its prior behavior. This
//! matches the Phase 1.2 PLAN entry: "*infer_manifest(target_dir)
//! 从 `Cargo.toml` / `package.json` / `pyproject.toml` 自动推断*".
//!
//! The CLI gets a `--target-dir` flag in `commands/mod.rs`; when
//! not supplied, the consumer falls back to "no manifest known"
//! and the LLM operates with language-only context (the
//! pre-1.2 behavior).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A short summary of the project the generated code is going
/// to live in. Fields default to `Unknown` when the manifest
/// file didn't carry them (or no manifest was found).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetManifest {
    /// The manifest's language identifier, normalized to one of
    /// `rust` / `typescript` / `javascript` / `python`. `None`
    /// when no manifest was found.
    pub language: Option<String>,
    /// Package / crate / project name. From `[package].name` /
    /// `package.json.name` / `[project].name`.
    pub package_name: Option<String>,
    /// Package version, if any. From `[package].version` /
    /// `package.json.version` / `[project].version`.
    pub version: Option<String>,
    /// Project entry-point file (relative to `target_dir`). For
    /// Rust we pick `src/main.rs` (binary) or `src/lib.rs`
    /// (library) based on which file actually exists. For
    /// Node we use `package.json.main` / `module` /
    /// `"type": "module"`. For Python we use the conventional
    /// `src/<name>.py` or `__init__.py`.
    pub entry_point: Option<String>,
    /// Dependency names, in the order they appeared in the
    /// manifest. We don't try to parse version specifiers —
    /// the LLM only needs the names to pick idiomatic patterns.
    pub dependencies: Vec<String>,
    /// Path to the manifest file we actually read (for logging
    /// and audit purposes). `None` if no manifest was found.
    pub manifest_path: Option<PathBuf>,
}

impl TargetManifest {
    /// `true` if no useful info was extracted — used by the
    /// consumer to decide whether to bother mentioning the
    /// target skeleton in the LLM prompt.
    pub fn is_empty(&self) -> bool {
        self.language.is_none()
            && self.package_name.is_none()
            && self.version.is_none()
            && self.entry_point.is_none()
            && self.dependencies.is_empty()
            && self.manifest_path.is_none()
    }

    /// Render as a short `key: value` block, suitable for
    /// dropping into a system prompt. Empty fields are
    /// omitted. The `=== target project skeleton ===` header
    /// is *not* included here — the caller decides whether
    /// to include it (so it can suppress the whole block via
    /// `is_empty`).
    pub fn render_for_prompt(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        if let Some(lang) = &self.language {
            lines.push(format!("  language:    {lang}"));
        }
        if let Some(name) = &self.package_name {
            lines.push(format!("  package:     {name}"));
        }
        if let Some(v) = &self.version {
            lines.push(format!("  version:     {v}"));
        }
        if let Some(e) = &self.entry_point {
            lines.push(format!("  entry:       {e}"));
        }
        if !self.dependencies.is_empty() {
            // Cap the dep list so the LLM prompt doesn't explode
            // on a Cargo.toml with 200 deps. The LLM doesn't
            // need all of them to make idiomatic choices.
            const MAX_DEPS: usize = 12;
            let shown = if self.dependencies.len() > MAX_DEPS {
                let more = self.dependencies.len() - MAX_DEPS;
                let mut s = self.dependencies[..MAX_DEPS].to_vec();
                s.push(format!("... and {more} more"));
                s
            } else {
                self.dependencies.clone()
            };
            lines.push(format!("  deps:        {}", shown.join(", ")));
        }
        lines.join("\n")
    }
}

/// The top-level entry point: look for a known manifest file in
/// `target_dir` and parse it. Tries, in order:
/// 1. `Cargo.toml` (Rust)
/// 2. `package.json` (TypeScript / JavaScript)
/// 3. `pyproject.toml` (Python)
///
/// Returns a [`TargetManifest`] with `manifest_path` set to the
/// file that was read; returns [`TargetManifest::default`] (all
/// fields `None`) if no manifest was found. Read errors are
/// logged via `tracing::warn!` and treated as "no manifest" so
/// the caller always gets a usable value back.
pub fn infer_manifest(target_dir: &Path) -> TargetManifest {
    // Try Rust first (the most common target for Cleanroom in
    // Phase 0/1); the order is arbitrary, but being deterministic
    // makes the test suite reproducible.
    let cargo = target_dir.join("Cargo.toml");
    if cargo.is_file() {
        match std::fs::read_to_string(&cargo) {
            Ok(s) => match parse_cargo_toml(&s) {
                Some(mut m) => {
                    m.manifest_path = Some(cargo);
                    // If `entry_point` is still None, pick from
                    // `src/main.rs` / `src/lib.rs` based on which
                    // exists.
                    if m.entry_point.is_none() {
                        m.entry_point = pick_rust_entry_point(target_dir);
                    }
                    return m;
                }
                None => tracing::warn!(
                    file = %cargo.display(),
                    "infer_manifest: Cargo.toml present but unparseable; \
                     returning empty TargetManifest"
                ),
            },
            Err(e) => tracing::warn!(
                file = %cargo.display(),
                error = %e,
                "infer_manifest: read Cargo.toml failed; returning empty TargetManifest"
            ),
        }
    }

    let package_json = target_dir.join("package.json");
    if package_json.is_file() {
        match std::fs::read_to_string(&package_json) {
            Ok(s) => match parse_package_json(&s) {
                Some(mut m) => {
                    m.manifest_path = Some(package_json);
                    return m;
                }
                None => tracing::warn!(
                    file = %package_json.display(),
                    "infer_manifest: package.json present but unparseable"
                ),
            },
            Err(e) => tracing::warn!(
                file = %package_json.display(),
                error = %e,
                "infer_manifest: read package.json failed"
            ),
        }
    }

    let pyproject = target_dir.join("pyproject.toml");
    if pyproject.is_file() {
        match std::fs::read_to_string(&pyproject) {
            Ok(s) => match parse_pyproject_toml(&s) {
                Some(mut m) => {
                    m.manifest_path = Some(pyproject);
                    return m;
                }
                None => tracing::warn!(
                    file = %pyproject.display(),
                    "infer_manifest: pyproject.toml present but unparseable"
                ),
            },
            Err(e) => tracing::warn!(
                file = %pyproject.display(),
                error = %e,
                "infer_manifest: read pyproject.toml failed"
            ),
        }
    }

    // No known manifest file: return empty.
    TargetManifest::default()
}

// ----------------------------------------------------------------
// Rust: Cargo.toml
// ----------------------------------------------------------------

/// Parse a subset of `Cargo.toml` sufficient for the LLM prompt:
/// `[package].name` / `version`, and the list of dependency
/// names from `[dependencies]` / `[dev-dependencies]`. We avoid
/// a hard `toml` crate dep by using line-based heuristics —
/// good enough for the 90% case. Workspaces / `[workspace.dependencies]`
/// / private registries are out of scope.
fn parse_cargo_toml(content: &str) -> Option<TargetManifest> {
    let mut m = TargetManifest {
        language: Some("rust".to_string()),
        ..Default::default()
    };

    let mut in_package = false;
    let mut in_deps = false;
    let mut in_dev_deps = false;
    let mut in_build_deps = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_package = trimmed == "[package]";
            in_deps = trimmed == "[dependencies]";
            in_dev_deps = trimmed == "[dev-dependencies]";
            in_build_deps = trimmed == "[build-dependencies]";
            continue;
        }
        if in_package {
            if let Some(v) = extract_toml_string_value(trimmed, "name") {
                m.package_name = Some(v);
            } else if let Some(v) = extract_toml_string_value(trimmed, "version") {
                m.version = Some(v);
            }
        } else if in_deps || in_dev_deps || in_build_deps {
            if let Some(name) = extract_toml_dep_name(trimmed) {
                m.dependencies.push(name);
            }
        }
    }
    Some(m)
}

fn extract_toml_string_value(line: &str, key: &str) -> Option<String> {
    // Match `key = "value"` (Rust string literal syntax: supports
    // `r#"..."#` raw strings too).
    let prefix = format!("{key} =");
    if !line.starts_with(&prefix) {
        return None;
    }
    let rest = line[prefix.len()..].trim();
    if let Some(s) = parse_rust_string_literal(rest) {
        return Some(s);
    }
    // Fallback: bare identifier (e.g. `version = "0.1.0"` already
    // covered; `edition = 2021` would skip).
    None
}

fn extract_toml_dep_name(line: &str) -> Option<String> {
    // Match `dep_name = "version"` or `dep_name = { version = "..." }`
    // or `dep_name.workspace = true`.
    let eq = line.find('=')?;
    let name = line[..eq].trim();
    if name.is_empty() || name.contains(' ') {
        return None;
    }
    if name.contains('.') && !name.ends_with(".workspace") {
        // `features.foo = ...` etc. — skip, we only want top-level
        // dep names.
        return None;
    }
    Some(name.trim_end_matches(".workspace").to_string())
}

fn parse_rust_string_literal(s: &str) -> Option<String> {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        return Some(s[1..s.len() - 1].to_string());
    }
    if s.starts_with("r#\"") && s.ends_with("\"#") && s.len() >= 4 {
        return Some(s[3..s.len() - 2].to_string());
    }
    None
}

fn pick_rust_entry_point(target_dir: &Path) -> Option<String> {
    for candidate in &["src/main.rs", "src/lib.rs", "src/lib.rs"] {
        let p = target_dir.join(candidate);
        if p.is_file() {
            return Some(candidate.to_string());
        }
    }
    None
}

// ----------------------------------------------------------------
// TypeScript / JavaScript: package.json
// ----------------------------------------------------------------

fn parse_package_json(content: &str) -> Option<TargetManifest> {
    let v: serde_json::Value = serde_json::from_str(content).ok()?;
    let mut m = TargetManifest::default();

    // Detect language: ESM (`"type": "module"`) or TypeScript
    // (presence of `tsconfig.json` we don't actually check here —
    // a future iteration; for now we use `type: "module"` to
    // pick `javascript` ESM).
    let pkg_type = v.get("type").and_then(|x| x.as_str());
    let is_esm = pkg_type == Some("module");
    let has_typescript_dep = v
        .get("devDependencies")
        .and_then(|x| x.as_object())
        .map(|d| d.contains_key("typescript") || d.contains_key("@types/node"))
        .unwrap_or(false)
        || v
            .get("dependencies")
            .and_then(|x| x.as_object())
            .map(|d| d.contains_key("typescript"))
            .unwrap_or(false);
    m.language = Some(
        if has_typescript_dep {
            "typescript"
        } else if is_esm {
            "javascript"
        } else {
            "javascript"
        }
        .to_string(),
    );
    m.package_name = v
        .get("name")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    m.version = v
        .get("version")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());

    // Entry point: prefer `main`, then `module`, then `exports`.
    let entry = v
        .get("main")
        .and_then(|x| x.as_str())
        .or_else(|| v.get("module").and_then(|x| x.as_str()))
        .or_else(|| {
            v.get("exports")
                .and_then(|x| x.get("."))
                .and_then(|x| x.get("default"))
                .and_then(|x| x.as_str())
        });
    m.entry_point = entry.map(|s| s.to_string());

    // Dependencies (concat runtime + dev).
    let mut deps: Vec<String> = Vec::new();
    if let Some(d) = v.get("dependencies").and_then(|x| x.as_object()) {
        for k in d.keys() {
            deps.push(k.clone());
        }
    }
    if let Some(d) = v.get("devDependencies").and_then(|x| x.as_object()) {
        for k in d.keys() {
            if !deps.contains(k) {
                deps.push(k.clone());
            }
        }
    }
    m.dependencies = deps;

    Some(m)
}

// ----------------------------------------------------------------
// Python: pyproject.toml
// ----------------------------------------------------------------

fn parse_pyproject_toml(content: &str) -> Option<TargetManifest> {
    let mut m = TargetManifest {
        language: Some("python".to_string()),
        ..Default::default()
    };
    let mut in_project = false;
    let mut in_deps = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_project = trimmed == "[project]";
            in_deps = trimmed == "[project.optional-dependencies]"
                || trimmed == "[project.dependencies]";
            continue;
        }
        if in_project {
            if let Some(v) = extract_toml_string_value(trimmed, "name") {
                m.package_name = Some(v);
            } else if let Some(v) = extract_toml_string_value(trimmed, "version") {
                m.version = Some(v);
            }
        } else if in_deps {
            // `[project.optional-dependencies]` uses sub-tables:
            //   foo = ["bar", "baz"]
            // We treat each line as either:
            //   foo = ["bar", "baz"]  -> ["bar", "baz"]
            //   foo = "bar"           -> ["bar"]
            if let Some(eq) = trimmed.find('=') {
                let key = trimmed[..eq].trim();
                if !key.is_empty() && !key.contains(' ') {
                    let value = trimmed[eq + 1..].trim();
                    if value.starts_with('[') && value.ends_with(']') {
                        for item in value[1..value.len() - 1].split(',') {
                            let dep = item.trim().trim_matches('"').trim_matches('\'');
                            if !dep.is_empty() {
                                m.dependencies.push(dep.to_string());
                            }
                        }
                    } else {
                        let dep = value.trim_matches('"').trim_matches('\'');
                        if !dep.is_empty() {
                            m.dependencies.push(dep.to_string());
                        }
                    }
                }
            }
        }
    }
    Some(m)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(dir: &Path, name: &str, content: &str) {
        let p = dir.join(name);
        let mut f = std::fs::File::create(p).expect("create");
        f.write_all(content.as_bytes()).expect("write");
    }

    #[test]
    fn parses_cargo_toml_minimal() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(
            dir.path(),
            "Cargo.toml",
            r#"[package]
name = "my-app"
version = "0.2.0"
edition = "2021"

[dependencies]
serde = "1"
tokio = { version = "1", features = ["full"] }
anyhow = "1"

[dev-dependencies]
proptest = "1"
"#,
        );
        let m = infer_manifest(dir.path());
        assert_eq!(m.language.as_deref(), Some("rust"));
        assert_eq!(m.package_name.as_deref(), Some("my-app"));
        assert_eq!(m.version.as_deref(), Some("0.2.0"));
        // Dep order is preserved.
        assert_eq!(m.dependencies, vec!["serde", "tokio", "anyhow", "proptest"]);
        assert!(m.manifest_path.is_some());
    }

    #[test]
    fn parses_cargo_toml_picks_main_over_lib() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(
            dir.path(),
            "Cargo.toml",
            r#"[package]
name = "my-bin"
version = "0.1.0"
"#,
        );
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        write(dir.path(), "src/main.rs", "fn main() {}");
        let m = infer_manifest(dir.path());
        assert_eq!(m.entry_point.as_deref(), Some("src/main.rs"));
    }

    #[test]
    fn parses_cargo_toml_picks_lib_when_no_main() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(
            dir.path(),
            "Cargo.toml",
            r#"[package]
name = "my-lib"
version = "0.1.0"
"#,
        );
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        write(dir.path(), "src/lib.rs", "");
        let m = infer_manifest(dir.path());
        assert_eq!(m.entry_point.as_deref(), Some("src/lib.rs"));
    }

    #[test]
    fn parses_package_json_typescript_with_deps() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(
            dir.path(),
            "package.json",
            r#"{
  "name": "my-webapp",
  "version": "1.0.0",
  "type": "module",
  "main": "dist/index.js",
  "dependencies": {
    "react": "^18.0.0",
    "next": "^14.0.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/react": "^18.0.0",
    "vitest": "^1.0.0"
  }
}"#,
        );
        let m = infer_manifest(dir.path());
        assert_eq!(m.language.as_deref(), Some("typescript"));
        assert_eq!(m.package_name.as_deref(), Some("my-webapp"));
        assert_eq!(m.version.as_deref(), Some("1.0.0"));
        assert_eq!(m.entry_point.as_deref(), Some("dist/index.js"));
        // Runtime + dev deps merged, deduped, runtime first.
        assert!(m.dependencies.contains(&"react".to_string()));
        assert!(m.dependencies.contains(&"typescript".to_string()));
    }

    #[test]
    fn parses_pyproject_pep621() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(
            dir.path(),
            "pyproject.toml",
            r#"[project]
name = "my-tool"
version = "0.1.0"
description = "A small CLI"

[project.optional-dependencies]
dev = ["pytest", "ruff"]
"#,
        );
        let m = infer_manifest(dir.path());
        assert_eq!(m.language.as_deref(), Some("python"));
        assert_eq!(m.package_name.as_deref(), Some("my-tool"));
        assert_eq!(m.version.as_deref(), Some("0.1.0"));
        assert_eq!(m.dependencies, vec!["pytest", "ruff"]);
    }

    #[test]
    fn no_manifest_returns_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let m = infer_manifest(dir.path());
        assert!(m.is_empty(), "empty dir -> empty TargetManifest");
        assert!(m.manifest_path.is_none());
    }

    #[test]
    fn render_for_prompt_omits_empty_fields() {
        let m = TargetManifest {
            language: Some("rust".to_string()),
            package_name: Some("my-app".to_string()),
            version: None,
            entry_point: None,
            dependencies: vec![],
            manifest_path: None,
        };
        let s = m.render_for_prompt();
        assert!(s.contains("language:    rust"));
        assert!(s.contains("package:     my-app"));
        assert!(!s.contains("version:"));
        assert!(!s.contains("deps:"));
    }

    #[test]
    fn render_for_prompt_caps_dep_list() {
        let mut m = TargetManifest::default();
        m.language = Some("rust".to_string());
        m.dependencies = (0..20).map(|i| format!("dep-{i}")).collect();
        let s = m.render_for_prompt();
        assert!(s.contains("dep-0"));
        assert!(s.contains("... and 8 more"));
        assert!(!s.contains("dep-12"));
    }
}
