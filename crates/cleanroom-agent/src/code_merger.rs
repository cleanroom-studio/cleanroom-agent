//! Code merger — combines generated code from multiple agents into a coherent output.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, instrument};

/// A generated file fragment from a single agent.
#[derive(Debug, Clone)]
pub struct CodeFragment {
    /// Agent ID that generated this fragment.
    pub agent_id: String,
    /// Relative file path.
    pub file_path: String,
    /// Source code content.
    pub content: String,
}

/// Merge conflict when two agents generate different code for the same file.
#[derive(Debug, Clone)]
pub struct MergeConflict {
    pub file_path: String,
    pub agent_a: String,
    pub agent_b: String,
    pub content_a: String,
    pub content_b: String,
}

/// Result of code merging.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Final set of merged files (path → content).
    pub files: HashMap<String, String>,
    /// Conflicts that were resolved.
    pub resolved_conflicts: Vec<MergeConflict>,
    /// Module registration files generated.
    pub registration_files: HashMap<String, String>,
}

/// Code merger configuration.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Language for module registration file generation.
    pub language: String,
    /// How to resolve conflicts: "keep_first", "keep_last", "concatenate".
    pub conflict_strategy: String,
    /// Generated module directory.
    pub output_dir: PathBuf,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            language: "typescript".to_string(),
            conflict_strategy: "keep_last".to_string(),
            output_dir: PathBuf::from("./generated"),
        }
    }
}

/// Merges code fragments from multiple agents.
pub struct CodeMerger {
    config: MergeConfig,
}

impl CodeMerger {
    pub fn new(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Merge code fragments from multiple agents.
    #[instrument(skip(self, fragments))]
    pub fn merge(&self, fragments: Vec<CodeFragment>) -> MergeResult {
        let mut files: HashMap<String, String> = HashMap::new();
        let mut resolved_conflicts = Vec::new();

        for fragment in fragments {
            match files.get(&fragment.file_path) {
                Some(existing) if *existing != fragment.content => {
                    // Conflict detected
                    let conflict = MergeConflict {
                        file_path: fragment.file_path.clone(),
                        agent_a: "previous".to_string(),
                        agent_b: fragment.agent_id.clone(),
                        content_a: existing.clone(),
                        content_b: fragment.content.clone(),
                    };

                    let merged = match self.config.conflict_strategy.as_str() {
                        "keep_first" => existing.clone(),
                        "concatenate" => format!("{}\n\n// === From {} ===\n{}", existing, fragment.agent_id, fragment.content),
                        _ => fragment.content, // keep_last (default)
                    };

                    files.insert(fragment.file_path.clone(), merged);
                    resolved_conflicts.push(conflict);
                }
                Some(_) => {
                    // Identical content, skip
                }
                None => {
                    files.insert(fragment.file_path.clone(), fragment.content);
                }
            }
        }

        // Generate registration files
        let registration_files = self.generate_registration_files(&files);

        info!(
            files = files.len(),
            conflicts = resolved_conflicts.len(),
            "Code merge complete"
        );

        MergeResult {
            files,
            resolved_conflicts,
            registration_files,
        }
    }

    /// Generate module registration files (mod.rs, __init__.py, index.ts).
    fn generate_registration_files(&self, files: &HashMap<String, String>) -> HashMap<String, String> {
        let mut reg_files = HashMap::new();

        // Group files by directory
        let mut dirs: HashMap<String, Vec<String>> = HashMap::new();
        for file_path in files.keys() {
            if let Some(dir) = Path::new(file_path).parent() {
                let dir_str = dir.to_string_lossy().to_string();
                let stem = Path::new(file_path).file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                dirs.entry(dir_str).or_default().push(stem);
            }
        }

        for (dir, stems) in &dirs {
            match self.config.language.as_str() {
                "rust" => {
                    let mut content = String::new();
                    for stem in stems {
                        content.push_str(&format!("pub mod {};\n", stem));
                    }
                    reg_files.insert(format!("{}/mod.rs", dir), content);
                }
                "python" => {
                    let mut content = String::new();
                    for stem in stems {
                        let module = stem.replace('-', "_");
                        content.push_str(&format!("from .{} import *\n", module));
                    }
                    reg_files.insert(format!("{}/__init__.py", dir), content);
                }
                "typescript" | "javascript" => {
                    let mut content = String::new();
                    for stem in stems {
                        content.push_str(&format!("export * from './{}';\n", stem));
                    }
                    reg_files.insert(format!("{}/index.ts", dir), content);
                }
                _ => {}
            }
        }

        reg_files
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fragment(agent: &str, path: &str, content: &str) -> CodeFragment {
        CodeFragment {
            agent_id: agent.to_string(),
            file_path: path.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn test_merge_no_conflicts() {
        let merger = CodeMerger::new(MergeConfig::default());
        let frags = vec![
            fragment("a", "src/user.rs", "pub struct User {}"),
            fragment("b", "src/order.rs", "pub struct Order {}"),
        ];
        let result = merger.merge(frags);
        assert_eq!(result.files.len(), 2);
        assert!(result.resolved_conflicts.is_empty());
    }

    #[test]
    fn test_merge_duplicate_skipped() {
        let merger = CodeMerger::new(MergeConfig::default());
        let frags = vec![
            fragment("a", "src/user.rs", "pub struct User {}"),
            fragment("b", "src/user.rs", "pub struct User {}"),
        ];
        let result = merger.merge(frags);
        assert_eq!(result.files.len(), 1);
    }

    #[test]
    fn test_merge_conflict_keep_last() {
        let merger = CodeMerger::new(MergeConfig::default());
        let frags = vec![
            fragment("a", "src/user.rs", "pub struct UserV1 {}"),
            fragment("b", "src/user.rs", "pub struct UserV2 {}"),
        ];
        let result = merger.merge(frags);
        assert_eq!(result.files.len(), 1);
        assert!(result.files.get("src/user.rs").unwrap().contains("UserV2"));
        assert_eq!(result.resolved_conflicts.len(), 1);
    }

    #[test]
    fn test_merge_conflict_keep_first() {
        let mut config = MergeConfig::default();
        config.conflict_strategy = "keep_first".to_string();
        let merger = CodeMerger::new(config);
        let frags = vec![
            fragment("a", "src/user.rs", "pub struct UserV1 {}"),
            fragment("b", "src/user.rs", "pub struct UserV2 {}"),
        ];
        let result = merger.merge(frags);
        assert!(result.files.get("src/user.rs").unwrap().contains("UserV1"));
    }

    #[test]
    fn test_merge_conflict_concatenate() {
        let mut config = MergeConfig::default();
        config.conflict_strategy = "concatenate".to_string();
        let merger = CodeMerger::new(config);
        let frags = vec![
            fragment("a", "src/user.rs", "// V1"),
            fragment("b", "src/user.rs", "// V2"),
        ];
        let result = merger.merge(frags);
        assert!(result.files.get("src/user.rs").unwrap().contains("V1"));
        assert!(result.files.get("src/user.rs").unwrap().contains("V2"));
    }

    #[test]
    fn test_generate_mod_rs() {
        let mut files = HashMap::new();
        files.insert("src/user.rs".to_string(), "".to_string());
        files.insert("src/order.rs".to_string(), "".to_string());

        let config = MergeConfig {
            language: "rust".to_string(),
            ..MergeConfig::default()
        };
        let merger = CodeMerger::new(config);
        let regs = merger.generate_registration_files(&files);
        assert!(regs.contains_key("src/mod.rs"));
        let mod_rs = regs.get("src/mod.rs").unwrap();
        assert!(mod_rs.contains("pub mod user"));
        assert!(mod_rs.contains("pub mod order"));
    }

    #[test]
    fn test_generate_index_ts() {
        let mut files = HashMap::new();
        files.insert("lib/user.ts".to_string(), "".to_string());

        let config = MergeConfig {
            language: "typescript".to_string(),
            ..MergeConfig::default()
        };
        let merger = CodeMerger::new(config);
        let regs = merger.generate_registration_files(&files);
        assert!(regs.contains_key("lib/index.ts"));
    }

    #[test]
    fn test_generate_init_py() {
        let mut files = HashMap::new();
        files.insert("models/user.py".to_string(), "".to_string());

        let config = MergeConfig {
            language: "python".to_string(),
            ..MergeConfig::default()
        };
        let merger = CodeMerger::new(config);
        let regs = merger.generate_registration_files(&files);
        assert!(regs.contains_key("models/__init__.py"));
    }
}