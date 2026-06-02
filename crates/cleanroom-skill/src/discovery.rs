//! Skill discovery — scan the filesystem for SKILL.md files.
//!
//! See `docs/21-skills-system.md` §4.3 for the directory conventions:
//! `<root>/.cleanroom/skills/<name>/SKILL.md` (project scope, highest priority),
//! `<root>/.agents/skills/<name>/SKILL.md` (cross-client convention),
//! `~/.cleanroom/skills/...`, `~/.agents/skills/...` (user scope),
//! `<crate>/skills/...` (built-in, compiled into the binary).

use std::path::{Path, PathBuf};

use ignore::WalkBuilder;
use tracing::debug;

/// Heavy directories we never descend into.
const IGNORED_DIRS: &[&str] = &[
    ".git", ".hg", ".svn", "target", "node_modules", ".next", "dist", "build", "coverage",
    ".idea", ".vscode", "__pycache__", ".pytest_cache", "venv", ".venv",
];

/// Subdirectories under a skill directory that contain *support* files, not
/// skill definitions themselves.
const SUPPORT_DIRS: &[&str] = &["references", "scripts", "assets"];

/// Recursively scan a list of skill roots for SKILL.md files.
///
/// `root` is the project root (used to construct the default skill paths).
/// `extra_dirs` are additional roots to scan (e.g. user-level dirs, built-in
/// crate assets).
pub fn discover_skill_files_with_extras(
    root: Option<&Path>,
    extra_dirs: &[PathBuf],
) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(r) = root {
        dirs.push(r.join(".cleanroom").join("skills"));
        dirs.push(r.join(".agents").join("skills"));
    }
    dirs.extend(extra_dirs.iter().cloned());
    discover_skill_files_from_dirs(&dirs)
}

/// Recursively scan `<root>/.cleanroom/skills` and `<root>/.agents/skills`.
pub fn discover_skill_files(root: &Path) -> Vec<PathBuf> {
    discover_skill_files_with_extras(Some(root), &[])
}

fn discover_skill_files_from_dirs(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();

    for dir in dirs {
        if !dir.exists() || !dir.is_dir() {
            continue;
        }
        debug!(dir = %dir.display(), "scanning skill directory");

        let walker = WalkBuilder::new(dir)
            .standard_filters(false)
            .build();

        for entry in walker.flatten() {
            let path = entry.path();
            // Skip ignored directories.
            if path
                .components()
                .any(|c| IGNORED_DIRS.iter().any(|d| c.as_os_str() == *d))
            {
                continue;
            }
            // Skip support subdirectories.
            if path
                .components()
                .any(|c| SUPPORT_DIRS.iter().any(|d| c.as_os_str() == *d))
            {
                continue;
            }
            // Only regular files named SKILL.md.
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            if path.file_name().and_then(|n| n.to_str()) != Some("SKILL.md") {
                continue;
            }
            files.push(path.to_path_buf());
        }
    }

    files.sort();
    files.dedup();
    files
}

/// Return the built-in skill directory (compiled into the binary).
///
/// Resolves to `$CARGO_MANIFEST_DIR/skills` at compile time when used inside
/// this crate. External callers should pass this as an `extra_dir` to
/// `discover_skill_files_with_extras`.
pub fn builtin_skill_dir() -> Option<PathBuf> {
    // When the crate is built, the `skills/` subdirectory is shipped as part
    // of the source tree (alongside `src/` and `Cargo.toml`).
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let p = PathBuf::from(manifest_dir).join("skills");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn discovers_skill_md_files() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let skill_dir = root.join(".cleanroom").join("skills").join("my-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: my-skill\ndescription: d\n---\nbody",
        )
        .unwrap();

        let found = discover_skill_files(root);
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("SKILL.md"));
    }

    #[test]
    fn ignores_support_dirs() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let skill_dir = root.join(".cleanroom").join("skills").join("x");
        fs::create_dir_all(skill_dir.join("scripts")).unwrap();
        fs::write(skill_dir.join("scripts").join("foo.md"), "# not a skill").unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: x\ndescription: d\n---\nbody",
        )
        .unwrap();

        let found = discover_skill_files(root);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn ignores_target_and_git() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let dir = root.join(".cleanroom").join("skills").join("good");
        fs::create_dir_all(&dir).unwrap();
        fs::create_dir_all(dir.join("target").join("nested")).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            "---\nname: good\ndescription: d\n---\nbody",
        )
        .unwrap();
        fs::write(dir.join("target").join("nested").join("SKILL.md"), "x").unwrap();

        let found = discover_skill_files(root);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn dedup_across_dirs() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let dir = root.join(".cleanroom").join("skills").join("s");
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join("SKILL.md");
        fs::write(&f, "---\nname: s\ndescription: d\n---\nbody").unwrap();

        let found = discover_skill_files_with_extras(
            Some(root),
            &[root.join(".cleanroom").join("skills")],
        );
        assert_eq!(found.len(), 1);
    }
}
