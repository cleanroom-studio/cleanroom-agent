//! `discover_skills` — scan a directory tree for SKILL.md files and print
//! the discovered skills' Tier-1 summaries. Run with:
//!
//! ```bash
//! cargo run -p cleanroom-skill --example discover_skills -- /path/to/project
//! ```

use cleanroom_skill::{load_skill_index_strict, SkillSummary};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    println!("Scanning {} for skills...\n", root.display());

    let index = load_skill_index_strict(&root)?;
    if index.is_empty() {
        println!("(no skills found)");
        return Ok(());
    }

    println!("{} skill(s) discovered:\n", index.len());
    for SkillSummary {
        name,
        scope,
        priority,
        token_budget,
        description,
        ..
    } in index.summaries()
    {
        println!(
            "  - {name}  [{scope:?}]  priority={priority}  token_budget={token_budget}"
        );
        println!("    {description}");
    }

    Ok(())
}
