//! `select_topk` — query a directory's Skills index with a free-form
//! query string and print the top-k matches (default k=3) with their
//! scores. Run with:
//!
//! ```bash
//! cargo run -p cleanroom-skill --example select_topk -- /path/to/project "rust trait impl"
//! ```

use cleanroom_skill::{load_skill_index_strict, select_skills, SelectionPolicy};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let root: PathBuf = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let query: &str = args.get(2).map(String::as_str).unwrap_or("");
    let top_k: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    if query.is_empty() {
        eprintln!("usage: select_topk <root> <query> [top_k]");
        std::process::exit(2);
    }

    let index = load_skill_index_strict(&root)?;
    if index.is_empty() {
        println!("(no skills found)");
        return Ok(());
    }

    let policy = SelectionPolicy {
        top_k,
        min_score: 0.1,
        ..Default::default()
    };
    let matches = select_skills(&index, query, &policy);

    println!("Query: {query:?}");
    println!("Index: {} skills, top_k={top_k}\n", index.len());

    if matches.is_empty() {
        println!("(no matches above min_score)");
        return Ok(());
    }

    for m in &matches {
        println!(
            "  score={:.2}  {name}  [{scope:?}]  priority={priority}",
            m.score,
            name = m.skill.name,
            scope = m.skill.scope,
            priority = m.skill.priority,
        );
        println!("    {}", m.skill.description);
    }

    Ok(())
}
