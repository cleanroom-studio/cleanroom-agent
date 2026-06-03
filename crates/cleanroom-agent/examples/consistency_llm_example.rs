//! `examples/consistency_llm_example.rs`
//!
//! Phase 1.5 end-to-end demo: simulate a fingerprint mismatch
//! in the `fingerprints` table, then ask the LLM to explain
//! why the three hashes (sdef / db / code) disagree. The
//! resulting explanation is written to the new
//! `fingerprints.llm_explanation` column (migration 012) and
//! the example re-reads it to verify.
//!
//! ## Flow
//!
//! 1. Build a temp SQLite DB.
//! 2. Insert a `sdef_documents` row + a `fingerprints` row
//!    with **deliberately mismatched** hashes (`aaa` / `bbb` /
//!    `ccc`).
//! 3. Build a `consistency_llm::FingerprintSnapshot` from the
//!    row.
//! 4. Call `consistency_llm::explain_mismatch(llm, db,
//!    snapshot).await` — this calls the LLM, then UPDATEs
//!    the row with the explanation.
//! 5. Re-read the `llm_explanation` column and print it.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --manifest-path cleanroom-agent/Cargo.toml \
//!   -p cleanroom-agent --example consistency_llm_example
//! ```

use std::sync::Arc;

use cleanroom_agent::consistency_llm::{explain_mismatch, FingerprintSnapshot};
use cleanroom_db::Database;
use cleanroom_meta_llm::backends::openai::OpenAiProvider;
use cleanroom_meta_llm::builder::MetaBuilder;
use cleanroom_meta_llm::MetaLlm;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 0. Tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,cleanroom_meta=warn")),
        )
        .init();

    // 1. Build the LLM (same as the other Phase 1 examples).
    let api_key = std::env::var("MINIMAX_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .map_err(|_| "MINIMAX_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY not set")?;
    if std::env::var_os("OPENAI_BASE_URL").is_none() {
        std::env::set_var("OPENAI_BASE_URL", "https://api.minimaxi.com/v1");
    }
    let model = std::env::var("EVAL_MODEL").unwrap_or_else(|_| "MiniMax-M3".to_string());
    let llm: Arc<OpenAiProvider> = MetaBuilder::<OpenAiProvider>::new()
        .api_key(api_key)
        .base_url(
            std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.minimaxi.com/v1".into()),
        )
        .model(model.clone())
        .max_tokens(512)
        .temperature(0.0)
        .build()?;
    let llm_dyn: Arc<dyn MetaLlm> = llm;

    // 2. Set up a temp DB.
    let tmpdir = tempfile::tempdir()?;
    let db_path = tmpdir.path().join("consistency-llm.db");
    let migrations_dir = workspace_migrations_dir();
    let db = Arc::new(Database::open_with_migrations_from(&db_path, Some(&migrations_dir))?);

    // 3. Seed a sdef_documents row + a fingerprints row with
    //    **deliberately mismatched** hashes so the LLM has
    //    something to explain.
    let conn = db.connection();
    conn.execute(
        "INSERT INTO sdef_documents (name, version, description, created_at, updated_at) \
         VALUES ('consistency-doc', '0.1.0', 'phase 1.5 demo', datetime(), datetime())",
        [],
    )
    .expect("insert sdef_documents");
    conn.execute(
        "INSERT INTO fingerprints \
         (entity_uri, document_name, entity_type, sdef_hash, db_hash, code_hash, code_path) \
         VALUES ('sdef://consistency-doc/User', 'consistency-doc', 'data_model', \
                 'aaa', 'bbb', 'ccc', 'src/user.rs')",
        [],
    )
    .expect("insert fingerprint");
    drop(conn);

    // 4. Build the snapshot.
    let snapshot = FingerprintSnapshot {
        document_name: "consistency-doc".to_string(),
        entity_uri: "sdef://consistency-doc/User".to_string(),
        entity_type: "data_model".to_string(),
        sdef_hash: Some("aaa".to_string()),
        db_hash: Some("bbb".to_string()),
        code_hash: Some("ccc".to_string()),
        code_path: Some("src/user.rs".to_string()),
        sdef_snippet: Some(
            r#"{"kind": "data_model", "name": "User", "attributes": [
              {"name": "id", "type": "String"},
              {"name": "email", "type": "String"}
            ]}"#
                .to_string(),
        ),
    };

    println!("== consistency_llm_example ==");
    println!("provider: openai");
    println!("model:    {model}");
    println!("snapshot: aaa / bbb / ccc (deliberately mismatched)");
    println!();

    // 5. Run the LLM + persist.
    let started = std::time::Instant::now();
    let explanation = explain_mismatch(llm_dyn, &db, &snapshot).await
        .map_err(|e| format!("explain_mismatch failed: {e}"))?;
    let elapsed_ms = started.elapsed().as_millis();

    println!("== explain_mismatch finished in {elapsed_ms}ms ==");
    println!("--- LLM explanation ---");
    println!("{}", explanation);
    println!("--- end ---");
    println!();

    // 6. Re-read the column and verify the persist path.
    let conn = db.connection();
    let stored: String = conn
        .query_row(
            "SELECT llm_explanation FROM fingerprints \
             WHERE document_name = ?1 AND entity_uri = ?2",
            rusqlite::params!["consistency-doc", "sdef://consistency-doc/User"],
            |row| row.get(0),
        )
        .expect("re-read llm_explanation");
    assert_eq!(stored, explanation, "stored explanation matches the LLM output");
    println!("OK: llm_explanation column persisted ({} bytes)", stored.len());
    println!("== consistency_llm_example: all checks passed ==");

    Ok(())
}

fn workspace_migrations_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .and_then(|p| p.parent()) // cleanroom-agent/
        .expect("cleanroom-agent crate layout has two parents")
        .join("migrations")
}
