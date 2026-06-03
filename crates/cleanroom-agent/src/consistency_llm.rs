//! `consistency_llm` — Phase 1.5: LLM-narrated fingerprint mismatch.
//!
//! The `fingerprints` table tracks three SHA-256 hashes per
//! entity (sdef_hash / db_hash / code_hash). When they
//! disagree, the partial index
//! `idx_fingerprints_inconsistent` flags the row. Until 1.5
//! the only signal a human got was a log line; Phase 1.5
//! adds an LLM pass that explains *why* the hashes disagree.
//!
//! # What
//!
//! [`explain_mismatch`] takes the LLM and a snapshot of one
//! mismatched row (entity_uri + the three hashes + a small
//! S.DEF snippet) and returns a short narrative
//! explanation: "the S.DEF hash and DB hash disagree because
//! the consumer dropped a column; the code hash differs
//! because...". The string is written to the new
//! `fingerprints.llm_explanation` column (migration 012) so a
//! human reviewer can see it on the next CLI `inspect
//! consistency` run.
//!
//! # Why now (Phase 1.5)
//!
//! The pre-1.5 consistency check was a bit of a black box: the
//! CLI said "0 inconsistent rows" or "5 inconsistent rows" and
//! left the user to grep the source. The LLM is well-suited to
//! the "blame the right layer" task — it can compare the three
//! hashes and the S.DEF snippet and produce a 1-2 sentence
//! hypothesis. It's a small cost (~$0.01 per call) and a big
//! DX win.
//!
//! # Limitations
//!
//! The LLM only sees the S.DEF snippet + the three hashes; it
//! doesn't see the actual source code (that would balloon the
//! prompt and burn tokens). The explanation is a *hypothesis*
//! — useful as a starting point for the human reviewer, not a
//! ground-truth diagnosis. A real "compare the bytes" check
//! is Phase 2 work.

use std::sync::Arc;

use cleanroom_meta_llm::MetaLlm;
use rusqlite::params;
use tracing::warn;

/// Phase 1.5: one row of the `fingerprints` table, snapped
/// before we send it to the LLM. Kept as a plain struct (not
/// the database row) so callers can construct one from
/// in-memory data without going through the DB.
#[derive(Debug, Clone)]
pub struct FingerprintSnapshot {
    pub document_name: String,
    pub entity_uri: String,
    pub entity_type: String,
    pub sdef_hash: Option<String>,
    pub db_hash: Option<String>,
    pub code_hash: Option<String>,
    pub code_path: Option<String>,
    /// Optional short S.DEF fragment (the entity itself, as
    /// pretty JSON). When the caller has it, including it
    /// gives the LLM much more context to work with.
    pub sdef_snippet: Option<String>,
}

/// Error type for [`explain_mismatch`]. `LlmCall` wraps
/// transport / rate-limit / context-length failures;
/// `Persist` wraps the `UPDATE fingerprints` failure
/// (we don't fail the surrounding consistency check on
/// persist failure — just `warn!` and move on).
#[derive(Debug, thiserror::Error)]
pub enum ConsistencyLlmError {
    #[error("LLM call failed: {0}")]
    LlmCall(String),
    #[error("DB persist failed: {0}")]
    Persist(String),
}

/// Ask the LLM to explain a fingerprint mismatch, and persist
/// the result. Returns the explanation string on success.
///
/// # Cost
///
/// One LLM call per invocation (~$0.01-0.02 with `MiniMax-M3`,
/// depending on the length of `sdef_snippet`). The caller
/// (typically `consistency_checker`) is expected to gate this
/// behind a `--explain` CLI flag so the cost is opt-in.
///
/// # Errors
///
/// - `LlmCall`: the LLM call failed (rate limit, transport,
///   etc.). Caller decides whether to retry or skip.
/// - `Persist`: the `UPDATE fingerprints SET llm_explanation`
///   failed. We log a `warn!` and return `Err` so the caller
///   knows the audit row wasn't updated, but we *don't*
///   silently swallow it — that would make diagnostics worse
///   than the pre-1.5 state.
pub async fn explain_mismatch(
    llm: Arc<dyn MetaLlm>,
    db: &Arc<cleanroom_db::Database>,
    snapshot: &FingerprintSnapshot,
) -> Result<String, ConsistencyLlmError> {
    let explanation = llm_explain(&llm, snapshot)
        .await
        .map_err(ConsistencyLlmError::LlmCall)?;
    persist_explanation(db, snapshot, &explanation)
        .map_err(ConsistencyLlmError::Persist)?;
    Ok(explanation)
}

/// Pure-LLM call; exposed separately so tests / CLI tools
/// can dry-run the prompt without touching the DB.
pub async fn llm_explain(
    llm: &Arc<dyn MetaLlm>,
    snapshot: &FingerprintSnapshot,
) -> Result<String, String> {
    let system_prompt = build_consistency_explain_system_prompt();
    let user_message = render_user_message(snapshot);
    let messages = vec![
        cleanroom_meta_llm::chat::MetaMessageBuilder::new(
            cleanroom_meta_llm::chat::MetaRole::System,
        )
        .content(system_prompt)
        .build(),
        cleanroom_meta_llm::chat::MetaMessageBuilder::new(
            cleanroom_meta_llm::chat::MetaRole::User,
        )
        .content(user_message)
        .build(),
    ];
    let response = llm
        .chat(&messages, None)
        .await
        .map_err(|e| e.to_string())?;
    Ok(response.text().unwrap_or_default().trim().to_string())
}

fn render_user_message(s: &FingerprintSnapshot) -> String {
    let mut out = String::new();
    out.push_str(&format!("Entity: {}\n", s.entity_uri));
    out.push_str(&format!("Document: {}\n", s.document_name));
    out.push_str(&format!("Entity type: {}\n", s.entity_type));
    out.push_str("\nThree SHA-256 fingerprints:\n");
    out.push_str(&format!("  sdef_hash:  {}\n", s.sdef_hash.as_deref().unwrap_or("(none)")));
    out.push_str(&format!("  db_hash:    {}\n", s.db_hash.as_deref().unwrap_or("(none)")));
    out.push_str(&format!("  code_hash:  {}\n", s.code_hash.as_deref().unwrap_or("(none)")));
    if let Some(p) = &s.code_path {
        out.push_str(&format!("  code_path:  {p}\n"));
    }
    if let Some(snippet) = &s.sdef_snippet {
        out.push_str(&format!(
            "\nS.DEF snippet (the contract the hashes were computed from):\n```json\n{snippet}\n```\n"
        ));
    }
    out.push_str(
        "\nWrite a 1-2 sentence hypothesis explaining WHY these three hashes disagree. \
         No JSON, no bullets, just prose.",
    );
    out
}

fn build_consistency_explain_system_prompt() -> String {
    String::from(
        "You are a code/S.DEF consistency analyst. You're given three SHA-256 fingerprints \
         for one entity:\n\
         \n\
         - `sdef_hash`:  computed from the S.DEF document on disk (the source of truth)\n\
         - `db_hash`:    computed from the row in the SQLite `data_models` / `contracts` table\n\
         - `code_hash`:  computed from the generated source code on disk\n\
         \n\
         When the three agree, the entity is consistent. When they disagree, you must \
         explain *why*. Likely causes:\n\
         - `sdef_hash != db_hash`  : the S.DEF importer dropped a field, or a migration \
           was missing, or the parser trimmed whitespace.\n\
         - `sdef_hash == db_hash, both != code_hash`  : the consumer generated code that \
           doesn't match the S.DEF (missing field, wrong type, hallucinated method).\n\
         - `db_hash == code_hash, both != sdef_hash`  : the S.DEF document was edited by hand \
           but the DB was not re-imported.\n\
         - all three differ  : the S.DEF, the DB, and the code all moved independently — \
           the importer needs to be re-run, then the consumer regenerated.\n\
         \n\
         Be terse (1-2 sentences). Cite the specific hash difference (which two of the three \
         match) so the human reviewer knows where to look. No JSON, no bullets, just prose.",
    )
}

fn persist_explanation(
    db: &Arc<cleanroom_db::Database>,
    snapshot: &FingerprintSnapshot,
    explanation: &str,
) -> Result<(), String> {
    let conn = db.connection();
    let updated = conn
        .execute(
            "UPDATE fingerprints SET llm_explanation = ?1 \
             WHERE document_name = ?2 AND entity_uri = ?3",
            params![explanation, snapshot.document_name, snapshot.entity_uri],
        )
        .map_err(|e| e.to_string())?;
    if updated == 0 {
        warn!(
            document = %snapshot.document_name,
            entity = %snapshot.entity_uri,
            "consistency_llm: UPDATE affected 0 rows (row missing or already deleted); \
             the audit trail won't be updated"
        );
        return Err(format!(
            "no fingerprints row for document={}, entity_uri={}",
            snapshot.document_name, snapshot.entity_uri
        ));
    }
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_user_message_includes_all_three_hashes() {
        let s = FingerprintSnapshot {
            document_name: "doc-1".to_string(),
            entity_uri: "sdef://doc-1/User".to_string(),
            entity_type: "data_model".to_string(),
            sdef_hash: Some("aaa".to_string()),
            db_hash: Some("aaa".to_string()),
            code_hash: Some("bbb".to_string()),
            code_path: Some("src/user.rs".to_string()),
            sdef_snippet: None,
        };
        let rendered = render_user_message(&s);
        assert!(rendered.contains("sdef_hash:  aaa"));
        assert!(rendered.contains("db_hash:    aaa"));
        assert!(rendered.contains("code_hash:  bbb"));
        assert!(rendered.contains("code_path:  src/user.rs"));
        assert!(rendered.contains("hypothesis"));
    }

    #[test]
    fn render_user_message_handles_none_hashes() {
        let s = FingerprintSnapshot {
            document_name: "doc-1".to_string(),
            entity_uri: "sdef://doc-1/User".to_string(),
            entity_type: "data_model".to_string(),
            sdef_hash: None,
            db_hash: None,
            code_hash: None,
            code_path: None,
            sdef_snippet: None,
        };
        let rendered = render_user_message(&s);
        // All-None hashes are rendered as "(none)" so the LLM
        // sees the gap instead of an empty field.
        assert!(rendered.contains("(none)"));
        // We don't render the `code_path:` line if `code_path`
        // is None (avoids `code_path: ` with empty value).
        assert!(!rendered.contains("code_path:"));
    }

    #[test]
    fn system_prompt_mentions_three_hash_types() {
        let s = build_consistency_explain_system_prompt();
        assert!(s.contains("sdef_hash"));
        assert!(s.contains("db_hash"));
        assert!(s.contains("code_hash"));
        // Terse guidance
        assert!(s.contains("1-2 sentences"));
    }

    /// Phase 1.5 MVP close-out: integration test that
    /// `persist_explanation` actually writes the
    /// `llm_explanation` column to the `fingerprints` table.
    /// We seed a 3-hash-mismatch row, call
    /// `persist_explanation`, then re-read the row to verify
    /// the column was updated. No LLM is involved — we're
    /// testing the persistence path, not the LLM call.
    #[test]
    fn persist_explanation_writes_to_db() {
        use cleanroom_db::Database;
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("consistency-test.db");
        let migrations_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("crate layout has two parents")
            .join("migrations");
        let db = std::sync::Arc::new(
            Database::open_with_migrations_from(&db_path, Some(&migrations_dir))
                .expect("open"),
        );
        // We need a parent sdef_documents row because
        // `fingerprints.document_name` has a FK to it.
        let conn = db.connection();
        conn.execute(
            "INSERT INTO sdef_documents (name, version, description, created_at, updated_at) \
             VALUES ('persistence-doc', '0.1.0', 'test', datetime(), datetime())",
            [],
        )
        .expect("insert sdef_documents");
        conn.execute(
            "INSERT INTO fingerprints \
             (entity_uri, document_name, entity_type, sdef_hash, db_hash, code_hash) \
             VALUES ('sdef://persistence-doc/User', 'persistence-doc', 'data_model', \
                     'aaa', 'bbb', 'ccc')",
            [],
        )
        .expect("insert fingerprint");
        drop(conn);

        // Run the persistence-only path (no LLM).
        let snapshot = FingerprintSnapshot {
            document_name: "persistence-doc".to_string(),
            entity_uri: "sdef://persistence-doc/User".to_string(),
            entity_type: "data_model".to_string(),
            sdef_hash: Some("aaa".to_string()),
            db_hash: Some("bbb".to_string()),
            code_hash: Some("ccc".to_string()),
            code_path: Some("src/user.rs".to_string()),
            sdef_snippet: None,
        };
        let explanation = "S.DEF hash and DB hash disagree: importer dropped a field. \
                          Code hash differs: consumer hallucinated a method.";
        persist_explanation(&db, &snapshot, explanation).expect("persist");

        // Re-read and verify the column is populated.
        let conn = db.connection();
        let stored: String = conn
            .query_row(
                "SELECT llm_explanation FROM fingerprints \
                 WHERE document_name = ?1 AND entity_uri = ?2",
                rusqlite::params!["persistence-doc", "sdef://persistence-doc/User"],
                |row| row.get(0),
            )
            .expect("re-read");
        assert_eq!(stored, explanation);
    }

    /// Negative test: `persist_explanation` against a
    /// non-existent fingerprint row returns `Err` (we don't
    /// silently swallow the audit loss).
    #[test]
    fn persist_explanation_errors_when_row_missing() {
        use cleanroom_db::Database;
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("consistency-empty.db");
        let migrations_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("crate layout has two parents")
            .join("migrations");
        let db = std::sync::Arc::new(
            Database::open_with_migrations_from(&db_path, Some(&migrations_dir))
                .expect("open"),
        );
        let snapshot = FingerprintSnapshot {
            document_name: "ghost-doc".to_string(),
            entity_uri: "sdef://ghost-doc/Phantom".to_string(),
            entity_type: "data_model".to_string(),
            sdef_hash: Some("xxx".to_string()),
            db_hash: Some("yyy".to_string()),
            code_hash: Some("zzz".to_string()),
            code_path: None,
            sdef_snippet: None,
        };
        let result = persist_explanation(&db, &snapshot, "anything");
        assert!(result.is_err(), "missing row must NOT be silently swallowed");
    }
}
