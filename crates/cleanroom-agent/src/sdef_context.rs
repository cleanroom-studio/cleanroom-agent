//! `sdef_context` — Phase 0.3 helper that loads the S.DEF entities a
//! given LLM task needs as `cleanroom_prompt::ContextItem`s.
//!
//! # Why
//!
//! The LLM doesn't read the entire S.DEF document at once. Per
//! `docs/01-sdef-integration.md §3.1`, S.DEF is split into 16K-token
//! shards. Each LlmAnalyzeFile / LlmGenerateCode task gets a slice
//! tailored to its `file_path` / `entity_uri`.
//!
//! For Phase 0.3 we keep the slicing simple: queries go against the
//! `data_models` / `contracts` / `data_attributes` tables (the
//! "current" S.DEF storage). When the shard subsystem is built
//! (Phase 1+), the same public API will swap over to read from
//! `sdef_shards` without breaking callers.
//!
//! # Budget
//!
//! All loaders respect a `ContextBudget` (16K tokens by default, per
//! `docs/01-sdef-integration.md`). Items are sorted by `Priority` and
//! trimmed from `Optional` downward until the budget fits.
//!
//! # Determinism
//!
//! `load_shard_for_task` is a pure read against the DB; same input
//! gives the same `ContextItem` set. Tests pin this.

use std::sync::Arc;

use cleanroom_db::{Database, DbError, Task, TaskRepository, TaskType};
use cleanroom_prompt::{ContextBudget, ContextItem, Priority};

/// Default per-task context budget. Matches `docs/01-sdef-integration.md
/// §3.1` (16K tokens).
pub const DEFAULT_BUDGET_TOKENS: usize = 16_000;

/// Build the S.DEF context for a task. The shape of the loaded shard
/// depends on the task type:
///
/// - `LlmAnalyzeFile` (input has `file_path`): all `data_models` whose
///   logical model mentions the file, plus the document's
///   `data_models` summary. Falls back to the whole document if no
///   file-path match is found (LLM benefits from a broad context).
/// - `LlmGenerateCode` (input has `entity_uri`): the single entity
///   with that URI.
/// - anything else: empty vec (no context for legacy task types).
pub fn load_shard_for_task(
    db: &Arc<Database>,
    task: &Task,
    budget: &ContextBudget,
) -> Result<Vec<ContextItem>, DbError> {
    let input: serde_json::Value = serde_json::from_str(&task.input_json)
        .unwrap_or_else(|_| serde_json::json!({}));
    match task.task_type {
        TaskType::LlmAnalyzeFile => {
            let file_path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            load_shard_for_file(db, file_path, budget)
        }
        TaskType::LlmGenerateCode => {
            let entity_uri = input
                .get("entity_uri")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            load_entity_with_attributes(db, entity_uri, budget)
        }
        TaskType::InferDesignDecisions => {
            // Phase 1.1: hand the LLM all the per-file design
            // decisions already inferred for the module (from
            // earlier `LlmAnalyzeFile` tasks) plus the document
            // summary, so it can synthesize the *module-level* set
            // without re-deriving per-file observations. The
            // `module_name` comes from the task input; the document
            // name is the S.DEF document under analysis.
            let module_name = input
                .get("module_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let document_name = input
                .get("document")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if document_name.is_empty() || module_name.is_empty() {
                return Ok(Vec::new());
            }
            // Doc summary as a Medium anchor; module decisions as
            // High; the rest (per-file data models / etc.) come from
            // a `load_shard_for_file` for the first file in the
            // module so the LLM can see the entities.
            let mut items: Vec<ContextItem> = Vec::new();
            items.push(ContextItem::new(
                format!("sdef://{document_name}"),
                format!("Document: {document_name}\nModule under analysis: {module_name}"),
                Priority::Medium,
            ));
            // Existing module-level decisions (if a re-run of the
            // task happens; the LLM benefits from seeing its own
            // previous answers).
            let mut module_decisions = load_module_design_decisions(
                db,
                document_name,
                module_name,
                budget,
            )?;
            items.append(&mut module_decisions);
            // Per-file context: scan all files in the module and load
            // each one's shard. We use `load_shard_for_file` for
            // every file path in the task input.
            if let Some(arr) = input.get("file_paths").and_then(|v| v.as_array()) {
                for f in arr {
                    if let Some(rel) = f.as_str() {
                        let mut per_file = load_shard_for_file(db, rel, budget)?;
                        items.append(&mut per_file);
                    }
                }
            }
            trim_to_budget(&mut items, budget);
            Ok(items)
        }
        _ => Ok(Vec::new()),
    }
}

/// Load the S.DEF context for a given file path: all data models in
/// the same document, with the file-matching ones marked `Must` and
/// the rest marked `High`. Trimmed to fit `budget`.
pub fn load_shard_for_file(
    db: &Arc<Database>,
    file_path: &str,
    budget: &ContextBudget,
) -> Result<Vec<ContextItem>, DbError> {
    let conn = db.connection();
    // Look up the document for this file path. For Phase 0.3 we
    // don't have a reliable `repo_path -> document` map, so we
    // gather all documents and let the caller pick.
    let mut stmt = conn
        .prepare("SELECT name, description FROM sdef_documents")
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    let mut docs: Vec<(String, Option<String>)> = Vec::new();
    let mut rows = stmt
        .query([])
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    while let Some(row) = rows.next().map_err(|e| DbError::QueryFailed(e.to_string()))? {
        docs.push((
            row.get::<_, String>(0).map_err(|e| DbError::QueryFailed(e.to_string()))?,
            row.get::<_, Option<String>>(1)
                .map_err(|e| DbError::QueryFailed(e.to_string()))?,
        ));
    }
    drop(rows);
    drop(stmt);

    let mut items: Vec<ContextItem> = Vec::new();
    for (doc_name, doc_desc) in &docs {
        // The document itself is a "Medium" priority summary.
        items.push(ContextItem::new(
            format!("sdef://{doc_name}"),
            format!(
                "Document: {doc_name}\nDescription: {}\nFile under analysis: {file_path}",
                doc_desc.as_deref().unwrap_or("(no description)")
            ),
            Priority::Medium,
        ));

        // Data models in this document.
        let models = query_data_models(&conn, doc_name)?;
        for model in models {
            let is_relevant = !file_path.is_empty()
                && model
                    .get("logical_model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.contains(file_path))
                    .unwrap_or(false);
            let priority = if is_relevant {
                Priority::Must
            } else {
                Priority::High
            };
            let entity = model
                .get("entity")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string();
            let description = model
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            items.push(ContextItem::new(
                format!("sdef://{doc_name}/{entity}"),
                format!(
                    "Data model: {entity}\nDescription: {description}\nDetails: {}",
                    serde_json::to_string_pretty(&model).unwrap_or_default()
                ),
                priority,
            ));
        }
    }

    // Drop the lowest-priority items until we fit the budget.
    trim_to_budget(&mut items, budget);
    Ok(items)
}

/// Load a single entity (data model or contract) with its attributes
/// as a `Vec<ContextItem>`. Returns an empty vec if the entity isn't
/// found.
pub fn load_entity_with_attributes(
    db: &Arc<Database>,
    entity_uri: &str,
    budget: &ContextBudget,
) -> Result<Vec<ContextItem>, DbError> {
    if entity_uri.is_empty() {
        return Ok(Vec::new());
    }
    // `entity_uri` is `sdef://<doc>/<entity>` -- parse it loosely.
    let stripped = entity_uri
        .strip_prefix("sdef://")
        .unwrap_or(entity_uri);
    let mut parts = stripped.splitn(2, '/');
    let doc_name = parts.next().unwrap_or("").to_string();
    let entity_name = parts.next().unwrap_or("").to_string();
    if doc_name.is_empty() || entity_name.is_empty() {
        return Ok(Vec::new());
    }

    let conn = db.connection();
    let mut items: Vec<ContextItem> = Vec::new();
    if let Some(model) = query_data_model_one(&conn, &doc_name, &entity_name)? {
        items.push(ContextItem::new(
            entity_uri.to_string(),
            format!(
                "Data model: {entity_name}\nDescription: {}\nDetails: {}",
                model
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
                serde_json::to_string_pretty(&model).unwrap_or_default()
            ),
            Priority::Must,
        ));
        // Attributes for this model.
        let attrs = query_attributes(&conn, &doc_name, &entity_name)?;
        for attr in attrs {
            let name = attr
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let ty = attr.get("attr_type").and_then(|v| v.as_str()).unwrap_or("?");
            let desc = attr
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            items.push(ContextItem::new(
                format!("{entity_uri}#{name}"),
                format!("  - {name}: {ty} -- {desc}"),
                Priority::High,
            ));
        }
    }
    trim_to_budget(&mut items, budget);
    Ok(items)
}

/// Placeholder for Phase 1+: load a module + all its submodules'
/// entities. For Phase 0.3, this delegates to the document-level
/// loader so the API exists for callers to use.
pub fn load_module_subtree(
    db: &Arc<Database>,
    _module_uri: &str,
    budget: &ContextBudget,
) -> Result<Vec<ContextItem>, DbError> {
    // Phase 0.3: no real module tree. Look up the first document name
    // and return the whole document as a single "module" so callers
    // can use this API today. NB: the connection is held only inside
    // the scope below -- if we held it across the recursive
    // `load_shard_for_file` call we'd deadlock on the DB's Mutex.
    let doc_name: Option<String> = {
        let conn = db.connection();
        let mut stmt = conn
            .prepare("SELECT name FROM sdef_documents LIMIT 1")
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        stmt.query_row([], |row| row.get(0)).ok()
        // conn + stmt drop here, releasing the Mutex.
    };
    let Some(doc_name) = doc_name else { return Ok(Vec::new()) };
    let mut items = load_shard_for_file(db, "", budget)?;
    items.push(ContextItem::new(
        format!("sdef://{doc_name}"),
        format!("Module: {doc_name} (fallback: no module tree in Phase 0.3)"),
        Priority::Medium,
    ));
    Ok(items)
}

/// Placeholder for Phase 1+: load a function spec + the data types
/// it references. For Phase 0.3, returns the entity as-is.
pub fn load_function_and_dependents(
    db: &Arc<Database>,
    entity_uri: &str,
    budget: &ContextBudget,
) -> Result<Vec<ContextItem>, DbError> {
    load_entity_with_attributes(db, entity_uri, budget)
}

/// Phase 1.1: load all *module-level* design decisions for a given
/// module name, in the context-loader style. Per-module decisions
/// are persisted by [`ProducerAgent::infer_design_decisions`]
/// (see `producer.rs`) with `module_name = '<name>'` (set in
/// `DesignDecisionRecord::module_name` since migration 013; before
/// that we LIKE-matched `context = 'module=<name>;%'` which was
/// the pre-1.1 workaround). Returns one `ContextItem` per
/// decision, all at `Priority::High` (these are useful summaries
/// but rarely Must-level "the LLM cannot proceed without them").
/// Trimmed to fit `budget`.
///
/// An empty `module_name` returns an empty vec (no filter matches).
pub fn load_module_design_decisions(
    db: &Arc<Database>,
    document_name: &str,
    module_name: &str,
    budget: &ContextBudget,
) -> Result<Vec<ContextItem>, DbError> {
    if module_name.is_empty() || document_name.is_empty() {
        return Ok(Vec::new());
    }
    let conn = db.connection();
    let mut stmt = conn
        .prepare(
            "SELECT id, topic, decision, rationale FROM design_decisions \
             WHERE document_name = ?1 AND module_name = ?2 \
             ORDER BY topic ASC",
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    let rows = stmt
        .query_map(
            rusqlite::params![document_name, module_name],
            |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "topic": row.get::<_, String>(1)?,
                    "decision": row.get::<_, String>(2)?,
                    "rationale": row.get::<_, String>(3)?,
                }))
            },
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    let mut items: Vec<ContextItem> = Vec::new();
    for r in rows {
        let row = r.map_err(|e| DbError::QueryFailed(e.to_string()))?;
        let topic = row.get("topic").and_then(|v| v.as_str()).unwrap_or("?");
        let decision = row.get("decision").and_then(|v| v.as_str()).unwrap_or("?");
        let rationale = row.get("rationale").and_then(|v| v.as_str()).unwrap_or("");
        items.push(ContextItem::new(
            format!("sdef://{document_name}/{module_name}/decisions#{topic}"),
            format!(
                "Module decision [{module_name}/{topic}]:\n  decision:  {decision}\n  rationale: {rationale}"
            ),
            Priority::High,
        ));
    }
    trim_to_budget(&mut items, budget);
    Ok(items)
}

// ============================================================================
// Internal helpers
// ============================================================================

fn query_data_models(
    conn: &rusqlite::Connection,
    document_name: &str,
) -> Result<Vec<serde_json::Value>, DbError> {
    let mut stmt = conn
        .prepare(
            "SELECT entity, description, version, logical_model FROM data_models \
             WHERE document_name = ?1",
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    let rows = stmt
        .query_map(rusqlite::params![document_name], |row| {
            Ok(serde_json::json!({
                "entity": row.get::<_, String>(0)?,
                "description": row.get::<_, Option<String>>(1)?,
                "version": row.get::<_, Option<String>>(2)?,
                "logical_model": row.get::<_, Option<String>>(3)?,
            }))
        })
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| DbError::QueryFailed(e.to_string()))?);
    }
    Ok(out)
}

fn query_data_model_one(
    conn: &rusqlite::Connection,
    document_name: &str,
    entity_name: &str,
) -> Result<Option<serde_json::Value>, DbError> {
    let mut stmt = conn
        .prepare(
            "SELECT entity, description, version, logical_model FROM data_models \
             WHERE document_name = ?1 AND entity = ?2",
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    let mut rows = stmt
        .query_map(rusqlite::params![document_name, entity_name], |row| {
            Ok(serde_json::json!({
                "entity": row.get::<_, String>(0)?,
                "description": row.get::<_, Option<String>>(1)?,
                "version": row.get::<_, Option<String>>(2)?,
                "logical_model": row.get::<_, Option<String>>(3)?,
            }))
        })
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    match rows.next() {
        Some(r) => Ok(Some(r.map_err(|e| DbError::QueryFailed(e.to_string()))?)),
        None => Ok(None),
    }
}

fn query_attributes(
    conn: &rusqlite::Connection,
    document_name: &str,
    entity_name: &str,
) -> Result<Vec<serde_json::Value>, DbError> {
    let mut stmt = conn
        .prepare(
            "SELECT name, attr_type, format, description FROM data_attributes \
             WHERE document_name = ?1 AND entity = ?2",
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    let rows = stmt
        .query_map(rusqlite::params![document_name, entity_name], |row| {
            Ok(serde_json::json!({
                "name": row.get::<_, String>(0)?,
                "attr_type": row.get::<_, String>(1)?,
                "format": row.get::<_, Option<String>>(2)?,
                "description": row.get::<_, Option<String>>(3)?,
            }))
        })
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| DbError::QueryFailed(e.to_string()))?);
    }
    Ok(out)
}

/// Sort items by `Priority` (Must first), then drop the lowest-priority
/// items until the total estimated tokens fit inside the budget.
fn trim_to_budget(items: &mut Vec<ContextItem>, budget: &ContextBudget) {
    items.sort_by_key(|i| (i.priority as i32, i.uri.clone()));
    let cap = budget.effective_max();
    let mut total: usize = 0;
    items.retain(|i| {
        total += i.estimated_tokens;
        // `Must` items are never trimmed.
        if i.priority == Priority::Must {
            return true;
        }
        total <= cap
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_db::{TaskStatus, TaskType};

    fn workspace_migrations_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("cleanroom-agent crate layout has two parents")
            .join("migrations")
    }

    fn make_db() -> (tempfile::TempDir, Arc<Database>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("sdef-ctx-test.db");
        let db = Database::open_with_migrations_from(&db_path, Some(&workspace_migrations_dir()))
            .expect("open db");
        (dir, Arc::new(db))
    }

    fn seed_one_document(dir: &tempfile::TempDir) {
        let conn = rusqlite::Connection::open(dir.path().join("sdef-ctx-test.db")).expect("open");
        conn.execute_batch(
            "INSERT INTO sdef_documents (name, version, description, created_at, updated_at) \
             VALUES ('doc-1', '0.1.0', 'Test document', datetime(), datetime());\
             INSERT INTO data_models (entity, document_name, status, description, logical_model) \
             VALUES ('User', 'doc-1', 'active', 'A user record', 'src/user.rs');\
             INSERT INTO data_models (entity, document_name, status, description, logical_model) \
             VALUES ('Order', 'doc-1', 'active', 'An order record', 'src/order.rs');\
             INSERT INTO data_attributes (document_name, entity, name, attr_type, description, required) \
             VALUES ('doc-1', 'User', 'id', 'String', 'Primary key', 1);\
             INSERT INTO data_attributes (document_name, entity, name, attr_type, description, required) \
             VALUES ('doc-1', 'User', 'email', 'String', 'Email address', 1);\
             INSERT INTO data_attributes (document_name, entity, name, attr_type, description) \
             VALUES ('doc-1', 'Order', 'id', 'String', 'Primary key');\
             INSERT INTO data_attributes (document_name, entity, name, attr_type, description) \
             VALUES ('doc-1', 'Order', 'total', 'i64', 'Order total in cents');",
        )
        .expect("seed");
    }

    fn make_task(task_type: TaskType, input: serde_json::Value) -> Task {
        Task {
            task_id: uuid::Uuid::new_v4().to_string(),
            task_type,
            status: TaskStatus::Pending,
            priority: 5,
            input_json: input.to_string(),
            output_json: None,
            error_message: None,
            assigned_to: None,
            progress: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 2,
            last_heartbeat: None,
            dependencies_json: "[]".to_string(),
            version: 1,
        }
    }

    #[test]
    fn test_load_shard_for_file_returns_relevant_model_first() {
        let (dir, db) = make_db();
        seed_one_document(&dir);
        let budget = ContextBudget::default();
        // The User model has logical_model 'src/user.rs' -- the relevant
        // model is marked Must; the Order model is High.
        let items = load_shard_for_file(&db, "src/user.rs", &budget).expect("ok");
        let must_uris: Vec<&str> = items
            .iter()
            .filter(|i| i.priority == Priority::Must)
            .map(|i| i.uri.as_str())
            .collect();
        assert!(
            must_uris.contains(&"sdef://doc-1/User"),
            "User should be Must (logical_model matches file_path)"
        );
        let high_uris: Vec<&str> = items
            .iter()
            .filter(|i| i.priority == Priority::High)
            .map(|i| i.uri.as_str())
            .collect();
        assert!(
            high_uris.contains(&"sdef://doc-1/Order"),
            "Order should be High (no match for file_path)"
        );
    }

    #[test]
    fn test_load_entity_with_attributes_returns_must_plus_high() {
        let (dir, db) = make_db();
        seed_one_document(&dir);
        let budget = ContextBudget::default();
        let items =
            load_entity_with_attributes(&db, "sdef://doc-1/User", &budget).expect("ok");
        let must = items
            .iter()
            .filter(|i| i.priority == Priority::Must)
            .count();
        let high = items
            .iter()
            .filter(|i| i.priority == Priority::High)
            .count();
        assert_eq!(must, 1, "entity itself is Must");
        assert_eq!(high, 2, "2 attributes are High");
    }

    #[test]
    fn test_load_shard_for_unknown_file_falls_back_to_no_must() {
        let (dir, db) = make_db();
        seed_one_document(&dir);
        let budget = ContextBudget::default();
        // An empty file_path: no model is "relevant" so nothing is Must.
        // The doc summary itself is Medium, the data models are High
        // (downgraded from Must because they don't match the file).
        let items = load_shard_for_file(&db, "", &budget).expect("ok");
        assert!(
            items.iter().all(|i| i.priority != Priority::Must),
            "unknown file -> nothing is Must"
        );
        // The User + Order models both appear, with the doc as a Medium summary.
        let uris: Vec<&str> = items.iter().map(|i| i.uri.as_str()).collect();
        assert!(uris.contains(&"sdef://doc-1/User"));
        assert!(uris.contains(&"sdef://doc-1/Order"));
        let must_count = items.iter().filter(|i| i.priority == Priority::Must).count();
        assert_eq!(must_count, 0);
    }

    #[test]
    fn test_load_shard_trims_low_priority_when_over_budget() {
        let (dir, db) = make_db();
        seed_one_document(&dir);
        // Tiny budget: 100 tokens. The 2 attributes alone already eat
        // through that. The "Must" entity stays; the "High" entity
        // gets dropped.
        let budget = ContextBudget {
            max_tokens: 100,
            tool_response_reserve: 0,
            safety_margin: 1.0,
        };
        let items = load_shard_for_file(&db, "src/user.rs", &budget).expect("ok");
        let total: usize = items.iter().map(|i| i.estimated_tokens).sum();
        assert!(total <= 100, "total {total} must fit budget");
        // Must items always kept.
        assert!(items.iter().any(|i| i.priority == Priority::Must));
    }

    #[test]
    fn test_load_shard_for_task_dispatches_by_task_type() {
        let (dir, db) = make_db();
        seed_one_document(&dir);
        let budget = ContextBudget::default();

        let analyze_task = make_task(
            TaskType::LlmAnalyzeFile,
            serde_json::json!({
                "file_path": "src/user.rs",
                "document": "doc-1",
            }),
        );
        let items = load_shard_for_task(&db, &analyze_task, &budget).expect("ok");
        assert!(!items.is_empty(), "analyze task -> non-empty shard");

        let generate_task = make_task(
            TaskType::LlmGenerateCode,
            serde_json::json!({
                "entity_uri": "sdef://doc-1/User",
                "document": "doc-1",
            }),
        );
        let items = load_shard_for_task(&db, &generate_task, &budget).expect("ok");
        assert_eq!(
            items.iter().filter(|i| i.priority == Priority::Must).count(),
            1
        );

        // Legacy task types return empty.
        let legacy_task = make_task(TaskType::RepoAnalyze, serde_json::json!({}));
        let items = load_shard_for_task(&db, &legacy_task, &budget).expect("ok");
        assert!(items.is_empty());
    }

    #[test]
    fn test_load_module_subtree_works() {
        let (dir, db) = make_db();
        seed_one_document(&dir);
        let budget = ContextBudget::default();
        let items = load_module_subtree(&db, "sdef://doc-1", &budget).expect("ok");
        assert!(!items.is_empty(), "Phase 0.3 fallback returns at least the doc summary");
    }

    // Phase 1.1: seed two module-level design decisions on top of the
    // baseline `seed_one_document` fixture and assert the helper finds
    // them by module name. Phase 1.1 (close-out): use the
    // dedicated `module_name` column (migration 013) instead of
    // stuffing the value into `context`.
    fn seed_module_decisions(dir: &tempfile::TempDir) {
        let conn =
            rusqlite::Connection::open(dir.path().join("sdef-ctx-test.db")).expect("open");
        conn.execute_batch(
            "INSERT INTO design_decisions \
             (id, document_name, topic, decision, rationale, module_name) VALUES \
             ('dd-mod-001', 'doc-1', 'Storage backend', 'In-memory Vec', \
              'Simplicity.', 'src');\
             INSERT INTO design_decisions \
             (id, document_name, topic, decision, rationale, module_name) VALUES \
             ('dd-mod-002', 'doc-1', 'Error handling', 'Result<T, E>', \
              'No panics.', 'src');\
             INSERT INTO design_decisions \
             (id, document_name, topic, decision, rationale, module_name) VALUES \
             ('dd-other', 'doc-1', 'Other module', 'n/a', 'n/a', 'other');",
        )
        .expect("seed module decisions");
    }

    #[test]
    fn test_load_module_design_decisions_filters_by_module() {
        let (dir, db) = make_db();
        seed_one_document(&dir);
        seed_module_decisions(&dir);
        let budget = ContextBudget::default();

        let items = load_module_design_decisions(&db, "doc-1", "src", &budget).expect("ok");
        assert_eq!(items.len(), 2, "two module-level decisions for 'src'");
        // The "Other module" decision must NOT be present (we filter
        // on `module=src;` specifically, prefix-safe).
        let uris: Vec<&str> = items.iter().map(|i| i.uri.as_str()).collect();
        assert!(uris.iter().any(|u| u.contains("Storage backend")));
        assert!(uris.iter().any(|u| u.contains("Error handling")));
        assert!(!uris.iter().any(|u| u.contains("Other module")));

        // The 'other' module returns its 1 decision, not the 'src' ones.
        let other = load_module_design_decisions(&db, "doc-1", "other", &budget).expect("ok");
        assert_eq!(other.len(), 1);

        // Empty module name returns empty.
        let empty =
            load_module_design_decisions(&db, "doc-1", "", &budget).expect("ok");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_load_shard_for_task_dispatches_infer_design_decisions() {
        // Phase 1.1: `InferDesignDecisions` tasks should get a
        // non-empty context: the doc summary + any prior module-level
        // decisions + per-file shards. This pins the contract that
        // the LLM sees prior observations when re-synthesizing.
        let (dir, db) = make_db();
        seed_one_document(&dir);
        seed_module_decisions(&dir);
        let budget = ContextBudget::default();

        let task = make_task(
            TaskType::InferDesignDecisions,
            serde_json::json!({
                "document": "doc-1",
                "module_name": "src",
                "file_paths": ["src/user.rs"],
            }),
        );
        let items = load_shard_for_task(&db, &task, &budget).expect("ok");
        assert!(
            !items.is_empty(),
            "InferDesignDecisions task should get a non-empty context"
        );
        // The doc summary is always present (Medium).
        assert!(items.iter().any(|i| i.priority == Priority::Medium));
        // At least one High item from the prior module decisions.
        assert!(items.iter().any(|i| i.priority == Priority::High));
    }

    #[test]
    fn test_default_budget_constant_matches_doc() {
        assert_eq!(DEFAULT_BUDGET_TOKENS, 16_000);
    }
}
