//! Parser + writer for the `sdef_output` schema the LLM emits in
//! response to `LlmAnalyzeFile` tasks. Phase 0.5+; replaces the
//! `parser: "pending (Phase 0.7)"` marker in `tasks.output_json` with
//! a real persistence path.
//!
//! Pipeline:
//! 1. LLM response text (with `<think>...</think>` thinking + a
//!    ```json ... ``` fenced JSON body) is fed to
//!    [`parse_llm_analyze_output`]
//! 2. The parser extracts the JSON, tolerates missing keys, and yields
//!    a [`ParsedEntities`] value with parallel `Vec`s of
//!    data-models / contracts / functions / design-decisions
//! 3. [`write_parsed_to_db`] maps those onto `cleanroom_db` model
//!    structs and persists them via `SdefRepository`
//!
//! Schema references (from `crates/cleanroom-db/src/repositories/sdef_repository.rs`):
//! - `DataModel` / `DataAttribute` — `data_models` / `data_attributes` tables
//! - `Contract` — `contracts` table (LLM `methods` JSON gets stuffed
//!   into `invariants_json` as a stand-in; the dedicated
//!   `contract_methods` table is not exposed via `SdefRepository` yet)
//! - `FunctionSpec` — `function_specs` table
//! - `DesignDecisionRecord` — `design_decisions` table
//!
//! Tests below cover parser happy-path + every error variant, plus a
//! writer round-trip against an in-memory migrated DB.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

use cleanroom_db::repositories::sdef_repository::{
    Contract, DataAttribute, DataModel, DesignDecisionRecord, FunctionSpec, SdefDocument,
    SdefRepository,
};
use cleanroom_db::DbError;

// =============================================================================
// Intermediate parsed entities (LLM JSON shape)
// =============================================================================

/// All entities extracted from one LLM response. Every field defaults
/// to an empty `Vec` so the LLM is free to emit any subset.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ParsedEntities {
    #[serde(default)]
    pub data_models: Vec<ParsedDataModel>,
    #[serde(default)]
    pub contracts: Vec<ParsedContract>,
    #[serde(default)]
    pub functions: Vec<ParsedFunction>,
    #[serde(default)]
    pub design_decisions: Vec<ParsedDesignDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedDataModel {
    pub name: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub visibility: Option<String>,
    #[serde(default)]
    pub fields: Vec<ParsedField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedField {
    pub name: String,
    #[serde(rename = "type", default)]
    pub type_: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub visibility: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedContract {
    pub name: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub visibility: Option<String>,
    #[serde(default)]
    pub methods: Vec<ParsedMethod>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedMethod {
    pub name: String,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedFunction {
    pub name: String,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub logic: Option<String>,
    #[serde(default)]
    pub visibility: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedDesignDecision {
    #[serde(default)]
    pub topic: Option<String>,
    // The LLM doesn't always emit a `decision` field (it often folds the
    // decision into `description` / `rationale`). Tolerate absence and
    // let the writer fall back to those fields (or `"unspecified"`).
    #[serde(default)]
    pub decision: Option<String>,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

// =============================================================================
// Parse error
// =============================================================================

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("LLM output contains no ```json ... ``` (or generic ``` ... ```) fence; cannot extract structured entities")]
    NoJsonFence,
    #[error("malformed JSON inside the fence: {0}")]
    JsonSyntax(#[from] serde_json::Error),
    #[error("LLM JSON root is not an object (got: {0})")]
    NotAnObject(String),
}

// =============================================================================
// Extraction + parse
// =============================================================================

const JSON_FENCE_OPEN: &str = "```json";
const FENCE_CLOSE: &str = "```";

/// Top-level keys that the `sdef_output` schema guarantees. The
/// recovery path uses this set to decide whether a candidate
/// `{...}` block from a fence-less LLM response is actually a
/// `sdef_output` payload (and not, say, a stray `serde_json::Value`
/// embedded in a Rust example).
const SDEF_TOP_LEVEL_KEYS: &[&str] = &[
    "data_models",
    "contracts",
    "functions",
    "design_decisions",
];

/// Extract the body of the first ```json ... ``` block (or, as a
/// fallback, the first generic ``` ... ``` block) from a free-form
/// LLM response. Returns `None` if no fence is present.
fn extract_json_block(raw: &str) -> Option<&str> {
    // Prefer ```json (the prompt asks for it explicitly); fall back to a
    // generic ``` fence if the LLM dropped the language tag.
    let after_open = if let Some(idx) = raw.find(JSON_FENCE_OPEN) {
        &raw[idx + JSON_FENCE_OPEN.len()..]
    } else if let Some(idx) = raw.find(FENCE_CLOSE) {
        // Heuristic: only treat a generic ``` as a JSON fence if the
        // next non-whitespace byte is `{` or `[`. Otherwise we'd parse
        // e.g. a ```rust code block.
        let rest = &raw[idx + FENCE_CLOSE.len()..];
        let trimmed = rest.trim_start();
        if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
            return None;
        }
        rest
    } else {
        return None;
    };
    let close_offset = after_open.find(FENCE_CLOSE)?;
    Some(after_open[..close_offset].trim())
}

/// Strip `<think>...</think>` blocks from a raw LLM response. The
/// MiniMax-M3 reasoning model emits these by default and they
/// precede the structured JSON output. By the time we reach the
/// recovery path the JSON fence is gone but the thinking block
/// is still noise we need to skip.
fn strip_think_blocks(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(open) = rest.find("<think>") {
        out.push_str(&rest[..open]);
        let after_open = &rest[open + "<think>".len()..];
        if let Some(close) = after_open.find("</think>") {
            rest = &after_open[close + "</think>".len()..];
        } else {
            // Unterminated `<think>` — treat the rest as thinking
            // content and stop.
            return out;
        }
    }
    out.push_str(rest);
    out
}

/// Recovery fallback: if the LLM forgot the ```json fence
/// (a known failure mode for `MiniMax-M3` on complex source
/// files ≥ 100 LoC where the `max_tokens=1024` cap cut the
/// response mid-thought), try to find a balanced top-level
/// `{...}` object in the raw text and return a slice of it.
/// Returns `None` if no candidate is found.
///
/// We deliberately use byte-level scanning with a brace counter
/// rather than a regex so that strings, escapes, and nested
/// objects are handled correctly. The candidate must contain at
/// least one of [`SDEF_TOP_LEVEL_KEYS`] for us to trust it.
fn find_raw_sdef_object(raw: &str) -> Option<&str> {
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            // Found a candidate `{`; walk forward, counting
            // unmatched braces (skipping over JSON strings so a
            // `"` inside a string doesn't confuse the count).
            let mut depth: i32 = 0;
            let mut j = i;
            let mut in_string = false;
            let mut escape_next = false;
            while j < bytes.len() {
                let c = bytes[j];
                if escape_next {
                    escape_next = false;
                } else if in_string {
                    match c {
                        b'\\' => escape_next = true,
                        b'"' => in_string = false,
                        _ => {}
                    }
                } else {
                    match c {
                        b'"' => in_string = true,
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                let candidate = &raw[i..=j];
                                if looks_like_sdef_payload(candidate) {
                                    return Some(candidate);
                                }
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }
    None
}

/// Cheap structural check: does this `{...}` blob look like an
/// `sdef_output` payload? We require at least one of the four
/// known top-level keys. This filters out the case where the LLM
/// produced a balanced but irrelevant JSON object (e.g. an
/// example `{"x": 1}` embedded in a prose explanation).
fn looks_like_sdef_payload(blob: &str) -> bool {
    SDEF_TOP_LEVEL_KEYS.iter().any(|k| {
        let needle = format!("\"{k}\"");
        blob.contains(&needle)
    })
}

/// Parse a `LlmAnalyzeFile` output blob into intermediate entities.
/// Tolerant of <think>...</mm:think> blocks preceding the JSON fence.
///
/// Recovery ladder (in order):
/// 1. Look for a ```json ... ``` fence — the prompt-asked format.
/// 2. Look for a generic ``` ... ``` fence whose body starts with `{`.
/// 3. (New in 0.5++) Strip `<think>...</think>` and scan for a
///    balanced top-level `{...}` object that contains at least one
///    of the `sdef_output` keys. This recovers from the failure
///    mode where the LLM produced valid JSON but ran out of
///    `max_tokens` before the closing ``` fence was emitted.
pub fn parse_llm_analyze_output(raw: &str) -> Result<ParsedEntities, ParseError> {
    let json_str = if let Some(fenced) = extract_json_block(raw) {
        fenced
    } else {
        // No fence — try the recovery path. We do all the work on
        // an owned String so the returned &str borrow is tied to a
        // local owned value (the `?` short-circuits if None).
        let stripped = strip_think_blocks(raw);
        find_raw_sdef_object(&stripped).ok_or(ParseError::NoJsonFence)?
    };
    let value: serde_json::Value = serde_json::from_str(json_str)?;
    let obj = value
        .as_object()
        .ok_or_else(|| ParseError::NotAnObject(value.to_string()))?;
    let entities: ParsedEntities =
        serde_json::from_value(serde_json::Value::Object(obj.clone()))?;
    Ok(entities)
}

// =============================================================================
// Writer
// =============================================================================

/// Counts of rows persisted for each entity kind, returned by
/// [`write_parsed_to_db`] so the caller can log / report progress.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WriteSummary {
    pub data_models: usize,
    pub attributes: usize,
    pub contracts: usize,
    pub functions: usize,
    pub design_decisions: usize,
}

impl WriteSummary {
    pub fn is_empty(&self) -> bool {
        self.data_models == 0
            && self.contracts == 0
            && self.functions == 0
            && self.design_decisions == 0
    }

    pub fn total(&self) -> usize {
        self.data_models
            + self.attributes
            + self.contracts
            + self.functions
            + self.design_decisions
    }
}

/// Persist `ParsedEntities` into the database via `SdefRepository`.
/// Also upserts the parent `SdefDocument` row so the `export` command
/// can find it (this is what `--mode llm` was missing in Phase 0.5).
pub fn write_parsed_to_db(
    repo: &SdefRepository,
    document_name: &str,
    entities: &ParsedEntities,
) -> Result<WriteSummary, DbError> {
    let mut summary = WriteSummary::default();
    let now = chrono::Utc::now().to_rfc3339();

    // Upsert the document so `export --document X` finds it.
    repo.upsert_document(&SdefDocument {
        name: document_name.to_string(),
        version: None,
        description: Some(format!(
            "LLM-analyzed document ({} data models, {} contracts, {} functions, {} design decisions)",
            entities.data_models.len(),
            entities.contracts.len(),
            entities.functions.len(),
            entities.design_decisions.len(),
        )),
        created_at: now.clone(),
        updated_at: now,
    })?;

    for dm in &entities.data_models {
        // `data_models.status` CHECK constraint: ('active', 'deprecated', 'legacy').
        // Map enum -> "active" (same default); legacy if explicitly marked.
        let status = if dm.kind.as_deref().is_some_and(|k| k.eq_ignore_ascii_case("legacy")) {
            "legacy"
        } else {
            "active"
        };
        repo.create_data_model(&DataModel {
            entity: dm.name.clone(),
            document_name: document_name.to_string(),
            status: status.to_string(),
            version: None,
            description: dm.description.clone(),
            logical_model: dm.visibility.clone(),
        })?;
        summary.data_models += 1;
        for f in &dm.fields {
            let attr_type = f.type_.clone().unwrap_or_else(|| "unknown".to_string());
            let internal = f
                .visibility
                .as_deref()
                .map(|v| !v.eq_ignore_ascii_case("pub"))
                .unwrap_or(false);
            repo.create_data_attribute(&DataAttribute {
                id: None,
                document_name: document_name.to_string(),
                entity: dm.name.clone(),
                name: f.name.clone(),
                attr_type,
                format: None,
                description: f.description.clone(),
                required: false,
                identity: false,
                generated: false,
                unique_flag: false,
                internal,
                deprecated: false,
                default_value: None,
                constraints_json: None,
            })?;
            summary.attributes += 1;
        }
    }

    for c in &entities.contracts {
        // `contracts.contract_type` CHECK constraint:
        //   ('interface', 'class', 'enum', 'api').
        // Map LLM's "trait" / unknown -> "interface" (closest match).
        let raw_kind = c.kind.as_deref().unwrap_or("trait");
        let contract_type = match raw_kind.to_ascii_lowercase().as_str() {
            "interface" | "trait" | "protocol" => "interface",
            "class" => "class",
            "enum" => "enum",
            "api" | "endpoint" | "route" => "api",
            _ => "interface",
        };
        let invariants_json = serde_json::to_string(&c.methods).ok();
        repo.create_contract(&Contract {
            name: c.name.clone(),
            document_name: document_name.to_string(),
            contract_type: contract_type.to_string(),
            status: "active".to_string(),
            version: None,
            is_abstract: contract_type == "interface",
            description: c.description.clone(),
            implements_json: None,
            dependencies_json: None,
            invariants_json,
            http_method: None,
            api_path: None,
            auth: None,
            rate_limit: None,
        })?;
        summary.contracts += 1;
    }

    for f in &entities.functions {
        let logic = match (&f.logic, &f.signature) {
            (Some(l), _) => Some(l.clone()),
            (None, Some(s)) => Some(s.clone()),
            (None, None) => None,
        };
        repo.create_function_spec(&FunctionSpec {
            id: None,
            document_name: document_name.to_string(),
            name: f.name.clone(),
            description: f.description.clone(),
            logic,
            complexity: None,
            pure_function: false,
        })?;
        summary.functions += 1;
    }

    for dd in &entities.design_decisions {
        let topic = dd
            .topic
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        // `design_decisions.rationale` is NOT NULL; the LLM may not
        // supply it, so fall back to the description (or empty string).
        let rationale = dd
            .rationale
            .clone()
            .or_else(|| dd.description.clone())
            .unwrap_or_default();
        // `decision` is also NOT NULL in the schema. The LLM sometimes
        // omits it, so fall back to the rationale / description / topic
        // (or the literal `"unspecified"` if everything is empty).
        let decision = dd
            .decision
            .clone()
            .or_else(|| dd.description.clone())
            .or_else(|| Some(rationale.clone()))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unspecified".to_string());
        let id = format!("dd-{}", uuid::Uuid::new_v4());
        repo.create_design_decision(&DesignDecisionRecord {
            id,
            document_name: document_name.to_string(),
            topic,
            decision,
            rationale,
            context: None,
            // Per-file decisions don't belong to a module —
            // they're observations of one file. Leave NULL so
            // `sdef_context::load_module_design_decisions` can
            // filter them out cleanly.
            module_name: None,
            alternatives_json: None,
            consequences_json: None,
        })?;
        summary.design_decisions += 1;
    }

    info!(
        document = %document_name,
        data_models = summary.data_models,
        attributes = summary.attributes,
        contracts = summary.contracts,
        functions = summary.functions,
        design_decisions = summary.design_decisions,
        "write_parsed_to_db: persisted LLM-analyzed entities"
    );
    Ok(summary)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_db::Database;

    /// A representative LLM response, modeled on the live `llm_analyze_file`
    /// example output: <think> block followed by a ```json fence.
    const FIXTURE_USER_RS: &str = r#"
<think>The user wants me to analyze a Rust file. Let me proceed.</think>
```json
{
  "data_models": [
    {
      "name": "User",
      "kind": "struct",
      "visibility": "pub",
      "description": "Plain data struct representing a user.",
      "fields": [
        {"name": "id", "type": "u64", "visibility": "pub", "description": "Unique numeric identifier."},
        {"name": "email", "type": "String", "visibility": "pub", "description": "Email."}
      ]
    }
  ],
  "contracts": [
    {
      "name": "UserStore",
      "kind": "trait",
      "description": "Storage contract.",
      "methods": [
        {"name": "get", "signature": "fn get(&self, id: u64) -> Option<User>", "description": "Lookup."}
      ]
    }
  ],
  "functions": [
    {"name": "validate_email", "signature": "fn validate_email(s: &str) -> bool", "description": "Email validation."}
  ],
  "design_decisions": [
    {"topic": "Storage", "decision": "In-memory Vec", "rationale": "Simplicity."}
  ]
}
```
"#;

    // ----- parser happy paths -----

    #[test]
    fn parse_extracts_fenced_json() {
        let entities = parse_llm_analyze_output(FIXTURE_USER_RS).expect("parse");
        assert_eq!(entities.data_models.len(), 1);
        assert_eq!(entities.data_models[0].name, "User");
        assert_eq!(entities.data_models[0].fields.len(), 2);
        assert_eq!(entities.data_models[0].fields[0].name, "id");
        assert_eq!(entities.data_models[0].fields[0].type_.as_deref(), Some("u64"));
        assert_eq!(entities.contracts.len(), 1);
        assert_eq!(entities.contracts[0].name, "UserStore");
        assert_eq!(entities.contracts[0].methods.len(), 1);
        assert_eq!(entities.functions.len(), 1);
        assert_eq!(entities.functions[0].name, "validate_email");
        assert_eq!(entities.design_decisions.len(), 1);
        assert_eq!(entities.design_decisions[0].topic.as_deref(), Some("Storage"));
        assert_eq!(
            entities.design_decisions[0].decision.as_deref(),
            Some("In-memory Vec")
        );
    }

    /// Bug 1 regression: the LLM often folds the decision into
    /// `description` and omits the `decision` field. The parser must
    /// accept that and the writer must fall back to a placeholder.
    #[test]
    fn parse_tolerates_design_decision_missing_decision_field() {
        let raw = r#"```json
{
  "data_models": [],
  "contracts": [],
  "functions": [],
  "design_decisions": [
    {"topic": "Performance", "description": "We pre-allocate the result vector."}
  ]
}
```"#;
        let entities = parse_llm_analyze_output(raw).expect("parse");
        assert_eq!(entities.design_decisions.len(), 1);
        assert!(entities.design_decisions[0].decision.is_none());
        // writer round-trip: the row must be persisted (count == 1)
        // even though the LLM never supplied a `decision` value.
        let repo = repo();
        let summary = write_parsed_to_db(&repo, "bug1-proj", &entities).expect("write");
        assert_eq!(summary.design_decisions, 1);
        // Document row should also be created so the `export` command finds it.
        let docs = repo.list_documents().expect("list documents");
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].name, "bug1-proj");
    }

    #[test]
    fn parse_tolerates_missing_entity_arrays() {
        let raw = "```json\n{}\n```";
        let entities = parse_llm_analyze_output(raw).expect("parse");
        assert!(entities.data_models.is_empty());
        assert!(entities.contracts.is_empty());
        assert!(entities.functions.is_empty());
        assert!(entities.design_decisions.is_empty());
    }

    #[test]
    fn parse_tolerates_optional_field_omission() {
        let raw = r#"```json
{
  "data_models": [
    {"name": "Foo"}
  ]
}
```"#;
        let entities = parse_llm_analyze_output(raw).expect("parse");
        assert_eq!(entities.data_models[0].name, "Foo");
        assert!(entities.data_models[0].fields.is_empty());
        assert!(entities.data_models[0].description.is_none());
        assert!(entities.data_models[0].visibility.is_none());
    }

    #[test]
    fn parse_falls_back_to_generic_fence_when_json_tag_missing() {
        // No ```json opener, but a generic ``` fence whose body starts with `{`.
        let raw = "```\n{\"data_models\": [], \"contracts\": [], \"functions\": [], \"design_decisions\": []}\n```";
        let entities = parse_llm_analyze_output(raw).expect("parse");
        assert!(entities.data_models.is_empty());
    }

    #[test]
    fn parse_skips_non_json_generic_fence() {
        // A ```rust code block must NOT be treated as JSON.
        let raw = "```rust\nfn foo() {}\n```";
        let err = parse_llm_analyze_output(raw).expect_err("should fail");
        assert!(matches!(err, ParseError::NoJsonFence));
    }

    // ----- parser error paths -----

    #[test]
    fn parse_handles_no_fence() {
        let raw = "I have no JSON to give you, only prose.";
        let err = parse_llm_analyze_output(raw).expect_err("should fail");
        assert!(matches!(err, ParseError::NoJsonFence));
    }

    #[test]
    fn parse_handles_malformed_json() {
        let raw = "```json\n{ this is not json\n```";
        let err = parse_llm_analyze_output(raw).expect_err("should fail");
        assert!(matches!(err, ParseError::JsonSyntax(_)));
    }

    #[test]
    fn parse_handles_non_object_root() {
        let raw = "```json\n[1, 2, 3]\n```";
        let err = parse_llm_analyze_output(raw).expect_err("should fail");
        assert!(matches!(err, ParseError::NotAnObject(_)));
    }

    // ----- writer round-trip against in-memory DB -----

    fn repo() -> SdefRepository {
        let db = Database::in_memory().expect("in-memory db");
        SdefRepository::new_with_arc(db.connection_arc())
    }

    #[test]
    fn write_upserts_document_and_persists_data_model_with_attributes() {
        let repo = repo();
        let entities = parse_llm_analyze_output(FIXTURE_USER_RS).expect("parse");
        let summary = write_parsed_to_db(&repo, "my-proj", &entities).expect("write");
        assert_eq!(summary.data_models, 1);
        assert_eq!(summary.attributes, 2);
        assert_eq!(summary.contracts, 1);
        assert_eq!(summary.functions, 1);
        assert_eq!(summary.design_decisions, 1);
        assert_eq!(summary.total(), 6);
        let (model, attrs) = repo
            .get_data_model("my-proj", "User")
            .expect("get data model");
        assert_eq!(model.entity, "User");
        assert_eq!(attrs.len(), 2);
        let attr_names: Vec<&str> = attrs.iter().map(|a| a.name.as_str()).collect();
        assert!(attr_names.contains(&"id"));
        assert!(attr_names.contains(&"email"));
    }

    #[test]
    fn write_creates_contract_with_method_json() {
        let repo = repo();
        let entities = parse_llm_analyze_output(FIXTURE_USER_RS).expect("parse");
        let summary = write_parsed_to_db(&repo, "my-proj", &entities).expect("write");
        assert_eq!(summary.contracts, 1);
        let contract = repo.get_contract("my-proj", "UserStore").expect("get contract");
        // LLM emits "trait" for the kind; the writer maps it to
        // "interface" because that's the closest match in the
        // `contracts.contract_type` CHECK constraint domain.
        assert_eq!(contract.contract_type, "interface");
        assert!(contract.is_abstract);
        assert!(contract.invariants_json.is_some());
        // methods JSON must contain the "get" entry.
        let parsed: serde_json::Value =
            serde_json::from_str(contract.invariants_json.as_ref().unwrap()).unwrap();
        let methods = parsed.as_array().expect("methods array");
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0]["name"], "get");
    }

    #[test]
    fn write_creates_function_spec() {
        let repo = repo();
        let entities = parse_llm_analyze_output(FIXTURE_USER_RS).expect("parse");
        write_parsed_to_db(&repo, "my-proj", &entities).expect("write");
        let funcs = repo
            .list_function_specs("my-proj")
            .expect("list function_specs");
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "validate_email");
        assert!(funcs[0].logic.is_some());
    }

    #[test]
    fn write_handles_empty_entities() {
        let repo = repo();
        let summary = write_parsed_to_db(&repo, "empty-proj", &ParsedEntities::default())
            .expect("write");
        assert!(summary.is_empty());
        // Document row should still be created so the `export` command finds it.
        let docs = repo.list_documents().expect("list documents");
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].name, "empty-proj");
    }

    #[test]
    fn write_visibility_private_marks_attribute_internal() {
        // LLM marks `items` as `visibility: "private"`; writer should map
        // that to `DataAttribute.internal = true` so downstream consumers
        // know it's not part of the public API.
        let repo = repo();
        let raw = r#"```json
{
  "data_models": [
    {
      "name": "Store",
      "kind": "struct",
      "fields": [
        {"name": "items", "type": "Vec<User>", "visibility": "private"}
      ]
    }
  ]
}
```"#;
        let entities = parse_llm_analyze_output(raw).expect("parse");
        write_parsed_to_db(&repo, "v-proj", &entities).expect("write");
        let (_model, attrs) = repo.get_data_model("v-proj", "Store").expect("get");
        assert_eq!(attrs.len(), 1);
        assert!(attrs[0].internal);
    }
}
