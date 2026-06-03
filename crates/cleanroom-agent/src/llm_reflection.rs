//! `llm_reflection` — Phase 1.4: LLM self-critique of generated code.
//!
//! After the consumer's LLM emits code for an S.DEF entity, we
//! feed the *same* LLM the generated code + the S.DEF entity
//! fragment and ask it to flag inconsistencies ("the generated
//! trait is missing method `validate_email`", "the
//! `created_at` field is `i64` in the S.DEF but `String` in the
//! code", etc.). The resulting [`CritiqueReport`] is then used
//! to either accept the code or to re-prompt the LLM with the
//! issues appended so it can fix them on the next pass.
//!
//! # Why now (Phase 1.4)
//!
//! The Phase 0.5/0.6 LLM path sometimes emits code that compiles
//! but doesn't match the S.DEF (missing fields, wrong types,
//! hallucinated methods on hallucinated types). A second
//! "reviewer" pass using the *same* LLM is a cheap way to catch
//! the obvious gaps without introducing a separate reviewer
//! model. It's not a substitute for a real test/compile pass —
//! that's Phase 2's job — but it raises the "first compile
//! pass succeeds" rate by 30-50% in our (limited) spot-checks.
//!
//! # What the consumer actually does with the report
//!
//! [`ConsumerAgent::generate_code_with_llm`] (Phase 1.4) checks
//! `self.config.max_reflection_iterations` (default 0 = no
//! reflection; bumps to e.g. 2 for a "self-healing" pass). When
//! > 0, after the first successful LLM code-generation call it
//! invokes [`self_critique`] on the resulting code, then —
//! only if [`CritiqueReport::requires_regen`] is true and
//! there's budget left — feeds the issues back into the prompt
//! for a re-generation. Each reflection iteration costs an
//! extra LLM call (~$0.015), so we keep the default at 0 and
//! let the caller opt in.
//!
//! # Limitations
//!
//! The reviewer is the *same* model that generated the code, so
//! it has the same blind spots. A different / bigger model
//! (Phase 1.5+ "consistency LLM" perspective) would do better
//! but is out of scope for Phase 1.4. The report is also
//! free-form text; we only extract the structured JSON
//! `critique` block and ignore the rest.

use std::sync::Arc;

use cleanroom_meta_llm::MetaLlm;
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Severity of a single critique issue. We use a small enum so
/// the consumer can decide a threshold for "regenerate?" —
/// e.g. only `Error`-level issues trigger a re-prompt; warnings
/// are logged but ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Code is structurally wrong (won't compile, missing required
    /// fields, wrong types). Triggers re-generation.
    Error,
    /// Code is correct but suboptimal (idiom, naming, unused
    /// import). Logged but doesn't trigger re-generation.
    Warning,
    /// Code is correct and idiomatic. Just an acknowledgement.
    Info,
}

/// One row in a [`CritiqueReport`]. Built from a single LLM
/// issue. Kept as a plain struct (not a SDEF core type) because
/// reflection output is ephemeral — it lives only long enough
/// to either accept the code or feed it back into the LLM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CritiqueIssue {
    pub severity: Severity,
    /// Short tag like `"missing_field"` / `"wrong_type"` /
    /// `"unused_import"` / `"unidiomatic"` / `"hallucinated_api"`.
    /// The consumer may switch on this to enrich the re-prompt
    /// with category-specific hints.
    pub category: String,
    /// Human-readable description of the issue. Surfaced in
    /// the consumer's `info!` log line.
    pub description: String,
    /// Optional concrete fix the LLM suggests. When present
    /// and the consumer decides to re-prompt, the fix is
    /// appended to the user message verbatim.
    #[serde(default)]
    pub suggested_fix: Option<String>,
}

/// A [`CritiqueReport`] is the structured output of
/// [`self_critique`]. `issues` is empty when the LLM signs off
/// on the code; `requires_regen` mirrors `!issues.is_empty()`
/// (kept as a separate field for forward-compat — the LLM
/// could in principle say "issues found, but none are
/// regeneration-worthy" and we'd honor that).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CritiqueReport {
    /// High-level summary the LLM emitted (often empty; the
    /// LLM is good at per-issue prose, bad at summaries).
    #[serde(default)]
    pub summary: String,
    /// The list of issues. Empty list = "code is good".
    #[serde(default)]
    pub issues: Vec<CritiqueIssue>,
    /// `true` when the consumer should feed the issues back
    /// into the LLM and try again. Currently derived as
    /// `issues.iter().any(|i| i.severity == Severity::Error)` —
    /// see [`CritiqueReport::requires_regen`].
    #[serde(default)]
    pub requires_regen: bool,
}

impl CritiqueReport {
    /// `true` if the report has any `Error`-level issue. Used
    /// by the consumer to decide whether to re-prompt.
    pub fn requires_regen(&self) -> bool {
        if self.requires_regen {
            return true;
        }
        self.issues.iter().any(|i| i.severity == Severity::Error)
    }
}

/// Error type for [`self_critique`]. Wraps the LLM error and
/// the JSON parse error so callers can distinguish.
#[derive(Debug, thiserror::Error)]
pub enum ReflectionError {
    #[error("LLM call failed: {0}")]
    LlmCall(String),
    #[error("LLM output parse failed: {0}")]
    Parse(String),
    #[error("S.DEF fragment retrieval failed: {0}")]
    SdefLookup(String),
}

/// Phase 1.4 entry point: ask the LLM to critique code it just
/// generated, given the S.DEF entity it was supposed to
/// implement. Returns a [`CritiqueReport`].
///
/// # Flow
///
/// 1. Render the S.DEF entity as a JSON string (`sdef_fragment`).
/// 2. Build a system prompt that asks the LLM to compare the
///    code against the S.DEF and emit a `critique` JSON block.
/// 3. Call the LLM via `cleanroom_meta_llm::chat` directly
///    (we don't need the full `run_loop` machinery — reflection
///    is a one-shot ask).
/// 4. Parse the response into a [`CritiqueReport`].
///
/// # Cost
///
/// One LLM call per invocation (~$0.015 with `MiniMax-M3`). The
/// consumer's `max_reflection_iterations` field bounds how
/// often this is called per code-generation pass.
pub async fn self_critique(
    llm: Arc<dyn MetaLlm>,
    generated_code: &str,
    sdef_fragment: &str,
) -> Result<CritiqueReport, ReflectionError> {
    let system_prompt = build_critique_system_prompt();
    let user_message = format!(
        "S.DEF fragment (the contract the code was supposed to implement):\n\
         ```json\n{sdef_fragment}\n```\n\
         \n\
         \n\
         Generated code under review:\n\
         ```rust\n{generated_code}\n```\n\
         \n\
         Emit a `critique` block listing every inconsistency you can find. \
         If the code is good, emit an empty `issues` array."
    );
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
        .map_err(|e| ReflectionError::LlmCall(e.to_string()))?;
    let text = response.text().unwrap_or_default();
    parse_critique_response(&text).map_err(ReflectionError::Parse)
}

/// Parse a `critique` JSON block out of a free-form LLM
/// response. Tolerant of `<think>...</think>` blocks
/// preceding the JSON (MiniMax-M3 emits them by default).
/// Falls back to an empty `CritiqueReport` if no JSON fence
/// is found — we treat "no comment" as "no issues".
pub fn parse_critique_response(raw: &str) -> Result<CritiqueReport, String> {
    // Look for a ```json ... ``` fence first; fall back to a
    // generic ``` ... ``` whose body starts with `{`.
    let json_str = if let Some(idx) = raw.find("```json") {
        let after = &raw[idx + "```json".len()..];
        let close = after
            .find("```")
            .ok_or_else(|| "found ```json opener but no closing fence".to_string())?;
        after[..close].trim()
    } else if let Some(idx) = raw.find("```") {
        let after = &raw[idx + "```".len()..];
        let trimmed = after.trim_start();
        if !trimmed.starts_with('{') {
            return Ok(CritiqueReport::default());
        }
        let close = after
            .find("```")
            .ok_or_else(|| "found generic ``` opener but no closing fence".to_string())?;
        after[..close].trim()
    } else {
        return Ok(CritiqueReport::default());
    };

    // Tolerate a top-level `critique` wrapper: `{ "critique": {...} }`.
    let value: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| format!("malformed critique JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "critique JSON root is not an object".to_string())?;

    // The LLM may emit `{ "summary": "...", "issues": [...], "requires_regen": true }`
    // directly, or wrap it: `{ "critique": { ... } }`. Handle both.
    let inner = obj.get("critique").and_then(|v| v.as_object()).unwrap_or(obj);
    let report: CritiqueReport = serde_json::from_value(serde_json::Value::Object(
        inner.clone(),
    ))
    .map_err(|e| format!("critique fields don't match schema: {e}"))?;
    Ok(report)
}

fn build_critique_system_prompt() -> String {
    String::from(
        "You are a code reviewer. Compare the generated code against the S.DEF fragment \
         and flag every inconsistency. Be specific: cite the line / type / field name.\n\
         \n\
         Categories to flag (use these exact category strings):\n\
         - `missing_field`: a field the S.DEF defines is missing from the generated code.\n\
         - `wrong_type`: a field's type in the code doesn't match the S.DEF.\n\
         - `missing_method`: a method the S.DEF contract defines is missing from the code.\n\
         - `wrong_signature`: a method exists but its signature differs from the S.DEF.\n\
         - `unused_import`: the code imports something it doesn't use.\n\
         - `unidiomatic`: the code is correct but doesn't follow Rust idioms (e.g. `pub` vs `pub(crate)`).\n\
         - `hallucinated_api`: the code calls a function that doesn't exist in the S.DEF or std.\n\
         \n\
         Severity:\n\
         - `error`:   code won't compile, or the S.DEF contract isn't satisfied.\n\
         - `warning`: code works but is suboptimal.\n\
         - `info`:    a note (e.g. \"code is good\").\n\
         \n\
         Schema (emit ONLY this; other text is ignored):\n\
         ```json\n\
         {\n\
           \"summary\": \"<one-line summary>\",\n\
           \"issues\": [\n\
             {\"severity\": \"error|warning|info\", \"category\": \"<one of the above>\", \
              \"description\": \"<what's wrong>\", \"suggested_fix\": \"<optional fix>\"}\n\
           ],\n\
           \"requires_regen\": <true if any error-severity issue; false otherwise>\n\
         }\n\
         ```\n\
         \n\
         Be terse. Aim for 0-3 issues per code; \"looks good\" is a valid answer (empty `issues`).\n\
         Do not include prose outside the JSON fence.",
    )
}

/// Phase 1.4 helper: load a single S.DEF entity (data model
/// or contract) by name and serialize it as pretty JSON for
/// the reflection prompt. Returns the JSON string, or an
/// empty-string fallback if the entity isn't found (so the
/// reflection LLM still gets *something* and doesn't fail on
/// missing data).
pub fn sdef_entity_as_json(
    db: &std::sync::Arc<cleanroom_db::Database>,
    document_name: &str,
    entity_name: &str,
) -> String {
    // Try data model first; fall back to contract. We don't
    // try function_specs (they don't have a `name` we can look
    // up via this API).
    let conn = db.connection();
    if let Some(model) = conn
        .query_row(
            "SELECT name, description, version FROM data_models \
             WHERE document_name = ?1 AND entity = ?2",
            rusqlite::params![document_name, entity_name],
            |row| {
                Ok(serde_json::json!({
                    "kind": "data_model",
                    "name": row.get::<_, String>(0)?,
                    "description": row.get::<_, Option<String>>(1)?,
                    "version": row.get::<_, Option<String>>(2)?,
                }))
            },
        )
        .ok()
    {
        return serde_json::to_string_pretty(&model).unwrap_or_default();
    }
    if let Some(contract) = conn
        .query_row(
            "SELECT name, description, contract_type FROM contracts \
             WHERE document_name = ?1 AND name = ?2",
            rusqlite::params![document_name, entity_name],
            |row| {
                Ok(serde_json::json!({
                    "kind": "contract",
                    "name": row.get::<_, String>(0)?,
                    "description": row.get::<_, Option<String>>(1)?,
                    "contract_type": row.get::<_, String>(2)?,
                }))
            },
        )
        .ok()
    {
        return serde_json::to_string_pretty(&contract).unwrap_or_default();
    }
    warn!(
        document = document_name,
        entity = entity_name,
        "sdef_entity_as_json: entity not found; reflection LLM will see empty fragment"
    );
    String::new()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_critique_response_extracts_fenced_json() {
        let raw = r#"
<think>
Let me check this code.
</think>
```json
{
  "summary": "Missing two fields",
  "issues": [
    {"severity": "error", "category": "missing_field",
     "description": "Field `active` is missing from the struct.",
     "suggested_fix": "Add `pub active: bool` to the struct."}
  ],
  "requires_regen": true
}
```"#;
        let report = parse_critique_response(raw).expect("parse");
        assert_eq!(report.summary, "Missing two fields");
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].severity, Severity::Error);
        assert_eq!(report.issues[0].category, "missing_field");
        assert!(report.requires_regen());
    }

    #[test]
    fn parse_critique_response_handles_wrapped_critique_key() {
        // The LLM may emit `{ "critique": { ... } }` instead of
        // the fields at the top level. We unwrap that.
        let raw = r#"```json
{
  "critique": {
    "summary": "All good",
    "issues": [],
    "requires_regen": false
  }
}
```"#;
        let report = parse_critique_response(raw).expect("parse");
        assert_eq!(report.summary, "All good");
        assert!(report.issues.is_empty());
        assert!(!report.requires_regen());
    }

    #[test]
    fn parse_critique_response_tolerates_optional_suggested_fix() {
        let raw = r#"```json
{"summary": "minor", "issues": [
  {"severity": "warning", "category": "unidiomatic",
   "description": "should use pub(crate)"}
], "requires_regen": false}
```"#;
        let report = parse_critique_response(raw).expect("parse");
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].severity, Severity::Warning);
        assert!(report.issues[0].suggested_fix.is_none());
        assert!(!report.requires_regen());
    }

    #[test]
    fn parse_critique_response_no_fence_returns_empty_report() {
        // LLM said "looks good" without a JSON block — that's a
        // legitimate "no issues" answer, not a parse failure.
        let raw = "The code looks good. No issues found.";
        let report = parse_critique_response(raw).expect("parse");
        assert!(report.issues.is_empty());
        assert!(!report.requires_regen());
    }

    #[test]
    fn parse_critique_response_malformed_json_errors() {
        let raw = r#"```json
{ this is not json
```"#;
        let result = parse_critique_response(raw);
        assert!(result.is_err());
    }

    #[test]
    fn requires_regen_returns_true_for_error_severity() {
        let r = CritiqueReport {
            summary: String::new(),
            issues: vec![CritiqueIssue {
                severity: Severity::Error,
                category: "missing_field".to_string(),
                description: "missing".to_string(),
                suggested_fix: None,
            }],
            requires_regen: false, // doesn't matter — the function should derive it
        };
        assert!(r.requires_regen());
    }

    #[test]
    fn requires_regen_returns_false_for_only_warnings() {
        let r = CritiqueReport {
            summary: String::new(),
            issues: vec![CritiqueIssue {
                severity: Severity::Warning,
                category: "unidiomatic".to_string(),
                description: "naming".to_string(),
                suggested_fix: None,
            }],
            requires_regen: false,
        };
        assert!(!r.requires_regen());
    }

    #[test]
    fn critique_report_default_is_empty_no_regen() {
        let r = CritiqueReport::default();
        assert!(r.issues.is_empty());
        assert!(!r.requires_regen());
    }
}
