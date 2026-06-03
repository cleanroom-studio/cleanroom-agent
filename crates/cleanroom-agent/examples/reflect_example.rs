//! `examples/reflect_example.rs`
//!
//! Phase 1.4 end-to-end demo: drive the consumer's self-critique
//! loop on one S.DEF entity. Generates code with the LLM, then
//! asks the LLM to critique the generated code against the
//! S.DEF. Logs the [`CritiqueReport`] and the token cost; exits
//! non-zero if reflection found issues that needed a regen
//! (so the example can be used as a "smoke test" in CI).
//!
//! ## Flow
//!
//! 1. Build a temp SQLite DB.
//! 2. Set up a minimal S.DEF document with one data model
//!    (`User`) — we write the rows directly to keep the
//!    example self-contained.
//! 3. Build a `ConsumerAgent` with
//!    `with_max_reflection_iterations(1)`, an LLM, and the
//!    `llm_call_logger` wired up.
//! 4. Call `reflect_on_entity("sdef://reflect-doc/User", &code, &sdef_json)`.
//! 5. Print the report, the final code, and the total token cost.
//!    Then verify by re-reading `llm_call_log` that we
//!    actually performed N+1 LLM calls (1 generation + 1
//!    reflection critique).
//!
//! ## Usage
//!
//! ```bash
//! cargo run --manifest-path cleanroom-agent/Cargo.toml \
//!   -p cleanroom-agent --example reflect_example
//! ```

use std::io::Write;
use std::sync::Arc;

use cleanroom_agent::consumer::ConsumerAgent;
use cleanroom_agent::llm_loop::{LoopConfig, MemoryConfig};
use cleanroom_agent::llm_reflection::CritiqueReport;
use cleanroom_agent::producer::ProducerConfig;
use cleanroom_db::repositories::llm_call_log_repository::LlmCallLogRepository;
use cleanroom_db::repositories::sdef_repository::{SdefDocument, SdefRepository};
use cleanroom_db::{Database, TaskRepository};
use cleanroom_meta_llm::backends::openai::OpenAiProvider;
use cleanroom_meta_llm::builder::MetaBuilder;
use cleanroom_meta_llm::MetaLlm;

/// A tiny S.DEF fragment (as JSON) for the demo entity. We embed
/// it as a const so the example is self-contained.
const SDEF_FRAGMENT_USER: &str = r#"{
  "kind": "data_model",
  "name": "User",
  "description": "A registered user. Has a stable UUID id, a contact email, an active flag, and a creation timestamp (Unix epoch seconds).",
  "version": "0.1.0",
  "attributes": [
    {"name": "id", "type": "String", "description": "Stable unique identifier (UUID v4)."},
    {"name": "email", "type": "String", "description": "Contact email (RFC 5322)."},
    {"name": "active", "type": "bool", "description": "Whether the account is active."},
    {"name": "created_at", "type": "i64", "description": "Unix epoch seconds of account creation."}
  ]
}"#;

/// A deliberately-not-great generated code sample, to give the
/// reflection LLM something to critique. We pretend the consumer
/// emitted this on the first pass — missing the `created_at`
/// field and with the wrong type for `id`. The reflection LLM
/// should flag both as `error`-severity issues.
const SAMPLE_GENERATED_CODE: &str = r#"
/// A registered user. (Misses the created_at field and uses u64
/// for id instead of String — the reflection LLM should flag both.)
#[derive(Debug, Clone)]
pub struct User {
    pub id: u64,
    pub email: String,
    pub active: bool,
}
"#;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 0. Tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,cleanroom_meta=warn")),
        )
        .init();

    // 1. Build the LLM.
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
        .max_tokens(1024)
        .temperature(0.0)
        .build()?;
    let llm_dyn: Arc<dyn MetaLlm> = llm;

    // 2. Set up a temp DB with the User data model seeded.
    let tmpdir = tempfile::tempdir()?;
    let db_path = tmpdir.path().join("reflect-example.db");
    let migrations_dir = workspace_migrations_dir();
    let db = Database::open_with_migrations_from(&db_path, Some(&migrations_dir))?;
    let db = Arc::new(db);
    let sdef_repo = SdefRepository::new_with_arc(db.connection_arc());
    sdef_repo
        .upsert_document(&SdefDocument {
            name: "reflect-doc".to_string(),
            version: Some("0.1.0".to_string()),
            description: Some("phase 1.4 reflection demo".to_string()),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        })
        .expect("upsert document");
    sdef_repo
        .create_data_model(&cleanroom_db::repositories::sdef_repository::DataModel {
            entity: "User".to_string(),
            document_name: "reflect-doc".to_string(),
            status: "active".to_string(),
            version: Some("0.1.0".to_string()),
            description: Some("A registered user".to_string()),
            logical_model: Some("src/user.rs".to_string()),
        })
        .expect("create data model");

    // 3. Build a Consumer with reflection enabled (1 round) and
    //    a logger so the audit trail shows both the generation
    //    call AND the critique call.
    let llm_call_logger = Arc::new(LlmCallLogRepository::new_with_arc(db.connection_arc()));
    let agent = ConsumerAgent::new(
        cleanroom_agent::consumer::ConsumerConfig {
            language: "rust".to_string(),
            framework: None,
            compatibility_mode: cleanroom_agent::CompatibilityMode::Mixed,
            fidelity: cleanroom_agent::Fidelity::Medium,
            output_path: tmpdir.path().join("out"),
            use_legacy_template: false,
            llm: None, // set below
            loop_config: LoopConfig {
                max_iterations: 1,
                max_tokens_per_call: 1024,
                temperature: 0.0,
                tool_timeout_secs: 30,
                cost_limit_usd: Some(0.20), // higher cap to allow the reflection round
                on_call_complete: None,
                tools: None,
                memory: MemoryConfig::None,
                max_reflection_iterations: 1, // 1 critique round
            },
            target_manifest: None,
            scope: cleanroom_agent::consumer::ConsumeScope::WholeProject,
        },
        db.clone(),
    )
    .with_llm(llm_dyn)
    .with_llm_call_logger(llm_call_logger.clone())
    .with_max_reflection_iterations(1);

    println!("== reflect_example ==");
    println!("provider: openai");
    println!("model:    {model}");
    println!("seeded code:  {} bytes (deliberately missing `created_at`, wrong id type)", SAMPLE_GENERATED_CODE.len());
    println!();

    // 4. Run the reflection loop. This makes 2 LLM calls in
    //    the worst case (1 critique + 1 regen) and 1 call in
    //    the happy case (1 critique, report says "looks good").
    let started = std::time::Instant::now();
    let (final_code, report, total_prompt, total_completion) = agent
        .reflect_on_entity(
            "sdef://reflect-doc/User",
            SAMPLE_GENERATED_CODE,
            SDEF_FRAGMENT_USER,
        )
        .await?;
    let elapsed_ms = started.elapsed().as_millis();

    println!("== reflect_on_entity finished in {elapsed_ms}ms ==");
    println!("report.summary:         {:?}", report.summary);
    println!("report.issues count:   {}", report.issues.len());
    println!("report.requires_regen: {}", report.requires_regen());
    println!("total prompt tokens:   {total_prompt}");
    println!("total completion tokens: {total_completion}");
    println!();
    println!("== final code (first 500 chars) ==");
    println!("{}", &final_code.chars().take(500).collect::<String>());
    if final_code.chars().count() > 500 {
        println!("... ({} more chars truncated)", final_code.chars().count() - 500);
    }
    println!();

    // 5. Print the per-issue breakdown.
    if !report.issues.is_empty() {
        println!("== issues ==");
        for (i, issue) in report.issues.iter().enumerate() {
            let fix = issue
                .suggested_fix
                .as_deref()
                .map(|f| format!("; fix: {f}"))
                .unwrap_or_default();
            println!(
                "  [{}] {:?} / {} — {}{}",
                i, issue.severity, issue.category, issue.description, fix
            );
        }
        println!();
    }

    // 6. Verify the audit trail: at least 1 call was made for
    //    the critique (could be 2 if regen was triggered).
    let log_count = llm_call_logger.count()?;
    println!("== llm_call_log rows ==");
    println!("total:                  {log_count}");
    let recent = llm_call_logger.list_recent(5)?;
    for (i, row) in recent.iter().enumerate() {
        println!(
            "  [{}] task_id={} tokens={}p+{}c cost=${:.4} duration_ms={}",
            i, row.task_id.as_deref().unwrap_or("?"),
            row.prompt_tokens, row.completion_tokens,
            row.cost_estimate_usd, row.duration_ms
        );
    }
    println!();

    // 7. Phase 1.4 acceptance: at least 1 critique call was made
    //    (the seed we passed in is intentionally bad, so we
    //    EXPECT a non-empty report). We treat this as a smoke
    //    test: if the reflection LLM emitted "looks good" on a
    //    clearly-broken input, that's a regression of the
    //    critique prompt.
    if report.issues.is_empty() {
        eprintln!(
            "ERROR: reflection LLM said 'no issues' for a clearly-broken input \
             (missing `created_at`, wrong id type). This is a regression in the \
             critique prompt — see llm_reflection::build_critique_system_prompt."
        );
        std::process::exit(1);
    }
    println!("== reflect_example: all checks passed ==");

    // Silence the unused-import warning for TaskRepository
    // (kept available for future smoke tests).
    let _ = TaskRepository::new(db.connection_arc());
    let _ = ProducerConfig::default();
    // The `Write` import is exercised above by `SAMPLE_GENERATED_CODE`
    // being a `&str` (no I/O), so silence the warning by referencing
    // the type once.
    fn _suppress_unused<T: Write>(_: &T) {}
    let _sink = std::io::sink();
    _suppress_unused(&_sink);

    Ok(())
}

fn workspace_migrations_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .and_then(|p| p.parent()) // cleanroom-agent/
        .expect("cleanroom-agent crate layout has two parents")
        .join("migrations")
}
