//! `examples/eval_llm_loop.rs`
//!
//! Phase 0.4.5 end-to-end smoke test of `llm_loop::run_loop_via_basic_agent`.
//!
//! Unlike `eval_meta.rs` (which exercises the raw `cleanroom_meta_llm`
//! `MetaProvider::chat()` call), this example goes through the full
//! `MetaBasicAgent` + `MetaAgentBuilder` + `MetaDirectAgent` path that
//! `llm_loop` uses internally. Phase 0.5 will switch Producer/Consumer over
//! to this path, so this example is the "canonical" shape of how the rest
//! of the project will eventually call the LLM.
//!
//! ## Usage
//!
//! ```bash
//! EVAL_PROVIDER=openai cargo run --example eval_llm_loop -p cleanroom-agent
//! ```
//!
//! Override any of these via env vars:
//! - `EVAL_PROVIDER`   — `openai` (default) / `anthropic` / `minimax`
//! - `EVAL_MODEL`      — `MiniMax-M3` (default)
//! - `EVAL_PROMPT`     — the user prompt (default is a simple math question)
//! - `EVAL_BASE_URL`   — override the default base URL
//! - `EVAL_SYSTEM`     — override the system prompt

use std::path::{Path, PathBuf};
use std::sync::Arc;

use cleanroom_meta_llm::backends::anthropic::AnthropicProvider;
use cleanroom_meta_llm::backends::minimax::MinimaxProvider;
use cleanroom_meta_llm::backends::openai::OpenAiProvider;
use cleanroom_meta_llm::builder::MetaBuilder;
use cleanroom_meta_llm::MetaLlm;

use cleanroom_agent::llm_loop::{
    run_loop_via_basic_agent, LoopConfig, LoopContext, LoopOutcome,
};

/// Minimal `.env` loader. See `eval_meta.rs` for the full rationale;
/// we duplicate it here rather than dragging the eval example into
/// `llm_loop`'s public API surface.
fn load_dotenv(path: &Path) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim();
        let val = v.trim().trim_matches('"').trim_matches('\'');
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, val);
        }
    }
}

fn key_preview(api_key: &str) -> String {
    if api_key.len() >= 8 {
        format!(
            "{}...{} ({} chars)",
            &api_key[..4],
            &api_key[api_key.len() - 4..],
            api_key.len()
        )
    } else {
        "<too short>".to_string()
    }
}

/// Build the `Arc<dyn MetaLlm>` the user picked via `EVAL_PROVIDER`.
///
/// This is the only place the example knows about `cleanroom_meta_llm`'s
/// `OpenAiProvider` / `AnthropicProvider` / `MinimaxProvider` types --
/// `llm_loop::run_loop_via_basic_agent` itself only sees the `MetaLlm`
/// trait.
fn build_llm(
    provider: &str,
    api_key: &str,
    model: &str,
) -> std::result::Result<Arc<dyn MetaLlm>, Box<dyn std::error::Error>> {
    match provider {
        "openai" => {
            if std::env::var_os("OPENAI_BASE_URL").is_none() {
                std::env::set_var("OPENAI_BASE_URL", "https://api.minimaxi.com/v1");
            }
            let llm: Arc<OpenAiProvider> = MetaBuilder::<OpenAiProvider>::new()
                .api_key(api_key.to_string())
                .base_url(
                    std::env::var("OPENAI_BASE_URL")
                        .unwrap_or_else(|_| "https://api.minimaxi.com/v1".to_string()),
                )
                .model(model.to_string())
                .max_tokens(256)
                .temperature(0.0)
                .build()?;
            Ok(llm)
        }
        "anthropic" => {
            if std::env::var_os("ANTHROPIC_BASE_URL").is_none() {
                std::env::set_var(
                    "ANTHROPIC_BASE_URL",
                    "https://api.minimaxi.com/anthropic",
                );
            }
            let llm: Arc<AnthropicProvider> = MetaBuilder::<AnthropicProvider>::new()
                .api_key(api_key.to_string())
                .base_url(
                    std::env::var("ANTHROPIC_BASE_URL")
                        .unwrap_or_else(|_| "https://api.minimaxi.com/anthropic".to_string()),
                )
                .model(model.to_string())
                .max_tokens(256)
                .temperature(0.0)
                .build()?;
            Ok(llm)
        }
        "minimax" => {
            let llm: Arc<MinimaxProvider> = MetaBuilder::<MinimaxProvider>::new()
                .api_key(api_key.to_string())
                .base_url(
                    std::env::var("EVAL_BASE_URL")
                        .unwrap_or_else(|_| "https://api.minimax.chat/v1/".to_string()),
                )
                .model(model.to_string())
                .max_tokens(256)
                .temperature(0.0)
                .build()?;
            Ok(llm)
        }
        other => Err(format!("unknown provider '{other}', use 'openai'/'anthropic'/'minimax'").into()),
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    tracing_subscriber::EnvFilter::new("info,cleanroom_meta=warn")
                }),
        )
        .init();

    for candidate in [".env", "../.env", "cleanroom-agent/.env"] {
        load_dotenv(&PathBuf::from(candidate));
    }

    let provider = std::env::var("EVAL_PROVIDER").unwrap_or_else(|_| "openai".to_string());
    let api_key = std::env::var("MINIMAX_API_KEY")
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .map_err(|_| "MINIMAX_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY not set")?;

    let model = std::env::var("EVAL_MODEL").unwrap_or_else(|_| "MiniMax-M3".to_string());
    let prompt = std::env::var("EVAL_PROMPT")
        .unwrap_or_else(|_| "What is 2+2? Answer in one sentence.".to_string());
    let system = std::env::var("EVAL_SYSTEM").unwrap_or_else(|_| {
        "You are a helpful assistant. Reply concisely.".to_string()
    });

    println!("== eval_llm_loop (via run_loop_via_basic_agent) ==");
    println!("provider: {provider}");
    println!("model:    {model}");
    println!("api_key:  {}", key_preview(&api_key));
    println!("system:   {system}");
    println!("prompt:   {prompt}");
    println!();

    let llm = build_llm(&provider, &api_key, &model)?;
    let ctx = LoopContext::new(
        "eval-llm-loop-1",
        "eval-session-1",
        "eval-llm-loop",
        system,
        prompt,
    );
    let cfg = LoopConfig {
        max_iterations: 4,
        max_tokens_per_call: 256,
        temperature: 0.0,
        tool_timeout_secs: 30,
        cost_limit_usd: Some(0.05),
    };

    println!("== run_loop_via_basic_agent start ==");
    let started = std::time::Instant::now();
    let outcome = run_loop_via_basic_agent(llm, ctx, &cfg).await?;
    let elapsed = started.elapsed();

    println!();
    println!("== result (elapsed: {}ms) ==", elapsed.as_millis());
    match &outcome {
        LoopOutcome::Done {
            result,
            iterations,
            prompt_tokens,
            completion_tokens,
        } => {
            println!("status:        Done");
            println!("iterations:    {iterations}");
            println!(
                "tokens:        {prompt_tokens} prompt + {completion_tokens} completion = {}",
                prompt_tokens + completion_tokens
            );
            println!("result:");
            println!("{result}");
        }
        LoopOutcome::MaxIter {
            iterations,
            last_text,
        } => {
            println!("status:        MaxIter");
            println!("iterations:    {iterations}");
            println!("last_text:     {last_text}");
        }
        LoopOutcome::Aborted { reason, iterations } => {
            println!("status:        Aborted");
            println!("iterations:    {iterations}");
            println!("reason:        {reason}");
        }
        LoopOutcome::LlmRefused { reason, iterations } => {
            println!("status:        LlmRefused");
            println!("iterations:    {iterations}");
            println!("reason:        {reason}");
        }
    }

    Ok(())
}
