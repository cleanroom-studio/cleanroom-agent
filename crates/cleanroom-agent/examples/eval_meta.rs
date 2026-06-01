//! `examples/eval_meta.rs`
//!
//! Phase 0 evaluation: real-world smoke test of the vendored `cleanroom-meta-*`
//! crates (the in-tree fork of autoagents 0.3.7) against MiniMax-M3 (both its
//! Anthropic-compatible and OpenAI-compatible endpoints).
//!
//! `cleanroom-meta-llm` ships both a dedicated `MinimaxProvider` provider and
//! an `AnthropicProvider` provider; we exercise three paths here:
//! 1. `MinimaxProvider` pointed at the legacy `api.minimax.chat/v1/` host.
//! 2. `AnthropicProvider` with `base_url` overridden to the new
//!    `api.minimaxi.com/anthropic` host.
//! 3. `OpenAiProvider` with `base_url` overridden to the new
//!    `api.minimaxi.com/v1/` host (OpenAI-compatible path).
//!
//! ## Usage
//!
//! ```bash
//! cargo run --manifest-path cleanroom-agent/Cargo.toml \
//!   -p cleanroom-agent --example eval_meta
//! ```
//!
//! Provider selection is driven by `EVAL_PROVIDER`:
//! - `EVAL_PROVIDER=openai`    (default; new host, OpenAI-compatible)
//! - `EVAL_PROVIDER=anthropic` (new host, Anthropic-compatible)
//! - `EVAL_PROVIDER=minimax`   (legacy host, MiniMax-native)

use std::path::{Path, PathBuf};
use std::sync::Arc;

use cleanroom_meta_llm::backends::anthropic::AnthropicProvider;
use cleanroom_meta_llm::backends::minimax::MinimaxProvider;
use cleanroom_meta_llm::backends::openai::OpenAiProvider;
use cleanroom_meta_llm::builder::MetaBuilder;
use cleanroom_meta_llm::chat::{MetaMessage, MetaProvider};

/// Minimal .env loader: parses `KEY=VALUE` lines, stripping surrounding
/// double or single quotes from the value. Sets the env var only when the
/// process hasn't already defined one. Empty lines and `#` comments are
/// skipped silently. This intentionally avoids pulling in the `dotenvy`
/// crate just for the eval example.
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
            // Silently set; never log the value to avoid leaking the key.
            std::env::set_var(key, val);
        }
    }
}

/// Compact, non-revealing preview of an API key (`abcd...wxyz (NN chars)`),
/// safe to print to stdout without exposing the full secret.
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
        .map_err(|_| "MINIMAX_API_KEY or ANTHROPIC_API_KEY not set")?;

    let model = std::env::var("EVAL_MODEL").unwrap_or_else(|_| "MiniMax-M3".to_string());
    let prompt = std::env::var("EVAL_PROMPT")
        .unwrap_or_else(|_| "What is 2+2? Answer in one sentence.".to_string());

    println!("== eval_meta ==");
    println!("provider: {provider}");
    println!("model:    {model}");
    println!("api_key:  {}", key_preview(&api_key));
    println!("prompt:   {prompt}");
    println!();

    let started = std::time::Instant::now();
    let messages = vec![MetaMessage::user().content(prompt.clone()).build()];

    match provider.as_str() {
        "anthropic" => {
            let base_url = std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.minimaxi.com/anthropic".to_string());
            println!("base_url: {base_url}");

            let llm: Arc<AnthropicProvider> = MetaBuilder::<AnthropicProvider>::new()
                .api_key(api_key)
                .base_url(base_url)
                .model(model.clone())
                .max_tokens(256)
                .temperature(0.0)
                .build()
                .map_err(|e| format!("Failed to build AnthropicProvider LLM: {e:?}"))?;

            println!("== calling cleanroom_meta_llm (AnthropicProvider provider) ==");
            let resp = llm
                .chat(&messages, None)
                .await
                .map_err(|e| format!("chat failed: {e:?}"))?;
            println!();
            println!("== result ==");
            println!("elapsed:  {}ms", started.elapsed().as_millis());
            println!("response: {resp:?}");
        }
        "minimax" => {
            let base_url = std::env::var("EVAL_BASE_URL")
                .unwrap_or_else(|_| "https://api.minimax.chat/v1/".to_string());
            println!("base_url: {base_url}");

            let llm: Arc<MinimaxProvider> = MetaBuilder::<MinimaxProvider>::new()
                .api_key(api_key)
                .base_url(base_url)
                .model(model.clone())
                .max_tokens(256)
                .temperature(0.0)
                .build()
                .map_err(|e| format!("Failed to build MinimaxProvider LLM: {e:?}"))?;

            println!("== calling cleanroom_meta_llm (MinimaxProvider provider) ==");
            let resp = llm
                .chat(&messages, None)
                .await
                .map_err(|e| format!("chat failed: {e:?}"))?;
            println!();
            println!("== result ==");
            println!("elapsed:  {}ms", started.elapsed().as_millis());
            println!("response: {resp:?}");
        }
        "openai" => {
            // The `OpenAiProvider` provider is a typed
            // `OpenAICompatibleProvider<OpenAIInternalCfg>` under the hood; we
            // override its hard-coded `https://api.openai.com/v1/` default
            // via `MetaBuilder::base_url` so it can reach MiniMax instead.
            let base_url = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.minimaxi.com/v1/".to_string());
            println!("base_url: {base_url}");

            let llm: Arc<OpenAiProvider> = MetaBuilder::<OpenAiProvider>::new()
                .api_key(api_key)
                .base_url(base_url)
                .model(model.clone())
                .max_tokens(256)
                .temperature(0.0)
                .build()
                .map_err(|e| format!("Failed to build OpenAiProvider LLM: {e:?}"))?;

            println!("== calling cleanroom_meta_llm (OpenAiProvider provider) ==");
            let resp = llm
                .chat(&messages, None)
                .await
                .map_err(|e| format!("chat failed: {e:?}"))?;
            println!();
            println!("== result ==");
            println!("elapsed:  {}ms", started.elapsed().as_millis());
            println!("response: {resp:?}");
        }
        other => {
            return Err(format!(
                "unknown provider '{other}', use 'anthropic', 'minimax', or 'openai'"
            )
            .into());
        }
    }

    Ok(())
}
