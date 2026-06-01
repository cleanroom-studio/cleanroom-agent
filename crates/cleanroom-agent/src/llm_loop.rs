//! `llm_loop` — generic LLM agent loop built on top of `autoagents`'s `ChatProvider`.
//!
//! # Motivation
//!
//! We originally considered `zavora-ai/adk-rust` 0.9, but its `ThinkingBlock.signature`
//! field is missing `#[serde(default)]`, which broke deserialization against
//! MiniMax-M3's streaming responses. In Phase 0 we evaluated three candidates
//! (`autoagents` / `rig` / `anda`) and ultimately picked **autoagents 0.3.7** —
//! the only one of the three that can talk to MiniMax-M3 (over both its
//! Anthropic-compatible and OpenAI-compatible endpoints) end-to-end.
//!
//! # Implementation strategy
//!
//! `autoagents`'s [`ChatProvider`] trait exposes a single-shot
//! `chat(messages, schema)` call. Multi-turn ReAct + tool-calling is provided by
//! autoagents's [`BasicAgent`](autoagents_core::agent::prebuilt::executor::BasicAgent)
//! and will be adopted in Phase 0.5 once we switch the main Producer/Consumer
//! path to it.
//!
//! `run_loop` here wraps the single-shot `chat()` call and exposes a
//! task-friendly surface:
//! - `LoopConfig` — tuning + cost guardrail
//!   (`max_iterations` / `max_tokens` / `temperature` / `cost_limit`)
//! - `LoopContext` — task input
//!   (system prompt + initial user message)
//! - `LoopOutcome` — flat result enum
//!   (`Done` / `MaxIter` / `Aborted` / `LlmRefused`)
//! - `LoopStats` — token usage / tool calls / elapsed time / cost estimate
//!
//! # Usage
//!
//! ```ignore
//! use cleanroom_agent::llm_loop::{run_loop, LoopConfig, LoopContext};
//! use autoagents::llm::backends::openai::OpenAI;
//! use autoagents::llm::builder::LLMBuilder;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
//!     .api_key(std::env::var("OPENAI_API_KEY")?)
//!     .base_url("https://api.minimaxi.com/v1".to_string())
//!     .model("MiniMax-M3".to_string())
//!     .max_tokens(256)
//!     .temperature(0.2)
//!     .build()?;
//!
//! let ctx = LoopContext::new(
//!     "task-1", "sess-1", "cleanroom-producer",
//!     "You are a code analysis agent.",
//!     "Analyze src/main.rs".to_string(),
//! );
//! let cfg = LoopConfig::default();
//! let outcome = run_loop(llm, ctx, &cfg).await?;
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use autoagents::llm::chat::{
    ChatMessage, ChatMessageBuilder, ChatProvider, ChatRole,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

// ============================================================================
// Public types
// ============================================================================

/// Tuning parameters + cost guardrail for one `run_loop` invocation.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Maximum number of LLM round-trips. Reserved in the Phase 0.1 single-shot
    /// chat implementation; will be used once Phase 0.5 switches to
    /// `BasicAgent` / `AgentBuilder`.
    pub max_iterations: u32,
    /// `max_tokens` for a single LLM call. `autoagents::llm::builder::LLMBuilder`
    /// already takes it at build time; this field is kept for stats / cost
    /// estimation and to allow post-validation of the response length.
    pub max_tokens_per_call: u32,
    /// Sampling temperature (`0.0` = deterministic, `1.0` = creative).
    /// `LLMBuilder` consumes it at build time.
    pub temperature: f32,
    /// Per-tool-call timeout. No-op in Phase 0.1 (single-shot); will take
    /// effect once Phase 0.5 wires `BasicAgent` / tool-call execution.
    pub tool_timeout_secs: u64,
    /// Hard cap on total cost in USD. When the estimate exceeds this value,
    /// `run_loop` short-circuits with `LoopOutcome::Aborted`. `None` disables
    /// the cap.
    pub cost_limit_usd: Option<f64>,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 16,
            max_tokens_per_call: 4096,
            temperature: 0.2,
            tool_timeout_secs: 60,
            cost_limit_usd: None,
        }
    }
}

/// Input context for one `run_loop` invocation (one task / one session).
///
/// Carries the system prompt, the initial user message, and an internal
/// stats handle for callers that want to observe in-flight progress.
pub struct LoopContext {
    /// Task identifier, used for log correlation.
    pub task_id: String,
    /// Session identifier. Currently used only for logging; Phase 0.5 will
    /// reuse it as the multi-turn conversation history key.
    pub session_id: String,
    /// Application name, used for logs and `BasicAgent` naming downstream.
    pub app_name: String,
    /// LLM system prompt. Maps to `role=system` for OpenAI-compatible providers
    /// and the `system` field for Anthropic-compatible providers.
    pub system_prompt: String,
    /// First user message, maps to `role=user`.
    pub initial_user_message: String,
    /// Internal handle to the live stats accumulator.
    stats: Arc<Mutex<LoopStats>>,
}

impl LoopContext {
    /// Construct a `LoopContext` with a freshly-allocated stats handle.
    pub fn new(
        task_id: impl Into<String>,
        session_id: impl Into<String>,
        app_name: impl Into<String>,
        system_prompt: impl Into<String>,
        initial_user_message: impl Into<String>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            session_id: session_id.into(),
            app_name: app_name.into(),
            system_prompt: system_prompt.into(),
            initial_user_message: initial_user_message.into(),
            stats: Arc::new(Mutex::new(LoopStats::default())),
        }
    }

    /// Borrow the internal stats handle; callers can `lock().snapshot()` to
    /// read the live state.
    pub fn stats_handle(&self) -> Arc<Mutex<LoopStats>> {
        self.stats.clone()
    }
}

/// Final outcome of a `run_loop` invocation.
///
/// Serializes as a tagged union (`#[serde(tag = "kind")]`) so it round-trips
/// cleanly through `llm_call_log` and downstream dashboards.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum LoopOutcome {
    /// LLM returned a non-empty text response.
    Done {
        result: String,
        iterations: u32,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    /// Hit `max_iterations` without producing a final response. Unreachable in
    /// the Phase 0.1 single-shot chat implementation; kept for the Phase 0.5
    /// multi-iter tool-calling path.
    MaxIter {
        iterations: u32,
        last_text: String,
    },
    /// Estimated cost exceeded `cfg.cost_limit_usd`.
    Aborted {
        reason: String,
        iterations: u32,
    },
    /// LLM returned a final response but with empty text (refusal, blank
    /// content, etc.).
    LlmRefused {
        reason: String,
        iterations: u32,
    },
}

/// `run_loop` error type. Stable across `error:Display` because we serialize
/// the variants for `llm_call_log` and we don't want log fields to drift.
#[derive(Debug, Error)]
pub enum LoopError {
    #[error("no LLM configured (need to construct an autoagents ChatProvider)")]
    NoLlm,
    #[error("LLM call failed: {0}")]
    LlmCall(String),
    #[error("response parsing failed: {0}")]
    ResponseParse(String),
}

/// Per-call accumulator for `run_loop`; intended for logging, monitoring,
/// and cost attribution.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LoopStats {
    pub iterations: u32,
    pub tool_calls: u32,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub elapsed_ms: u64,
    pub cost_estimate_usd: f64,
}

// ============================================================================
// Core entry point
// ============================================================================

/// Run one LLM agent loop.
///
/// Phase 0.1 implementation: a single `chat()` call, no multi-turn
/// tool-calling. Phase 0.5 will switch this to autoagents's `BasicAgent` +
/// `AgentBuilder` for the full ReAct + tool-call loop, keeping the public
/// API stable.
pub async fn run_loop(
    llm: Arc<dyn ChatProvider>,
    ctx: LoopContext,
    cfg: &LoopConfig,
) -> std::result::Result<LoopOutcome, LoopError> {
    let started = Instant::now();
    let stats = ctx.stats_handle();

    info!(
        task_id = %ctx.task_id,
        session_id = %ctx.session_id,
        app_name = %ctx.app_name,
        max_iterations = cfg.max_iterations,
        "llm_loop::run_loop start"
    );

    // 1. Build the two-message seed: system + user.
    let messages = vec![
        ChatMessageBuilder::new(ChatRole::System)
            .content(ctx.system_prompt.clone())
            .build(),
        ChatMessageBuilder::new(ChatRole::User)
            .content(ctx.initial_user_message.clone())
            .build(),
    ];

    // 2. Single LLM call.
    let response = llm
        .chat(&messages, None)
        .await
        .map_err(|e| LoopError::LlmCall(e.to_string()))?;

    // 3. Accumulate stats from the response.
    {
        let mut s = lock_stats(&stats);
        s.iterations = 1;
        s.tool_calls = response
            .tool_calls()
            .as_ref()
            .map(|tc| tc.len() as u32)
            .unwrap_or(0);
        if let Some(usage) = response.usage() {
            s.prompt_tokens = usage.prompt_tokens;
            s.completion_tokens = usage.completion_tokens;
        }
        s.elapsed_ms = started.elapsed().as_millis() as u64;
        s.cost_estimate_usd = estimate_cost(s.prompt_tokens, s.completion_tokens);
    }
    let snapshot = lock_stats(&stats).clone();

    info!(
        task_id = %ctx.task_id,
        elapsed_ms = snapshot.elapsed_ms,
        prompt = snapshot.prompt_tokens,
        completion = snapshot.completion_tokens,
        tool_calls = snapshot.tool_calls,
        cost_usd = snapshot.cost_estimate_usd,
        "llm_loop::run_loop end"
    );

    // 4. Cost guardrail.
    if let Some(limit) = cfg.cost_limit_usd {
        if snapshot.cost_estimate_usd > limit {
            return Ok(LoopOutcome::Aborted {
                reason: format!(
                    "cost limit ${:.4} exceeded (est. ${:.4})",
                    limit, snapshot.cost_estimate_usd
                ),
                iterations: 1,
            });
        }
    }

    // 5. Final decision based on response text.
    let text = response.text().unwrap_or_default();
    if text.is_empty() {
        return Ok(LoopOutcome::LlmRefused {
            reason: "empty text in LLM response".to_string(),
            iterations: 1,
        });
    }

    Ok(LoopOutcome::Done {
        result: text,
        iterations: 1,
        prompt_tokens: snapshot.prompt_tokens,
        completion_tokens: snapshot.completion_tokens,
    })
}

// ============================================================================
// Helpers
// ============================================================================

fn lock_stats(stats: &Arc<Mutex<LoopStats>>) -> MutexGuard<'_, LoopStats> {
    // Lock poisoning would be unrecoverable here; just propagate.
    stats.lock().unwrap_or_else(|p| p.into_inner())
}

/// Estimate dollar cost.
///
/// Defaults to Claude-Sonnet-3.5 pricing: `$3 / 1M input tokens`,
/// `$15 / 1M output tokens`. autoagents's `Usage` (0.3.7) does not yet carry a
/// provider-given `cost` field, so this is a rough estimate. Phase 0.9 will
/// switch to per-model price tables once the `llm_call_log` is in place.
fn estimate_cost(prompt_tokens: u32, completion_tokens: u32) -> f64 {
    let input_cost = (prompt_tokens as f64) * 3.0 / 1_000_000.0;
    let output_cost = (completion_tokens as f64) * 15.0 / 1_000_000.0;
    input_cost + output_cost
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_config_defaults() {
        let cfg = LoopConfig::default();
        assert_eq!(cfg.max_iterations, 16);
        assert_eq!(cfg.max_tokens_per_call, 4096);
        assert!((cfg.temperature - 0.2).abs() < 1e-6);
        assert_eq!(cfg.tool_timeout_secs, 60);
        assert!(cfg.cost_limit_usd.is_none());
    }

    #[test]
    fn test_loop_context_new() {
        let ctx = LoopContext::new(
            "task-1",
            "sess-1",
            "cleanroom",
            "you are a helpful assistant",
            "hi",
        );
        assert_eq!(ctx.task_id, "task-1");
        assert_eq!(ctx.session_id, "sess-1");
        assert_eq!(ctx.app_name, "cleanroom");
        assert_eq!(ctx.system_prompt, "you are a helpful assistant");
        assert_eq!(ctx.initial_user_message, "hi");
    }

    #[test]
    fn test_loop_stats_handle_snapshot() {
        let ctx = LoopContext::new("t", "s", "a", "p", "hi");
        let h = ctx.stats_handle();
        {
            let mut s = h.lock().unwrap();
            s.iterations = 1;
            s.tool_calls = 2;
            s.prompt_tokens = 100;
            s.completion_tokens = 50;
        }
        let snap = h.lock().unwrap().clone();
        assert_eq!(snap.iterations, 1);
        assert_eq!(snap.tool_calls, 2);
        assert_eq!(snap.prompt_tokens, 100);
        assert_eq!(snap.completion_tokens, 50);
    }

    #[test]
    fn test_estimate_cost_default() {
        // 1M input + 1M output -> $3 + $15 = $18
        let cost = estimate_cost(1_000_000, 1_000_000);
        assert!((cost - 18.0).abs() < 1e-6, "got {}", cost);
        // 0 tokens -> $0
        assert_eq!(estimate_cost(0, 0), 0.0);
    }

    #[test]
    fn test_loop_outcome_done_serde() {
        let o = LoopOutcome::Done {
            result: "ok".into(),
            iterations: 1,
            prompt_tokens: 100,
            completion_tokens: 50,
        };
        let s = serde_json::to_string(&o).unwrap();
        assert!(s.contains("\"kind\":\"Done\""));
        assert!(s.contains("\"result\":\"ok\""));
    }

    #[test]
    fn test_loop_outcome_max_iter_serde() {
        let o = LoopOutcome::MaxIter {
            iterations: 16,
            last_text: "partial".into(),
        };
        let s = serde_json::to_string(&o).unwrap();
        assert!(s.contains("\"kind\":\"MaxIter\""));
    }

    #[test]
    fn test_loop_outcome_aborted_serde() {
        let o = LoopOutcome::Aborted {
            reason: "cost limit".into(),
            iterations: 1,
        };
        let s = serde_json::to_string(&o).unwrap();
        assert!(s.contains("\"kind\":\"Aborted\""));
        assert!(s.contains("\"reason\":\"cost limit\""));
    }
}
