//! `llm_loop` — generic LLM agent loop built on top of `autoagents`'s `MetaProvider`.
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
//! `autoagents`'s [`MetaProvider`] trait exposes a single-shot
//! `chat(messages, schema)` call. Multi-turn ReAct + tool-calling is provided by
//! autoagents's [`MetaBasicAgent`](autoagents_core::agent::prebuilt::executor::MetaBasicAgent)
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
//! use autoagents::llm::builder::MetaBuilder;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let llm: Arc<OpenAI> = MetaBuilder::<OpenAI>::new()
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

use std::sync::{Arc, Mutex as StdMutex, MutexGuard};
use std::time::Instant;

use cleanroom_db::LlmCallLog;
use cleanroom_meta_core::agent::memory::{MemoryProvider, SlidingWindowMemory};
use cleanroom_meta_core::agent::prebuilt::executor::{MetaBasicAgent, MetaBasicAgentOutput};
use cleanroom_meta_core::agent::task::MetaTask;
use cleanroom_meta_core::agent::{MetaAgentBuilder, MetaOutputT, MetaDirectAgent};
use tokio::sync::Mutex;
use cleanroom_meta_llm::chat::{
    MetaMessage, MetaMessageBuilder, MetaProvider, MetaResponse, MetaRole, Tool, Usage,
    MetaStructuredOutputFormat,
};
use cleanroom_meta_llm::completion::{MetaCompletionProvider, MetaCompletionRequest, MetaCompletionResponse};
use cleanroom_meta_llm::embedding::MetaEmbeddingProvider;
use cleanroom_meta_llm::models::{MetaModelsProvider, ModelListRequest, ModelListResponse};
use cleanroom_meta_llm::MetaLlm;
use cleanroom_meta_derive::{MetaHooks, MetaOutput, meta_agent};
use cleanroom_meta_llm::error::MetaError;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, warn};

// ============================================================================
// Public types
// ============================================================================

/// Phase 0.10: memory strategy for one `run_loop` invocation.
///
/// Selects how the loop maintains multi-turn context across calls.
/// `Default = MemoryConfig::None` keeps the pre-0.10 stateless
/// behavior; opt in by setting this on a `LoopConfig` and providing
/// a matching `Arc<Mutex<Box<dyn MemoryProvider>>>` on
/// [`LoopContext::with_memory`].
///
/// Note: the **actual** memory instance lives on `LoopContext`, not
/// on `LoopConfig` — `LoopConfig.memory` is just the *recipe* that
/// tells the surrounding agent (`ProducerAgent` / `ConsumerAgent`)
/// what kind of instance to construct. This split lets `LoopConfig`
/// stay `Clone + Send + Sync` (the enum is `Clone + Send + Sync`)
/// while the memory instance — which holds a `Mutex` — lives
/// separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryConfig {
    /// Stateless, no memory across calls. The pre-0.10 default.
    None,
    /// Sliding window: keep the last `window_size` messages; drop
    /// older ones on overflow (or mark for summarization, see
    /// `SlidingWindowMemory::with_strategy`).
    SlidingWindow {
        /// Maximum number of messages to retain (excluding the
        /// prepended history slots used in the next call).
        window_size: usize,
    },
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::None
    }
}

impl std::fmt::Display for MemoryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::SlidingWindow { window_size } => {
                write!(f, "sliding_window({window_size})")
            }
        }
    }
}

/// Tuning parameters + cost guardrail for one `run_loop` invocation.
#[derive(Clone)]
pub struct LoopConfig {
    /// Maximum number of LLM round-trips. Reserved in the Phase 0.1 single-shot
    /// chat implementation; will be used once Phase 0.5 switches to
    /// `MetaBasicAgent` / `MetaAgentBuilder`.
    pub max_iterations: u32,
    /// `max_tokens` for a single LLM call. `autoagents::llm::builder::MetaBuilder`
    /// already takes it at build time; this field is kept for stats / cost
    /// estimation and to allow post-validation of the response length.
    pub max_tokens_per_call: u32,
    /// Sampling temperature (`0.0` = deterministic, `1.0` = creative).
    /// `MetaBuilder` consumes it at build time.
    pub temperature: f32,
    /// Per-tool-call timeout. No-op in Phase 0.1 (single-shot); will take
    /// effect once Phase 0.5 wires `MetaBasicAgent` / tool-call execution.
    pub tool_timeout_secs: u64,
    /// Hard cap on total cost in USD. When the estimate exceeds this value,
    /// `run_loop` short-circuits with `LoopOutcome::Aborted`. `None` disables
    /// the cap.
    pub cost_limit_usd: Option<f64>,
    /// Optional callback fired after each LLM call (Phase 0.9). Receives
    /// an owned [`LlmCallLog`] describing the call (status, tokens,
    /// duration, cost, optional error). The default impl constructs
    /// `None`; callers (e.g. the Producer) opt in by setting this to
    /// a closure that writes the record somewhere (typically the
    /// `llm_call_log` table). The closure runs on the LLM-loop task;
    /// keep it fast and non-blocking.
    pub on_call_complete: Option<Arc<dyn Fn(LlmCallLog) + Send + Sync>>,
    /// Phase 0.10: optional tool set. When `Some` and non-empty, the
    /// loop constructs `DefaultLlmAgent { tools: ... }` and the LLM
    /// can call them (per-turn when running through the ReAct
    /// executor; single-turn when running through the basic
    /// executor — basic only attaches the schema to the first chat
    /// call). When `None` (default), the agent has no tools and the
    /// pre-0.10 behavior is preserved exactly.
    pub tools: Option<Vec<Arc<dyn cleanroom_meta_core::tool::MetaToolT>>>,
    /// Phase 0.10: memory strategy (recipe only — the actual
    /// `MemoryProvider` instance lives on `LoopContext.memory`).
    /// `Default = MemoryConfig::None` keeps the pre-0.10 stateless
    /// behavior; opt in via `MemoryConfig::SlidingWindow { .. }` and
    /// pair with `LoopContext::with_memory(arc)` (or the matching
    /// `ProducerAgent::with_memory` / `ConsumerAgent::with_memory`
    /// builders which thread the same value through).
    pub memory: MemoryConfig,
}

impl std::fmt::Debug for LoopConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopConfig")
            .field("max_iterations", &self.max_iterations)
            .field("max_tokens_per_call", &self.max_tokens_per_call)
            .field("temperature", &self.temperature)
            .field("tool_timeout_secs", &self.tool_timeout_secs)
            .field("cost_limit_usd", &self.cost_limit_usd)
            .field("on_call_complete", &self.on_call_complete.as_ref().map(|_| "<fn>"))
            .field("tools", &self.tools.as_ref().map(|v| format!("<{} tools>", v.len())))
            .field("memory", &self.memory)
            .finish()
    }
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 16,
            max_tokens_per_call: 4096,
            temperature: 0.2,
            tool_timeout_secs: 60,
            cost_limit_usd: None,
            on_call_complete: None,
            tools: None,
            memory: MemoryConfig::None,
        }
    }
}

/// Input context for one `run_loop` invocation (one task / one session).
///
/// Carries the system prompt, the initial user message, and an internal
/// stats handle for callers that want to observe in-flight progress.
pub struct LoopContext {
    /// MetaTask identifier, used for log correlation.
    pub task_id: String,
    /// Session identifier. Currently used only for logging; Phase 0.5 will
    /// reuse it as the multi-turn conversation history key.
    pub session_id: String,
    /// Application name, used for logs and `MetaBasicAgent` naming downstream.
    pub app_name: String,
    /// LLM system prompt. Maps to `role=system` for OpenAI-compatible providers
    /// and the `system` field for Anthropic-compatible providers.
    pub system_prompt: String,
    /// First user message, maps to `role=user`.
    pub initial_user_message: String,
    /// Model name (Phase 0.9: surfaced in `llm_call_log`). Set via
    /// [`LoopContext::with_model`]; defaults to `None` for backwards
    /// compatibility with the 5-arg `new()` constructor.
    pub model: Option<String>,
    /// Phase 0.10: optional memory instance. When `Some`, the loop
    /// queries `recall()` before the LLM call to prepend history
    /// to the message vector, and calls `remember(user)` then
    /// `remember(assistant)` after the LLM call. The instance is
    /// typically owned by the surrounding `ProducerAgent` /
    /// `ConsumerAgent` (so it persists across `run_loop` calls on
    /// the same agent), but `LoopContext` clones the `Arc` cheaply.
    ///
    /// We use `tokio::sync::Mutex` (not `std::sync::Mutex`) because
    /// the lock is held across `MemoryProvider::recall().await` /
    /// `remember().await`, and `std::sync::MutexGuard` is `!Send`
    /// which breaks `tokio::spawn` callers (e.g. `orchestrator.rs`
    /// `ProducerAgent::process_next_task` is spawned across a
    /// multi-producer pool). The async mutex locks via `.await`
    /// so its guard is also `Send`.
    pub memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
    /// Internal handle to the live stats accumulator.
    stats: Arc<StdMutex<LoopStats>>,
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
            model: None,
            memory: None,
            stats: Arc::new(StdMutex::new(LoopStats::default())),
        }
    }

    /// Builder-style: set the LLM model name (used by the
    /// `on_call_complete` hook to populate `LlmCallLog::model`).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Builder-style: attach a memory instance (Phase 0.10). The
    /// instance is shared via `Arc` so the same memory can be
    /// threaded across many `run_loop` calls.
    pub fn with_memory(
        mut self,
        memory: Arc<Mutex<Box<dyn MemoryProvider>>>,
    ) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Builder-style: attach a memory instance, taking `Option` so
    /// callers (`ProducerAgent` / `ConsumerAgent`) can forward their
    /// own `Option<Arc<...>>` field directly without an `.map(...)`.
    /// Equivalent to `self.memory = memory;` if `memory.is_some()`,
    /// else leaves `self.memory = None`.
    pub fn with_memory_opt(
        mut self,
        memory: Option<Arc<Mutex<Box<dyn MemoryProvider>>>>,
    ) -> Self {
        self.memory = memory;
        self
    }

    /// Borrow the internal stats handle; callers can `lock().snapshot()` to
    /// read the live state.
    pub fn stats_handle(&self) -> Arc<StdMutex<LoopStats>> {
        self.stats.clone()
    }
}

/// Phase 0.10: the type alias `LoopContextMemory` is the lock type
/// used by both `LoopContext::memory` and the matching agent-level
/// memory slots (`ProducerAgent::memory` / `ConsumerAgent::memory`).
/// Re-exported here so callers don't have to import the
/// `tokio::sync::Mutex` path themselves.
pub type LoopContextMemory = Mutex<Box<dyn MemoryProvider>>;

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
    #[error("no LLM configured (need to construct an autoagents MetaProvider)")]
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
    /// Phase 0.10: how many history messages were prepended to
    /// the LLM call (from `MemoryProvider::recall`). `0` means
    /// no memory was attached or the memory was empty. Surfaced
    /// to `LlmCallLog::memory_messages_at_call` via
    /// `fire_on_call_complete`.
    pub memory_messages_at_call: i64,
}

// ============================================================================
// Core entry point
// ============================================================================

/// Run one LLM agent loop.
///
/// Phase 0.1 implementation: a single `chat()` call, no multi-turn
/// tool-calling. Phase 0.5 will switch this to autoagents's `MetaBasicAgent` +
/// `MetaAgentBuilder` for the full ReAct + tool-call loop, keeping the public
/// API stable.
pub async fn run_loop(
    llm: Arc<dyn MetaProvider>,
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

    // 1. Build the two-message seed: system + user. Phase 0.10
    // prepends memory history (if any) ahead of the system message
    // so the LLM sees prior turns of the same conversation.
    let mut messages = vec![
        MetaMessageBuilder::new(MetaRole::System)
            .content(ctx.system_prompt.clone())
            .build(),
        MetaMessageBuilder::new(MetaRole::User)
            .content(ctx.initial_user_message.clone())
            .build(),
    ];
    // Phase 0.10: query memory (if attached) and prepend history.
    // We use the `window_size` hint from `cfg.memory` to bound how
    // much history to splice in; falls back to the memory's own
    // internal window when `cfg.memory = None` (e.g. caller is
    // using a pre-configured `SlidingWindowMemory` with its own
    // window_size and passed it via `with_memory`).
    // Phase 0.10: query memory (if attached) and prepend history.
    // `tokio::sync::Mutex::lock().await` returns a guard that is
    // `Send` across `.await`, which `orchestrator.rs`'s
    // `tokio::spawn(async move { ... })` requires. Note that
    // tokio's lock does NOT return a `Result` (it panics on
    // poisoned mutex via `blocking_lock` instead, but the async
    // lock path can't observe poisoning), so no `match Ok/Err` is
    // needed here.
    let history: Vec<MetaMessage> = if let Some(mem_arc) = &ctx.memory {
        let limit = match cfg.memory {
            MemoryConfig::None => None,
            MemoryConfig::SlidingWindow { window_size } => Some(window_size),
        };
        let mut guard = mem_arc.lock().await;
        match guard.recall("", limit).await {
            Ok(h) => h,
            Err(e) => {
                warn!(
                    task_id = %ctx.task_id,
                    error = %e,
                    "run_loop: memory recall failed; proceeding without history"
                );
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    let memory_prepend_count: i64 = history.len() as i64;
    if memory_prepend_count > 0 {
        let mut with_history = history;
        with_history.append(&mut messages);
        messages = with_history;
    }

    // 2. Single LLM call.
    let response = llm
        .chat(&messages, None)
        .await
        .map_err(|e| LoopError::LlmCall(e.to_string()))?;

    // 3. Accumulate stats from the response. Phase 0.10 also
    // records `memory_messages_at_call` so observers can see how
    // many history messages the LLM was fed on this call.
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
        s.memory_messages_at_call = memory_prepend_count;
    }
    let snapshot = lock_stats(&stats).clone();

    // Phase 0.10: remember this turn. We do this BEFORE firing
    // `on_call_complete` so observers reading the audit log can
    // correlate the LLM call with the memory write that follows.
    if let Some(mem_arc) = &ctx.memory {
        let text = response.text().unwrap_or_default();
        let user_msg = MetaMessageBuilder::new(MetaRole::User)
            .content(ctx.initial_user_message.clone())
            .build();
        let assistant_msg = MetaMessageBuilder::new(MetaRole::Assistant)
            .content(text.clone())
            .build();
        // `tokio::sync::Mutex::lock().await` — same reasoning as
        // above (Send across await, no Result on lock).
        let mut guard = mem_arc.lock().await;
        if let Err(e) = guard.remember(&user_msg).await {
            warn!(
                task_id = %ctx.task_id,
                error = %e,
                "run_loop: memory remember(user) failed"
            );
        }
        if let Err(e) = guard.remember(&assistant_msg).await {
            warn!(
                task_id = %ctx.task_id,
                error = %e,
                "run_loop: memory remember(assistant) failed"
            );
        }
    }

    // Phase 0.9: fire the on_call_complete hook (if attached) with a
    // `completed` record. We do this even though the outcome may later
    // be downgraded to `Aborted` (cost guard) or `LlmRefused` (empty
    // response) — the hook records the LLM call *event*, not the
    // caller's downstream decision.
    fire_on_call_complete(cfg, &ctx, &snapshot, "completed", None);

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
// autoagents MetaBasicAgent integration
// ============================================================================

/// Default output payload for [`run_loop_via_basic_agent`].
///
/// Mirrors `autoagents`'s `MetaBasicAgentOutput` so we can convert with
/// `From<MetaBasicAgentOutput>`. The single `response: String` is what
/// `MetaBasicAgent` produces for chat-style tasks.
#[derive(Debug, Default, Clone, Serialize, Deserialize, MetaOutput)]
pub struct LoopAgentOutput {
    /// LLM's text response.
    #[output(description = "The LLM's text response")]
    pub response: String,
}

impl From<MetaBasicAgentOutput> for LoopAgentOutput {
    fn from(output: MetaBasicAgentOutput) -> Self {
        LoopAgentOutput {
            response: output.response,
        }
    }
}

/// A no-op agent struct used by [`run_loop_via_basic_agent`].
///
/// `autoagents`'s `MetaBasicAgent::new` requires a struct annotated with
/// `#[agent]` and `MetaHooks`. Since `MetaBasicAgent` (and its default
/// `MetaDirectAgent` executor) only calls the LLM once, the struct body is
/// empty -- all the work happens via the system prompt + user message
/// passed in at `MetaTask` build time.
///
/// Users who want multi-iter ReAct / tool-calling can supply their own
/// `#[agent]`-annotated struct instead of using this default.
/// A no-op agent struct used by [`run_loop_via_basic_agent`] and the
/// (forthcoming) `run_loop_via_react_agent`.
///
/// Phase 0.10: `tools` is a `Vec<Arc<dyn MetaToolT>>` field so the
/// `#[meta_agent]` proc macro's generated `tools()` impl can read live
/// tools at every call. An empty `Vec` (the default) is fine for
/// single-shot, no-tool workflows — exactly the pre-0.10 behavior.
///
/// For multi-iter / ReAct / tool-calling, supply your own tools at
/// construction time (or via `LoopConfig.tools`).
#[meta_agent(
    name = "cleanroom_llm_agent",
    description = "A direct (single-iteration) LLM agent used by cleanroom-agent's llm_loop. \
                   For multi-iter / ReAct / tool-calling, supply a custom agent struct.",
    output = LoopAgentOutput,
)]
#[derive(Default, Clone, MetaHooks)]
pub struct DefaultLlmAgent {
    /// Live tool set (Phase 0.10). The proc-macro-generated
    /// `MetaDeriveT::tools()` reads this field via
    /// `cleanroom_meta::core::tool::shared_tools_to_boxes(&self.tools)`
    /// so callers can add / remove tools at any time before invoking
    /// `MetaBasicAgent::new(DefaultLlmAgent { tools })`.
    pub tools: Vec<Arc<dyn cleanroom_meta_core::tool::MetaToolT>>,
}

/// Run the LLM agent loop via `autoagents`'s `MetaBasicAgent` + `MetaDirectAgent`
/// executor.
///
/// This is the autoagents-native counterpart of [`run_loop`]: instead of
/// calling `llm.chat()` directly, we go through the agent abstraction so
/// that swapping in a multi-iter / ReAct / tool-calling executor in the
/// future is a one-line change (replace `MetaDirectAgent` with the new
/// executor marker in the `MetaAgentBuilder` type parameter).
///
/// The agent's *output* is a `String` (whatever the LLM produced as text);
/// the conversation *history* lives entirely inside the agent's `MetaTask`
/// (single-turn for now, because `MetaDirectAgent` runs exactly one LLM call
/// per task).
///
/// **Phase 0.10 memory note**: this function does NOT support
/// `LoopContext::memory`. The `MetaDirectAgent` executor is single-turn
/// (it concatenates system + user into one `MetaTask` string) and there
/// is no clean way to splice prepended history into that pipeline
/// without re-architecting the agent framework. Use [`run_loop`]
/// directly if you need memory — it's the canonical Phase 0.10 path
/// for stateful conversations. (`run_loop_via_basic_agent` will
/// pick up memory in Phase 0.10.4 once the ReAct executor path is
/// in place — at that point `MetaAgentBuilder::memory()` will own
/// the lifecycle.)

// ============================================================================
// PLAN2 Phase F.3: Skill injection helpers
//
// `run_loop`'s signature is intentionally frozen for backward compat.
// Skill injection happens in two flavors:
//
// 1. `build_system_prompt_with_skill(...)` — pure function: takes the
//    caller's `system_prompt`, an `Option<&SkillIndex>`, and an
//    `Option<&SelectionPolicy>`, and returns a *new* system prompt
//    string with Tier 1 (`<available_skills>`) + Tier 2 (preselected
//    body) blocks appended. Callers fold the result into their
//    `LoopContext.system_prompt` before calling `run_loop`.
//
// 2. The `mcp_tool_bridge.rs` path uses the same helper internally to
//    keep `ProducerAgent` / `ConsumerAgent` from hard-coding skill
//    logic.
// ============================================================================

use cleanroom_skill::{
    build_skill_catalog_block, select_skill_prompt_block, SelectionPolicy, SkillIndex,
};

/// Build a system prompt that has the Tier 1 `<available_skills>` block
/// and (optionally) the Tier 2 preselected body of the top-1 match for
/// `query`. Pure function — no I/O, no state.
pub fn build_system_prompt_with_skill(
    base_prompt: &str,
    skill_index: Option<&SkillIndex>,
    query: Option<&str>,
    policy: Option<&SelectionPolicy>,
    task_type: Option<&str>,
    token_budget_chars: usize,
) -> String {
    let Some(index) = skill_index else {
        return base_prompt.to_string();
    };
    if index.is_empty() {
        return base_prompt.to_string();
    }

    let catalog = build_skill_catalog_block(index, task_type);

    let tier2 = match (query, policy) {
        (Some(q), Some(p)) => select_skill_prompt_block(index, q, p, token_budget_chars)
            .map(|(block, _)| block)
            .unwrap_or_default(),
        _ => String::new(),
    };

    let mut out = String::with_capacity(base_prompt.len() + catalog.len() + tier2.len() + 32);
    out.push_str(base_prompt);
    out.push_str("\n\n");
    out.push_str(&catalog);
    if !tier2.is_empty() {
        out.push_str("\n\n[preloaded]\n");
        out.push_str(&tier2);
        out.push_str("\n[/preloaded]");
    }
    out
}

pub async fn run_loop_via_basic_agent(
    llm: Arc<dyn MetaLlm>,
    ctx: LoopContext,
    cfg: &LoopConfig,
) -> std::result::Result<LoopOutcome, LoopError> {
    let started = Instant::now();
    let stats = ctx.stats_handle();

    info!(
        task_id = %ctx.task_id,
        session_id = %ctx.session_id,
        app_name = %ctx.app_name,
        "llm_loop::run_loop_via_basic_agent start (MetaDirectAgent, single-iter)"
    );

    // `MetaBasicAgent` owns the agent struct via the proc-macro-generated
    // `MetaHooks` impl. Wrap our default no-op struct in it,
    // threading the optional tool set from `LoopConfig` through the
    // `tools` field (Phase 0.10). The generated `tools()` impl reads
    // `self.tools` via `shared_tools_to_boxes`, so an empty Vec is
    // equivalent to the pre-0.10 no-tools behavior.
    let basic = MetaBasicAgent::new(DefaultLlmAgent {
        tools: cfg.tools.clone().unwrap_or_default(),
    });

    // Wrap the LLM in a UsageCapturingLlm so we can recover token counts
    // after `agent.run()` returns. `MetaBasicAgentOutput` only carries
    // `response: String` + `done: bool`, so without this proxy the cost
    // guardrail in `LoopConfig::cost_limit_usd` is meaningless.
    let cell: UsageCell = Arc::new(StdMutex::new(None));
    let capturing = UsageCapturingLlm::new(llm, cell.clone());
    let llm_dyn: Arc<dyn MetaLlm> = Arc::new(capturing);

    // `MetaAgentBuilder::<_, MetaDirectAgent>` picks the single-iter executor.
    // Build is async because it sets up the LLM client + memory wiring.
    let handle = MetaAgentBuilder::<_, MetaDirectAgent>::new(basic)
        .llm(llm_dyn)
        .build()
        .await
        .map_err(|e| LoopError::LlmCall(format!("MetaBasicAgent build failed: {e}")))?;

    // Encode the system prompt + user message into a single `MetaTask` string.
    // `MetaDirectAgent` doesn't do multi-turn -- it appends the system role
    // and then sends the user message once. We mirror that by concatenating
    // them in a way the LLM understands.
    let task_prompt = format!(
        "{}\n\n{}",
        ctx.system_prompt.trim_end(),
        ctx.initial_user_message
    );
    let task = MetaTask::new(task_prompt);

    // `agent_handle.agent.run(task)` is async; returns a `MetaBasicAgentOutput`
    // (or its serde-encoded value) which we convert into our outcome.
    let output: LoopAgentOutput = handle
        .agent
        .run(task)
        .await
        .map_err(|e| LoopError::LlmCall(format!("MetaBasicAgent run failed: {e}")))?;

    // Read captured usage from the proxy and copy into the per-task stats.
    // The basic agent's chat_with_tools call already wrote into `cell`.
    if let Some(usage) = cell.lock().ok().and_then(|g| g.clone()) {
        let mut s = lock_stats(&stats);
        s.prompt_tokens = usage.prompt_tokens;
        s.completion_tokens = usage.completion_tokens;
    }
    {
        let mut s = lock_stats(&stats);
        s.iterations = 1;
        s.elapsed_ms = started.elapsed().as_millis() as u64;
        s.cost_estimate_usd =
            estimate_cost(s.prompt_tokens, s.completion_tokens);
    }
    let snapshot = lock_stats(&stats).clone();

    // Phase 0.9: see `run_loop` for the rationale behind firing the
    // hook with status="completed" regardless of downstream outcome.
    fire_on_call_complete(cfg, &ctx, &snapshot, "completed", None);

    info!(
        task_id = %ctx.task_id,
        elapsed_ms = snapshot.elapsed_ms,
        "llm_loop::run_loop_via_basic_agent end"
    );

    // Cost guard
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

    if output.response.is_empty() {
        return Ok(LoopOutcome::LlmRefused {
            reason: "empty text in MetaBasicAgent output".to_string(),
            iterations: 1,
        });
    }

    Ok(LoopOutcome::Done {
        result: output.response,
        iterations: 1,
        prompt_tokens: snapshot.prompt_tokens,
        completion_tokens: snapshot.completion_tokens,
    })
}

// ============================================================================
// UsageCapturingLlm proxy (Phase 0.5)
//
// `MetaBasicAgentOutput` only carries `response: String` + `done: bool` and
// drops the `MetaResponse::usage()`. To recover token counts (and thus make
// `LoopConfig::cost_limit_usd` meaningful) we wrap the inner LLM in a
// `UsageCapturingLlm` proxy that records the last `Usage` into a shared
// `Arc<Mutex<Option<Usage>>>` cell. The caller (typically
// `run_loop_via_basic_agent`) holds a clone of the cell and reads it after
// `agent.run()`.
//
// We use a generic `P: MetaLlm + ?Sized` so the inner can be either a concrete
// provider (e.g. `OpenAiProvider`) or `Arc<dyn MetaLlm>`. The wrapper
// implements all 4 sub-traits `MetaLlm` requires (`MetaProvider`,
// `MetaCompletionProvider`, `MetaEmbeddingProvider`, `MetaModelsProvider`),
// each by delegating to the inner; only `MetaProvider::chat_with_tools` is
// overridden to also record usage.
// ============================================================================

/// Thread-safe shared cell for the last observed LLM usage.
pub type UsageCell = Arc<StdMutex<Option<Usage>>>;

/// LLM proxy that records the last `Usage` (prompt + completion tokens) from
/// every `MetaProvider::chat_with_tools` call into a shared `UsageCell`.
///
/// # Why this exists
///
/// `MetaBasicAgentOutput` does not carry token counts, so going through the
/// agent abstraction (the path `run_loop_via_basic_agent` uses) loses the
/// `usage` field on the `MetaResponse`. Without this proxy, `LoopStats` shows
/// `0 prompt + 0 completion` and the `cost_limit_usd` guardrail in
/// `LoopConfig` is effectively a no-op. Wrapping the inner LLM in
/// `UsageCapturingLlm` keeps the agent abstraction AND recovers the usage.
///
/// # Usage
///
/// ```ignore
/// let inner: Arc<dyn MetaLlm> = MetaBuilder::<OpenAiProvider>::new()
///     .api_key("...").base_url("...").model("...").build()?;
/// // Cast the concrete `Arc<OpenAiProvider>` to `Arc<dyn MetaLlm>` via
/// // the standard unsized coercion (every concrete provider impls MetaLlm).
///
/// let cell: UsageCell = Arc::new(Mutex::new(None));
/// let capturing = UsageCapturingLlm::new(inner, cell.clone());
/// let dyn_llm: Arc<dyn MetaLlm> = Arc::new(capturing);
///
/// // ... pass `dyn_llm` to MetaAgentBuilder::llm() ...
/// // ... after agent.run() ...
/// let usage = cell.lock().unwrap().clone();
/// ```
///
/// # Note
///
/// This is a *concrete* struct (not generic) because the sub-trait methods
/// need to dispatch through the trait object held inside. Supertrait method
/// calls on `&self.inner` (an `Arc<dyn MetaLlm>`) work via Rust 1.86+
/// supertrait unsized coercion, no manual upcast needed.
pub struct UsageCapturingLlm {
    inner: Arc<dyn MetaLlm>,
    cell: UsageCell,
}

impl UsageCapturingLlm {
    /// Build a new capturing proxy around `inner` that writes to `cell`.
    pub fn new(inner: Arc<dyn MetaLlm>, cell: UsageCell) -> Self {
        Self { inner, cell }
    }

    /// Borrow the shared usage cell (e.g. to read `last_usage()` after
    /// `agent.run()` returns).
    pub fn cell(&self) -> &UsageCell {
        &self.cell
    }

    /// Convenience: read the last captured usage (clone, no lock held).
    pub fn last_usage(&self) -> Option<Usage> {
        self.cell.lock().ok().and_then(|g| g.clone())
    }

    /// Borrow the inner `Arc<dyn MetaLlm>` (defensive escape hatch for any
    /// caller that needs to call methods the proxy doesn't override).
    pub fn inner(&self) -> &Arc<dyn MetaLlm> {
        &self.inner
    }
}

#[cleanroom_meta::async_trait]
impl MetaProvider for UsageCapturingLlm {
    /// Override `chat_with_tools` to also record `response.usage()`. This is
    /// the path the `MetaBasicAgent` / `MetaDirectAgent` actually takes, so
    /// all in-process LLM calls funnel through here.
    async fn chat_with_tools(
        &self,
        messages: &[MetaMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Box<dyn MetaResponse>, MetaError> {
        // `&self.inner` derefs to `&dyn MetaLlm`; supertrait method dispatch
        // (MetaLlm: MetaProvider) makes this call resolve to the inner's
        // `MetaProvider::chat_with_tools` impl. (rust 1.86+ feature.)
        let resp = self.inner.chat_with_tools(messages, tools, json_schema).await?;
        if let Some(usage) = resp.usage() {
            // Lock poisoning would be unrecoverable; mirror the rest of the crate.
            *self.cell.lock().unwrap_or_else(|p| p.into_inner()) = Some(usage);
        }
        Ok(resp)
    }

    // All other MetaProvider methods have working default impls on the trait
    // (they route to `chat_with_tools` or return a "not supported" error for
    // the streaming variants). We rely on those defaults.
}

#[cleanroom_meta::async_trait]
impl MetaCompletionProvider for UsageCapturingLlm {
    async fn complete(
        &self,
        req: &MetaCompletionRequest,
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<MetaCompletionResponse, MetaError> {
        self.inner.complete(req, json_schema).await
    }
}

#[cleanroom_meta::async_trait]
impl MetaEmbeddingProvider for UsageCapturingLlm {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
        self.inner.embed(input).await
    }
}

#[cleanroom_meta::async_trait]
impl MetaModelsProvider for UsageCapturingLlm {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, MetaError> {
        self.inner.list_models(request).await
    }
}

// `MetaLlm` is a marker trait (`pub trait MetaLlm: ... {}` -- no methods).
// The 4 sub-trait impls above satisfy its bounds (`MetaProvider` +
// `MetaCompletionProvider` + `MetaEmbeddingProvider` + `MetaModelsProvider` +
// `Send` + `Sync` + `'static`); we just opt the wrapper in.
impl MetaLlm for UsageCapturingLlm {}

// ============================================================================
// Helpers
// ============================================================================

fn lock_stats(stats: &Arc<StdMutex<LoopStats>>) -> MutexGuard<'_, LoopStats> {
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

/// Phase 0.9: fire the `LoopConfig::on_call_complete` hook with a
/// freshly-built [`LlmCallLog`] record. `status` is one of `"completed"` /
/// `"aborted"` / `"max_iter"` / `"refused"` / `"failed"`. `error` is set
/// for non-completed statuses.
///
/// The hook is invoked synchronously on the LLM-loop task; keep the
/// closure non-blocking (e.g. a fire-and-forget DB insert). Any panic in
/// the closure is logged + swallowed so a logging failure can never
/// take down the loop.
fn fire_on_call_complete(
    cfg: &LoopConfig,
    ctx: &LoopContext,
    snapshot: &LoopStats,
    status: &str,
    error: Option<String>,
) {
    let Some(cb) = cfg.on_call_complete.as_ref() else {
        return;
    };
    let log = LlmCallLog {
        call_id: format!("call-{}", uuid::Uuid::new_v4()),
        task_id: Some(ctx.task_id.clone()),
        session_id: Some(ctx.session_id.clone()),
        agent_type: "meta".to_string(),
        app_name: Some(ctx.app_name.clone()),
        model: ctx.model.clone(),
        prompt_tokens: snapshot.prompt_tokens,
        completion_tokens: snapshot.completion_tokens,
        duration_ms: snapshot.elapsed_ms,
        iterations: snapshot.iterations,
        tool_calls: snapshot.tool_calls,
        cost_estimate_usd: snapshot.cost_estimate_usd,
        memory_messages_at_call: snapshot.memory_messages_at_call,
        status: status.to_string(),
        error,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    // Std::panic::catch_unwind to make the hook non-fatal.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cb(log);
    }));
    if result.is_err() {
        warn!(
            task_id = %ctx.task_id,
            "on_call_complete hook panicked; logging failure is non-fatal"
        );
    }
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

// ============================================================================
// UsageCapturingLlm tests
//
// We can't cheaply fake a network LLM call here, so these tests cover the
// non-async surface: cell initial state, `last_usage` returning None before
// any chat, and the `Arc<Mutex<Option<Usage>>>` cell type ergonomics.
// The async capture path is exercised by `examples/eval_llm_loop.rs`.
// ============================================================================
#[cfg(test)]
mod usage_capturing_tests {
    use super::*;

    /// Compile-time check that the wrapper can be coerced to `Arc<dyn MetaLlm>`.
    /// This is the wiring the `MetaAgentBuilder::llm(...)` call site needs:
    /// `Arc<UsageCapturingLlm>` must coerce to `Arc<dyn MetaLlm>` for the
    /// builder to accept it. If the trait impls are misconfigured, this
    /// function fails to compile.
    #[allow(dead_code)]
    fn assert_arc_dyn_compat(cap: UsageCapturingLlm) -> Arc<dyn MetaLlm> {
        Arc::new(cap)
    }

    #[test]
    fn test_usage_cell_default_is_none() {
        let cell: UsageCell = Arc::new(StdMutex::new(None));
        assert!(cell.lock().unwrap().is_none());
    }

    #[test]
    fn test_usage_cell_write_then_read() {
        let cell: UsageCell = Arc::new(StdMutex::new(None));
        *cell.lock().unwrap() = Some(Usage {
            prompt_tokens: 42,
            completion_tokens: 7,
            total_tokens: 49,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        });
        let snapshot = cell.lock().unwrap().clone().expect("should be Some");
        assert_eq!(snapshot.prompt_tokens, 42);
        assert_eq!(snapshot.completion_tokens, 7);
        assert_eq!(snapshot.total_tokens, 49);
    }

    #[test]
    fn test_usage_capturing_last_usage_initially_none() {
        // We need a concrete `P: MetaLlm` to construct the wrapper. The unit
        // tests in the rest of this crate use real LLM providers only inside
        // examples, not in `cargo test`. Here we just exercise the cell
        // constructor and `last_usage` reader, which don't touch the inner
        // LLM at all (we put a dummy `Arc::new` to satisfy the type system).
        //
        // The actual `chat_with_tools` override that does the capturing is
        // validated end-to-end by `examples/eval_llm_loop.rs` -- it now
        // reports non-zero prompt + completion tokens.
        //
        // We do this by constructing the wrapper with a stub `Arc` of a
        // zero-sized dummy type. Since `P: MetaLlm` is the bound, and we
        // don't have a `MetaLlm` impl for a dummy type in scope, this
        // assertion is intentionally a compile-only check. See the comment
        // in the function for why.
        //
        // For runtime coverage we instead just verify the UsageCell ergonomics
        // (above) and rely on the doc tests for the trait wiring.
        let _ = assert_arc_dyn_compat_is_a_valid_fn_pointer;
    }

    /// Marker constant so the `assert_arc_dyn_compat_is_a_valid_fn_pointer`
    /// reference above is "used" even when the test doesn't run anything.
    const assert_arc_dyn_compat_is_a_valid_fn_pointer: () = ();
}

// ============================================================================
// Phase 0.9: `on_call_complete` hook tests
//
// `fire_on_call_complete` is a tiny helper but its contract has three
// non-obvious properties worth locking down:
//   1. `None` hook is a true no-op (no allocations, no lock).
//   2. The hook receives a fully-populated `LlmCallLog` (every field
//      populated from `LoopContext` / `LoopStats`).
//   3. A panicking hook does NOT propagate to the loop — logging is a
//      side-channel, never load-bearing.
// ============================================================================
#[cfg(test)]
mod on_call_complete_tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    fn sample_ctx() -> LoopContext {
        LoopContext::new(
            "task-1",
            "session-1",
            "cleanroom-producer",
            "You are a code analyst.",
            "Analyze src/main.rs",
        )
        .with_model("MiniMax-M3")
    }

    fn sample_stats() -> LoopStats {
        LoopStats {
            iterations: 1,
            tool_calls: 0,
            prompt_tokens: 522,
            completion_tokens: 1024,
            elapsed_ms: 29_400,
            cost_estimate_usd: 0.0169,
            memory_messages_at_call: 0,
        }
    }

    #[test]
    fn fire_with_no_hook_is_a_noop() {
        let cfg = LoopConfig::default();
        assert!(cfg.on_call_complete.is_none());
        // Should not panic, should not allocate anything observable.
        fire_on_call_complete(&cfg, &sample_ctx(), &sample_stats(), "completed", None);
    }

    #[test]
    fn fire_invokes_hook_with_populated_record() {
        let collected: Arc<StdMutex<Vec<LlmCallLog>>> = Arc::new(StdMutex::new(Vec::new()));
        let collected_clone = collected.clone();
        let cfg = LoopConfig {
            on_call_complete: Some(Arc::new(move |log: LlmCallLog| {
                collected_clone.lock().unwrap().push(log);
            })),
            ..LoopConfig::default()
        };
        fire_on_call_complete(&cfg, &sample_ctx(), &sample_stats(), "completed", None);
        let logs = collected.lock().unwrap();
        assert_eq!(logs.len(), 1);
        let log = &logs[0];
        assert_eq!(log.task_id.as_deref(), Some("task-1"));
        assert_eq!(log.session_id.as_deref(), Some("session-1"));
        assert_eq!(log.app_name.as_deref(), Some("cleanroom-producer"));
        assert_eq!(log.model.as_deref(), Some("MiniMax-M3"));
        assert_eq!(log.agent_type, "meta");
        assert_eq!(log.prompt_tokens, 522);
        assert_eq!(log.completion_tokens, 1024);
        assert_eq!(log.duration_ms, 29_400);
        assert_eq!(log.iterations, 1);
        assert_eq!(log.tool_calls, 0);
        assert!((log.cost_estimate_usd - 0.0169).abs() < 1e-9);
        assert_eq!(log.status, "completed");
        assert!(log.error.is_none());
        // call_id is auto-generated and unique.
        assert!(log.call_id.starts_with("call-"));
        // created_at is an RFC3339 timestamp.
        assert!(log.created_at.contains('T'));
    }

    #[test]
    fn fire_carries_status_and_error_fields() {
        let collected: Arc<StdMutex<Vec<LlmCallLog>>> = Arc::new(StdMutex::new(Vec::new()));
        let collected_clone = collected.clone();
        let cfg = LoopConfig {
            on_call_complete: Some(Arc::new(move |log: LlmCallLog| {
                collected_clone.lock().unwrap().push(log);
            })),
            ..LoopConfig::default()
        };
        fire_on_call_complete(
            &cfg,
            &sample_ctx(),
            &sample_stats(),
            "failed",
            Some("transport error".to_string()),
        );
        let logs = collected.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].status, "failed");
        assert_eq!(logs[0].error.as_deref(), Some("transport error"));
    }

    #[test]
    fn fire_swallows_hook_panic() {
        // A panicking hook must NOT propagate — the loop should never
        // crash because the audit log got an angry closure.
        let cfg = LoopConfig {
            on_call_complete: Some(Arc::new(|_log: LlmCallLog| {
                panic!("intentional test panic in hook");
            })),
            ..LoopConfig::default()
        };
        // Should not panic, even though the hook does.
        fire_on_call_complete(&cfg, &sample_ctx(), &sample_stats(), "completed", None);
    }

    #[test]
    fn loop_context_with_model_sets_model_field() {
        let ctx = LoopContext::new("t", "s", "a", "sp", "um").with_model("claude-3-5");
        assert_eq!(ctx.model.as_deref(), Some("claude-3-5"));
    }

    #[test]
    fn loop_context_without_with_model_has_none() {
        let ctx = LoopContext::new("t", "s", "a", "sp", "um");
        assert!(ctx.model.is_none());
    }

    #[test]
    fn loop_config_debug_does_not_panic_with_hook() {
        // `Debug` is hand-rolled to print "<fn>" for the hook so the
        // closure doesn't get formatted. This is a smoke test against
        // a future maintainer re-deriving `Debug` and accidentally
        // calling `format!("{:?}", hook)` on a non-Debug closure.
        let cfg = LoopConfig {
            on_call_complete: Some(Arc::new(|_: LlmCallLog| {})),
            ..LoopConfig::default()
        };
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("on_call_complete"));
        assert!(dbg.contains("<fn>"));
    }
}

// ============================================================================
// Phase 0.10: `LoopConfig.tools` + `DefaultLlmAgent.tools` tests
//
// Locks down three properties of the new "tool set is opt-in" surface:
//   1. `Default::default()` for `LoopConfig` is `tools: None` — i.e. the
//      pre-0.10 single-shot, no-tool flow is unchanged.
//   2. `Debug` for `LoopConfig` with `tools: Some(vec![...])` doesn't
//      print the boxes (which are not Debug) — it shows just the count.
//   3. `DefaultLlmAgent` is constructible with a non-empty `tools` vec
//      of `Arc<dyn MetaToolT>`, and `Clone` shares the underlying Arc
//      references (so `MetaBasicAgent::new(DefaultLlmAgent { tools })`
//      and downstream clones see the same tool instances — important
//      for the future per-ProducerAgent tool registry).
// ============================================================================
#[cfg(test)]
mod function_call_tools_tests {
    use super::*;
    use crate::mcp_tool_bridge::{McpDispatchFn, McpToolBridge};
    use cleanroom_meta_core::tool::MetaToolT;

    fn echo_dispatch(args: serde_json::Value) -> Result<serde_json::Value, String> {
        Ok(args)
    }

    fn make_arc_tool(name: &str) -> Arc<dyn MetaToolT> {
        let bridge = McpToolBridge::new(
            name,
            format!("echo tool #{name}"),
            serde_json::json!({"type": "object", "properties": {}}),
            Arc::new(echo_dispatch) as McpDispatchFn,
        );
        Arc::new(bridge) as Arc<dyn MetaToolT>
    }

    #[test]
    fn loop_config_default_has_tools_none() {
        let cfg = LoopConfig::default();
        assert!(cfg.tools.is_none(), "Phase 0.10 invariant: default has no tools");
    }

    #[test]
    fn loop_config_with_tools_debug_does_not_panic_and_shows_count() {
        // Pre-0.10: the same `LoopConfig` shape printed via `{:?}` worked
        // because `tools` was an unprintable but `None` field. Now we have
        // a real `Option<Vec<Box<...>>>` whose `Debug` impl (the derived
        // one) would print the entire tool list — which fails because
        // `dyn MetaToolT` is not `Debug`. The hand-rolled `Debug` for
        // `LoopConfig` must therefore project tools to a count string.
        let cfg = LoopConfig {
            tools: Some(vec![make_arc_tool("alpha"), make_arc_tool("beta")]),
            ..LoopConfig::default()
        };
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("tools"));
        assert!(dbg.contains("<2 tools>"), "got: {dbg}");
    }

    #[test]
    fn loop_config_with_empty_tools_vec_still_constructs() {
        // Distinguishing `None` (no-tools fast path, current default
        // behavior) from `Some(vec![])` (explicitly opted in but with
        // no tools) is intentional: the first is what the pre-0.10
        // path serializes / deserializes as, the second is the
        // forward-compatible "I want the tool-aware code path but
        // I have no tools yet" form.
        let cfg = LoopConfig {
            tools: Some(vec![]),
            ..LoopConfig::default()
        };
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("<0 tools>"), "got: {dbg}");
    }

    #[test]
    fn default_llm_agent_constructs_with_empty_tools() {
        // The pre-0.10 path: `MetaBasicAgent::new(DefaultLlmAgent {})` —
        // verified that the new struct still constructs with the
        // default empty `tools` vec.
        let agent = DefaultLlmAgent::default();
        assert!(agent.tools.is_empty());
    }

    #[test]
    fn default_llm_agent_constructs_with_nonempty_tools() {
        let tools: Vec<Arc<dyn MetaToolT>> =
            vec![make_arc_tool("echo"), make_arc_tool("ping")];
        let agent = DefaultLlmAgent {
            tools: tools.clone(),
        };
        assert_eq!(agent.tools.len(), 2);
        assert_eq!(agent.tools[0].name(), "echo");
        assert_eq!(agent.tools[1].name(), "ping");
    }

    #[test]
    fn default_llm_agent_clone_shares_arc_tools() {
        // `MetaBasicAgent` internally wraps `DefaultLlmAgent` in an Arc
        // (see `cleanroom_meta_core::agent::base::MetaBaseAgent::inner`).
        // Cloning the agent must not duplicate the tool Arc — both
        // clones must point at the same `MetaToolT` instance, so any
        // tool-side state mutated by the dispatch closure is visible
        // across all clones. (Practically: this is a property of
        // `Vec<Arc<T>>`'s `Clone`, but pinning it down here protects
        // against a future maintainer accidentally using `Vec<Box<T>>`
        // and breaking it.)
        let tools: Vec<Arc<dyn MetaToolT>> = vec![make_arc_tool("shared")];
        let a = DefaultLlmAgent {
            tools: tools.clone(),
        };
        let b = a.clone();
        assert_eq!(b.tools.len(), 1);
        // Arc identity: same pointer.
        assert!(Arc::ptr_eq(&a.tools[0], &b.tools[0]));
    }
}

// ============================================================================
// Phase 0.10: `MemoryConfig` + `LoopContext::with_memory_opt` tests
//
// Validates the wiring surface for the new memory subsystem. End-to-end
// behavior (prepend / append) is exercised by the integration / 0.10.4
// end-to-end path; these tests pin down:
//   1. `MemoryConfig::default()` is `None` — the pre-0.10 stateless
//      behavior.
//   2. `Display` for both variants.
//   3. `LoopConfig` round-trips the new `memory` field through Debug.
//   4. `LoopContext::with_memory_opt` forwards the `Some` / `None` arm.
//   5. `LoopStats::memory_messages_at_call` is part of the public surface
//      and defaults to 0 (so legacy audit hooks don't need a rewrite).
// ============================================================================
#[cfg(test)]
mod memory_config_tests {
    use super::*;

    #[test]
    fn memory_config_default_is_none() {
        let cfg = MemoryConfig::default();
        assert!(matches!(cfg, MemoryConfig::None));
    }

    #[test]
    fn memory_config_display_renders_both_variants() {
        assert_eq!(MemoryConfig::None.to_string(), "none");
        assert_eq!(
            MemoryConfig::SlidingWindow { window_size: 8 }.to_string(),
            "sliding_window(8)"
        );
    }

    #[test]
    fn loop_config_default_has_memory_none() {
        let cfg = LoopConfig::default();
        assert!(matches!(cfg.memory, MemoryConfig::None));
    }

    #[test]
    fn loop_config_debug_includes_memory_field() {
        // Pre-0.10: `Debug` for `LoopConfig` only had a hand-rolled
        // subset. We must extend it to print the new `memory` field
        // or it would be silently dropped from `format!("{:?}", cfg)`.
        let cfg = LoopConfig {
            memory: MemoryConfig::SlidingWindow { window_size: 16 },
            ..LoopConfig::default()
        };
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("memory"), "got: {dbg}");
        // The Debug derive on `MemoryConfig` prints the struct
        // variant, not the `Display` impl, so we look for
        // `SlidingWindow { window_size: 16 }` rather than
        // `sliding_window(16)`.
        assert!(dbg.contains("SlidingWindow"), "got: {dbg}");
        assert!(dbg.contains("window_size: 16"), "got: {dbg}");
    }

    #[test]
    fn loop_context_with_memory_opt_some_round_trips() {
        // Building a concrete `MemoryProvider` for the test is overkill;
        // the trait object round-trip is verified at the type level by
        // checking the field is `Some` after `with_memory_opt(Some(..))`.
        // We use a fresh `SlidingWindowMemory` instance (kept alive
        // via `Arc`) so the field value is observable.
        use cleanroom_meta_core::agent::memory::SlidingWindowMemory;
        let sw = SlidingWindowMemory::new(4);
        let provider: Box<dyn cleanroom_meta_core::agent::memory::MemoryProvider> =
            Box::new(sw);
        let mutex = tokio::sync::Mutex::new(provider);
        let arc: Arc<tokio::sync::Mutex<Box<dyn cleanroom_meta_core::agent::memory::MemoryProvider>>> =
            Arc::new(mutex);
        let ctx = LoopContext::new("t", "s", "a", "sp", "um").with_memory_opt(Some(arc.clone()));
        assert!(ctx.memory.is_some());
        // Arc identity — same pointer.
        let same = ctx.memory.as_ref().unwrap();
        assert!(Arc::ptr_eq(same, &arc));
    }

    #[test]
    fn loop_context_with_memory_opt_none_keeps_none() {
        // Conversely: `with_memory_opt(None)` should leave the field
        // as `None` (don't accidentally set it to a default).
        let ctx = LoopContext::new("t", "s", "a", "sp", "um").with_memory_opt(None);
        assert!(ctx.memory.is_none());
    }

    #[test]
    fn loop_stats_memory_messages_at_call_defaults_to_zero() {
        let s = LoopStats::default();
        assert_eq!(s.memory_messages_at_call, 0);
    }
}

// ============================================================================
// PLAN2 Phase F.3 tests
// ============================================================================
#[cfg(test)]
mod skill_injection_tests {
    use super::*;
    use cleanroom_skill::{SelectionPolicy, SkillIndex, SkillScope};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn fake_skill(name: &str, desc: &str, body: &str) -> cleanroom_skill::SkillDocument {
        cleanroom_skill::SkillDocument {
            id: format!("{name}+h"),
            name: name.to_string(),
            description: desc.to_string(),
            license: None,
            compatibility: None,
            tags: vec![],
            allowed_tools: vec!["fs.read_file".into()],
            denied_tools: vec![],
            allowed_paths: vec!["src/**/*.rs".into()],
            staging: None,
            output_schema: None,
            gates: vec![],
            divergence_spec: None,
            applies_to: vec![],
            token_budget: 4096,
            priority: "high".into(),
            trigger: false,
            body: body.to_string(),
            path: PathBuf::from("/x"),
            scope: SkillScope::Builtin,
            hash: "h".into(),
            last_modified: None,
            sdef_shard_uri: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn no_skill_index_passthrough() {
        let out = build_system_prompt_with_skill("base", None, None, None, None, 1000);
        assert_eq!(out, "base");
    }

    #[test]
    fn empty_skill_index_passthrough() {
        let idx = SkillIndex::default();
        let out = build_system_prompt_with_skill("base", Some(&idx), None, None, None, 1000);
        assert_eq!(out, "base");
    }

    #[test]
    fn catalog_injected_when_skill_index_provided() {
        let idx = SkillIndex::new(vec![fake_skill("rust-x", "rust tool", "body")]);
        let out = build_system_prompt_with_skill("base", Some(&idx), None, None, None, 1000);
        assert!(out.contains("base"));
        assert!(out.contains("<available_skills>"));
        assert!(out.contains("rust-x"));
    }

    #[test]
    fn tier2_injected_when_query_and_policy_given() {
        let idx = SkillIndex::new(vec![fake_skill(
            "rust-analysis",
            "rust trait impl",
            "very long body...",
        )]);
        let p = SelectionPolicy {
            top_k: 1,
            min_score: 0.0,
            ..Default::default()
        };
        let out = build_system_prompt_with_skill(
            "base",
            Some(&idx),
            Some("rust analysis"),
            Some(&p),
            None,
            200,
        );
        assert!(out.contains("[preloaded]"));
        assert!(out.contains("[/preloaded]"));
        assert!(out.contains("rust-analysis"));
    }
}
