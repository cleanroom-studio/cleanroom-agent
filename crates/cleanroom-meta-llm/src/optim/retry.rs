//! Retry layer with exponential back-off and full jitter.
//!
//! # Retry semantics
//! - All non-streaming calls are retried transparently.
//! - Streaming methods are retried on the **initial async call** only (i.e.
//!   before the stream starts delivering items). Mid-stream errors are not
//!   retried — the caller must restart the stream explicitly.
//! - [`AuthError`], [`InvalidRequest`], [`JsonError`], [`ToolConfigError`],
//!   and [`NoToolSupport`] are **never** retried by the default policy (they
//!   cannot succeed on a subsequent attempt without user intervention).
//! - [`HttpError`], [`ProviderError`], and [`Generic`] errors that carry
//!   rate-limit or server-error signals are retried up to
//!   `max_attempts − 1` additional times with exponential back-off.
//!
//! # Hot-path overhead
//! On a successful first attempt the only overhead over a bare provider call
//! is one extra match arm — no allocation, no timer, no log.
//!
//! [`AuthError`]: crate::error::MetaError::AuthError
//! [`InvalidRequest`]: crate::error::MetaError::InvalidRequest
//! [`JsonError`]: crate::error::MetaError::JsonError
//! [`ToolConfigError`]: crate::error::MetaError::ToolConfigError
//! [`NoToolSupport`]: crate::error::MetaError::NoToolSupport
//! [`HttpError`]: crate::error::MetaError::HttpError
//! [`ProviderError`]: crate::error::MetaError::ProviderError
//! [`Generic`]: crate::error::MetaError::Generic

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::Stream;

use crate::{
    MetaLlm,
    chat::{
        MetaMessage, MetaProvider, MetaResponse, StreamChunk, StreamResponse,
        MetaStructuredOutputFormat, Tool,
    },
    completion::{MetaCompletionProvider, MetaCompletionRequest, MetaCompletionResponse},
    embedding::MetaEmbeddingProvider,
    error::MetaError,
    models::{ModelListRequest, ModelListResponse, MetaModelsProvider},
    pipeline::LLMLayer,
};

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for [`RetryLayer`].
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Total number of attempts, including the first one (≥ 1). Default: `3`.
    pub max_attempts: u32,
    /// Delay before the second attempt. Default: `200 ms`.
    pub initial_backoff: Duration,
    /// Upper bound on the computed back-off interval. Default: `30 s`.
    pub max_backoff: Duration,
    /// Apply **full jitter** to the computed delay (recommended). Default: `true`.
    ///
    /// Full jitter draws the sleep duration uniformly from `[0, ceiling]`,
    /// preventing thundering-herd retries across concurrent callers.
    pub jitter: bool,
    /// Returns `true` if an error should trigger a retry.
    ///
    /// Swap with a custom `fn` to adjust the policy without allocating a
    /// trait object.  The default is [`default_is_retryable`].
    pub retryable: fn(&MetaError) -> bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(30),
            jitter: true,
            retryable: default_is_retryable,
        }
    }
}

/// Default retryability predicate.
///
/// Retries when the error message contains rate-limit (429) or server-error
/// (5xx) signals.  Never retries auth, invalid-request, or structural errors.
pub fn default_is_retryable(err: &MetaError) -> bool {
    match err {
        MetaError::HttpError(msg) | MetaError::ProviderError(msg) => {
            let m = msg.to_lowercase();
            m.contains("429")
                || m.contains("500")
                || m.contains("502")
                || m.contains("503")
                || m.contains("504")
                || m.contains("529") // AnthropicProvider overload
                || m.contains("rate limit")
                || m.contains("too many requests")
                || m.contains("overloaded")
                || m.contains("server error")
                || m.contains("service unavailable")
        }
        MetaError::Generic(_) => true,
        MetaError::AuthError(_)
        | MetaError::InvalidRequest(_)
        | MetaError::GuardrailBlocked { .. }
        | MetaError::GuardrailExecutionFailed { .. }
        | MetaError::ResponseFormatError { .. }
        | MetaError::JsonError(_)
        | MetaError::ToolConfigError(_)
        | MetaError::NoToolSupport(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// An [`LLMLayer`] that wraps the downstream provider with automatic retry.
///
/// Compose it first (outermost) in the pipeline when you also use
/// [`FallbackLayer`](super::FallbackLayer) so that each candidate provider is
/// retried before the next fallback is tried.
///
/// # Example
///
/// ```ignore
/// use cleanroom_meta_llm::{pipeline::PipelineBuilder, optim::{RetryLayer, RetryConfig}};
/// use std::time::Duration;
///
/// let llm = PipelineBuilder::new(base)
///     .add_layer(RetryLayer::new(RetryConfig {
///         max_attempts: 5,
///         initial_backoff: Duration::from_millis(100),
///         ..RetryConfig::default()
///     }))
///     .build();
/// ```
pub struct RetryLayer {
    config: RetryConfig,
}

impl RetryLayer {
    /// Create a layer with the given configuration.
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Create a layer with default configuration (3 attempts, 200 ms initial
    /// back-off, 30 s cap, full jitter).
    pub fn with_defaults() -> Self {
        Self::new(RetryConfig::default())
    }
}

impl LLMLayer for RetryLayer {
    fn build(self: Box<Self>, next: Arc<dyn MetaLlm>) -> Arc<dyn MetaLlm> {
        Arc::new(RetryProvider {
            inner: next,
            config: self.config,
        })
    }
}

// ---------------------------------------------------------------------------
// Provider wrapper
// ---------------------------------------------------------------------------

struct RetryProvider {
    inner: Arc<dyn MetaLlm>,
    config: RetryConfig,
}

// ---------------------------------------------------------------------------
// Back-off helpers (no external RNG — uses subsecond system-time entropy)
// ---------------------------------------------------------------------------

/// Full-jitter sleep duration in `[0, ceiling]`.
///
/// Uses `SystemTime::subsec_nanos()` as a cheap entropy source.  Not
/// cryptographically uniform, but sufficient for back-off anti-thundering-herd.
#[inline]
fn jitter_duration(ceiling: Duration) -> Duration {
    let nanos = ceiling.as_nanos();
    if nanos == 0 {
        return Duration::ZERO;
    }
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u128;
    Duration::from_nanos((seed % nanos) as u64)
}

/// Back-off ceiling for zero-based `attempt` index.
/// `ceiling = min(max_backoff, initial * 2^attempt)`
#[inline]
fn compute_backoff(config: &RetryConfig, attempt: u32) -> Duration {
    let initial_ns = config.initial_backoff.as_nanos().min(u64::MAX as u128) as u64;
    // Use checked_shl so very large attempt counts saturate instead of panicking.
    let multiplier = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
    let max_ns = config.max_backoff.as_nanos().min(u64::MAX as u128) as u64;
    let ceiling = Duration::from_nanos(initial_ns.saturating_mul(multiplier).min(max_ns));
    if config.jitter {
        jitter_duration(ceiling)
    } else {
        ceiling
    }
}

// ---------------------------------------------------------------------------
// Core retry loop
// ---------------------------------------------------------------------------

/// Execute `f` up to `config.max_attempts` times.
///
/// Returns immediately on the first `Ok`.  On a retryable `Err` sleeps for
/// the computed back-off then retries.  On a non-retryable `Err` or when all
/// attempts are exhausted, returns the error.
///
/// # Hot path (first attempt succeeds)
/// No allocation, no timer — just `f().await` and a match.
async fn retry_call<F, Fut, T>(config: &RetryConfig, mut f: F) -> Result<T, MetaError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, MetaError>>,
{
    let max = config.max_attempts.max(1);
    let mut attempt = 0u32;
    loop {
        match f().await {
            Ok(v) => return Ok(v),
            Err(e) if attempt + 1 < max && (config.retryable)(&e) => {
                let backoff = compute_backoff(config, attempt);
                log::warn!(
                    "LLM call failed (attempt {}/{}): {e}. Retrying in {backoff:?}.",
                    attempt + 1,
                    max,
                );
                tokio::time::sleep(backoff).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// MetaProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl MetaProvider for RetryProvider {
    async fn chat(
        &self,
        messages: &[MetaMessage],
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Box<dyn MetaResponse>, MetaError> {
        retry_call(&self.config, || {
            self.inner.chat(messages, json_schema.clone())
        })
        .await
    }

    async fn chat_with_tools(
        &self,
        messages: &[MetaMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Box<dyn MetaResponse>, MetaError> {
        retry_call(&self.config, || {
            self.inner
                .chat_with_tools(messages, tools, json_schema.clone())
        })
        .await
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn MetaResponse>, MetaError> {
        retry_call(&self.config, || {
            self.inner.chat_with_web_search(input.clone())
        })
        .await
    }

    async fn chat_stream(
        &self,
        messages: &[MetaMessage],
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, MetaError>> + Send>>, MetaError> {
        retry_call(&self.config, || {
            self.inner.chat_stream(messages, json_schema.clone())
        })
        .await
    }

    async fn chat_stream_struct(
        &self,
        messages: &[MetaMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, MetaError>> + Send>>, MetaError>
    {
        retry_call(&self.config, || {
            self.inner
                .chat_stream_struct(messages, tools, json_schema.clone())
        })
        .await
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[MetaMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, MetaError>> + Send>>, MetaError> {
        retry_call(&self.config, || {
            self.inner
                .chat_stream_with_tools(messages, tools, json_schema.clone())
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// MetaCompletionProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl MetaCompletionProvider for RetryProvider {
    async fn complete(
        &self,
        req: &MetaCompletionRequest,
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<MetaCompletionResponse, MetaError> {
        retry_call(&self.config, || {
            self.inner.complete(req, json_schema.clone())
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// MetaEmbeddingProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl MetaEmbeddingProvider for RetryProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
        retry_call(&self.config, || self.inner.embed(input.clone())).await
    }
}

// ---------------------------------------------------------------------------
// MetaModelsProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl MetaModelsProvider for RetryProvider {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, MetaError> {
        // `Box<dyn ModelListResponse>` is !Send so cannot go through the generic
        // retry_call helper (which requires T: Send to produce a Send future).
        // Models listing is an administrative call; simple delegation suffices.
        self.inner.list_models(request).await
    }
}

impl MetaLlm for RetryProvider {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FunctionCall, ToolCall,
        chat::{MetaResponse, MetaStructuredOutputFormat, Tool},
        completion::MetaCompletionRequest,
        error::MetaError,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    // -----------------------------------------------------------------------
    // Minimal mock provider
    // -----------------------------------------------------------------------

    struct MockResponse(String);

    impl MetaResponse for MockResponse {
        fn text(&self) -> Option<String> {
            Some(self.0.clone())
        }
        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            None
        }
    }
    impl std::fmt::Debug for MockResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockResponse({})", self.0)
        }
    }
    impl std::fmt::Display for MockResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    /// Succeeds on `success_after`-th call (1-based); returns `err` before that.
    struct CountingMock {
        calls: AtomicU32,
        chat_calls: AtomicU32,
        chat_with_tools_calls: AtomicU32,
        success_after: u32,
        err: MetaError,
    }

    impl CountingMock {
        fn new(success_after: u32, err: MetaError) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicU32::new(0),
                chat_calls: AtomicU32::new(0),
                chat_with_tools_calls: AtomicU32::new(0),
                success_after,
                err,
            })
        }

        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::Relaxed)
        }

        fn next_result(&self) -> Result<Box<dyn MetaResponse>, MetaError> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if n >= self.success_after {
                Ok(Box::new(MockResponse("ok".into())))
            } else {
                // Return a clone of the error variant by re-constructing it.
                Err(match &self.err {
                    MetaError::HttpError(m) => MetaError::HttpError(m.clone()),
                    MetaError::ProviderError(m) => MetaError::ProviderError(m.clone()),
                    MetaError::Generic(m) => MetaError::Generic(m.clone()),
                    MetaError::AuthError(m) => MetaError::AuthError(m.clone()),
                    MetaError::InvalidRequest(m) => MetaError::InvalidRequest(m.clone()),
                    other => MetaError::Generic(other.to_string()),
                })
            }
        }
    }

    #[async_trait]
    impl MetaProvider for CountingMock {
        async fn chat(
            &self,
            _messages: &[MetaMessage],
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<Box<dyn MetaResponse>, MetaError> {
            self.chat_calls.fetch_add(1, Ordering::Relaxed);
            self.next_result()
        }

        async fn chat_with_tools(
            &self,
            _messages: &[MetaMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<Box<dyn MetaResponse>, MetaError> {
            self.chat_with_tools_calls.fetch_add(1, Ordering::Relaxed);
            self.next_result()
        }
    }

    #[async_trait]
    impl MetaCompletionProvider for CountingMock {
        async fn complete(
            &self,
            _req: &MetaCompletionRequest,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<MetaCompletionResponse, MetaError> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if n >= self.success_after {
                Ok(MetaCompletionResponse {
                    text: "done".into(),
                })
            } else {
                Err(MetaError::HttpError("503 service unavailable".into()))
            }
        }
    }

    #[async_trait]
    impl MetaEmbeddingProvider for CountingMock {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if n >= self.success_after {
                Ok(vec![vec![1.0, 2.0]])
            } else {
                Err(MetaError::HttpError("429 rate limit".into()))
            }
        }
    }

    #[async_trait]
    impl MetaModelsProvider for CountingMock {}

    impl MetaLlm for CountingMock {}

    impl crate::MetaHasConfig for CountingMock {
        type Config = crate::MetaNoConfig;
    }

    // -----------------------------------------------------------------------
    // Back-off unit tests (no I/O)
    // -----------------------------------------------------------------------

    #[test]
    fn backoff_grows_exponentially() {
        let cfg = RetryConfig {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(60),
            jitter: false,
            ..RetryConfig::default()
        };
        assert_eq!(compute_backoff(&cfg, 0), Duration::from_millis(100));
        assert_eq!(compute_backoff(&cfg, 1), Duration::from_millis(200));
        assert_eq!(compute_backoff(&cfg, 2), Duration::from_millis(400));
        assert_eq!(compute_backoff(&cfg, 3), Duration::from_millis(800));
    }

    #[test]
    fn backoff_capped_at_max() {
        let cfg = RetryConfig {
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_millis(1000),
            jitter: false,
            ..RetryConfig::default()
        };
        // attempt 2: 500 * 4 = 2000 ms → capped at 1000 ms
        assert_eq!(compute_backoff(&cfg, 2), Duration::from_millis(1000));
    }

    #[test]
    fn backoff_with_jitter_within_bounds() {
        let cfg = RetryConfig {
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(30),
            jitter: true,
            ..RetryConfig::default()
        };
        let ceiling = Duration::from_millis(200);
        for _ in 0..20 {
            let b = compute_backoff(&cfg, 0);
            assert!(b <= ceiling, "jitter exceeded ceiling: {b:?}");
        }
    }

    #[test]
    fn large_attempt_does_not_overflow() {
        let cfg = RetryConfig {
            jitter: false,
            ..RetryConfig::default()
        };
        // Should saturate at max_backoff, not panic.
        let b = compute_backoff(&cfg, 200);
        assert_eq!(b, cfg.max_backoff);
    }

    // -----------------------------------------------------------------------
    // Retryability predicate
    // -----------------------------------------------------------------------

    #[test]
    fn retryable_errors() {
        assert!(default_is_retryable(&MetaError::HttpError(
            "429 rate limit exceeded".into()
        )));
        assert!(default_is_retryable(&MetaError::HttpError(
            "503 service unavailable".into()
        )));
        assert!(default_is_retryable(&MetaError::ProviderError(
            "overloaded".into()
        )));
        assert!(default_is_retryable(&MetaError::Generic(
            "connection reset".into()
        )));
    }

    #[test]
    fn non_retryable_errors() {
        assert!(!default_is_retryable(&MetaError::AuthError(
            "invalid key".into()
        )));
        assert!(!default_is_retryable(&MetaError::InvalidRequest(
            "bad param".into()
        )));
        assert!(!default_is_retryable(&MetaError::GuardrailBlocked {
            phase: crate::error::GuardrailPhase::Input,
            guard: "prompt-injection".into(),
            rule_id: "prompt_injection_detected".into(),
            category: "prompt_injection".into(),
            severity: "high".into(),
            message: "detected suspicious instruction pattern".into(),
        }));
        assert!(!default_is_retryable(&MetaError::GuardrailExecutionFailed {
            guard: "prompt-injection".into(),
            message: "guard runtime error".into(),
        }));
        assert!(!default_is_retryable(&MetaError::JsonError(
            "parse error".into()
        )));
        assert!(!default_is_retryable(&MetaError::ToolConfigError(
            "bad tool".into()
        )));
        assert!(!default_is_retryable(&MetaError::NoToolSupport(
            "unsupported".into()
        )));
    }

    // -----------------------------------------------------------------------
    // Integration: RetryProvider behaviour
    // -----------------------------------------------------------------------

    fn build_retry(mock: Arc<CountingMock>, cfg: RetryConfig) -> Arc<dyn MetaLlm> {
        RetryLayer::new(cfg).build_arc(mock as Arc<dyn MetaLlm>)
    }

    // Helper on RetryLayer to avoid Box ceremony in tests.
    impl RetryLayer {
        fn build_arc(self, next: Arc<dyn MetaLlm>) -> Arc<dyn MetaLlm> {
            Box::new(self).build(next)
        }
    }

    #[tokio::test]
    async fn success_on_first_attempt_makes_one_call() {
        let mock = CountingMock::new(1, MetaError::Generic("never".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 3,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = MetaMessage::user().content("hi").build();
        provider.chat(&[msg], None).await.unwrap();
        assert_eq!(mock.call_count(), 1, "should call inner exactly once");
    }

    #[tokio::test]
    async fn retries_on_retryable_error_and_succeeds() {
        // Fails on attempts 1 and 2, succeeds on attempt 3.
        let mock = CountingMock::new(3, MetaError::HttpError("429 rate limit".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 5,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = MetaMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "ok");
        assert_eq!(mock.call_count(), 3);
    }

    #[tokio::test]
    async fn exhausts_attempts_and_returns_last_error() {
        let mock = CountingMock::new(99, MetaError::HttpError("503 unavailable".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 3,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = MetaMessage::user().content("hi").build();
        let err = provider.chat(&[msg], None).await.unwrap_err();
        assert!(err.to_string().contains("503"));
        assert_eq!(mock.call_count(), 3);
    }

    #[tokio::test]
    async fn non_retryable_error_is_not_retried() {
        let mock = CountingMock::new(99, MetaError::AuthError("invalid key".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 5,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = MetaMessage::user().content("hi").build();
        provider.chat(&[msg], None).await.unwrap_err();
        assert_eq!(mock.call_count(), 1, "auth error must not be retried");
    }

    #[tokio::test]
    async fn max_attempts_one_means_no_retry() {
        let mock = CountingMock::new(99, MetaError::HttpError("429".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 1,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = MetaMessage::user().content("hi").build();
        provider.chat(&[msg], None).await.unwrap_err();
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn chat_preserves_chat_method_shape() {
        let mock = CountingMock::new(1, MetaError::Generic("never".into()));
        let provider = build_retry(mock.clone(), RetryConfig::default());
        let msg = MetaMessage::user().content("Hello").build();

        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().as_deref(), Some("ok"));
        assert_eq!(mock.chat_calls.load(Ordering::Relaxed), 1);
        assert_eq!(mock.chat_with_tools_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn completion_is_retried() {
        let mock = CountingMock::new(2, MetaError::HttpError("503".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 3,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let req = MetaCompletionRequest::new("test");
        let resp = provider.complete(&req, None).await.unwrap();
        assert_eq!(resp.text, "done");
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn embedding_is_retried() {
        let mock = CountingMock::new(2, MetaError::HttpError("429 rate limit".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 3,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let result = provider.embed(vec!["hello".into()]).await.unwrap();
        assert_eq!(result, vec![vec![1.0, 2.0]]);
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn custom_retryable_predicate() {
        // Custom predicate: only retry on auth errors (unusual, but proves override works).
        let mock = CountingMock::new(3, MetaError::AuthError("retry me".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 5,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                retryable: |err| matches!(err, MetaError::AuthError(_)),
                ..RetryConfig::default()
            },
        );
        let msg = MetaMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "ok");
        assert_eq!(mock.call_count(), 3);
    }

    // Verify the FunctionCall import used in mock is available.
    #[test]
    fn function_call_construction() {
        let _ = FunctionCall {
            name: "f".into(),
            arguments: "{}".into(),
        };
    }
}
