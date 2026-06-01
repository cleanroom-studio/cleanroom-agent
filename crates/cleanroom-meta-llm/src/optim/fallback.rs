//! Fallback layer — routes to backup providers on failure.
//!
//! # Fallback semantics
//! - On a **fallbackable** error from the primary provider, each fallback is
//!   tried in the order they were added until one succeeds.
//! - On a **non-fallbackable** error (e.g. [`InvalidRequest`],
//!   [`AuthError`]) the error is propagated immediately without trying further
//!   providers — retrying a bad request on a different provider is wasteful.
//! - [`NoToolSupport`] *is* fallbackable so that a local/lite model can be
//!   backed by a full-featured remote provider for tool-calling tasks.
//! - Streaming methods fall back on the **initial async call** only.
//!
//! # Hot-path overhead
//! When `providers[0]` (primary) succeeds, the only overhead over a bare
//! provider is one call to the `fallbackable` function pointer and one slice
//! element access — no allocation, no iteration.
//!
//! # Composing with RetryLayer
//! `FallbackLayer` uses fallback providers exactly as passed to
//! [`FallbackLayer::new`]. Inner pipeline layers only wrap the primary `next`
//! provider.
//!
//! To retry each provider independently, pre-wrap each fallback with
//! [`RetryLayer`](super::RetryLayer) before passing it to fallback. If a single
//! outer retry is acceptable, add `RetryLayer` outside `FallbackLayer`.
//!
//! Example with a single outer retry:
//!
//! ```ignore
//! PipelineBuilder::new(openai)
//!     .add_layer(RetryLayer::with_defaults())
//!     .add_layer(FallbackLayer::new(vec![anthropic, ollama]))
//!     .build()
//! // Request flow: RetryLayer → FallbackLayer → primary/fallback providers
//! ```
//!
//! [`InvalidRequest`]: crate::error::MetaError::InvalidRequest
//! [`AuthError`]: crate::error::MetaError::AuthError
//! [`NoToolSupport`]: crate::error::MetaError::NoToolSupport

use std::{future::Future, pin::Pin, sync::Arc};

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

/// Configuration for [`FallbackLayer`].
#[derive(Debug, Clone)]
pub struct FallbackConfig {
    /// Returns `true` when a provider error should trigger a fallback attempt.
    ///
    /// Swap with a custom `fn` to adjust the policy without allocating a
    /// trait object.  The default is [`default_is_fallbackable`].
    pub fallbackable: fn(&MetaError) -> bool,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            fallbackable: default_is_fallbackable,
        }
    }
}

/// Default fallbackability predicate.
///
/// Falls back on network, provider, and structural-response errors as well as
/// [`NoToolSupport`](crate::error::MetaError::NoToolSupport) (enables routing
/// tool-calling tasks to a capable fallback).
///
/// Does **not** fall back on auth or invalid-request errors.
pub fn default_is_fallbackable(err: &MetaError) -> bool {
    matches!(
        err,
        MetaError::HttpError(_)
            | MetaError::ProviderError(_)
            | MetaError::Generic(_)
            | MetaError::ResponseFormatError { .. }
            | MetaError::NoToolSupport(_)
    )
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// An [`LLMLayer`] that routes to backup providers when the primary fails.
///
/// The fallback list is tried **in addition to** the primary provider injected
/// by [`PipelineBuilder`](crate::pipeline::PipelineBuilder) at build time, so
/// total providers = 1 (primary) + `fallbacks.len()`.
///
/// # Example
///
/// ```ignore
/// use cleanroom_meta_llm::{pipeline::PipelineBuilder, optim::FallbackLayer};
///
/// let llm = PipelineBuilder::new(openai)
///     .add_layer(FallbackLayer::new(vec![anthropic, ollama]))
///     .build();
/// ```
pub struct FallbackLayer {
    fallbacks: Vec<Arc<dyn MetaLlm>>,
    config: FallbackConfig,
}

impl FallbackLayer {
    /// Create a layer with the given fallback providers and default config.
    ///
    /// Fallback providers are used as-is. They are not automatically wrapped by
    /// other pipeline layers that may exist around the primary provider.
    pub fn new(fallbacks: Vec<Arc<dyn MetaLlm>>) -> Self {
        Self {
            fallbacks,
            config: FallbackConfig::default(),
        }
    }

    /// Create a layer with a single fallback provider.
    pub fn single(fallback: Arc<dyn MetaLlm>) -> Self {
        Self::new(vec![fallback])
    }

    /// Override the fallbackability predicate.
    pub fn with_config(mut self, config: FallbackConfig) -> Self {
        self.config = config;
        self
    }
}

impl LLMLayer for FallbackLayer {
    fn build(self: Box<Self>, next: Arc<dyn MetaLlm>) -> Arc<dyn MetaLlm> {
        // Pre-build the provider list: primary first, then fallbacks.
        // Avoids any allocation on the hot call path.
        let mut providers = Vec::with_capacity(1 + self.fallbacks.len());
        providers.push(next);
        providers.extend(self.fallbacks);
        Arc::new(FallbackProvider {
            providers,
            config: self.config,
        })
    }
}

// ---------------------------------------------------------------------------
// Provider wrapper
// ---------------------------------------------------------------------------

struct FallbackProvider {
    /// `providers[0]` is always the primary; the rest are fallbacks in order.
    providers: Vec<Arc<dyn MetaLlm>>,
    config: FallbackConfig,
}

// ---------------------------------------------------------------------------
// Core fallback loop
// ---------------------------------------------------------------------------

/// Try each provider with `f` in order.
///
/// Receives an owned `Arc<dyn MetaLlm>` (cloned from the slice) so that
/// callers can wrap the call in `async move { p.method(...).await }` without
/// the future ever borrowing from an iteration-scoped variable.
///
/// Returns the first `Ok`.  On a fallbackable `Err` logs a warning and
/// advances to the next provider.  On a non-fallbackable `Err` returns
/// immediately — retrying on a different provider would be pointless.
///
/// # Hot path (primary succeeds)
/// Single `f(providers[0].clone()).await` + one match arm.  No allocation
/// beyond the Arc clone.
async fn try_fallback<F, Fut, T>(
    providers: &[Arc<dyn MetaLlm>],
    config: &FallbackConfig,
    mut f: F,
) -> Result<T, MetaError>
where
    F: FnMut(Arc<dyn MetaLlm>) -> Fut,
    Fut: Future<Output = Result<T, MetaError>>,
{
    let mut last_err: Option<MetaError> = None;
    for (idx, provider) in providers.iter().enumerate() {
        match f(Arc::clone(provider)).await {
            Ok(v) => return Ok(v),
            Err(e) if (config.fallbackable)(&e) => {
                let label = if idx == 0 { "primary" } else { "fallback" };
                log::warn!(
                    "LLM {label}[{idx}] failed: {e}. Trying next provider ({}/{}).",
                    idx + 1,
                    providers.len(),
                );
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or_else(|| MetaError::Generic("No providers available".into())))
}

// ---------------------------------------------------------------------------
// MetaProvider
// ---------------------------------------------------------------------------

// Each method uses `async move` so that the owned `Arc` (and any cloned data)
// are moved into the future rather than borrowing from the closure parameter.
// This is required because `async_trait` futures borrow `&self`, and if `p`
// were only borrowed from the closure's scope the future would not live long
// enough.

#[async_trait]
impl MetaProvider for FallbackProvider {
    async fn chat(
        &self,
        messages: &[MetaMessage],
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Box<dyn MetaResponse>, MetaError> {
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.chat(messages, js).await }
        })
        .await
    }

    async fn chat_with_tools(
        &self,
        messages: &[MetaMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Box<dyn MetaResponse>, MetaError> {
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.chat_with_tools(messages, tools, js).await }
        })
        .await
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn MetaResponse>, MetaError> {
        try_fallback(&self.providers, &self.config, |p| {
            let input = input.clone();
            async move { p.chat_with_web_search(input).await }
        })
        .await
    }

    async fn chat_stream(
        &self,
        messages: &[MetaMessage],
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, MetaError>> + Send>>, MetaError> {
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.chat_stream(messages, js).await }
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
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.chat_stream_struct(messages, tools, js).await }
        })
        .await
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[MetaMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, MetaError>> + Send>>, MetaError> {
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.chat_stream_with_tools(messages, tools, js).await }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// MetaCompletionProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl MetaCompletionProvider for FallbackProvider {
    async fn complete(
        &self,
        req: &MetaCompletionRequest,
        json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<MetaCompletionResponse, MetaError> {
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.complete(req, js).await }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// MetaEmbeddingProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl MetaEmbeddingProvider for FallbackProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
        try_fallback(&self.providers, &self.config, |p| {
            let input = input.clone();
            async move { p.embed(input).await }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// MetaModelsProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl MetaModelsProvider for FallbackProvider {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, MetaError> {
        // `Box<dyn ModelListResponse>` is !Send so cannot go through the generic
        // try_fallback helper.  Manual loop is equivalent for this low-frequency
        // administrative call.
        let mut last_err: Option<MetaError> = None;
        for (idx, provider) in self.providers.iter().enumerate() {
            match provider.list_models(request).await {
                Ok(r) => return Ok(r),
                Err(e) if (self.config.fallbackable)(&e) => {
                    let label = if idx == 0 { "primary" } else { "fallback" };
                    log::warn!("list_models {label}[{idx}] failed: {e}. Trying next provider.");
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or_else(|| MetaError::Generic("No providers available".into())))
    }
}

impl MetaLlm for FallbackProvider {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ToolCall,
        chat::{MetaResponse, MetaStructuredOutputFormat, Tool},
        completion::MetaCompletionRequest,
        error::MetaError,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    // -----------------------------------------------------------------------
    // Mock helpers
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

    /// Always fails with the given error.
    struct AlwaysFails {
        err_msg: String,
        calls: AtomicU32,
    }

    impl AlwaysFails {
        fn new(err_msg: impl Into<String>) -> Arc<Self> {
            Arc::new(Self {
                err_msg: err_msg.into(),
                calls: AtomicU32::new(0),
            })
        }
        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl MetaProvider for AlwaysFails {
        async fn chat_with_tools(
            &self,
            _messages: &[MetaMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<Box<dyn MetaResponse>, MetaError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(MetaError::ProviderError(self.err_msg.clone()))
        }
    }
    #[async_trait]
    impl MetaCompletionProvider for AlwaysFails {
        async fn complete(
            &self,
            _req: &MetaCompletionRequest,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<MetaCompletionResponse, MetaError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(MetaError::ProviderError(self.err_msg.clone()))
        }
    }
    #[async_trait]
    impl MetaEmbeddingProvider for AlwaysFails {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(MetaError::HttpError(self.err_msg.clone()))
        }
    }
    #[async_trait]
    impl MetaModelsProvider for AlwaysFails {}
    impl MetaLlm for AlwaysFails {}
    impl crate::MetaHasConfig for AlwaysFails {
        type Config = crate::MetaNoConfig;
    }

    /// Always succeeds with `response_text`.
    struct AlwaysSucceeds {
        text: String,
        calls: AtomicU32,
        chat_calls: AtomicU32,
        chat_with_tools_calls: AtomicU32,
    }

    impl AlwaysSucceeds {
        fn new(text: impl Into<String>) -> Arc<Self> {
            Arc::new(Self {
                text: text.into(),
                calls: AtomicU32::new(0),
                chat_calls: AtomicU32::new(0),
                chat_with_tools_calls: AtomicU32::new(0),
            })
        }
        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl MetaProvider for AlwaysSucceeds {
        async fn chat(
            &self,
            _messages: &[MetaMessage],
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<Box<dyn MetaResponse>, MetaError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.chat_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(MockResponse(self.text.clone())))
        }

        async fn chat_with_tools(
            &self,
            _messages: &[MetaMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<Box<dyn MetaResponse>, MetaError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.chat_with_tools_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(MockResponse(self.text.clone())))
        }
    }
    #[async_trait]
    impl MetaCompletionProvider for AlwaysSucceeds {
        async fn complete(
            &self,
            _req: &MetaCompletionRequest,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<MetaCompletionResponse, MetaError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(MetaCompletionResponse {
                text: self.text.clone(),
            })
        }
    }
    #[async_trait]
    impl MetaEmbeddingProvider for AlwaysSucceeds {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(vec![vec![0.5]])
        }
    }
    #[async_trait]
    impl MetaModelsProvider for AlwaysSucceeds {}
    impl MetaLlm for AlwaysSucceeds {}
    impl crate::MetaHasConfig for AlwaysSucceeds {
        type Config = crate::MetaNoConfig;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    impl FallbackLayer {
        fn build_arc(self, next: Arc<dyn MetaLlm>) -> Arc<dyn MetaLlm> {
            Box::new(self).build(next)
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn primary_success_no_fallback_called() {
        let primary = AlwaysSucceeds::new("primary");
        let fallback = AlwaysSucceeds::new("fallback");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn MetaLlm>])
            .build_arc(primary.clone() as Arc<dyn MetaLlm>);

        let msg = MetaMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "primary");
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 0, "fallback must not be called");
        assert_eq!(primary.chat_calls.load(Ordering::Relaxed), 1);
        assert_eq!(primary.chat_with_tools_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn primary_fails_fallback_is_tried() {
        let primary = AlwaysFails::new("provider down");
        let fallback = AlwaysSucceeds::new("fallback_ok");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn MetaLlm>])
            .build_arc(primary.clone() as Arc<dyn MetaLlm>);

        let msg = MetaMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "fallback_ok");
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn all_providers_fail_returns_last_error() {
        let p1 = AlwaysFails::new("p1 down");
        let p2 = AlwaysFails::new("p2 down");
        let p3 = AlwaysFails::new("p3 down");

        let provider = FallbackLayer::new(vec![
            p2.clone() as Arc<dyn MetaLlm>,
            p3.clone() as Arc<dyn MetaLlm>,
        ])
        .build_arc(p1.clone() as Arc<dyn MetaLlm>);

        let msg = MetaMessage::user().content("hi").build();
        let err = provider.chat(&[msg], None).await.unwrap_err();
        assert!(
            err.to_string().contains("p3 down"),
            "last error should be from p3: {err}"
        );
        assert_eq!(p1.call_count(), 1);
        assert_eq!(p2.call_count(), 1);
        assert_eq!(p3.call_count(), 1);
    }

    #[tokio::test]
    async fn non_fallbackable_error_stops_immediately() {
        let primary = Arc::new(AuthFailProvider);
        let fallback = AlwaysSucceeds::new("should_not_reach");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn MetaLlm>])
            .build_arc(primary as Arc<dyn MetaLlm>);

        let msg = MetaMessage::user().content("hi").build();
        let err = provider.chat(&[msg], None).await.unwrap_err();
        assert!(matches!(err, MetaError::AuthError(_)));
        assert_eq!(
            fallback.call_count(),
            0,
            "fallback must not be called on auth error"
        );
    }

    #[tokio::test]
    async fn no_tool_support_triggers_fallback() {
        let primary = Arc::new(NoToolProvider);
        let fallback = AlwaysSucceeds::new("tool_capable");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn MetaLlm>])
            .build_arc(primary as Arc<dyn MetaLlm>);

        let msg = MetaMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "tool_capable");
        assert_eq!(fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn fallback_second_in_chain_succeeds() {
        let p1 = AlwaysFails::new("p1 down");
        let p2 = AlwaysFails::new("p2 down");
        let p3 = AlwaysSucceeds::new("p3_ok");

        let provider = FallbackLayer::new(vec![
            p2.clone() as Arc<dyn MetaLlm>,
            p3.clone() as Arc<dyn MetaLlm>,
        ])
        .build_arc(p1.clone() as Arc<dyn MetaLlm>);

        let msg = MetaMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "p3_ok");
        assert_eq!(p1.call_count(), 1);
        assert_eq!(p2.call_count(), 1);
        assert_eq!(p3.call_count(), 1);
    }

    #[tokio::test]
    async fn completion_fallback() {
        let primary = AlwaysFails::new("down");
        let fallback = AlwaysSucceeds::new("fallback_completion");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn MetaLlm>])
            .build_arc(primary.clone() as Arc<dyn MetaLlm>);

        let req = MetaCompletionRequest::new("prompt");
        let resp = provider.complete(&req, None).await.unwrap();
        assert_eq!(resp.text, "fallback_completion");
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn embedding_fallback() {
        let primary = AlwaysFails::new("embed down");
        let fallback = AlwaysSucceeds::new("embed_ok");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn MetaLlm>])
            .build_arc(primary.clone() as Arc<dyn MetaLlm>);

        let result = provider.embed(vec!["text".into()]).await.unwrap();
        assert_eq!(result, vec![vec![0.5_f32]]);
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn custom_fallbackable_predicate() {
        // Custom: only fallback on auth errors (unusual — proves override works).
        let primary = Arc::new(AuthFailProvider);
        let fallback = AlwaysSucceeds::new("custom_fallback");

        let config = FallbackConfig {
            fallbackable: |err| matches!(err, MetaError::AuthError(_)),
        };
        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn MetaLlm>])
            .with_config(config)
            .build_arc(primary as Arc<dyn MetaLlm>);

        let msg = MetaMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "custom_fallback");
        assert_eq!(fallback.call_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Auxiliary mock providers
    // -----------------------------------------------------------------------

    struct AuthFailProvider;

    #[async_trait]
    impl MetaProvider for AuthFailProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[MetaMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<Box<dyn MetaResponse>, MetaError> {
            Err(MetaError::AuthError("invalid key".into()))
        }
    }
    #[async_trait]
    impl MetaCompletionProvider for AuthFailProvider {
        async fn complete(
            &self,
            _req: &MetaCompletionRequest,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<MetaCompletionResponse, MetaError> {
            Err(MetaError::AuthError("invalid key".into()))
        }
    }
    #[async_trait]
    impl MetaEmbeddingProvider for AuthFailProvider {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
            Err(MetaError::AuthError("invalid key".into()))
        }
    }
    #[async_trait]
    impl MetaModelsProvider for AuthFailProvider {}
    impl MetaLlm for AuthFailProvider {}
    impl crate::MetaHasConfig for AuthFailProvider {
        type Config = crate::MetaNoConfig;
    }

    struct NoToolProvider;

    #[async_trait]
    impl MetaProvider for NoToolProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[MetaMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<Box<dyn MetaResponse>, MetaError> {
            Err(MetaError::NoToolSupport("no tools".into()))
        }
    }
    #[async_trait]
    impl MetaCompletionProvider for NoToolProvider {
        async fn complete(
            &self,
            _req: &MetaCompletionRequest,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<MetaCompletionResponse, MetaError> {
            Err(MetaError::NoToolSupport("no tools".into()))
        }
    }
    #[async_trait]
    impl MetaEmbeddingProvider for NoToolProvider {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
            Err(MetaError::NoToolSupport("no tools".into()))
        }
    }
    #[async_trait]
    impl MetaModelsProvider for NoToolProvider {}
    impl MetaLlm for NoToolProvider {}
    impl crate::MetaHasConfig for NoToolProvider {
        type Config = crate::MetaNoConfig;
    }

    // -----------------------------------------------------------------------
    // Default-predicate unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn fallbackable_errors() {
        assert!(default_is_fallbackable(&MetaError::HttpError(
            "timeout".into()
        )));
        assert!(default_is_fallbackable(&MetaError::ProviderError(
            "down".into()
        )));
        assert!(default_is_fallbackable(&MetaError::Generic(
            "network".into()
        )));
        assert!(default_is_fallbackable(&MetaError::NoToolSupport(
            "unsupported".into()
        )));
        assert!(default_is_fallbackable(&MetaError::ResponseFormatError {
            message: "bad".into(),
            raw_response: "{}".into()
        }));
    }

    #[test]
    fn non_fallbackable_errors() {
        assert!(!default_is_fallbackable(&MetaError::AuthError(
            "bad key".into()
        )));
        assert!(!default_is_fallbackable(&MetaError::InvalidRequest(
            "bad param".into()
        )));
        assert!(!default_is_fallbackable(&MetaError::JsonError(
            "parse".into()
        )));
        assert!(!default_is_fallbackable(&MetaError::ToolConfigError(
            "bad".into()
        )));
    }
}
