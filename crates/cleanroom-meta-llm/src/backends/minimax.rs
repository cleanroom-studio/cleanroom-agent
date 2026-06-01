//! MinimaxProvider API client implementation for chat functionality.
//!
//! This module provides integration with MinimaxProvider's LLM models through their
//! OpenAiProvider-compatible API. Supports MinimaxProvider-M2.5 and MinimaxProvider-M2.5-highspeed models.

use crate::builder::MetaBuilder;
use crate::{
    MetaLlm,
    builder::MetaBackend,
    chat::{MetaStructuredOutputFormat, MetaToolChoice},
    completion::{MetaCompletionProvider, MetaCompletionRequest, MetaCompletionResponse},
    embedding::MetaEmbeddingProvider,
    error::MetaError,
    models::{ModelListRequest, ModelListResponse, MetaModelsProvider, StandardModelListResponse},
    providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig},
};
use async_trait::async_trait;
use std::sync::Arc;

/// MinimaxProvider configuration for the generic OpenAiProvider-compatible provider.
pub struct MiniMaxConfig;

impl OpenAIProviderConfig for MiniMaxConfig {
    const PROVIDER_NAME: &'static str = "MinimaxProvider";
    const DEFAULT_BASE_URL: &'static str = "https://api.minimaxi.chat/v1/";
    const DEFAULT_MODEL: &'static str = "MinimaxProvider-M2.5";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = false;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = false;
}

/// MinimaxProvider LLM provider backed by the generic OpenAiProvider-compatible implementation.
pub type MinimaxProvider = OpenAICompatibleProvider<MiniMaxConfig>;

impl MinimaxProvider {
    /// Creates a new MinimaxProvider client with the specified configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tool_choice: Option<MetaToolChoice>,
        reasoning_effort: Option<String>,
        parallel_tool_calls: Option<bool>,
        normalize_response: Option<bool>,
        extra_body: Option<serde_json::Value>,
    ) -> Self {
        OpenAICompatibleProvider::<MiniMaxConfig>::new(
            api_key,
            base_url,
            model,
            max_tokens,
            temperature,
            timeout_seconds,
            top_p,
            top_k,
            tool_choice,
            reasoning_effort,
            None, // voice - not supported by MinimaxProvider
            extra_body,
            parallel_tool_calls,
            normalize_response,
            None, // embedding_encoding_format - not supported
            None, // embedding_dimensions - not supported
        )
    }
}

impl MetaLlm for MinimaxProvider {}

impl crate::MetaHasConfig for MinimaxProvider {
    type Config = crate::MetaNoConfig;
}

#[async_trait]
impl MetaCompletionProvider for MinimaxProvider {
    async fn complete(
        &self,
        _req: &MetaCompletionRequest,
        _json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<MetaCompletionResponse, MetaError> {
        Ok(MetaCompletionResponse {
            text: "MinimaxProvider completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl MetaEmbeddingProvider for MinimaxProvider {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
        Err(MetaError::ProviderError(
            "Embedding not supported by MinimaxProvider".to_string(),
        ))
    }
}

#[async_trait]
impl MetaModelsProvider for MinimaxProvider {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, MetaError> {
        if self.api_key.is_empty() {
            return Err(MetaError::AuthError("Missing MinimaxProvider API key".to_string()));
        }

        let url = format!("{}models", MiniMaxConfig::DEFAULT_BASE_URL);

        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: MetaBackend::MiniMax,
        };
        Ok(Box::new(result))
    }
}

impl MetaBuilder<MinimaxProvider> {
    /// Builds the MinimaxProvider provider from the configured builder.
    pub fn build(self) -> Result<Arc<MinimaxProvider>, MetaError> {
        let api_key = self.api_key.ok_or_else(|| {
            MetaError::InvalidRequest("No API key provided for MinimaxProvider".to_string())
        })?;

        let minimax = MinimaxProvider::with_config(
            api_key,
            self.base_url,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
            self.tool_choice,
            self.reasoning_effort,
            self.enable_parallel_tool_use,
            self.normalize_response,
            self.extra_body,
        );

        Ok(Arc::new(minimax))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::MetaBuilder;
    use crate::completion::MetaCompletionRequest;

    #[test]
    fn test_with_config_defaults() {
        let provider = MinimaxProvider::with_config(
            "key",
            None,
            None,
            Some(200),
            Some(0.5),
            Some(12),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(provider.api_key, "key");
        assert_eq!(provider.model, MiniMaxConfig::DEFAULT_MODEL);
        assert_eq!(provider.max_tokens, Some(200));
        assert_eq!(provider.temperature, Some(0.5));
        assert_eq!(provider.timeout_seconds, Some(12));
    }

    #[test]
    fn test_with_config_custom_model() {
        let provider = MinimaxProvider::with_config(
            "key",
            None,
            Some("MinimaxProvider-M2.5-highspeed".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(provider.model, "MinimaxProvider-M2.5-highspeed");
    }

    #[test]
    fn test_with_config_custom_base_url() {
        let provider = MinimaxProvider::with_config(
            "key",
            Some("https://api.minimax.chat/v1/".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(provider.base_url.as_str(), "https://api.minimax.chat/v1/");
    }

    #[tokio::test]
    async fn test_list_models_missing_key() {
        let provider = MinimaxProvider::with_config(
            "", None, None, None, None, None, None, None, None, None, None, None, None,
        );
        let err = provider.list_models(None).await.unwrap_err();
        assert!(err.to_string().contains("Missing MinimaxProvider API key"));
    }

    #[tokio::test]
    async fn test_complete_returns_placeholder() {
        let provider = MinimaxProvider::with_config(
            "key", None, None, None, None, None, None, None, None, None, None, None, None,
        );
        let response = provider
            .complete(
                &MetaCompletionRequest {
                    prompt: "hi".to_string(),
                    max_tokens: None,
                    temperature: None,
                },
                None,
            )
            .await
            .unwrap();
        assert!(response.text.contains("MinimaxProvider completion not implemented"));
    }

    #[tokio::test]
    async fn test_embed_not_supported() {
        let provider = MinimaxProvider::with_config(
            "key", None, None, None, None, None, None, None, None, None, None, None, None,
        );
        let err = provider.embed(vec!["hello".to_string()]).await.unwrap_err();
        assert!(err.to_string().contains("Embedding not supported"));
    }

    #[test]
    fn test_builder_requires_api_key() {
        let result = MetaBuilder::<MinimaxProvider>::new().build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("No API key provided for MinimaxProvider"));
    }

    #[test]
    fn test_builder_with_api_key() {
        let result = MetaBuilder::<MinimaxProvider>::new().api_key("test-key").build();
        assert!(result.is_ok());
        let provider = result.unwrap();
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.model, MiniMaxConfig::DEFAULT_MODEL);
    }

    #[test]
    fn test_builder_with_highspeed_model() {
        let result = MetaBuilder::<MinimaxProvider>::new()
            .api_key("test-key")
            .model("MinimaxProvider-M2.5-highspeed")
            .build();
        assert!(result.is_ok());
        let provider = result.unwrap();
        assert_eq!(provider.model, "MinimaxProvider-M2.5-highspeed");
    }
}
