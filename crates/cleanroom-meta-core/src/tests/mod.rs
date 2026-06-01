//! Shared test fixtures for `cleanroom-meta-core`.
//!
//! In the upstream `autoagents-core 0.3.7` crate this lives in
//! `src/tests/mod.rs` and exposes shared `MockAgentImpl`,
//! `ConfigurableLLMProvider`, and `StaticChatResponse` mocks that the
//! ReAct executor tests reuse. The vendoring commit only copied the
//! `mod tests;` declaration from `lib.rs`; this file is the re-created
//! body, with the same renames applied as everywhere else under
//! `cleanroom-meta-*` (e.g. `AgentDeriveT` → `MetaDeriveT`,
//! `LLMProvider` → `MetaLlm`, `autoagents_llm` → `cleanroom_meta_llm`).
//!
//! The original also bundled two submodules
//! (`actor_integration_tests`, `agent_integration_tests`) that exercised
//! the runtime end-to-end. They are *not* part of the vendored crate
//! (no other vendored file imports them), so we omit them here.

use async_trait::async_trait;
use futures::Stream;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::MetaHooks;
use crate::agent::error::RunnableAgentError;
use crate::agent::task::MetaTask;
use crate::agent::{Context, ExecutorConfig, MetaDeriveT, MetaExecutor, MetaOutputT};
use crate::tool::MetaToolT;
use cleanroom_meta_llm::builder::MetaBackend;
use cleanroom_meta_llm::{
    MetaLlm, ToolCall,
    chat::{
        MetaMessage, MetaProvider, MetaResponse, MetaStructuredOutputFormat, StreamChunk,
        StreamResponse, Tool, Usage,
    },
    completion::{MetaCompletionProvider, MetaCompletionRequest, MetaCompletionResponse},
    embedding::MetaEmbeddingProvider,
    error::MetaError,
    models::{
        MetaModelsProvider, ModelListRequest, ModelListResponse, StandardModelEntry,
        StandardModelListResponse, StandardModelListResponseInner,
    },
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum TestError {
    #[error("Test error: {0}")]
    ExecutionFailed(String),
}

impl From<TestError> for RunnableAgentError {
    fn from(error: TestError) -> Self {
        RunnableAgentError::ExecutorError(error.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TestAgentOutput {
    pub(crate) result: String,
}

impl MetaOutputT for TestAgentOutput {
    fn output_schema() -> &'static str {
        r#"{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]}"#
    }

    fn structured_output_format() -> Value {
        serde_json::json!({
            "name": "TestAgentOutput",
            "description": "Test agent output schema",
            "schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                },
                "required": ["result"]
            },
            "strict": true
        })
    }
}

impl From<TestAgentOutput> for Value {
    fn from(output: TestAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MockAgentImpl {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) should_fail: bool,
}

impl MockAgentImpl {
    pub(crate) fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            should_fail: false,
        }
    }

    pub(crate) fn with_failure(mut self, should_fail: bool) -> Self {
        self.should_fail = should_fail;
        self
    }
}

#[async_trait]
impl MetaDeriveT for MockAgentImpl {
    type Output = TestAgentOutput;

    fn description(&self) -> &'static str {
        Box::leak(self.description.clone().into_boxed_str())
    }

    fn output_schema(&self) -> Option<Value> {
        Some(TestAgentOutput::structured_output_format())
    }

    fn name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn tools(&self) -> Vec<Box<dyn MetaToolT>> {
        vec![]
    }
}

#[async_trait]
impl MetaExecutor for MockAgentImpl {
    type Output = TestAgentOutput;
    type Error = TestError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig::default()
    }

    async fn execute(
        &self,
        task: &MetaTask,
        _context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        if self.should_fail {
            return Err(TestError::ExecutionFailed(
                "Mock execution failed".to_string(),
            ));
        }

        Ok(TestAgentOutput {
            result: format!("Processed: {}", task.prompt),
        })
    }

    async fn execute_stream(
        &self,
        _task: &MetaTask,
        _context: Arc<Context>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>, Self::Error>
    {
        unimplemented!()
    }
}

impl MetaHooks for MockAgentImpl {}

#[derive(Debug, Clone)]
pub(crate) struct StaticChatResponse {
    pub(crate) text: Option<String>,
    pub(crate) tool_calls: Option<Vec<ToolCall>>,
    pub(crate) usage: Option<Usage>,
    pub(crate) thinking: Option<String>,
}

impl std::fmt::Display for StaticChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(text) = &self.text {
            write!(f, "{text}")
        } else {
            write!(f, "")
        }
    }
}

impl MetaResponse for StaticChatResponse {
    fn text(&self) -> Option<String> {
        self.text.clone()
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.tool_calls.clone()
    }

    fn thinking(&self) -> Option<String> {
        self.thinking.clone()
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ConfigurableLLMProvider {
    pub(crate) chat_response: StaticChatResponse,
    pub(crate) stream_chunks: Vec<StreamChunk>,
    pub(crate) structured_stream: Vec<StreamResponse>,
    pub(crate) completion_response: MetaCompletionResponse,
    pub(crate) embeddings: Vec<Vec<f32>>,
    pub(crate) models: Vec<String>,
}

impl Default for ConfigurableLLMProvider {
    fn default() -> Self {
        Self {
            chat_response: StaticChatResponse {
                text: Some("Mock response".to_string()),
                tool_calls: None,
                usage: None,
                thinking: None,
            },
            stream_chunks: Vec::new(),
            structured_stream: Vec::new(),
            completion_response: MetaCompletionResponse {
                text: "Mock completion".to_string(),
            },
            embeddings: vec![vec![0.1, 0.2, 0.3]],
            models: vec!["test-model".to_string()],
        }
    }
}

#[async_trait]
impl MetaProvider for ConfigurableLLMProvider {
    async fn chat(
        &self,
        _messages: &[MetaMessage],
        _json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Box<dyn MetaResponse>, MetaError> {
        Ok(Box::new(self.chat_response.clone()))
    }

    async fn chat_with_tools(
        &self,
        _messages: &[MetaMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Box<dyn MetaResponse>, MetaError> {
        Ok(Box::new(self.chat_response.clone()))
    }

    async fn chat_stream_struct(
        &self,
        _messages: &[MetaMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, MetaError>> + Send>>, MetaError>
    {
        let stream = stream::iter(self.structured_stream.clone().into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    async fn chat_stream_with_tools(
        &self,
        _messages: &[MetaMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, MetaError>> + Send>>, MetaError> {
        let stream = stream::iter(self.stream_chunks.clone().into_iter().map(Ok));
        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl MetaCompletionProvider for ConfigurableLLMProvider {
    async fn complete(
        &self,
        _req: &MetaCompletionRequest,
        _json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<MetaCompletionResponse, MetaError> {
        Ok(self.completion_response.clone())
    }
}

#[async_trait]
impl MetaEmbeddingProvider for ConfigurableLLMProvider {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
        Ok(self.embeddings.clone())
    }
}

#[async_trait]
impl MetaModelsProvider for ConfigurableLLMProvider {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, MetaError> {
        let data = self
            .models
            .iter()
            .cloned()
            .map(|id| StandardModelEntry {
                id,
                created: None,
                extra: Value::Null,
            })
            .collect::<Vec<_>>();
        let response = StandardModelListResponse {
            inner: StandardModelListResponseInner { data },
            backend: MetaBackend::OpenAI,
        };
        Ok(Box::new(response))
    }
}

impl MetaLlm for ConfigurableLLMProvider {}

pub(crate) struct MockLLMProvider;

#[async_trait]
impl MetaProvider for MockLLMProvider {
    async fn chat(
        &self,
        _messages: &[MetaMessage],
        _json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Box<dyn MetaResponse>, MetaError> {
        Ok(Box::new(StaticChatResponse {
            text: Some("Mock response".to_string()),
            tool_calls: None,
            usage: None,
            thinking: None,
        }))
    }

    async fn chat_with_tools(
        &self,
        _messages: &[MetaMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<Box<dyn MetaResponse>, MetaError> {
        Ok(Box::new(StaticChatResponse {
            text: Some("Mock response".to_string()),
            tool_calls: None,
            usage: None,
            thinking: None,
        }))
    }
}

#[async_trait]
impl MetaCompletionProvider for MockLLMProvider {
    async fn complete(
        &self,
        _req: &MetaCompletionRequest,
        _json_schema: Option<MetaStructuredOutputFormat>,
    ) -> Result<MetaCompletionResponse, MetaError> {
        Ok(MetaCompletionResponse {
            text: "Mock completion".to_string(),
        })
    }
}

#[async_trait]
impl MetaEmbeddingProvider for MockLLMProvider {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
        Ok(vec![vec![0.1, 0.2, 0.3]])
    }
}

#[async_trait]
impl MetaModelsProvider for MockLLMProvider {}

impl MetaLlm for MockLLMProvider {}
