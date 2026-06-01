//! Module for evaluating and comparing responses from multiple LLM providers.
//!
//! This module provides functionality to run the same prompt through multiple LLMs
//! and score their responses using custom evaluation functions.

mod parallel;
#[cfg(test)]
mod parallel_tests;

use crate::{MetaLlm, chat::MetaMessage, error::MetaError};

pub use parallel::{ParallelEvalResult, ParallelEvaluator};

/// Type alias for scoring functions that evaluate LLM responses
pub type ScoringFn = dyn Fn(&str) -> f32 + Send + Sync + 'static;

/// Evaluator for comparing responses from multiple LLM providers
pub struct LLMEvaluator {
    /// Collection of LLM providers to evaluate
    llms: Vec<Box<dyn MetaLlm>>,
    /// Optional scoring function to evaluate responses
    scorings_fns: Vec<Box<ScoringFn>>,
}

impl LLMEvaluator {
    /// Creates a new evaluator with the given LLM providers
    ///
    /// # Arguments
    /// * `llms` - Vector of LLM providers to evaluate
    pub fn new(llms: Vec<Box<dyn MetaLlm>>) -> Self {
        Self {
            llms,
            scorings_fns: Vec::new(),
        }
    }

    /// Adds a scoring function to evaluate LLM responses
    ///
    /// # Arguments
    /// * `f` - Function that takes a response string and returns a score
    pub fn scoring<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> f32 + Send + Sync + 'static,
    {
        self.scorings_fns.push(Box::new(f));
        self
    }

    /// Evaluates chat responses from all providers for the given messages
    ///
    /// # Arguments
    /// * `messages` - Chat messages to send to each provider
    ///
    /// # Returns
    /// Vector of evaluation results containing responses and scores
    pub async fn evaluate_chat(
        &self,
        messages: &[MetaMessage],
    ) -> Result<Vec<EvalResult>, MetaError> {
        let mut results = Vec::new();
        for llm in &self.llms {
            let response = llm.chat(messages, None).await?;
            let score = self.compute_score(&response.text().unwrap_or_default());
            results.push(EvalResult {
                text: response.text().unwrap_or_default(),
                score,
            });
        }
        Ok(results)
    }

    /// Computes the score for a given response
    ///
    /// # Arguments
    /// * `response` - The response to score
    ///
    /// # Returns
    /// The computed score
    fn compute_score(&self, response: &str) -> f32 {
        let mut total = 0.0;
        for sc in &self.scorings_fns {
            total += sc(response);
        }
        total
    }
}

/// Result of evaluating an LLM response
#[derive(Debug)]
pub struct EvalResult {
    /// The text response from the LLM
    pub text: String,
    /// Score assigned by the scoring function, if any
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolCall;
    use crate::chat::{
        MetaMessage, MetaProvider, MetaResponse, MetaRole, MessageType, MetaStructuredOutputFormat,
        Tool,
    };
    use crate::completion::{MetaCompletionProvider, MetaCompletionRequest, MetaCompletionResponse};
    use crate::embedding::MetaEmbeddingProvider;
    use crate::error::MetaError;
    use crate::models::MetaModelsProvider;
    use async_trait::async_trait;

    // Mock LLM Provider for testing
    struct MockLLMProvider {
        response_text: String,
        should_fail: bool,
    }

    impl MockLLMProvider {
        fn new(response_text: &str) -> Self {
            Self {
                response_text: response_text.to_string(),
                should_fail: false,
            }
        }

        fn with_failure(response_text: &str) -> Self {
            Self {
                response_text: response_text.to_string(),
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl MetaProvider for MockLLMProvider {
        async fn chat(
            &self,
            _messages: &[MetaMessage],
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<Box<dyn MetaResponse>, MetaError> {
            if self.should_fail {
                return Err(MetaError::ProviderError("Mock provider failed".to_string()));
            }
            Ok(Box::new(MockChatResponse {
                text: Some(self.response_text.clone()),
            }))
        }

        async fn chat_with_tools(
            &self,
            _messages: &[MetaMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<MetaStructuredOutputFormat>,
        ) -> Result<Box<dyn MetaResponse>, MetaError> {
            if self.should_fail {
                return Err(MetaError::ProviderError("Mock provider failed".to_string()));
            }
            Ok(Box::new(MockChatResponse {
                text: Some(self.response_text.clone()),
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
            if self.should_fail {
                return Err(MetaError::ProviderError("Mock provider failed".to_string()));
            }
            Ok(MetaCompletionResponse {
                text: self.response_text.clone(),
            })
        }
    }

    #[async_trait]
    impl MetaEmbeddingProvider for MockLLMProvider {
        async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, MetaError> {
            if self.should_fail {
                return Err(MetaError::ProviderError("Mock provider failed".to_string()));
            }
            Ok(vec![vec![0.1, 0.2, 0.3]])
        }
    }

    #[async_trait]
    impl MetaModelsProvider for MockLLMProvider {}

    impl MetaLlm for MockLLMProvider {}

    struct MockChatResponse {
        text: Option<String>,
    }

    impl MetaResponse for MockChatResponse {
        fn text(&self) -> Option<String> {
            self.text.clone()
        }

        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            None
        }
    }

    impl std::fmt::Debug for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockChatResponse")
        }
    }

    impl std::fmt::Display for MockChatResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.text.as_deref().unwrap_or(""))
        }
    }

    #[test]
    fn test_llm_evaluator_new() {
        let llm1 = Box::new(MockLLMProvider::new("response1")) as Box<dyn MetaLlm>;
        let llm2 = Box::new(MockLLMProvider::new("response2")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm1, llm2]);

        assert_eq!(evaluator.llms.len(), 2);
        assert_eq!(evaluator.scorings_fns.len(), 0);
    }

    #[test]
    fn test_llm_evaluator_with_scoring() {
        let llm = Box::new(MockLLMProvider::new("response")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm])
            .scoring(|response| response.len() as f32)
            .scoring(|response| if response.contains("good") { 10.0 } else { 0.0 });

        assert_eq!(evaluator.llms.len(), 1);
        assert_eq!(evaluator.scorings_fns.len(), 2);
    }

    #[test]
    fn test_llm_evaluator_compute_score() {
        let llm = Box::new(MockLLMProvider::new("response")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm])
            .scoring(|response| response.len() as f32)
            .scoring(|_| 5.0);

        let score = evaluator.compute_score("hello");
        assert_eq!(score, 10.0); // 5 (length) + 5 (fixed score)
    }

    #[test]
    fn test_llm_evaluator_compute_score_no_functions() {
        let llm = Box::new(MockLLMProvider::new("response")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm]);

        let score = evaluator.compute_score("hello");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_llm_evaluator_compute_score_complex() {
        let llm = Box::new(MockLLMProvider::new("response")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm])
            .scoring(|response| if response.contains("good") { 10.0 } else { 0.0 })
            .scoring(|response| if response.len() > 10 { 5.0 } else { 0.0 })
            .scoring(|response| response.chars().count() as f32 * 0.1);

        let score = evaluator.compute_score("this is a good response");
        assert_eq!(score, 17.3); // 10 (contains good) + 5 (length > 10) + 2.3 (char count * 0.1)
    }

    #[tokio::test]
    async fn test_llm_evaluator_evaluate_chat_single_provider() {
        let llm = Box::new(MockLLMProvider::new("test response")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm]).scoring(|response| response.len() as f32);

        let messages = vec![MetaMessage {
            role: MetaRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        }];

        let results = evaluator.evaluate_chat(&messages).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "test response");
        assert_eq!(results[0].score, 13.0); // "test response".len()
    }

    #[tokio::test]
    async fn test_llm_evaluator_evaluate_chat_multiple_providers() {
        let llm1 = Box::new(MockLLMProvider::new("short")) as Box<dyn MetaLlm>;
        let llm2 =
            Box::new(MockLLMProvider::new("this is a longer response")) as Box<dyn MetaLlm>;
        let evaluator =
            LLMEvaluator::new(vec![llm1, llm2]).scoring(|response| response.len() as f32);

        let messages = vec![MetaMessage {
            role: MetaRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        }];

        let results = evaluator.evaluate_chat(&messages).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].text, "short");
        assert_eq!(results[0].score, 5.0);
        assert_eq!(results[1].text, "this is a longer response");
        assert_eq!(results[1].score, 25.0);
    }

    #[tokio::test]
    async fn test_llm_evaluator_evaluate_chat_with_failure() {
        let llm =
            Box::new(MockLLMProvider::with_failure("failed response")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm]);

        let messages = vec![MetaMessage {
            role: MetaRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        }];

        let result = evaluator.evaluate_chat(&messages).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Mock provider failed")
        );
    }

    #[tokio::test]
    async fn test_llm_evaluator_evaluate_chat_no_scoring() {
        let llm = Box::new(MockLLMProvider::new("response")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm]);

        let messages = vec![MetaMessage {
            role: MetaRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        }];

        let results = evaluator.evaluate_chat(&messages).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "response");
        assert_eq!(results[0].score, 0.0);
    }

    #[tokio::test]
    async fn test_llm_evaluator_evaluate_chat_empty_response() {
        let llm = Box::new(MockLLMProvider::new("")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm])
            .scoring(|response| if response.is_empty() { -1.0 } else { 1.0 });

        let messages = vec![MetaMessage {
            role: MetaRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        }];

        let results = evaluator.evaluate_chat(&messages).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "");
        assert_eq!(results[0].score, -1.0);
    }

    #[tokio::test]
    async fn test_llm_evaluator_evaluate_chat_complex_messages() {
        let llm = Box::new(MockLLMProvider::new("assistant response")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm]).scoring(|response| response.len() as f32);

        let messages = vec![
            MetaMessage {
                role: MetaRole::System,
                message_type: MessageType::Text,
                content: "You are a helpful assistant".to_string(),
            },
            MetaMessage {
                role: MetaRole::User,
                message_type: MessageType::Text,
                content: "Hello, how are you?".to_string(),
            },
            MetaMessage {
                role: MetaRole::Assistant,
                message_type: MessageType::Text,
                content: "I'm doing well, thank you!".to_string(),
            },
        ];

        let results = evaluator.evaluate_chat(&messages).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "assistant response");
        assert_eq!(results[0].score, 18.0);
    }

    #[test]
    fn test_eval_result_creation() {
        let result = EvalResult {
            text: "test response".to_string(),
            score: 15.5,
        };

        assert_eq!(result.text, "test response");
        assert_eq!(result.score, 15.5);
    }

    #[test]
    fn test_eval_result_with_negative_score() {
        let result = EvalResult {
            text: "bad response".to_string(),
            score: -5.0,
        };

        assert_eq!(result.text, "bad response");
        assert_eq!(result.score, -5.0);
    }

    #[test]
    fn test_eval_result_with_zero_score() {
        let result = EvalResult {
            text: "neutral response".to_string(),
            score: 0.0,
        };

        assert_eq!(result.text, "neutral response");
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_scoring_function_types() {
        let llm = Box::new(MockLLMProvider::new("test")) as Box<dyn MetaLlm>;

        // Test different scoring function types
        let evaluator = LLMEvaluator::new(vec![llm])
            .scoring(|response| response.len() as f32)
            .scoring(|response| if response.contains("test") { 1.0 } else { 0.0 })
            .scoring(|response| response.chars().filter(|c| c.is_alphabetic()).count() as f32)
            .scoring(|response| response.to_lowercase().matches("a").count() as f32);

        let score = evaluator.compute_score("test data");
        assert_eq!(score, 20.0); // 9 (len) + 1 (contains test) + 8 (alphabetic) + 2 (contains 'a')
    }

    #[tokio::test]
    async fn test_llm_evaluator_mixed_providers() {
        let llm1 = Box::new(MockLLMProvider::new("good response")) as Box<dyn MetaLlm>;
        let llm2 = Box::new(MockLLMProvider::with_failure("bad response")) as Box<dyn MetaLlm>;
        let llm3 = Box::new(MockLLMProvider::new("another good response")) as Box<dyn MetaLlm>;

        let evaluator =
            LLMEvaluator::new(vec![llm1, llm2, llm3]).scoring(|response| response.len() as f32);

        let messages = vec![MetaMessage {
            role: MetaRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        }];

        let result = evaluator.evaluate_chat(&messages).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_llm_evaluator_empty_providers() {
        let evaluator = LLMEvaluator::new(vec![]);
        assert_eq!(evaluator.llms.len(), 0);
        assert_eq!(evaluator.scorings_fns.len(), 0);
    }

    #[tokio::test]
    async fn test_llm_evaluator_empty_providers_evaluation() {
        let evaluator = LLMEvaluator::new(vec![]);

        let messages = vec![MetaMessage {
            role: MetaRole::User,
            message_type: MessageType::Text,
            content: "test message".to_string(),
        }];

        let results = evaluator.evaluate_chat(&messages).await.unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_scoring_fn_type_alias() {
        // Test that we can use the ScoringFn type alias
        let _scoring_fn: Box<ScoringFn> = Box::new(|response| response.len() as f32);
        let _another_scoring_fn: Box<ScoringFn> =
            Box::new(
                |response| {
                    if response.contains("good") { 10.0 } else { 0.0 }
                },
            );
    }

    #[test]
    fn test_complex_scoring_logic() {
        let llm = Box::new(MockLLMProvider::new("response")) as Box<dyn MetaLlm>;
        let evaluator = LLMEvaluator::new(vec![llm]).scoring(|response| {
            // Complex scoring logic
            let mut score = 0.0;
            if response.len() > 5 {
                score += 5.0;
            }
            if response.contains("good") {
                score += 10.0;
            }
            if response.chars().any(|c| c.is_uppercase()) {
                score += 2.0;
            }
            score
        });

        let score1 = evaluator.compute_score("short");
        assert_eq!(score1, 0.0);

        let score2 = evaluator.compute_score("this is a good response");
        assert_eq!(score2, 15.0);

        let score3 = evaluator.compute_score("Good Response");
        assert_eq!(score3, 7.0);
    }
}
