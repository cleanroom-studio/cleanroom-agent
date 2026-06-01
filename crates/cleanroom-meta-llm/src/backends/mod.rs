//! Backend implementations for the LLM providers we support.
//!
//! As of v0.1, only three backends are vendored: OpenAiProvider (covers any
//! OpenAiProvider-compatible endpoint, including MinimaxProvider's openai-compatible API),
//! AnthropicProvider (Claude), and MinimaxProvider (first-class MinimaxProvider provider).

pub mod openai;
pub mod anthropic;
pub mod minimax;
