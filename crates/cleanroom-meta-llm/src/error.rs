use std::fmt;

/// Phase where a guardrail violation occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailPhase {
    Input,
    Output,
}

/// Error types that can occur when interacting with LLM providers.
#[derive(Debug, Clone)]
pub enum MetaError {
    /// HTTP request/response errors
    HttpError(String),
    /// Authentication and authorization errors
    AuthError(String),
    /// Invalid request parameters or format
    InvalidRequest(String),
    /// Errors returned by the LLM provider
    ProviderError(String),
    /// API response parsing or format error
    ResponseFormatError {
        message: String,
        raw_response: String,
    },
    /// Generic error
    Generic(String),
    /// JSON serialization/deserialization errors
    JsonError(String),
    /// Tool configuration error
    ToolConfigError(String),
    /// No Tool Support
    NoToolSupport(String),
    /// Guardrail blocked the request/response
    GuardrailBlocked {
        phase: GuardrailPhase,
        guard: Box<str>,
        rule_id: Box<str>,
        category: Box<str>,
        severity: Box<str>,
        message: Box<str>,
    },
    /// Guardrail execution failed unexpectedly
    GuardrailExecutionFailed { guard: String, message: String },
}

impl fmt::Display for MetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaError::HttpError(e) => write!(f, "HTTP Error: {e}"),
            MetaError::AuthError(e) => write!(f, "Auth Error: {e}"),
            MetaError::InvalidRequest(e) => write!(f, "Invalid Request: {e}"),
            MetaError::ProviderError(e) => write!(f, "Provider Error: {e}"),
            MetaError::Generic(e) => write!(f, "Generic Error : {e}"),
            MetaError::ResponseFormatError {
                message,
                raw_response,
            } => {
                write!(
                    f,
                    "Response Format Error: {message}. Raw response: {raw_response}"
                )
            }
            MetaError::JsonError(e) => write!(f, "JSON Parse Error: {e}"),
            MetaError::ToolConfigError(e) => write!(f, "Tool Configuration Error: {e}"),
            MetaError::NoToolSupport(e) => write!(f, "No Tool Support: {e}"),
            MetaError::GuardrailBlocked {
                phase,
                guard,
                rule_id,
                category,
                severity,
                message,
            } => {
                let phase = match phase {
                    GuardrailPhase::Input => "input",
                    GuardrailPhase::Output => "output",
                };
                write!(
                    f,
                    "guardrail blocked {phase}: guard={guard}, rule={rule_id}, category={category}, severity={severity}, message={message}"
                )
            }
            MetaError::GuardrailExecutionFailed { guard, message } => {
                write!(
                    f,
                    "guardrail execution failed: guard={guard}, error={message}"
                )
            }
        }
    }
}

impl std::error::Error for MetaError {}

/// Converts reqwest HTTP errors into LLMErrors
#[cfg(not(target_arch = "wasm32"))]
impl From<reqwest::Error> for MetaError {
    fn from(err: reqwest::Error) -> Self {
        MetaError::HttpError(err.to_string())
    }
}

impl From<serde_json::Error> for MetaError {
    fn from(err: serde_json::Error) -> Self {
        MetaError::JsonError(format!(
            "{} at line {} column {}",
            err,
            err.line(),
            err.column()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Error as JsonError;
    use std::error::Error;

    #[test]
    fn test_llm_error_display_http_error() {
        let error = MetaError::HttpError("Connection failed".to_string());
        assert_eq!(error.to_string(), "HTTP Error: Connection failed");
    }

    #[test]
    fn test_llm_error_display_auth_error() {
        let error = MetaError::AuthError("Invalid API key".to_string());
        assert_eq!(error.to_string(), "Auth Error: Invalid API key");
    }

    #[test]
    fn test_llm_error_display_invalid_request() {
        let error = MetaError::InvalidRequest("Missing required parameter".to_string());
        assert_eq!(
            error.to_string(),
            "Invalid Request: Missing required parameter"
        );
    }

    #[test]
    fn test_llm_error_display_provider_error() {
        let error = MetaError::ProviderError("Model not found".to_string());
        assert_eq!(error.to_string(), "Provider Error: Model not found");
    }

    #[test]
    fn test_llm_error_display_generic_error() {
        let error = MetaError::Generic("Something went wrong".to_string());
        assert_eq!(error.to_string(), "Generic Error : Something went wrong");
    }

    #[test]
    fn test_llm_error_display_response_format_error() {
        let error = MetaError::ResponseFormatError {
            message: "Invalid JSON".to_string(),
            raw_response: "{invalid json}".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Response Format Error: Invalid JSON. Raw response: {invalid json}"
        );
    }

    #[test]
    fn test_llm_error_display_json_error() {
        let error = MetaError::JsonError("Parse error at line 5 column 10".to_string());
        assert_eq!(
            error.to_string(),
            "JSON Parse Error: Parse error at line 5 column 10"
        );
    }

    #[test]
    fn test_llm_error_display_tool_config_error() {
        let error = MetaError::ToolConfigError("Invalid tool configuration".to_string());
        assert_eq!(
            error.to_string(),
            "Tool Configuration Error: Invalid tool configuration"
        );
    }

    #[test]
    fn test_llm_error_is_error_trait() {
        let error = MetaError::Generic("test error".to_string());
        assert!(error.source().is_none());
    }

    #[test]
    fn test_llm_error_debug_format() {
        let error = MetaError::HttpError("test".to_string());
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("HttpError"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_from_reqwest_error() {
        // Create a mock reqwest error by trying to make a request to an invalid URL
        let client = reqwest::Client::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let reqwest_error = rt
            .block_on(async {
                client
                    .get("https://invalid-url-that-does-not-exist-12345.com/")
                    .timeout(std::time::Duration::from_millis(100))
                    .send()
                    .await
            })
            .unwrap_err();

        let llm_error: MetaError = reqwest_error.into();

        match llm_error {
            MetaError::HttpError(msg) => {
                assert!(!msg.is_empty());
            }
            _ => panic!("Expected HttpError"),
        }
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_str = r#"{"invalid": json}"#;
        let json_error: JsonError =
            serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();

        let llm_error: MetaError = json_error.into();

        match llm_error {
            MetaError::JsonError(msg) => {
                assert!(msg.contains("line"));
                assert!(msg.contains("column"));
            }
            _ => panic!("Expected JsonError"),
        }
    }

    #[test]
    fn test_error_variants_equality() {
        let error1 = MetaError::HttpError("test".to_string());
        let error2 = MetaError::HttpError("test".to_string());
        let error3 = MetaError::HttpError("different".to_string());
        let error4 = MetaError::AuthError("test".to_string());

        // Note: MetaError doesn't implement PartialEq, so we test via string representation
        assert_eq!(error1.to_string(), error2.to_string());
        assert_ne!(error1.to_string(), error3.to_string());
        assert_ne!(error1.to_string(), error4.to_string());
    }

    #[test]
    fn test_response_format_error_fields() {
        let error = MetaError::ResponseFormatError {
            message: "Parse failed".to_string(),
            raw_response: "raw content".to_string(),
        };

        let display_str = error.to_string();
        assert!(display_str.contains("Parse failed"));
        assert!(display_str.contains("raw content"));
    }

    #[test]
    fn test_all_error_variants_have_display() {
        let errors = vec![
            MetaError::HttpError("http".to_string()),
            MetaError::AuthError("auth".to_string()),
            MetaError::InvalidRequest("invalid".to_string()),
            MetaError::ProviderError("provider".to_string()),
            MetaError::Generic("generic".to_string()),
            MetaError::ResponseFormatError {
                message: "format".to_string(),
                raw_response: "raw".to_string(),
            },
            MetaError::JsonError("json".to_string()),
            MetaError::ToolConfigError("tool".to_string()),
        ];

        for error in errors {
            let display_str = error.to_string();
            assert!(!display_str.is_empty());
        }
    }

    #[test]
    fn test_error_type_classification() {
        // Test that we can pattern match on different error types
        let http_error = MetaError::HttpError("test".to_string());
        match http_error {
            MetaError::HttpError(_) => {}
            _ => panic!("Expected HttpError"),
        }

        let auth_error = MetaError::AuthError("test".to_string());
        match auth_error {
            MetaError::AuthError(_) => {}
            _ => panic!("Expected AuthError"),
        }

        let response_error = MetaError::ResponseFormatError {
            message: "test".to_string(),
            raw_response: "test".to_string(),
        };
        match response_error {
            MetaError::ResponseFormatError { .. } => {}
            _ => panic!("Expected ResponseFormatError"),
        }
    }
}
