use crate::agent::base::MetaAgentType;
use crate::agent::error::{AgentBuildError, RunnableAgentError};
use crate::agent::task::MetaTask;
use crate::agent::{MetaAgentBuilder, MetaDeriveT, MetaExecutor, MetaHooks, MetaBaseAgent, MetaHookOutcome};
use crate::error::Error;
use cleanroom_meta_protocol::Event;
use futures::Stream;

use crate::agent::constants::DEFAULT_CHANNEL_BUFFER;

use crate::channel::{Receiver, Sender, channel};

#[cfg(not(target_arch = "wasm32"))]
use crate::event_fanout::EventFanout;
use crate::utils::{BoxEventStream, receiver_into_stream};
#[cfg(not(target_arch = "wasm32"))]
use futures_util::stream;

/// Marker type for direct (non-actor) agents.
///
/// Direct agents execute immediately within the caller's task without
/// requiring a runtime or event wiring. Use this for simple one-shot
/// invocations and unit tests.
pub struct MetaDirectAgent {}

impl MetaAgentType for MetaDirectAgent {
    fn type_name() -> &'static str {
        "direct_agent"
    }
}

/// Handle for a direct agent containing the agent instance and an event stream
/// receiver. Use `agent.run(...)` for one-shot calls or `agent.run_stream(...)`
/// to receive streaming outputs.
pub struct MetaDirectAgentHandle<T: MetaDeriveT + MetaExecutor + MetaHooks + Send + Sync> {
    pub agent: MetaBaseAgent<T, MetaDirectAgent>,
    pub rx: BoxEventStream<Event>,
    #[cfg(not(target_arch = "wasm32"))]
    fanout: Option<EventFanout>,
}

impl<T: MetaDeriveT + MetaExecutor + MetaHooks> MetaDirectAgentHandle<T> {
    pub fn new(agent: MetaBaseAgent<T, MetaDirectAgent>, rx: BoxEventStream<Event>) -> Self {
        Self {
            agent,
            rx,
            #[cfg(not(target_arch = "wasm32"))]
            fanout: None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn subscribe_events(&mut self) -> BoxEventStream<Event> {
        if let Some(fanout) = &self.fanout {
            return fanout.subscribe();
        }

        let stream = std::mem::replace(&mut self.rx, Box::pin(stream::empty::<Event>()));
        let fanout = EventFanout::new(stream, DEFAULT_CHANNEL_BUFFER);
        self.rx = fanout.subscribe();
        let stream = fanout.subscribe();
        self.fanout = Some(fanout);
        stream
    }
}

impl<T: MetaDeriveT + MetaExecutor + MetaHooks> MetaAgentBuilder<T, MetaDirectAgent> {
    /// Build the MetaBaseAgent and return a wrapper
    #[allow(clippy::result_large_err)]
    pub async fn build(self) -> Result<MetaDirectAgentHandle<T>, Error> {
        let llm = self.llm.ok_or(AgentBuildError::BuildFailure(
            "LLM provider is required".to_string(),
        ))?;
        let (tx, rx): (Sender<Event>, Receiver<Event>) = channel(DEFAULT_CHANNEL_BUFFER);
        let agent: MetaBaseAgent<T, MetaDirectAgent> =
            MetaBaseAgent::<T, MetaDirectAgent>::new(self.inner, llm, self.memory, tx, self.stream).await?;
        let stream = receiver_into_stream(rx);
        Ok(MetaDirectAgentHandle::new(agent, stream))
    }
}

impl<T: MetaDeriveT + MetaExecutor + MetaHooks> MetaBaseAgent<T, MetaDirectAgent> {
    /// Execute the agent for a single task and return the final agent output.
    pub async fn run(&self, task: MetaTask) -> Result<<T as MetaDeriveT>::Output, RunnableAgentError>
    where
        <T as MetaDeriveT>::Output: From<<T as MetaExecutor>::Output>,
        <T as MetaExecutor>::Error: Into<RunnableAgentError>,
    {
        let context = self.create_context();

        //Run Hook
        let hook_outcome = self.inner.on_run_start(&task, &context).await;
        match hook_outcome {
            MetaHookOutcome::Abort => return Err(RunnableAgentError::Abort),
            MetaHookOutcome::Continue => {}
        }

        // Execute the agent's logic using the executor
        match self.inner().execute(&task, context.clone()).await {
            Ok(output) => {
                let output: <T as MetaExecutor>::Output = output;

                //Extract Agent output into the desired type
                let agent_out: <T as MetaDeriveT>::Output = output.into();

                //Run On complete Hook
                self.inner
                    .on_run_complete(&task, &agent_out, &context)
                    .await;
                Ok(agent_out)
            }
            Err(e) => {
                // Send error event
                Err(e.into())
            }
        }
    }

    /// Execute the agent with streaming enabled and receive a stream of
    /// partial outputs which culminate in a final chunk with `done=true`.
    pub async fn run_stream(
        &self,
        task: MetaTask,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<<T as MetaDeriveT>::Output, Error>> + Send>>,
        RunnableAgentError,
    >
    where
        <T as MetaDeriveT>::Output: From<<T as MetaExecutor>::Output>,
        <T as MetaExecutor>::Error: Into<RunnableAgentError>,
    {
        let context = self.create_context();

        //Run Hook
        let hook_outcome = self.inner.on_run_start(&task, &context).await;
        match hook_outcome {
            MetaHookOutcome::Abort => return Err(RunnableAgentError::Abort),
            MetaHookOutcome::Continue => {}
        }

        // Execute the agent's streaming logic using the executor
        match self.inner().execute_stream(&task, context.clone()).await {
            Ok(stream) => {
                use futures::TryStreamExt;
                // Convert stream output/error without returning large Result err types from closures.
                let transformed_stream = stream
                    .map_ok(Into::into)
                    .map_err(Into::<RunnableAgentError>::into)
                    .map_err(Error::from);

                Ok(Box::pin(transformed_stream))
            }
            Err(e) => {
                // Send error event for stream creation failure
                Err(e.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hooks::MetaHookOutcome;
    use crate::agent::output::MetaOutputT;
    use crate::agent::prebuilt::executor::{
        MetaBasicAgent as StableBasicAgent, MetaBasicAgentOutput, ReActAgent as StableReActAgent,
        ReActAgentOutput,
    };
    use crate::agent::task::MetaTask;
    use crate::agent::{Context, ExecutorConfig};
    use crate::tests::{ConfigurableLLMProvider, MockAgentImpl, TestAgentOutput, TestError};
    use crate::tool::MetaToolT;
    use async_trait::async_trait;
    use futures::StreamExt;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    #[tokio::test]
    async fn test_direct_agent_build_requires_llm() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent");
        let err = match MetaAgentBuilder::<_, MetaDirectAgent>::new(mock_agent)
            .build()
            .await
        {
            Ok(_) => panic!("expected missing llm error"),
            Err(err) => err,
        };

        assert!(matches!(err, crate::error::Error::AgentBuildError(_)));
    }

    #[tokio::test]
    async fn test_direct_agent_run_success() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent");
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle = MetaAgentBuilder::<_, MetaDirectAgent>::new(mock_agent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = MetaTask::new("hello");
        let result = handle.agent.run(task).await.expect("run should succeed");
        assert_eq!(result.result, "Processed: hello");
    }

    #[tokio::test]
    async fn test_direct_agent_run_executor_error() {
        let mock_agent = MockAgentImpl::new("direct", "direct agent").with_failure(true);
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle = MetaAgentBuilder::<_, MetaDirectAgent>::new(mock_agent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = MetaTask::new("fail");
        let err = handle.agent.run(task).await.expect_err("expected error");
        assert!(matches!(err, RunnableAgentError::ExecutorError(_)));
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct HookCountOutput {
        result: String,
    }

    impl MetaOutputT for HookCountOutput {
        fn output_schema() -> &'static str {
            r#"{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]}"#
        }

        fn structured_output_format() -> Value {
            serde_json::json!({
                "name": "HookCountOutput",
                "description": "Hook count output",
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

    impl From<MetaBasicAgentOutput> for HookCountOutput {
        fn from(output: MetaBasicAgentOutput) -> Self {
            Self {
                result: output.response,
            }
        }
    }

    impl From<ReActAgentOutput> for HookCountOutput {
        fn from(output: ReActAgentOutput) -> Self {
            Self {
                result: output.response,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct CountingHookAgent {
        on_run_start_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl MetaDeriveT for CountingHookAgent {
        type Output = HookCountOutput;

        fn description(&self) -> &'static str {
            "counting hook agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(serde_json::json!({
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"]
            }))
        }

        fn name(&self) -> &'static str {
            "counting_hook_agent"
        }

        fn tools(&self) -> Vec<Box<dyn MetaToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl MetaHooks for CountingHookAgent {
        async fn on_run_start(&self, _task: &MetaTask, _ctx: &Context) -> MetaHookOutcome {
            self.on_run_start_calls.fetch_add(1, Ordering::SeqCst);
            MetaHookOutcome::Continue
        }
    }

    #[tokio::test]
    async fn test_direct_basic_agent_run_calls_on_run_start_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle =
            MetaAgentBuilder::<_, MetaDirectAgent>::new(StableBasicAgent::new(CountingHookAgent {
                on_run_start_calls: Arc::clone(&calls),
            }))
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = MetaTask::new("hello");
        let result = handle.agent.run(task).await.expect("run should succeed");

        assert_eq!(result.result, "Mock response");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_direct_react_agent_run_calls_on_run_start_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle =
            MetaAgentBuilder::<_, MetaDirectAgent>::new(StableReActAgent::new(CountingHookAgent {
                on_run_start_calls: Arc::clone(&calls),
            }))
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = MetaTask::new("hello");
        let result = handle.agent.run(task).await.expect("run should succeed");

        assert_eq!(result.result, "Mock response");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[derive(Clone, Debug)]
    struct StreamAgent;

    #[async_trait]
    impl MetaDeriveT for StreamAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "stream agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "stream_agent"
        }

        fn tools(&self) -> Vec<Box<dyn MetaToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl MetaExecutor for StreamAgent {
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
            Ok(TestAgentOutput {
                result: format!("Streamed: {}", task.prompt),
            })
        }
    }

    impl MetaHooks for StreamAgent {}

    #[tokio::test]
    async fn test_direct_agent_run_stream_default_executes_once() {
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle = MetaAgentBuilder::<_, MetaDirectAgent>::new(StreamAgent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = MetaTask::new("stream");
        let stream = handle
            .agent
            .run_stream(task)
            .await
            .expect("stream should succeed");
        let outputs: Vec<_> = stream.collect().await;
        assert_eq!(outputs.len(), 1);
        let output = outputs[0].as_ref().expect("expected Ok output");
        assert_eq!(output.result, "Streamed: stream");
    }

    #[derive(Debug)]
    struct AbortAgent {
        executed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl MetaDeriveT for AbortAgent {
        type Output = TestAgentOutput;

        fn description(&self) -> &'static str {
            "abort agent"
        }

        fn output_schema(&self) -> Option<Value> {
            Some(TestAgentOutput::structured_output_format())
        }

        fn name(&self) -> &'static str {
            "abort_agent"
        }

        fn tools(&self) -> Vec<Box<dyn MetaToolT>> {
            vec![]
        }
    }

    #[async_trait]
    impl MetaExecutor for AbortAgent {
        type Output = TestAgentOutput;
        type Error = TestError;

        fn config(&self) -> ExecutorConfig {
            ExecutorConfig::default()
        }

        async fn execute(
            &self,
            _task: &MetaTask,
            _context: Arc<Context>,
        ) -> Result<Self::Output, Self::Error> {
            self.executed.store(true, Ordering::SeqCst);
            Ok(TestAgentOutput {
                result: "should-not-run".to_string(),
            })
        }
    }

    #[async_trait]
    impl MetaHooks for AbortAgent {
        async fn on_run_start(&self, _task: &MetaTask, _ctx: &Context) -> MetaHookOutcome {
            MetaHookOutcome::Abort
        }
    }

    #[tokio::test]
    async fn test_direct_agent_run_aborts_before_execute() {
        let executed = Arc::new(AtomicBool::new(false));
        let agent = AbortAgent {
            executed: Arc::clone(&executed),
        };
        let llm = Arc::new(ConfigurableLLMProvider::default());
        let handle = MetaAgentBuilder::<_, MetaDirectAgent>::new(agent)
            .llm(llm)
            .build()
            .await
            .expect("build should succeed");

        let task = MetaTask::new("abort");
        let err = handle.agent.run(task).await.expect_err("expected abort");
        assert!(matches!(err, RunnableAgentError::Abort));
        assert!(!executed.load(Ordering::SeqCst));
    }
}
