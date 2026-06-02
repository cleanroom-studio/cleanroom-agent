//! Behavior types.

use serde::{Deserialize, Serialize};

use super::reconstruction_policy::ElementOrigin;

/// Behavior — function signatures, flows, and state machines.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Behavior {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<FunctionSpec>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub flows: Option<Vec<FlowSpec>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_machines: Option<Vec<StateMachine>>,
}

/// Function specification with pseudocode logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSpec {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub inputs: Option<Vec<FunctionParam>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<FunctionParam>>,

    /// Pseudocode logic (S.DEF standard syntax).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logic: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub complexity: Option<String>,

    #[serde(default)]
    pub pure_function: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_cases: Option<Vec<EdgeCase>>,

    /// Reconstruction provenance (PTDL) — see [`ElementOrigin`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<ElementOrigin>,
}

/// A function parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParam {
    pub name: String,

    /// Logical type.
    #[serde(rename = "type")]
    pub param_type: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// An edge case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCase {
    pub condition: String,
    pub expected_behavior: String,
}

/// A multi-step flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowSpec {
    pub id: String,
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub participants: Option<Vec<FlowParticipant>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<FlowStep>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub compensation: Option<CompensationStrategy>,
}

/// A participant in a flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowParticipant {
    pub role: String,
    pub type_: String,
}

/// A step in a flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowStep {
    pub step: u32,
    pub actor: String,
    pub action: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_handling: Option<Vec<FlowErrorHandler>>,
}

/// Error handler for a flow step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowErrorHandler {
    pub condition: String,
    pub action: String,
}

/// Compensation strategy if rollback is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompensationStrategy {
    pub description: String,
    pub action: String,
}

/// A state machine for key entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachine {
    /// The entity this state machine applies to.
    pub entity: String,
    pub states: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub transitions: Option<Vec<StateTransition>>,
}

/// A state transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from: String,
    pub to: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard_conditions: Option<Vec<String>>,
}
