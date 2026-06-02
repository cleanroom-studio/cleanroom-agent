//! Reconstruction Policy (PTDL) types — cross-language/paradigm directives.
//!
//! PTDL (Paradigm Translation & Decoupling Layer) tells the consumer
//! agent which S.DEF elements are source-language implementation
//! artifacts and should be omitted, replaced, or translated when
//! rebuilding in a different language or paradigm.
//!
//! See `docs/19-reconstruction-quality.md` for the full design and
//! `S.DEF/proposals/0000-ptdl-reconstruction-policy.md` for the
//! specification proposal.
//!
//! # Four-tier classification
//!
//! Each S.DEF element carries an [`ElementOrigin`] that places it in
//! one of four tiers:
//!
//! | Tier | Variant                | Consumer action |
//! |------|------------------------|-----------------|
//! | A    | `BehaviorContract`     | Preserve 1:1 (signatures, semantics, error cases) |
//! | B    | `Algorithm`            | Keep algorithm body, swap data structures |
//! | C    | `Idiom`                | Translate to target-language idioms |
//! | D    | `Incidental`           | Omit — let the target language fill the gap |

use serde::{Deserialize, Serialize};

/// The top-level reconstruction policy block on `SoftwareDefinition`.
///
/// All fields are optional. When the entire block is absent the
/// consumer agent should default to a same-language, same-paradigm
/// translation (i.e. behave as if PTDL were not in use).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReconstructionPolicy {
    /// Default strategy for Tier C elements when no per-element override exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_tier_c_strategy: Option<TierStrategy>,

    /// Default strategy for Tier D elements when no per-element override exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_tier_d_strategy: Option<TierStrategy>,

    /// Source-language paradigm fingerprint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_paradigm: Option<ParadigmMetadata>,

    /// Target-language paradigm fingerprint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_paradigm: Option<ParadigmMetadata>,

    /// Library substitution suggestions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library_substitutions: Option<Vec<LibrarySubstitution>>,

    /// Paradigm translation rules.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transformation_hints: Option<Vec<TransformationHint>>,

    /// Whether the consumer agent may introduce new dependencies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_extra_dependencies: Option<bool>,

    /// Whether the consumer agent may alter externally observable behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_behavior_drift: Option<bool>,
}

/// What the consumer agent should do with elements of a given tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierStrategy {
    /// Rewrite using target-language idioms.
    Translate,
    /// Keep the source-language structure as-is.
    Preserve,
    /// Drop entirely.
    Omit,
}

/// Language paradigm fingerprint — describes how a language "thinks."
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParadigmMetadata {
    /// Primary paradigm.
    pub primary: ParadigmPrimary,

    /// Optional secondary paradigm (e.g. Kotlin: oop + functional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secondary: Option<ParadigmSecondary>,

    /// Memory management model.
    pub memory_model: MemoryModel,

    /// Type system strength.
    pub type_system: TypeSystem,

    /// Default error-propagation idiom.
    pub error_handling: ErrorHandling,

    /// Concurrency model.
    pub concurrency: ConcurrencyModel,

    /// Whether the language has first-class functions / closures.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_first_class_functions: Option<bool>,

    /// Whether the language has syntactic macros / compile-time code generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_macros: Option<bool>,

    /// How unsafe operations are expressed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unsafe_constructs: Option<UnsafeConstructs>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParadigmPrimary {
    Imperative,
    ObjectOriented,
    Functional,
    Logic,
    Procedural,
    Scripting,
    Systems,
    #[default]
    Reactive,
    ConcurrentActor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParadigmSecondary {
    ObjectOriented,
    Functional,
    #[default]
    Generic,
    Reflection,
    Metaprogramming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryModel {
    Manual,
    #[default]
    Gc,
    Rc,
    Ownership,
    Region,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeSystem {
    Dynamic,
    StaticWeak,
    #[default]
    StaticStrong,
    StaticDependent,
    Gradual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorHandling {
    #[default]
    Exceptions,
    ErrorCodes,
    ResultType,
    Panics,
    Longjmp,
    MultipleValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcurrencyModel {
    #[default]
    Threads,
    AsyncAwait,
    Actors,
    Goroutines,
    EventLoop,
    SingleThreaded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnsafeConstructs {
    ExplicitUnsafe,
    ImplicitEverywhere,
    /// The S.DEF value `"none"` is used to express "the language does not
    /// expose unsafe operations at all". Named `NoUnsafe` here to avoid
    /// the `Option::None` collision in derive macros; serialized as `"none"`.
    #[serde(rename = "none")]
    #[default]
    NoUnsafe,
}

/// Per-element reconstruction provenance — the four-tier classification.
///
/// The Producer attaches this to every entity it extracts. The Consumer
/// reads it to decide what to generate, replace, or omit.
///
/// # Confidence
///
/// `confidence` is a 0–100 integer from the Producer's classifier.
/// Values < 60 are advisory only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementOrigin {
    /// Tier classification.
    pub reconstruction_class: ReconstructionClass,

    /// Producer-inferred confidence in the 0–100 range.
    pub confidence: u8,

    /// Evidence used to make the classification.
    pub evidence: Vec<String>,

    /// Human-readable rationale.
    pub rationale: String,

    /// Optional per-element override of the document-level default strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub override_strategy: Option<TierStrategy>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionClass {
    /// Tier A — public API, wire format. Preserve 1:1.
    BehaviorContract,
    /// Tier B — algorithm kernel. Keep algorithm, swap data structures.
    Algorithm,
    /// Tier C — implementation idiom. Translate to target-language idiom.
    Idiom,
    /// Tier D — source-language workaround. Omit in target.
    Incidental,
}

impl ReconstructionClass {
    /// String form matching the S.DEF JSON schema.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BehaviorContract => "behavior_contract",
            Self::Algorithm => "algorithm",
            Self::Idiom => "idiom",
            Self::Incidental => "incidental",
        }
    }

    /// Default consumer action for elements of this tier when no override is set.
    pub fn default_strategy(self) -> TierStrategy {
        match self {
            Self::BehaviorContract => TierStrategy::Preserve,
            Self::Algorithm => TierStrategy::Translate,
            Self::Idiom => TierStrategy::Translate,
            Self::Incidental => TierStrategy::Omit,
        }
    }
}

/// Suggestion to replace an original implementation with a library in the
/// target ecosystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibrarySubstitution {
    /// Unique ID (also used in the symbol registry).
    pub id: String,

    /// Signature of the function being replaced, in target-language notation.
    pub function_signature: String,

    /// What the original software actually used.
    pub original_implementation: OriginalImplementation,

    /// Ranked candidates.
    pub candidates: Vec<LibraryCandidate>,

    /// Optional selection rule (DSL string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection_rule: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginalImplementation {
    /// Display name (e.g. "SHA1", "zmalloc").
    pub name: String,

    /// Source file (e.g. "src/sha1.c").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,

    /// Lines of code in the original implementation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines_of_code: Option<u32>,

    /// SPDX license identifier of the original.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

/// A single library candidate for a substitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryCandidate {
    /// Target package ecosystem.
    pub ecosystem: Ecosystem,

    /// Package / crate / module name. For `std`, the full path.
    pub name: String,

    /// Version constraint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Why this candidate is recommended.
    pub rationale: String,

    /// Known risks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risks: Option<Vec<String>>,

    /// Trust score in 0–100 (integer). Higher = stronger preference.
    pub trust: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ecosystem {
    RustCrate,
    Npm,
    Pypi,
    Maven,
    GoModule,
    JavaJar,
    DotnetNuget,
    Std,
}

/// A source-pattern → target-pattern translation rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationHint {
    /// Unique ID.
    pub id: String,

    /// Source-side pattern (DSL).
    pub source_pattern: String,

    /// Target-side pattern (DSL).
    pub target_pattern: String,

    /// Human-readable transformation rule + pseudocode.
    pub transformation: String,

    /// Paradigm pairs this rule applies to.
    pub applies_to: TransformationParadigmScope,

    /// Worked examples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<TransformationExample>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationParadigmScope {
    /// Source paradigm primary values.
    pub source: Vec<String>,

    /// Target paradigm primary values.
    pub target: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationExample {
    /// S.DEF excerpt of the source pattern.
    pub source_sdef_excerpt: String,

    /// Generated target code.
    pub target_code: String,
}
