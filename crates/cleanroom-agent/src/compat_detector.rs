//! Compatibility Layer Detector.
//!
//! Scans source code for compatibility patterns across six recognition features:
//! 1. Version Bridging — code mapping between API versions
//! 2. Deprecation Wrappers — deprecated functions wrapping new implementations
//! 3. Adapter/Transformer — classes transforming one interface to another
//! 4. Conditional Version Logic — runtime version checks
//! 5. Compatibility Annotations — `@Deprecated`, `#[deprecated]`, etc.
//! 6. Pure Forwarding — methods that delegate without modification
//!
//! Produces a `CompatibilityModule` with confidence scoring.

use sdef_core::CompatibilityModule;

/// A detected compatibility pattern.
#[derive(Debug, Clone)]
pub struct CompatPattern {
    /// Human-readable description.
    pub description: String,
    /// Pattern category.
    pub category: CompatCategory,
    /// Confidence score (0.0-1.0).
    pub confidence: f64,
    /// Source locations where the pattern was found.
    pub locations: Vec<String>,
}

/// Category of compatibility pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatCategory {
    VersionBridging,
    DeprecationWrapper,
    AdapterTransformer,
    ConditionalVersion,
    CompatibilityAnnotation,
    PureForwarding,
}

impl CompatCategory {
    pub fn name(self) -> &'static str {
        match self {
            Self::VersionBridging => "version_bridging",
            Self::DeprecationWrapper => "deprecation_wrapper",
            Self::AdapterTransformer => "adapter_transformer",
            Self::ConditionalVersion => "conditional_version",
            Self::CompatibilityAnnotation => "compatibility_annotation",
            Self::PureForwarding => "pure_forwarding",
        }
    }
}

/// Detection result.
#[derive(Debug, Clone, Default)]
pub struct DetectionResult {
    /// All detected patterns.
    pub patterns: Vec<CompatPattern>,
    /// Aggregate confidence score.
    pub overall_confidence: f64,
    /// Whether pure forwarding was the primary pattern found.
    pub is_primarily_forwarding: bool,
}

/// Detects compatibility layers in source code.
pub struct CompatDetector {
    /// Source files: (relative_path, content)
    files: Vec<(String, String)>,
}

impl CompatDetector {
    /// Create a new detector with source files.
    pub fn new(files: Vec<(String, String)>) -> Self {
        Self { files }
    }

    /// Run all detection checks.
    pub fn detect(&self) -> DetectionResult {
        let mut result = DetectionResult::default();
        let mut pure_forwarding_count = 0;

        for (path, content) in &self.files {
            // 1. Version Bridging
            result.patterns.extend(self.detect_version_bridging(path, content));

            // 2. Deprecation Wrappers
            result.patterns.extend(self.detect_deprecation_wrappers(path, content));

            // 3. Adapter/Transformer
            result.patterns.extend(self.detect_adapter_patterns(path, content));

            // 4. Conditional Version Logic
            result.patterns.extend(self.detect_conditional_version(path, content));

            // 5. Compatibility Annotations
            result.patterns.extend(self.detect_compat_annotations(path, content));

            // 6. Pure Forwarding
            let forwards = self.detect_pure_forwarding(path, content);
            pure_forwarding_count += forwards.len();
            result.patterns.extend(forwards);
        }

        // If pure forwarding dominates (>80% of patterns), flag it
        let total = result.patterns.len();
        if total > 0 {
            let forwarding_ratio = pure_forwarding_count as f64 / total as f64;
            result.is_primarily_forwarding = forwarding_ratio > 0.8;
        }

        // Calculate overall confidence (weighted average)
        if !result.patterns.is_empty() {
            let total_conf: f64 = result.patterns.iter().map(|p| p.confidence).sum();
            result.overall_confidence = total_conf / result.patterns.len() as f64;
        }

        result
    }

    /// 1. Version Bridging: detect functions with version-related names.
    fn detect_version_bridging(&self, path: &str, content: &str) -> Vec<CompatPattern> {
        let mut patterns = Vec::new();
        let bridge_keywords = [
            "toV", "fromV", "v1", "v2", "migrate", "upgrade", "downgrade",
            "_v1_", "_v2_", "compat", "legacy", "backward",
        ];
        let line_re = regex::Regex::new(r"(?m)^.*(fn |def |func |function |public |private ).*$")
            .unwrap();

        for cap in line_re.captures_iter(content) {
            let line = cap[0].to_lowercase();
            let match_count = bridge_keywords.iter().filter(|kw| line.contains(*kw)).count();
            if match_count >= 2 {
                patterns.push(CompatPattern {
                    description: format!("Version bridging function detected: {}", cap[0].trim()),
                    category: CompatCategory::VersionBridging,
                    confidence: 0.5 + (match_count as f64 * 0.1),
                    locations: vec![format!("{}:{}", path, line_find(content, cap[0].trim()))],
                });
            }
        }
        patterns
    }

    /// 2. Deprecation Wrappers: detect deprecated functions.
    fn detect_deprecation_wrappers(&self, path: &str, content: &str) -> Vec<CompatPattern> {
        let mut patterns = Vec::new();
        // Find deprecated functions that call another function
        let dep_re = regex::Regex::new(r"(?m)^.*(?:#\[deprecated|@Deprecated|# Deprecated|// Deprecated).*$")
            .unwrap();
        let call_re = regex::Regex::new(r"(?m)^\s*(?:pub\s+)?fn\s+\w+\s*\(.*\).*\{.*\}").unwrap();

        for cap in dep_re.captures_iter(content) {
            patterns.push(CompatPattern {
                description: format!("Deprecation annotation: {}", cap[0].trim()),
                category: CompatCategory::DeprecationWrapper,
                confidence: 0.8,
                locations: vec![format!("{}:{}", path, line_find(content, cap[0].trim()))],
            });
        }
        // Also check for functions that wrap deprecated calls
        for cap in call_re.captures_iter(content) {
            if content.contains("deprecated") {
                patterns.push(CompatPattern {
                    description: format!("Deprecation wrapper method: {}", cap[0].trim()),
                    category: CompatCategory::DeprecationWrapper,
                    confidence: 0.6,
                    locations: vec![format!("{}:{}", path, line_find(content, cap[0].trim()))],
                });
            }
        }
        patterns
    }

    /// 3. Adapter/Transformer: detect adapter pattern classes.
    fn detect_adapter_patterns(&self, path: &str, content: &str) -> Vec<CompatPattern> {
        let mut patterns = Vec::new();
        let adapter_keywords = ["adapter", "adapter", "wrapper", "translator", "transformer", "converter"];
        let lower = content.to_lowercase();

        let present: Vec<_> = adapter_keywords
            .iter()
            .filter(|kw| lower.contains(*kw))
            .collect();

        if !present.is_empty() {
            patterns.push(CompatPattern {
                description: format!(
                    "Adapter/Transformer pattern: contains {}",
                    present.iter().map(|s| *s).copied().collect::<Vec<_>>().join(", ")
                ),
                category: CompatCategory::AdapterTransformer,
                confidence: 0.5 + (present.len() as f64 * 0.1),
                locations: vec![path.to_string()],
            });
        }
        patterns
    }

    /// 4. Conditional Version Logic: detect runtime version checks.
    fn detect_conditional_version(&self, path: &str, content: &str) -> Vec<CompatPattern> {
        let mut patterns = Vec::new();
        let version_conditions = [
            r"if\s+.*version", r"if\s+.*api_version", r"if\s+.*VERSION",
            r"switch\s+.*version", r"match\s+.*version",
            r">=\s*.?\d+\.\d+", r"<=\s*.?\d+\.\d+",
        ];

        let re = regex::Regex::new(&version_conditions.join("|")).unwrap();
        for cap in re.captures_iter(content) {
            patterns.push(CompatPattern {
                description: format!("Conditional version logic: {}", cap[0].trim()),
                category: CompatCategory::ConditionalVersion,
                confidence: 0.7,
                locations: vec![format!("{}:{}", path, line_find(content, cap[0].trim()))],
            });
        }
        patterns
    }

    /// 5. Compatibility Annotations: find @Deprecated, #[deprecated], etc.
    fn detect_compat_annotations(&self, path: &str, content: &str) -> Vec<CompatPattern> {
        let mut patterns = Vec::new();
        let annot_re = regex::Regex::new(r"(?m)^\s*((?:#\[deprecated|@Deprecated|//?\s*deprecated|# deprecated))")
            .unwrap();

        for cap in annot_re.captures_iter(content) {
            patterns.push(CompatPattern {
                description: format!("Compatibility annotation: {}", cap[0].trim()),
                category: CompatCategory::CompatibilityAnnotation,
                confidence: 0.9,
                locations: vec![format!("{}:{}", path, line_find(content, cap[0].trim()))],
            });
        }
        patterns
    }

    /// 6. Pure Forwarding: methods that simply delegate to another.
    fn detect_pure_forwarding(&self, path: &str, content: &str) -> Vec<CompatPattern> {
        let mut patterns = Vec::new();
        let forward_re = regex::Regex::new(
            r"(?m)^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\([^)]*\)[^{]*\{[^}]*(\w+)\([^)]*\)[^}]*\}"
        ).unwrap();

        for cap in forward_re.captures_iter(content) {
            let fn_name = cap[1].to_string();
            let called = cap[2].to_string();
            // Only flag if the function name differs from what it calls
            if fn_name != called && !fn_name.starts_with("test") {
                patterns.push(CompatPattern {
                    description: format!("Pure forwarding: '{}' delegates to '{}'", fn_name, called),
                    category: CompatCategory::PureForwarding,
                    confidence: 0.5,
                    locations: vec![format!("{}:{}", path, line_find(content, cap[0].trim()))],
                });
            }
        }
        patterns
    }
}

/// Convert detection results to a CompatibilityModule for the S.DEF.
pub fn build_compat_module(result: &DetectionResult, document: &str) -> CompatibilityModule {
    let mut interfaces: Vec<String> = Vec::new();
    let mut functions: Vec<String> = Vec::new();

    for pattern in &result.patterns {
        let desc = &pattern.description;
        let location = pattern.locations.first().cloned().unwrap_or_default();

        match pattern.category {
            CompatCategory::AdapterTransformer => interfaces.push(format!("{} @ {}", desc, location)),
            CompatCategory::DeprecationWrapper => functions.push(format!("deprecated: {} @ {}", desc, location)),
            _ => functions.push(format!("{} ({:.0}% confidence) @ {}", pattern.category.name(), pattern.confidence * 100.0, location)),
        }
    }

    CompatibilityModule {
        id: format!("compat-{}", document.replace('.', "-")),
        name: format!("Compatibility Layer for {}", document),
        description: Some(format!(
            "Auto-detected compatibility layer ({} patterns, {:.0}% confidence{})",
            result.patterns.len(),
            result.overall_confidence * 100.0,
            if result.is_primarily_forwarding { ", primarily pure forwarding" } else { "" },
        )),
        targets_versions: None,
        interfaces: if interfaces.is_empty() { None } else { Some(interfaces) },
        functions: if functions.is_empty() { None } else { Some(functions) },
    }
}

/// Find the line number where text appears.
fn line_find(content: &str, text: &str) -> usize {
    content
        .lines()
        .position(|l| l.contains(text.trim()))
        .unwrap_or(0)
        + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_version_bridging() {
        let content = r#"
fn convert_v1_to_v2(input: V1Input) -> V2Output {
    V2Output { id: input.id, name: input.name }
}
"#;
        let detector = CompatDetector::new(vec![("src/compat.rs".to_string(), content.to_string())]);
        let result = detector.detect();
        let bridge = result.patterns.iter().find(|p| matches!(p.category, CompatCategory::VersionBridging));
        assert!(bridge.is_some(), "Should detect version bridging");
    }

    #[test]
    fn test_detect_deprecation() {
        let content = r#"
#[deprecated(note = "Use new_api instead")]
pub fn old_api(input: &str) -> String {
    new_api(input)
}

fn new_api(input: &str) -> String {
    format!("Hello, {}", input)
}
"#;
        let detector = CompatDetector::new(vec![("src/lib.rs".to_string(), content.to_string())]);
        let result = detector.detect();
        let dep = result.patterns.iter().find(|p| matches!(p.category, CompatCategory::DeprecationWrapper));
        assert!(dep.is_some(), "Should detect deprecation");
    }

    #[test]
    fn test_detect_compat_annotation() {
        let content = r#"
@Deprecated
class OldService {
    // ...
}
"#;
        let detector = CompatDetector::new(vec![("src/OldService.java".to_string(), content.to_string())]);
        let result = detector.detect();
        let ann = result.patterns.iter().find(|p| matches!(p.category, CompatCategory::CompatibilityAnnotation));
        assert!(ann.is_some(), "Should detect annotation");
    }

    #[test]
    fn test_detect_pure_forwarding() {
        let content = r#"
fn get_user(id: &str) -> User {
    user_service.get_user(id)
}
"#;
        let detector = CompatDetector::new(vec![("src/service.rs".to_string(), content.to_string())]);
        let result = detector.detect();
        let fwd = result.patterns.iter().find(|p| matches!(p.category, CompatCategory::PureForwarding));
        assert!(fwd.is_some(), "Should detect pure forwarding");
    }

    #[test]
    fn test_no_compat_in_normal_code() {
        let content = r#"
fn calculate_total(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let detector = CompatDetector::new(vec![("src/calc.rs".to_string(), content.to_string())]);
        let result = detector.detect();
        assert!(result.patterns.is_empty(), "Normal code should have no compat patterns, got {} patterns: {:?}",
            result.patterns.len(),
            result.patterns.iter().map(|p| &p.description).collect::<Vec<_>>());
    }

    #[test]
    fn test_build_compat_module() {
        let mut result = DetectionResult::default();
        result.patterns.push(CompatPattern {
            description: "Test pattern".to_string(),
            category: CompatCategory::DeprecationWrapper,
            confidence: 0.8,
            locations: vec!["file.rs:10".to_string()],
        });
        result.overall_confidence = 0.8;

        let module = build_compat_module(&result, "test-doc");
        assert!(module.description.unwrap().contains("1 patterns"));
    }
}
