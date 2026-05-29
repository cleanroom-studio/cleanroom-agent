//! Test Contract Extractor — extracts `TestContract` entities from source code.
//!
//! Identifies test patterns across multiple languages:
//! - Rust: `#[test]`, `fn test_*`
//! - TypeScript/JavaScript: `describe()`, `it()`, `test()`
//! - Python: `def test_*`, `unittest.TestCase`
//! - Go: `func Test*`

use sdef_core::types::tests::{IntegrationTest, TestContract, UnitTestCase, UnitTestGroup};
use std::path::Path;

/// Result of extracting test contracts from a set of files.
#[derive(Debug, Clone, Default)]
pub struct ExtractionResult {
    pub unit_groups: Vec<UnitTestGroup>,
    pub integration_tests: Vec<IntegrationTest>,
    pub file_count: usize,
}

enum TestFramework {
    RustNative, Jest, Mocha, Pytest, GoTest,
}

impl TestFramework {
    fn detect(file_name: &str, content: &str) -> Option<Self> {
        let ext = Path::new(file_name).extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "rs" if content.contains("#[test]") || content.contains("#[tokio::test]") || content.contains("fn test_") => Some(Self::RustNative),
            "ts" | "tsx" if content.contains("describe(") || content.contains("test(") => Some(Self::Jest),
            "js" | "jsx" if content.contains("describe(") || content.contains("it(") => Some(Self::Mocha),
            "py" if content.contains("def test_") || content.contains("unittest") => Some(Self::Pytest),
            "go" if content.contains("func Test") => Some(Self::GoTest),
            _ => None,
        }
    }
}

/// Extract test contracts from source files.
pub fn extract_tests(files: &[(&str, &str)]) -> ExtractionResult {
    let mut result = ExtractionResult::default();
    for (file_path, content) in files {
        result.file_count += 1;
        match TestFramework::detect(file_path, content) {
            Some(TestFramework::RustNative) => extract_rust_tests(file_path, content, &mut result),
            Some(TestFramework::Jest) | Some(TestFramework::Mocha) => extract_jest_tests(file_path, content, &mut result),
            Some(TestFramework::Pytest) => extract_python_tests(file_path, content, &mut result),
            Some(TestFramework::GoTest) => extract_go_tests(file_path, content, &mut result),
            None => {}
        }
    }
    result
}

fn extract_rust_tests(file_path: &str, content: &str, result: &mut ExtractionResult) {
    let mut test_cases = Vec::new();
    let test_re = regex::Regex::new(r"(?m)^\s*(?:fn\s+(test_\w+|.*_test)\s*\(|#\[test\]\s*\n\s*fn\s+(\w+)\s*\()").unwrap();
    for caps in test_re.captures_iter(content) {
        let name = caps.iter().skip(1).find_map(|m| m.map(|m| m.as_str().to_string())).unwrap_or_default();
        let desc = extract_fn_description(content, &name);
        let (given, when_part, then_part) = extract_gwt(content, &name);
        test_cases.push(UnitTestCase {
            id: format!("ut_{}", name),
            description: if desc.is_empty() { format!("Test: {}", name) } else { desc },
            given: given.map(|s| serde_json::Value::String(s)),
            when: when_part,
            then: then_part.map(|s| serde_json::Value::String(s)),
            expected_exception: None,
            expected_side_effects: None,
        });
    }
    if !test_cases.is_empty() {
        result.unit_groups.push(UnitTestGroup {
            module_id: Some(file_path.to_string()),
            interface_id: None,
            test_cases: Some(test_cases),
        });
    }
}

fn extract_jest_tests(file_path: &str, content: &str, result: &mut ExtractionResult) {
    let mut test_cases = Vec::new();
    let describe_re = regex::Regex::new(r#"describe\(['"]([^'"]+)['"]"#).unwrap();
    let it_re = regex::Regex::new(r#"(?:it|test)\(['"]([^'"]+)['"]"#).unwrap();
    let module = describe_re.captures(content).and_then(|c| c.get(1)).map(|m| m.as_str().to_string()).unwrap_or_else(|| file_path.to_string());
    for cap in it_re.captures_iter(content) {
        let name = cap[1].to_string();
        test_cases.push(UnitTestCase {
            id: format!("it_{}", name),
            description: format!("Test: {}", name),
            given: None,
            when: None,
            then: None,
            expected_exception: None,
            expected_side_effects: None,
        });
    }
    if !test_cases.is_empty() {
        result.unit_groups.push(UnitTestGroup {
            module_id: Some(module),
            interface_id: None,
            test_cases: Some(test_cases),
        });
    }
}

fn extract_python_tests(file_path: &str, content: &str, result: &mut ExtractionResult) {
    let mut test_cases = Vec::new();
    let test_re = regex::Regex::new(r"def (test_\w+)").unwrap();
    let class_re = regex::Regex::new(r"class (\w+Test|Test\w+)").unwrap();
    let module = class_re.captures(content).and_then(|c| c.get(1)).map(|m| m.as_str().to_string()).unwrap_or_else(|| file_path.to_string());
    for cap in test_re.captures_iter(content) {
        let name = cap[1].to_string();
        test_cases.push(UnitTestCase {
            id: format!("py_{}", name),
            description: format!("Python test: {}", name),
            given: None,
            when: None,
            then: None,
            expected_exception: None,
            expected_side_effects: None,
        });
    }
    if !test_cases.is_empty() {
        result.unit_groups.push(UnitTestGroup {
            module_id: Some(module),
            interface_id: None,
            test_cases: Some(test_cases),
        });
    }
}

fn extract_go_tests(file_path: &str, content: &str, result: &mut ExtractionResult) {
    let mut test_cases = Vec::new();
    let test_re = regex::Regex::new(r"func (Test\w+)").unwrap();
    for cap in test_re.captures_iter(content) {
        let name = cap[1].to_string();
        test_cases.push(UnitTestCase {
            id: format!("go_{}", name),
            description: format!("Go test: {}", name),
            given: None,
            when: None,
            then: None,
            expected_exception: None,
            expected_side_effects: None,
        });
    }
    if !test_cases.is_empty() {
        result.unit_groups.push(UnitTestGroup {
            module_id: Some(file_path.to_string()),
            interface_id: None,
            test_cases: Some(test_cases),
        });
    }
}

fn extract_fn_description(content: &str, fn_name: &str) -> String {
    let escaped = regex::escape(fn_name);
    let pattern = format!(r"(?s)///\s*(.*?)\nfn {}\(", escaped);
    regex::Regex::new(&pattern).ok().and_then(|re| {
        re.captures(content).and_then(|c| c.get(1)).map(|m| m.as_str().trim().to_string())
    }).unwrap_or_default()
}

fn extract_gwt(content: &str, fn_name: &str) -> (Option<String>, Option<String>, Option<String>) {
    let escaped = regex::escape(fn_name);
    let pattern = format!(r"(?s)fn {}\(.*?\{{(.*?)\n\}}", escaped);
    let body = regex::Regex::new(&pattern).ok().and_then(|re| {
        re.captures(content).and_then(|c| c.get(1)).map(|m| m.as_str())
    }).unwrap_or("");

    let mut given = None;
    let mut when = None;
    let mut then = None;
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with("// Given") || t.starts_with("// Arrange") {
            given = Some(t.trim_start_matches('/').trim()
                .trim_start_matches("Given").trim_start_matches("given")
                .trim_start_matches("Arrange").trim().to_string());
        } else if t.starts_with("// When") || t.starts_with("// Act") {
            when = Some(t.trim_start_matches('/').trim()
                .trim_start_matches("When").trim_start_matches("when")
                .trim_start_matches("Act").trim().to_string());
        } else if t.starts_with("// Then") || t.starts_with("// Assert") {
            then = Some(t.trim_start_matches('/').trim()
                .trim_start_matches("Then").trim_start_matches("then")
                .trim_start_matches("Assert").trim().to_string());
        }
    }
    (given, when, then)
}

/// Build an `sdef_core::TestContract` from the extraction result.
pub fn build_test_contract(result: &ExtractionResult) -> TestContract {
    TestContract {
        unit_tests: if result.unit_groups.is_empty() { None } else { Some(result.unit_groups.clone()) },
        integration_tests: if result.integration_tests.is_empty() { None } else { Some(result.integration_tests.clone()) },
        acceptance_criteria: None,
    }
}

/// Persist extracted test contracts into the database.
pub fn persist_test_contract(
    db: &cleanroom_db::Database,
    document_name: &str,
    contract: &TestContract,
) -> Result<(), cleanroom_db::DbError> {
    let conn = db.connection();
    if let Some(groups) = &contract.unit_tests {
        for group in groups {
            if let Some(cases) = &group.test_cases {
                for tc in cases {
                    let given_str = tc.given.as_ref().map(|v| v.to_string());
                    let then_str = tc.then.as_ref().map(|v| v.to_string());
                    conn.execute(
                        "INSERT OR IGNORE INTO test_cases
                         (document_name, group_module, case_id, description, given, when_cond, then_expect)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        rusqlite::params![
                            document_name,
                            group.module_id.as_deref().unwrap_or("unknown"),
                            tc.id, tc.description,
                            given_str, tc.when, then_str,
                        ],
                    ).map_err(|e| cleanroom_db::DbError::QueryFailed(e.to_string()))?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_rust_tests() {
        let content = r#"
/// Creates a new user
fn test_create_user() {
    // Given a valid user input
    let input = "Alice";
    // When create_user is called
    let result = create_user(input);
    // Then user should be created
    assert!(result.is_ok());
}

fn test_delete_user() {
    let user = create_user("Bob");
    assert!(delete_user(user.id()).is_ok());
}
"#;
        let results = extract_tests(&[("tests/user_test.rs", content)]);
        assert!(results.file_count == 1);
        assert!(results.unit_groups.len() == 1);
        let cases = results.unit_groups[0].test_cases.as_ref().unwrap();
        assert_eq!(cases.len(), 2);
        // Check given/when/then extraction
        let gwt: String = cases[0].given.as_ref().map(|v| v.to_string()).unwrap_or_default()
            + &cases[0].when.as_ref().cloned().unwrap_or_default()
            + &cases[0].then.as_ref().map(|v| v.to_string()).unwrap_or_default();
        assert!(gwt.contains("valid user input"), "Should extract Given: {}", gwt);
    }

    #[test]
    fn test_extract_jest_tests() {
        let content = r#"describe("UserService", () => {
    it("should create a user", () => { expect(1).toBe(1); });
    test("should delete a user", () => { expect(1).toBe(1); });
});"#;
        let results = extract_tests(&[("tests/user.test.ts", content)]);
        assert_eq!(results.unit_groups.len(), 1);
        let cases = results.unit_groups[0].test_cases.as_ref().unwrap();
        assert!(cases.len() >= 2);
    }

    #[test]
    fn test_no_tests_in_regular_code() {
        let content = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let results = extract_tests(&[("src/lib.rs", content)]);
        assert_eq!(results.unit_groups.len(), 0);
    }

    #[test]
    fn test_build_contract() {
        let mut result = ExtractionResult::default();
        result.unit_groups.push(UnitTestGroup {
            module_id: Some("mod".to_string()),
            interface_id: None,
            test_cases: Some(vec![UnitTestCase {
                id: "ut_1".to_string(),
                description: "Test foo".to_string(),
                given: Some(serde_json::json!("state")),
                when: Some("action".to_string()),
                then: Some(serde_json::json!("result")),
                expected_exception: None,
                expected_side_effects: None,
            }]),
        });
        let contract = build_test_contract(&result);
        assert!(contract.unit_tests.is_some());
    }
}
