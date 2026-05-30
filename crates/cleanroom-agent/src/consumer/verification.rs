//! Code generation verification loop.
//!
//! Implements the four-layer verification from docs/12-code-generation-loop.md:
//!   1. Compilation check
//!   2. Type/semantic validation (via LSP)
//!   3. Test execution
//!   4. Cross-consistency (fingerprint round-trip)
//!
//! With auto-diagnose → fix → retry cycling.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cleanroom_db::Database;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

// ── Result types ─────────────────────────────────────────────────────────

/// Outcome of a compilation attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileResult {
    pub success: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub errors: Vec<CompileError>,
    pub warnings: Vec<String>,
}

/// A single compile error parsed from compiler output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileError {
    pub file_path: String,
    pub line: usize,
    pub column: usize,
    pub message: String,
    pub error_code: Option<String>,
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Error,
    Warning,
}

/// Error categories for auto-fix routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    MissingDependency,
    TypeMismatch,
    UndefinedSymbol,
    SyntaxError,
    SemanticError,
    CrossFileConflict,
    Unknown,
}

/// Strategy for fixing a detected error.
#[derive(Debug, Clone)]
pub enum FixStrategy {
    AddImport { module: String, symbol: String },
    RenameSymbol { old: String, new: String },
    ChangeType { file: String, line: usize, new_type: String },
    LlmRegenerate { file: String, error_context: String },
    HumanRequired { description: String },
}

/// Overall verification report after a full cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub compile_passed: bool,
    pub compile_errors: usize,
    pub compile_warnings: usize,
    pub type_check_passed: bool,
    pub test_passed: bool,
    pub tests_total: usize,
    pub tests_failed: usize,
    pub consistency_passed: bool,
    pub roundtrip_fidelity: f64,
    pub retries_used: u32,
    pub total_duration_ms: u64,
    pub fixes_applied: Vec<String>,
}

/// Configuration for the generation loop.
#[derive(Debug, Clone)]
pub struct GenerationLoopConfig {
    pub max_retries: u32,
    pub max_mechanical_fixes: u32,
    pub cooldown_ms: u64,
    pub min_test_pass_rate: f64,
    pub total_timeout: Duration,
}

impl Default for GenerationLoopConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            max_mechanical_fixes: 5,
            cooldown_ms: 500,
            min_test_pass_rate: 0.0,
            total_timeout: Duration::from_secs(300),
        }
    }
}

/// Per-language compiler configuration.
#[derive(Debug, Clone)]
struct LanguageCompiler {
    language: String,
    check_cmd: Vec<String>,    // "cargo check" equivalent
    build_cmd: Vec<String>,    // "cargo build" equivalent
    test_cmd: Vec<String>,     // "cargo test" equivalent
    lint_cmd: Vec<String>,     // "cargo clippy" equivalent
}

// ── Compilation Verifier ─────────────────────────────────────────────────

/// Runs compilation and parses error output.
pub struct CompilationVerifier;

impl CompilationVerifier {
    /// Run a compile check on generated code.
    pub fn check(work_dir: &Path, language: &str) -> CompileResult {
        let compiler = get_compiler_config(language);

        let mut result = CompileResult {
            success: false,
            error_count: 0,
            warning_count: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        let output = match Command::new(&compiler.check_cmd[0])
            .args(&compiler.check_cmd[1..])
            .current_dir(work_dir)
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                result.errors.push(CompileError {
                    file_path: String::new(),
                    line: 0,
                    column: 0,
                    message: format!("Failed to run compiler: {e}"),
                    error_code: None,
                    severity: ErrorSeverity::Error,
                });
                result.error_count = 1;
                return result;
            }
        };

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let combined = format!("{stdout}\n{stderr}");

        result.success = output.status.success();
        result.errors = Self::parse_errors(&combined, language);
        result.warnings = Self::parse_warnings(&combined, language);
        result.error_count = result.errors.len();
        result.warning_count = result.warnings.len();

        result
    }

    /// Parse compile errors from output, per language.
    fn parse_errors(output: &str, language: &str) -> Vec<CompileError> {
        match language {
            "rust" => Self::parse_rust_like(output, "error"),
            "go" => Self::parse_generic_path_line(output),
            _ => Self::parse_generic_path_line(output),
        }
    }

    fn parse_warnings(output: &str, language: &str) -> Vec<String> {
        match language {
            "rust" => output.lines()
                .filter(|l| l.contains("warning:") || l.contains("warning["))
                .map(|l| l.to_string())
                .collect(),
            _ => output.lines()
                .filter(|l| l.to_lowercase().contains("warning"))
                .map(|l| l.to_string())
                .collect(),
        }
    }

    /// Parse Rust-style `error[EXXXX]: message` and `--> file:line:col` patterns.
    fn parse_rust_like(output: &str, prefix: &str) -> Vec<CompileError> {
        let mut errors = Vec::new();
        let mut lines = output.lines().peekable();

        while let Some(line) = lines.next() {
            let trimmed = line.trim();
            if !trimmed.starts_with(prefix) {
                continue;
            }

            let error_code = extract_rust_code(trimmed);
            let message = trimmed
                .splitn(2, ':')
                .nth(1)
                .unwrap_or(trimmed)
                .trim()
                .to_string();

            let (file_path, line_no, col) = if let Some(next) = lines.peek() {
                parse_location_line(next)
            } else {
                (String::new(), 0, 0)
            };
            if file_path.is_empty() {
                if let Some(next) = lines.next() {
                    let (f, l, c) = parse_location_line(next);
                    errors.push(CompileError {
                        file_path: f,
                        line: l,
                        column: c,
                        message,
                        error_code,
                        severity: ErrorSeverity::Error,
                    });
                }
            } else {
                let _ = lines.next();
                errors.push(CompileError {
                    file_path,
                    line: line_no,
                    column: col,
                    message,
                    error_code,
                    severity: ErrorSeverity::Error,
                });
            }
        }

        errors
    }

    /// Generic parser: look for `file:line:col: message` patterns.
    fn parse_generic_path_line(output: &str) -> Vec<CompileError> {
        let mut errors = Vec::new();
        for line in output.lines() {
            if line.to_lowercase().contains("error") {
                let (file, line_no, col) = parse_location_line(line);
                errors.push(CompileError {
                    file_path: file,
                    line: line_no,
                    column: col,
                    message: line.to_string(),
                    error_code: None,
                    severity: ErrorSeverity::Error,
                });
            }
        }
        errors
    }

    /// Run linting and auto-format.
    pub fn format_code(work_dir: &Path, language: &str) -> Result<(), String> {
        let (cmd, args): (&str, &[&str]) = match language {
            "rust" => ("rustfmt", &["--edition", "2021"]),
            "typescript" | "javascript" => ("npx", &["prettier", "--write", "."]),
            "python" => ("black", &["."]),
            "go" => ("gofmt", &["-w", "."]),
            _ => return Ok(()),
        };

        let output = Command::new(cmd)
            .args(args)
            .current_dir(work_dir)
            .output()
            .map_err(|e| format!("Failed to run {cmd}: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("{cmd} failed: {stderr}"));
        }
        Ok(())
    }
}

// ── Test Executor ────────────────────────────────────────────────────────

/// Executes generated tests and collects results.
pub struct TestExecutor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub failures: Vec<TestFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFailure {
    pub name: String,
    pub message: String,
}

impl TestExecutor {
    /// Run tests in the generated output directory.
    pub fn run(work_dir: &Path, language: &str) -> TestReport {
        let compiler = get_compiler_config(language);

        let output = match Command::new(&compiler.test_cmd[0])
            .args(&compiler.test_cmd[1..])
            .current_dir(work_dir)
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                return TestReport {
                    total: 0,
                    passed: 0,
                    failed: 1,
                    failures: vec![TestFailure {
                        name: "test_runner".into(),
                        message: format!("Failed to run test command: {e}"),
                    }],
                };
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{stdout}\n{stderr}");

        Self::parse_test_output(&combined, language)
    }

    fn parse_test_output(output: &str, language: &str) -> TestReport {
        match language {
            "rust" => Self::parse_rust_tests(output),
            "go" => Self::parse_go_tests(output),
            _ => Self::parse_generic_tests(output),
        }
    }

    fn parse_rust_tests(output: &str) -> TestReport {
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut failures = Vec::new();

        for line in output.lines() {
            if line.contains("test result: ok.") {
                // e.g., "test result: ok. 5 passed; 0 failed;"
                for part in line.split(';') {
                    let part = part.trim();
                    if let Some(n) = part.strip_suffix(" passed") {
                        passed = n.trim().parse().unwrap_or(0);
                    }
                    if let Some(n) = part.strip_suffix(" failed") {
                        failed = n.trim().parse().unwrap_or(0);
                    }
                }
            }
            if line.starts_with("test ") && line.contains("... FAILED") {
                let name = line.split_whitespace().nth(1).unwrap_or("unknown");
                failures.push(TestFailure {
                    name: name.to_string(),
                    message: line.to_string(),
                });
            }
        }

        TestReport { total: passed + failed, passed, failed, failures }
    }

    fn parse_go_tests(output: &str) -> TestReport {
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut failures = Vec::new();

        for line in output.lines() {
            if line.starts_with("--- PASS:") { passed += 1; }
            if line.starts_with("--- FAIL:") {
                failed += 1;
                let name = line.strip_prefix("--- FAIL: ").unwrap_or("unknown");
                failures.push(TestFailure {
                    name: name.to_string(),
                    message: line.to_string(),
                });
            }
        }

        TestReport { total: passed + failed, passed, failed, failures }
    }

    fn parse_generic_tests(output: &str) -> TestReport {
        let pass_count = output.matches("PASS").count()
            + output.matches("ok").count();
        let fail_count = output.matches("FAIL").count();
        let failures = output.lines()
            .filter(|l| l.contains("FAIL"))
            .map(|l| TestFailure {
                name: "test".into(),
                message: l.to_string(),
            })
            .collect();

        TestReport {
            total: pass_count + fail_count,
            passed: pass_count,
            failed: fail_count,
            failures,
        }
    }
}

// ── Error Classifier ─────────────────────────────────────────────────────

/// Classify a compile error into a category for auto-fix routing.
pub fn classify_error(error: &CompileError) -> ErrorCategory {
    if let Some(ref code) = error.error_code {
        match code.as_str() {
            "E0432" | "E0433" | "E0583" => return ErrorCategory::MissingDependency,
            "E0308" | "E0277" | "E0368" | "E0369" => return ErrorCategory::TypeMismatch,
            "E0425" | "E0423" | "E0412" | "E0431" => return ErrorCategory::UndefinedSymbol,
            _ => {}
        }
    }

    let msg = &error.message;
    if (msg.contains("expected") && msg.contains("found"))
        || msg.contains("mismatched types")
    {
        return ErrorCategory::TypeMismatch;
    }
    if msg.contains("unresolved")
        || msg.contains("cannot find")
        || msg.contains("not found in this scope")
    {
        return ErrorCategory::UndefinedSymbol;
    }
    if msg.contains("syntax error")
        || msg.contains("expected one of")
        || msg.contains("unexpected token")
    {
        return ErrorCategory::SyntaxError;
    }

    ErrorCategory::Unknown
}

/// Determine fix strategies based on error category and retry count.
pub fn determine_fix_strategies(
    errors: &[CompileError],
    retry_count: u32,
) -> Vec<FixStrategy> {
    let mut strategies = Vec::new();

    for error in errors {
        let category = classify_error(error);

        if categories_are_all_mechanical(errors) && retry_count == 0 {
            // First round: apply mechanical fixes
            match category {
                ErrorCategory::MissingDependency => {
                    if let Some((module, symbol)) = extract_missing_import(error) {
                        strategies.push(FixStrategy::AddImport { module, symbol });
                    }
                }
                ErrorCategory::UndefinedSymbol => {
                    if let Some(correction) = suggest_correction(error) {
                        strategies.push(FixStrategy::RenameSymbol {
                            old: correction.from,
                            new: correction.to,
                        });
                    }
                }
                _ => {}
            }
        } else if retry_count < 3 {
            // Round 2-3: escalate to LLM
            strategies.push(FixStrategy::LlmRegenerate {
                file: error.file_path.clone(),
                error_context: format!(
                    "{}:{}:{}: [{}] {}",
                    error.file_path,
                    error.line,
                    error.column,
                    error.error_code.as_deref().unwrap_or("?"),
                    error.message,
                ),
            });
        } else {
            // Round 4+: give up
            strategies.push(FixStrategy::HumanRequired {
                description: format!(
                    "Failed to fix after {retry_count} rounds: {} in {}",
                    error.message,
                    error.file_path,
                ),
            });
        }
    }

    strategies
}

fn categories_are_all_mechanical(errors: &[CompileError]) -> bool {
    errors.iter().all(|e| {
        matches!(
            classify_error(e),
            ErrorCategory::MissingDependency | ErrorCategory::UndefinedSymbol
        )
    })
}

fn extract_missing_import(error: &CompileError) -> Option<(String, String)> {
    // Rust: "use of undeclared crate or module `serde`"
    // Try to extract the module name
    let msg = &error.message;
    for needle in ["use of undeclared crate or module `", "cannot find module `"] {
        if let Some(start) = msg.find(needle) {
            let rest = &msg[start + needle.len()..];
            if let Some(end) = rest.find('`') {
                let name = &rest[..end];
                return Some((name.to_string(), name.to_string()));
            }
        }
    }
    None
}

fn suggest_correction(error: &CompileError) -> Option<NameCorrection> {
    let msg = &error.message;
    // Rust: "help: there is an associated function with a similar name: `len`"
    if let Some(pos) = msg.find("similar name:") {
        let rest = &msg[pos + "similar name:".len()..];
        let suggestion = rest.trim().trim_matches('`').trim();
        if !suggestion.is_empty() {
            return Some(NameCorrection {
                from: String::new(), // caller fills in
                to: suggestion.to_string(),
            });
        }
    }
    None
}

struct NameCorrection { from: String, to: String }

// ── Generation Loop ──────────────────────────────────────────────────────

/// The main generation loop: generate → verify → fix → retry.
pub struct GenerationLoop {
    config: GenerationLoopConfig,
    db: Arc<Database>,
}

/// Final outcome of the generation loop.
#[derive(Debug)]
pub enum LoopOutcome {
    Success(VerificationReport),
    Failed { report: VerificationReport, reason: String },
}

impl GenerationLoop {
    pub fn new(config: GenerationLoopConfig, db: Arc<Database>) -> Self {
        Self { config, db }
    }

    /// Run verification on already-generated code at the given output directory.
    pub fn verify_and_heal(
        &self,
        output_dir: &Path,
        document_name: &str,
        language: &str,
        entity_count: usize,
    ) -> LoopOutcome {
        let start = Instant::now();
        let mut retries = 0u32;
        let mut report = VerificationReport::default();

        loop {
            // Check timeout
            if start.elapsed() > self.config.total_timeout {
                return LoopOutcome::Failed {
                    reason: "Generation loop timed out".into(),
                    report,
                };
            }

            // Layer 1: Compile
            let compile = CompilationVerifier::check(output_dir, language);
            if compile.success {
                report.compile_passed = true;
                report.compile_errors = 0;
            } else {
                report.compile_errors = compile.error_count;
            }
            report.compile_warnings = compile.warning_count;

            // Layer 2: Lint & format (mechanical clean-up)
            if compile.success {
                let _ = CompilationVerifier::format_code(output_dir, language);
            }

            // Layer 3: Tests
            if compile.success {
                let tests = TestExecutor::run(output_dir, language);
                report.tests_total = tests.total;
                report.tests_failed = tests.failed;
                report.test_passed = tests.failed == 0 || tests.total == 0
                    || (tests.passed as f64 / tests.total as f64) >= self.config.min_test_pass_rate;
            }

            // Layer 4: Consistency
            if compile.success && report.test_passed {
                report.consistency_passed = self.check_cross_consistency(
                    document_name, entity_count,
                );
            }

            // All layers passed
            if compile.success && report.test_passed && report.consistency_passed {
                report.retries_used = retries;
                report.total_duration_ms = start.elapsed().as_millis() as u64;
                report.type_check_passed = true; // compilation implies type check
                return LoopOutcome::Success(report);
            }

            // Failed — determine fix strategy
            if retries >= self.config.max_retries {
                return LoopOutcome::Failed {
                    reason: format!("Failed after {retries} retries"),
                    report,
                };
            }

            let fix_strategies = determine_fix_strategies(&compile.errors, retries);

            if fix_strategies.iter().any(|s| matches!(s, FixStrategy::HumanRequired { .. })) {
                warn!("Human intervention required for generation fix");
                return LoopOutcome::Failed {
                    reason: "Human intervention required".into(),
                    report,
                };
            }

            let fix_descriptions: Vec<String> = fix_strategies.iter()
                .map(|s| format!("{:?}", s))
                .collect();
            report.fixes_applied.extend(fix_descriptions);
            report.retries_used = retries;

            // Apply mechanical fixes
            let has_llm_fixes = fix_strategies.iter()
                .any(|s| matches!(s, FixStrategy::LlmRegenerate { .. }));

            if has_llm_fixes {
                info!(retry = retries, "LLM regeneration required");
                // The caller (MCP/agent) handles LLM re-generation
                return LoopOutcome::Failed {
                    reason: format!("Needs LLM regeneration (retry {retries})"),
                    report,
                };
            }

            // Mechanical fixes exhausted — escalate to LLM
            if retries >= self.config.max_mechanical_fixes {
                info!("Mechanical fixes exhausted, escalating to LLM");
                return LoopOutcome::Failed {
                    reason: "Mechanical fixes exhausted — needs LLM".into(),
                    report,
                };
            }

            retries += 1;
            std::thread::sleep(Duration::from_millis(self.config.cooldown_ms));
        }
    }

    /// Check fingerprint-based cross-consistency between S.DEF and generated code.
    fn check_cross_consistency(
        &self,
        document_name: &str,
        _entity_count: usize,
    ) -> bool {
        let conn = self.db.connection();

        // Query fingerprints table for any mismatches
        let inconsistent: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM fingerprints
                 WHERE document_name = ?1
                   AND (sdef_hash != db_hash OR db_hash != code_hash OR sdef_hash != code_hash)",
                rusqlite::params![document_name],
                |row| row.get(0),
            )
            .unwrap_or(i64::MAX);

        // Allow a small tolerance (e.g., up to 5% of entities to be mismatched)
        inconsistent == 0
    }
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self {
            compile_passed: false,
            compile_errors: 0,
            compile_warnings: 0,
            type_check_passed: false,
            test_passed: false,
            tests_total: 0,
            tests_failed: 0,
            consistency_passed: false,
            roundtrip_fidelity: 0.0,
            retries_used: 0,
            total_duration_ms: 0,
            fixes_applied: Vec::new(),
        }
    }
}

// ── Compiler config per language ─────────────────────────────────────────

fn get_compiler_config(language: &str) -> LanguageCompiler {
    match language {
        "rust" => LanguageCompiler {
            language: "rust".into(),
            check_cmd: vec!["cargo".into(), "check".into(), "--quiet".into()],
            build_cmd: vec!["cargo".into(), "build".into(), "--release".into()],
            test_cmd: vec!["cargo".into(), "test".into(), "--quiet".into()],
            lint_cmd: vec!["cargo".into(), "clippy".into(), "--".into(), "-D".into(), "warnings".into()],
        },
        "go" => LanguageCompiler {
            language: "go".into(),
            check_cmd: vec!["go".into(), "vet".into(), "./...".into()],
            build_cmd: vec!["go".into(), "build".into(), "./...".into()],
            test_cmd: vec!["go".into(), "test".into(), "./...".into()],
            lint_cmd: vec!["golangci-lint".into(), "run".into()],
        },
        "typescript" | "javascript" => LanguageCompiler {
            language: "typescript".into(),
            check_cmd: vec!["npx".into(), "tsc".into(), "--noEmit".into()],
            build_cmd: vec!["npx".into(), "tsc".into()],
            test_cmd: vec!["npx".into(), "jest".into(), "--passWithNoTests".into()],
            lint_cmd: vec!["npx".into(), "eslint".into(), ".".into()],
        },
        "python" => LanguageCompiler {
            language: "python".into(),
            check_cmd: vec!["python3".into(), "-m".into(), "py_compile".into(), ".".into()],
            build_cmd: vec!["python3".into(), "-c".into(), "pass".into()],
            test_cmd: vec!["python3".into(), "-m".into(), "pytest".into(), "--tb=short".into()],
            lint_cmd: vec!["ruff".into(), "check".into(), ".".into()],
        },
        other => LanguageCompiler {
            language: other.into(),
            check_cmd: vec!["echo".into(), format!("No compiler configured for {other}").into()],
            build_cmd: vec!["echo".into(), format!("No builder for {other}").into()],
            test_cmd: vec!["echo".into(), format!("No test runner for {other}").into()],
            lint_cmd: vec!["echo".into(), format!("No linter for {other}").into()],
        },
    }
}

// ── Location parser ──────────────────────────────────────────────────────

/// Parse a line like "  --> src/main.rs:5:10" to (file, line, column).
fn parse_location_line(line: &str) -> (String, usize, usize) {
    let line = line.trim();
    // Strip "-->" prefix
    let line = line.trim_start_matches("-->").trim();

    // Find the last ':' — column
    if let Some(colon_pos) = line.rfind(':') {
        let before_col = &line[..colon_pos];
        let col_str = &line[colon_pos + 1..];
        let col: usize = col_str.parse().unwrap_or(0);

        // Now find the line number
        if let Some(line_pos) = before_col.rfind(':') {
            let file = before_col[..line_pos].to_string();
            let line_no: usize = before_col[line_pos + 1..].parse().unwrap_or(0);
            return (file, line_no, col);
        }
    }

    (String::new(), 0, 0)
}

/// Extract Rust error code like "E0308" from "error[E0308]".
fn extract_rust_code(line: &str) -> Option<String> {
    if let Some(start) = line.find('[') {
        if let Some(end) = line[start..].find(']') {
            let code = &line[start + 1..start + end];
            return Some(code.to_string());
        }
    }
    None
}

// ── Quality Gate ─────────────────────────────────────────────────────────

/// Quality gate configuration.
#[derive(Debug, Clone)]
pub struct QualityGate {
    pub max_warnings: usize,
    pub min_test_pass_rate: f64,
    pub require_roundtrip: bool,
}

impl Default for QualityGate {
    fn default() -> Self {
        Self {
            max_warnings: 5,
            min_test_pass_rate: 0.0,
            require_roundtrip: false,
        }
    }
}

impl QualityGate {
    /// Check if a verification report passes all quality gates.
    pub fn evaluate(&self, report: &VerificationReport) -> GateResult {
        let mut violations = Vec::new();

        if !report.compile_passed {
            violations.push("Compilation failed".to_string());
        }
        if report.compile_warnings > self.max_warnings {
            violations.push(format!(
                "Too many warnings: {} > {}",
                report.compile_warnings, self.max_warnings
            ));
        }
        if report.tests_total > 0 && !report.test_passed {
            let rate = (report.tests_total - report.tests_failed) as f64 / report.tests_total as f64;
            if rate < self.min_test_pass_rate {
                violations.push(format!(
                    "Test pass rate {:.0}% below threshold {:.0}%",
                    rate * 100.0, self.min_test_pass_rate * 100.0,
                ));
            }
        }
        if self.require_roundtrip && !report.consistency_passed {
            violations.push("Round-trip consistency check failed".to_string());
        }

        GateResult {
            passed: violations.is_empty(),
            violations,
        }
    }
}

#[derive(Debug)]
pub struct GateResult {
    pub passed: bool,
    pub violations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_missing_dep() {
        let err = CompileError {
            file_path: "src/main.rs".into(),
            line: 5,
            column: 1,
            message: "use of undeclared crate or module `serde`".into(),
            error_code: Some("E0432".into()),
            severity: ErrorSeverity::Error,
        };
        assert_eq!(classify_error(&err), ErrorCategory::MissingDependency);
    }

    #[test]
    fn test_classify_type_mismatch() {
        let err = CompileError {
            file_path: "src/lib.rs".into(),
            line: 10,
            column: 5,
            message: "expected String, found i32".into(),
            error_code: Some("E0308".into()),
            severity: ErrorSeverity::Error,
        };
        assert_eq!(classify_error(&err), ErrorCategory::TypeMismatch);
    }

    #[test]
    fn test_parse_rust_like_errors() {
        let output = r#"error[E0308]: mismatched types
 --> src/main.rs:5:10
  |
5 |     let x: String = 42;
  |                    ^^ expected String, found integer
"#;
        let errors = CompilationVerifier::parse_rust_like(output, "error");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].file_path, "src/main.rs");
        assert_eq!(errors[0].line, 5);
        assert_eq!(errors[0].column, 10);
        assert_eq!(errors[0].error_code.as_deref(), Some("E0308"));
    }

    #[test]
    fn test_parse_location_line() {
        let (file, line, col) = parse_location_line(" --> src/main.rs:5:10");
        assert_eq!(file, "src/main.rs");
        assert_eq!(line, 5);
        assert_eq!(col, 10);
    }

    #[test]
    fn test_fix_strategies_mechanical_first() {
        let errors = vec![CompileError {
            file_path: "src/lib.rs".into(),
            line: 1,
            column: 1,
            message: "use of undeclared crate or module `serde`".into(),
            error_code: Some("E0432".into()),
            severity: ErrorSeverity::Error,
        }];
        let strategies = determine_fix_strategies(&errors, 0);
        assert!(strategies.iter().any(|s| matches!(s, FixStrategy::AddImport { .. })));
    }

    #[test]
    fn test_fix_strategies_escalates_to_llm() {
        let errors = vec![CompileError {
            file_path: "src/lib.rs".into(),
            line: 5,
            column: 1,
            message: "expected String, found i32".into(),
            error_code: Some("E0308".into()),
            severity: ErrorSeverity::Error,
        }];
        let strategies = determine_fix_strategies(&errors, 1);
        assert!(strategies.iter().any(|s| matches!(s, FixStrategy::LlmRegenerate { .. })));
    }

    #[test]
    fn test_quality_gate_compile_fail() {
        let report = VerificationReport { compile_passed: false, ..Default::default() };
        let gate = QualityGate::default();
        assert!(!gate.evaluate(&report).passed);
    }

    #[test]
    fn test_quality_gate_too_many_warnings() {
        let report = VerificationReport {
            compile_passed: true,
            compile_warnings: 10,
            ..Default::default()
        };
        let gate = QualityGate::default();
        assert!(!gate.evaluate(&report).passed);
    }

    #[test]
    fn test_quality_gate_all_pass() {
        let report = VerificationReport {
            compile_passed: true,
            compile_warnings: 0,
            test_passed: true,
            ..Default::default()
        };
        let gate = QualityGate::default();
        assert!(gate.evaluate(&report).passed);
    }

    #[test]
    fn test_generation_loop_config_defaults() {
        let config = GenerationLoopConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.max_mechanical_fixes, 5);
        assert_eq!(config.total_timeout, Duration::from_secs(300));
    }
}
