//! Integration test: full produce flow — scan a test repo → generate S.DEF.
//!
//! Creates a temporary Rust project, runs the producer pipeline,
//! and verifies the resulting SoftwareDefinition.

use std::path::PathBuf;
use std::sync::Arc;

use cleanroom_agent::{
    run_analysis_pipeline,
    repo_scanner::{scan_repository, ScanConfig},
    module_partitioner::{partition_files, PartitionConfig},
    dependency_graph::DependencyGraph,
};
use cleanroom_db::Database;
use tracing::info;

/// Create a temporary directory with a minimal Rust project.
fn create_test_project(tmp_dir: &std::path::Path) {
    std::fs::create_dir_all(tmp_dir.join("src")).unwrap();
    std::fs::write(
        tmp_dir.join("Cargo.toml"),
        r#"[package]
name = "test-crate"
version = "0.1.0"
edition = "2021"
"#,
    ).unwrap();
    std::fs::write(
        tmp_dir.join("src").join("main.rs"),
        r#"fn main() {
    let user = User::new("Alice");
    println!("{}", user.greet());
}

pub struct User {
    name: String,
}

impl User {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }

    pub fn greet(&self) -> String {
        format!("Hello, {}!", self.name)
    }
}
"#,
    ).unwrap();
}

fn test_dir(name: &str) -> PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("cleanroom_test_{}_{}", name, ts))
}

#[test]
fn test_produce_flow_rust_project() {
    let tmp = test_dir("produce_rust");
    let _ = std::fs::remove_dir_all(&tmp);
    create_test_project(&tmp);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let db = Arc::new(Database::in_memory().unwrap());
        let result = run_analysis_pipeline(
            db,
            &tmp,
            "test-crate",
            "0.1.0",
            Some("Test Rust crate".to_string()),
        ).await.unwrap();

        // Verify pipeline results
        assert!(result.file_count > 0, "Should discover source files");
        assert!(result.module_count > 0, "Should identify modules");
        assert!(!result.languages.is_empty(), "Should detect languages");
        assert!(result.languages.contains(&"rust".to_string()), "Should detect Rust");

        // Verify S.DEF output
        let sdef = &result.sdef;
        assert_eq!(sdef.name, "test-crate");
        assert_eq!(sdef.version.as_deref(), Some("0.1.0"));

        // Should have at least architecture info
        assert!(sdef.architecture.is_some(), "Should have architecture");

        info!(
            files = result.file_count,
            modules = result.module_count,
            data_models = sdef.data_models.as_ref().map(|v| v.len()).unwrap_or(0),
            functions = sdef.behavior.as_ref().and_then(|b| b.functions.as_ref()).map(|v| v.len()).unwrap_or(0),
            decisions = sdef.design_decisions.as_ref().map(|v| v.len()).unwrap_or(0),
            "Produce flow test passed"
        );
    });

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_produce_flow_empty_project() {
    let tmp = test_dir("produce_empty");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    // No files at all

    let rt = tokio::runtime::Runtime::new().unwrap();
    let err = rt.block_on(async {
        let db = Arc::new(Database::in_memory().unwrap());
        run_analysis_pipeline(db, &tmp, "empty", "0.1.0", None).await
    });

    assert!(err.is_err(), "Empty repo should produce an error");
    let _ = std::fs::remove_dir_all(&tmp);
}
