//! Integration test: full consume flow — read S.DEF → generate code.
//!
//! Imports a known S.DEF fixture, then runs the consumer code generators
//! to produce Rust, TypeScript, and Python output.

use std::path::PathBuf;

use cleanroom_db::Database;
use cleanroom_db::export_import::SdefImporter;
use sdef_core::SoftwareDefinition;

/// Locate a fixture file by searching relative paths.
fn fixture_path(relative: &str) -> String {
    if std::path::Path::new(relative).exists() {
        return relative.to_string();
    }
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let candidate = dir.join(relative);
        if candidate.exists() {
            return candidate.to_string_lossy().to_string();
        }
        if !dir.pop() {
            panic!("Cannot find fixture: {}", relative);
        }
    }
}

/// Create a temporary file-based database using embedded schema.
fn create_temp_db() -> (Database, PathBuf) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("cleanroom_consume_{}", ts));
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("state.db");
    let db = Database::open_embedded(&db_path).unwrap();
    (db, db_path)
}

#[test]
fn test_consume_flow_rust_generation() {
    let path = fixture_path("tests/fixtures/todo-app/expected.sdef.json");
    let content = std::fs::read_to_string(path).unwrap();
    let sdef: SoftwareDefinition = serde_json::from_str(&content).unwrap();

    let (_db, db_path) = create_temp_db();

    // Import S.DEF into database
    let doc_name = SdefImporter::new(
        rusqlite::Connection::open(&db_path).unwrap()
    ).import(&sdef).unwrap();

    assert_eq!(doc_name, sdef.name);

    // Verify data models were imported
    let verify_conn = rusqlite::Connection::open(&db_path).unwrap();
    let model_count: i64 = verify_conn
        .query_row("SELECT COUNT(*) FROM data_models", [], |r| r.get(0))
        .unwrap();
    let expected = sdef.data_models.as_ref().map(|v| v.len() as i64).unwrap_or(0);
    assert_eq!(model_count, expected, "Data model count should match");

    // Test code generation via Rust generator
    let generator = cleanroom_agent::consumer::code_generator::create_generator("rust")
        .expect("Rust generator should exist");

    assert_eq!(generator.language_id(), "rust");
    assert_eq!(generator.file_extension(), "rs");

    // Generate code for each data model
    if let Some(models) = &sdef.data_models {
        for model in models {
            let code = generator.generate_data_model(model);
            assert!(!code.is_empty(), "Should generate code for '{}'", model.entity);
            for piece in &code {
                assert!(!piece.content.is_empty(), "Generated content should not be empty");
                assert!(piece.file_path.ends_with(".rs"), "File should be .rs");
                assert_eq!(piece.language, "rust");
            }
        }
    }

    // Generate code for functions
    if let Some(behavior) = &sdef.behavior {
        if let Some(functions) = &behavior.functions {
            for func in functions {
                let code = generator.generate_function(func);
                assert!(!code.content.is_empty(), "Function '{}' should generate code", func.name);
            }
        }
    }
}

#[test]
fn test_consume_flow_typescript_generation() {
    let path = fixture_path("tests/fixtures/todo-app/expected.sdef.json");
    let content = std::fs::read_to_string(path).unwrap();
    let sdef: SoftwareDefinition = serde_json::from_str(&content).unwrap();

    let generator = cleanroom_agent::consumer::code_generator::create_generator("typescript")
        .expect("TypeScript generator should exist");

    assert_eq!(generator.language_id(), "typescript");

    if let Some(models) = &sdef.data_models {
        for model in models {
            let code = generator.generate_data_model(model);
            assert!(!code.is_empty(), "Should generate TS code for '{}'", model.entity);
            for piece in &code {
                assert!(piece.file_path.ends_with(".ts"), "File should be .ts");
            }
        }
    }
}

#[test]
fn test_consume_flow_python_generation() {
    let path = fixture_path("tests/fixtures/todo-app/expected.sdef.json");
    let content = std::fs::read_to_string(path).unwrap();
    let sdef: SoftwareDefinition = serde_json::from_str(&content).unwrap();

    let generator = cleanroom_agent::consumer::code_generator::create_generator("python")
        .expect("Python generator should exist");

    assert_eq!(generator.language_id(), "python");

    if let Some(models) = &sdef.data_models {
        for model in models {
            let code = generator.generate_data_model(model);
            assert!(!code.is_empty(), "Should generate Python code for '{}'", model.entity);
            for piece in &code {
                assert!(piece.file_path.ends_with(".py"), "File should be .py");
            }
        }
    }
}
