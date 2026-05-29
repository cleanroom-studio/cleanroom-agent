//! Integration test: consistency checking + fingerprint computation.
//!
//! Verifies that:
//! - Fingerprints can be computed and stored
//! - Consistency checking detects mismatches
//! - Auto-fix strategies can be applied

use std::sync::Arc;
use std::time::Duration;

use cleanroom_agent::{
    ConsistencyService, ConsistencyChecker, ConsistencyCheckerConfig,
    CheckLevel, FixStrategy, Inconsistency,
};
use cleanroom_db::{Database, FingerprintRepository, Fingerprint};

/// Helper to create an in-memory database with a prepared document.
fn setup_db() -> Arc<Database> {
    let db = Arc::new(Database::in_memory().unwrap());
    let conn = db.connection();

    // Create document
    conn.execute_batch(
        "INSERT INTO sdef_documents (name, version, description, created_at, updated_at)
         VALUES ('test-doc', '1.0', 'Test', datetime(), datetime());
         INSERT INTO sdef_documents (name, version, description, created_at, updated_at)
         VALUES ('other-doc', '1.0', 'Other', datetime(), datetime());"
    ).unwrap();

    drop(conn);
    db
}

#[test]
fn test_consistency_service_check_no_inconsistencies() {
    let db = setup_db();
    let service = ConsistencyService::new(db.clone());

    // Without any fingerprints, there should be no inconsistencies
    let issues = service.check("test-doc", CheckLevel::Fast).unwrap();
    assert!(issues.is_empty(), "No fingerprints → no inconsistencies");
}

#[test]
fn test_consistency_service_compute_hash_deterministic() {
    let hash1 = ConsistencyService::compute_hash("Hello, World!");
    let hash2 = ConsistencyService::compute_hash("Hello, World!");
    assert_eq!(hash1, hash2, "Same input must produce same hash");

    let hash3 = ConsistencyService::compute_hash("Hello, World?");
    assert_ne!(hash1, hash3, "Different input must produce different hash");
}

#[test]
fn test_consistency_service_hash_format() {
    let hash = ConsistencyService::compute_hash("test");
    assert_eq!(hash.len(), 64, "SHA-256 hex should be 64 characters");
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "Should be hex");
}

#[test]
fn test_consistency_checker_creation() {
    let db = setup_db();
    let checker = ConsistencyChecker::new(
        db.clone(),
        ConsistencyCheckerConfig {
            interval: Duration::from_secs(60),
            document_names: vec!["test-doc".to_string()],
            ..Default::default()
        },
    );
    // Verify the checker was created successfully (no panic)
    let result = checker.run_once().unwrap();
    assert_eq!(result, 0);
}

#[test]
fn test_fingerprint_repository_crud() {
    let db = setup_db();
    let repo = FingerprintRepository::from_arc(db.connection_arc());

    // Create fingerprint with all hashes matching (consistent)
    let fp = Fingerprint {
        entity_uri: "sdef://test-doc/entity/User".to_string(),
        document_name: "test-doc".to_string(),
        entity_type: "data_model".to_string(),
        sdef_hash: Some("abc123".to_string()),
        db_hash: Some("abc123".to_string()),
        code_hash: Some("abc123".to_string()),
        code_path: Some("src/user.rs".to_string()),
        last_checked_at: String::new(),
        last_consistent_at: None,
    };

    // Insert
    repo.upsert(&fp).expect("Upsert should succeed");

    // Query by document
    let fps = repo.list_by_document("test-doc").unwrap();
    assert_eq!(fps.len(), 1, "Should have 1 fingerprint");
    assert_eq!(fps[0].entity_uri, "sdef://test-doc/entity/User");

    // No inconsistencies when hashes match
    let inconsistent = repo.list_inconsistent("test-doc").unwrap();
    assert!(inconsistent.is_empty(), "Matching hashes should be consistent");

    // Update to be inconsistent
    let fp_inconsistent = Fingerprint {
        sdef_hash: Some("xyz789".to_string()),
        ..fp
    };
    repo.upsert(&fp_inconsistent).expect("Upsert updated");

    // Now there should be inconsistencies
    let inconsistent = repo.list_inconsistent("test-doc").unwrap();
    assert!(!inconsistent.is_empty(), "Different hashes should be inconsistent");
}

#[test]
fn test_fix_strategies_apply() {
    let db = setup_db();
    let service = ConsistencyService::new(db);

    // Fix with each strategy - should not error
    let inc = Inconsistency {
        entity_uri: "sdef://test-doc/entity/User".to_string(),
        sdef_hash: Some("abc".to_string()),
        db_hash: Some("def".to_string()),
        code_hash: Some("abc".to_string()),
    };

    for strategy in &[
        FixStrategy::SyncCodeToSdef,
        FixStrategy::RegenerateCode,
        FixStrategy::SyncDbToSdef,
        FixStrategy::SyncSdefToDb,
        FixStrategy::AcceptExternal,
    ] {
        service.fix(&inc, *strategy).expect("Fix should succeed");
    }
}

#[test]
fn test_consistency_across_documents() {
    let db = setup_db();
    let repo = FingerprintRepository::from_arc(db.connection_arc());

    // Add fingerprints for two different documents
    for (doc, uri, hash) in &[
        ("test-doc", "sdef://test-doc/entity/User", "aaa"),
        ("test-doc", "sdef://test-doc/entity/Task", "bbb"),
        ("other-doc", "sdef://other-doc/entity/Order", "ccc"),
    ] {
        repo.upsert(&Fingerprint {
            entity_uri: uri.to_string(),
            document_name: doc.to_string(),
            entity_type: "data_model".to_string(),
            sdef_hash: Some(hash.to_string()),
            db_hash: Some(hash.to_string()),
            code_hash: Some(hash.to_string()),
            code_path: None,
            last_checked_at: String::new(),
            last_consistent_at: None,
        }).unwrap();
    }

    // Each document should have its own fingerprints
    let test_doc_fps = repo.list_by_document("test-doc").unwrap();
    assert_eq!(test_doc_fps.len(), 2, "test-doc should have 2 fingerprints");

    let other_doc_fps = repo.list_by_document("other-doc").unwrap();
    assert_eq!(other_doc_fps.len(), 1, "other-doc should have 1 fingerprint");
}
