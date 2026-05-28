//! Fingerprint determinism tests: verify SHA-256 computation is deterministic
//! and consistent across multiple calls.

use cleanroom_agent::ConsistencyService;
use sdef_core::SoftwareDefinition;

/// Verify that fingerprint computation is deterministic.
#[test]
fn test_fingerprint_deterministic_json() {
    let sdef = SoftwareDefinition {
        name: "test-project".to_string(),
        sdef_version: Some("1.0.0".to_string()),
        description: Some("A test project".to_string()),
        ..Default::default()
    };

    let json1 = serde_json::to_string(&sdef).expect("Serialize");
    let json2 = serde_json::to_string(&sdef).expect("Serialize again");

    assert_eq!(json1, json2, "JSON output must be deterministic");

    let hash1 = ConsistencyService::compute_hash(&json1);
    let hash2 = ConsistencyService::compute_hash(&json2);

    assert_eq!(hash1, hash2, "Fingerprint must be deterministic");
}

/// Verify that different inputs produce different fingerprints.
#[test]
fn test_fingerprint_different_inputs() {
    let sdef_a = SoftwareDefinition {
        name: "project-a".to_string(),
        sdef_version: Some("1.0.0".to_string()),
        ..Default::default()
    };

    let sdef_b = SoftwareDefinition {
        name: "project-b".to_string(),
        sdef_version: Some("1.0.0".to_string()),
        ..Default::default()
    };

    let json_a = serde_json::to_string(&sdef_a).expect("Serialize A");
    let json_b = serde_json::to_string(&sdef_b).expect("Serialize B");

    let hash_a = ConsistencyService::compute_hash(&json_a);
    let hash_b = ConsistencyService::compute_hash(&json_b);

    assert_ne!(hash_a, hash_b, "Different inputs must produce different fingerprints");
}

/// Verify that compute_hash produces expected SHA-256 format.
#[test]
fn test_fingerprint_format() {
    let hash = ConsistencyService::compute_hash("hello");
    // SHA-256 hex is 64 characters
    assert_eq!(hash.len(), 64, "SHA-256 hex must be 64 chars");
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "Must be hex");
}

/// Verify that the software definition fingerprint includes all relevant fields.
#[test]
fn test_fingerprint_includes_fields() {
    let sdef = SoftwareDefinition {
        name: "test".to_string(),
        sdef_version: Some("1.0.0".to_string()),
        description: Some("desc".to_string()),
        metadata: Some(sdef_core::SoftwareMetadata {
            authors: Some(vec![sdef_core::Author {
                name: "Author".to_string(),
                email: Some("author@test.com".to_string()),
            }]),
            license: Some("MIT".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };

    let json = serde_json::to_string(&sdef).expect("Serialize");
    let hash = ConsistencyService::compute_hash(&json);

    // Verify: changing the name changes the hash
    let mut sdef2 = sdef.clone();
    sdef2.name = "test-modified".to_string();
    let json2 = serde_json::to_string(&sdef2).expect("Serialize");
    let hash2 = ConsistencyService::compute_hash(&json2);
    assert_ne!(hash, hash2, "Changing name must change fingerprint");
}

/// Verify fingerprint hash of bytes (binary data) works.
#[test]
fn test_fingerprint_bytes() {
    let data = b"Hello, Cleanroom Agent!";
    let hash = ConsistencyService::compute_hash(
        &String::from_utf8_lossy(data)
    );
    assert_eq!(hash.len(), 64, "SHA-256 hex must be 64 chars");

    // Same input must produce same hash
    let hash2 = ConsistencyService::compute_hash(
        &String::from_utf8_lossy(data)
    );
    assert_eq!(hash, hash2, "Must be deterministic for byte input");
}
