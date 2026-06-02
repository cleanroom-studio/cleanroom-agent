//! # cleanroom-sketch
//!
//! Sketch → S.DEF UI import pipeline.
//!
//! This crate converts Sketch designs into S.DEF `UserInterface` documents.
//! It supports two input paths:
//!
//! 1. **Sketch Cloud REST API** — live data fetched with a personal access
//!    token (see [`sketch_api::SketchCloudClient`]).
//! 2. **Local `.sketch` file** — offline / version-controlled Sketch
//!    exports. The `.sketch` format is a ZIP archive of JSON files
//!    (one per page) plus image previews (see [`sketch_zip`]).
//!
//! Both paths feed into the same intermediate representation
//! ([`sketch_ir`]), which is then mapped to S.DEF UI types
//! ([`sketch_to_sdef`]).
//!
//! ## Pipeline
//!
//! ```text
//! Sketch Cloud API ──┐
//!                    ├──► SketchIr ──► sketch_to_sdef ──► sdef_core::UserInterface
//! .sketch (zip)  ────┘
//! ```
//!
//! ## Entry point
//!
//! Use [`SketchImporter::import`] with either a [`SketchSource::Cloud`] or
//! a [`SketchSource::File`] variant.
//!
//! ## Relationship to cleanroom-figma
//!
//! This crate is a sibling of [`cleanroom_figma`]. Both produce
//! `sdef_core::UserInterface` documents and stamp a
//! `UIImportProvenance` block whose `source` field is `"figma"` or
//! `"sketch"` respectively. The S.DEF schema was designed to be
//! tool-agnostic enough that both importers can land in the same
//! document without conflicting.
//!
//! See:
//! - `S.DEF/proposals/0000-figma-ui-import.md`
//! - `S.DEF/proposals/0001-sketch-ui-import.md`

#![deny(missing_debug_implementations)]
#![warn(unreachable_pub)]

pub mod sketch_api;
pub mod sketch_zip;
pub mod sketch_ir;
pub mod sketch_to_sdef;

pub use sketch_api::SketchCloudClient;
pub use sketch_zip::SketchFile;
pub use sketch_ir::{SketchIr, SketchIrNode, SketchIrPage};

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use sha2::{Digest, Sha256};
use sdef_core::types::ui::UserInterface;
use sdef_core::types::ui_figma::UIImportSource;
use tracing::info;

/// Input source for the Sketch importer.
#[derive(Debug, Clone)]
pub enum SketchSource {
    /// Fetch from the Sketch Cloud REST API.
    Cloud {
        /// Sketch document share URL or document id.
        document_id: String,
        /// Optional Sketch access token. Falls back to the
        /// `SKETCH_TOKEN` environment variable.
        token: Option<String>,
    },
    /// Parse a local `.sketch` file from disk.
    File {
        /// Path to the `.sketch` file.
        path: std::path::PathBuf,
    },
}

/// The imported S.DEF UI document plus its provenance.
#[derive(Debug, Clone)]
pub struct SketchImportResult {
    /// The S.DEF UI document.
    pub user_interface: UserInterface,

    /// SHA-256 of the source content (HTTP body bytes or file bytes).
    pub content_hash: String,

    /// Per-page mapping: Sketch page id → S.DEF UI shard id.
    pub page_map: BTreeMap<String, String>,

    /// Per-element mapping: Sketch layer id → S.DEF element id.
    pub node_map: BTreeMap<String, String>,
}

/// The high-level entry point for Sketch imports.
#[derive(Debug, Default)]
pub struct SketchImporter {
    _private: (),
}

impl SketchImporter {
    /// Create a new importer with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Import a Sketch source and produce an S.DEF UI document.
    pub async fn import(&self, source: SketchSource) -> Result<SketchImportResult> {
        let started = std::time::Instant::now();

        // Stage 1: fetch / parse → raw bytes
        let (raw_bytes, source_id) = match &source {
            SketchSource::Cloud { document_id, token } => {
                let token = token
                    .clone()
                    .or_else(|| std::env::var("SKETCH_TOKEN").ok())
                    .ok_or_else(|| {
                        anyhow!(
                            "Sketch Cloud API requires a token. Pass `token` or set the \
                             SKETCH_TOKEN environment variable."
                        )
                    })?;
                let client = SketchCloudClient::new(token);
                let body = client
                    .fetch_document(document_id)
                    .await
                    .with_context(|| format!("fetching Sketch document {document_id}"))?;
                (body, document_id.clone())
            }
            SketchSource::File { path } => {
                let bytes = std::fs::read(path)
                    .with_context(|| format!("reading .sketch file {}", path.display()))?;
                let id = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                (bytes, id)
            }
        };

        // Hash the raw source for round-trip integrity.
        let content_hash: String = {
            let mut hasher = Sha256::new();
            hasher.update(&raw_bytes);
            hex::encode(hasher.finalize())
        };
        info!(
            source = "sketch",
            source_id = %source_id,
            bytes = raw_bytes.len(),
            content_hash = %content_hash,
            "Sketch source fetched"
        );

        // Stage 2: raw bytes → SketchIr
        let ir = match &source {
            SketchSource::Cloud { .. } => sketch_zip::parse_sketch_zip(&raw_bytes, &source_id)?,
            SketchSource::File { .. } => sketch_zip::parse_sketch_zip(&raw_bytes, &source_id)?,
        };
        info!(pages = ir.pages.len(), "SketchIr built");

        // Stage 3: SketchIr → S.DEF UserInterface
        let mapper = sketch_to_sdef::SketchToSdef::new();
        let (user_interface, page_map, node_map) = mapper.map(&ir, &source_id);

        // Stamp provenance onto the S.DEF document.
        let mut ui = user_interface;
        let prov = sdef_core::types::ui_figma::UIImportProvenance {
            source: UIImportSource::Sketch,
            source_id: source_id.clone(),
            source_version: ir.version.clone(),
            imported_at: Some(Utc::now().to_rfc3339()),
            content_hash: Some(content_hash.clone()),
            page_map: Some(page_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            node_map: Some(node_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        };
        ui.ui_provenance = Some(prov);

        info!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Sketch import complete"
        );

        Ok(SketchImportResult {
            user_interface: ui,
            content_hash,
            page_map,
            node_map,
        })
    }

    /// Compute the SHA-256 of a file on disk.
    pub fn hash_file(path: impl AsRef<Path>) -> Result<String> {
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        Ok(hex::encode(hasher.finalize()))
    }
}

/// Sanity-check a Sketch Cloud document id before doing any work.
pub fn validate_document_id(id: &str) -> Result<()> {
    if id.is_empty() {
        return Err(anyhow!("Sketch document id is empty"));
    }
    if id.len() < 8 {
        return Err(anyhow!("Sketch document id too short: {id:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_document_id_accepts_typical() {
        // Sketch Cloud document ids are typically 32-char hex (UUID-like).
        assert!(validate_document_id("a1b2c3d4e5f6789012345678abcdef01").is_ok());
    }

    #[test]
    fn validate_document_id_rejects_empty() {
        assert!(validate_document_id("").is_err());
    }

    #[test]
    fn validate_document_id_rejects_too_short() {
        assert!(validate_document_id("abc").is_err());
    }

    #[test]
    fn hash_file_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.sketch");
        std::fs::write(&path, b"hello world").unwrap();
        let hash = SketchImporter::hash_file(&path).unwrap();
        // sha256("hello world")
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }
}
