//! # cleanroom-figma
//!
//! Figma → S.DEF UI import pipeline.
//!
//! This crate converts Figma designs into S.DEF `UserInterface` documents.
//! It supports two input paths:
//!
//! 1. **Figma REST API** — live data fetched with a personal access token
//!    (see [`figma_api::FigmaClient`]).
//! 2. **Binary `.fig` file** — offline / version-controlled Figma desktop
//!    exports (see [`figma_fig`]).
//!
//! Both paths feed into the same intermediate representation ([`figma_ir`]),
//! which is then mapped to S.DEF UI types ([`figma_to_sdef`]).
//!
//! ## Pipeline
//!
//! ```text
//! Figma REST API  ──┐
//!                   ├──► FigmaIr ──► figma_to_sdef ──► sdef_core::UserInterface
//! .fig binary file ─┘
//! ```
//!
//! ## Entry point
//!
//! Use [`FigmaImporter::import`] with either an [`FigmaSource::Rest`] or
//! an [`FigmaSource::FigFile`] variant.
//!
//! ## Schema extensions used
//!
//! This crate depends on the Figma proposal extensions to S.DEF
//! (`S.DEF/proposals/0000-figma-ui-import.md`):
//!
//! - `UIImportProvenance` — `UserInterface::ui_provenance`
//! - `UIComponentVariant` / `UIComponentProperty` — `UIComponentType`
//! - `UIVariableMode` / `UIVariable.mode_values` — multi-mode tokens
//! - `UILayoutGrid` — Figma layout grids
//! - `UIFrame.primary_axis_sizing_mode` / `counter_axis_sizing_mode` /
//!   `layout_align` / `layout_grow` — auto-layout round-trip
//!
//! See [`S.DEF/proposals/0000-figma-ui-import.md`] for the full spec.

#![deny(missing_debug_implementations)]
#![warn(unreachable_pub)]

pub mod figma_api;
pub mod figma_fig;
pub mod figma_ir;
pub mod figma_to_sdef;

pub use figma_api::FigmaClient;
pub use figma_fig::FigFile;
pub use figma_ir::{FigmaIr, FigmaIrNode, FigmaIrPage, FigmaIrVariable};

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use sha2::{Digest, Sha256};
use sdef_core::types::ui::UserInterface;
use sdef_core::types::ui_figma::UIImportSource;
use tracing::{info, warn};

/// Input source for the Figma importer.
#[derive(Debug, Clone)]
pub enum FigmaSource {
    /// Fetch from the Figma REST API.
    Rest {
        /// Figma file key (the segment after `figma.com/file/` or
        /// `figma.com/design/`).
        file_key: String,
        /// Optional Figma personal access token. Falls back to the
        /// `FIGMA_TOKEN` environment variable.
        token: Option<String>,
        /// Optional: import only these page ids. `None` imports all pages.
        page_ids: Option<Vec<String>>,
    },
    /// Parse a binary `.fig` file from disk.
    FigFile {
        /// Path to the `.fig` file.
        path: std::path::PathBuf,
    },
}

/// The imported S.DEF UI document plus its provenance.
#[derive(Debug, Clone)]
pub struct FigmaImportResult {
    /// The S.DEF UI document.
    pub user_interface: UserInterface,

    /// SHA-256 of the source content (HTTP body bytes or file bytes).
    /// Stored in `UserInterface::ui_provenance.content_hash`.
    pub content_hash: String,

    /// Per-page mapping: Figma page id → S.DEF UI shard id.
    pub page_map: BTreeMap<String, String>,

    /// Per-element mapping: Figma node id → S.DEF element id.
    pub node_map: BTreeMap<String, String>,
}

/// The high-level entry point for Figma imports.
#[derive(Debug, Default)]
pub struct FigmaImporter {
    // Reserved for future configuration (e.g. import filters, retry policy).
    _private: (),
}

impl FigmaImporter {
    /// Create a new importer with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Import a Figma source and produce an S.DEF UI document.
    pub async fn import(&self, source: FigmaSource) -> Result<FigmaImportResult> {
        let started = std::time::Instant::now();

        // Stage 1: fetch / parse → raw bytes
        let (raw_bytes, source_kind, source_id) = match &source {
            FigmaSource::Rest { file_key, token, page_ids: _ } => {
                let token = token
                    .clone()
                    .or_else(|| std::env::var("FIGMA_TOKEN").ok())
                    .ok_or_else(|| {
                        anyhow!(
                            "Figma REST API requires a token. Pass `token` or set the \
                             FIGMA_TOKEN environment variable."
                        )
                    })?;
                let client = FigmaClient::new(token);
                let body = client
                    .fetch_file_body(file_key)
                    .await
                    .with_context(|| format!("fetching Figma file {file_key}"))?;
                (body, UIImportSource::Figma, file_key.clone())
            }
            FigmaSource::FigFile { path } => {
                let bytes = std::fs::read(path)
                    .with_context(|| format!("reading .fig file {}", path.display()))?;
                let id = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                (bytes, UIImportSource::Figma, id)
            }
        };

        // Hash the raw source for round-trip integrity.
        let content_hash: String = {
            let mut hasher = Sha256::new();
            hasher.update(&raw_bytes);
            hex::encode(hasher.finalize())
        };
        info!(
            source = ?source_kind,
            source_id = %source_id,
            bytes = raw_bytes.len(),
            content_hash = %content_hash,
            "Figma source fetched"
        );

        // Stage 2: raw bytes → FigmaIr
        let ir = match &source {
            FigmaSource::Rest { .. } => {
                let api = figma_api::FigmaClient::new(String::new());
                api.parse_file_json(&raw_bytes)?
            }
            FigmaSource::FigFile { .. } => figma_fig::FigFile::parse(&raw_bytes)?.into_ir()?,
        };
        info!(pages = ir.pages.len(), "FigmaIr built");

        // Stage 3: FigmaIr → S.DEF UserInterface
        let mapper = figma_to_sdef::FigmaToSdef::new();
        let (user_interface, page_map, node_map) = mapper.map(&ir, source_kind, &source_id);

        // Stamp provenance onto the S.DEF document.
        let mut ui = user_interface;
        let prov = sdef_core::types::ui_figma::UIImportProvenance {
            source: source_kind,
            source_id: source_id.clone(),
            source_version: ir.last_modified.clone(),
            imported_at: Some(Utc::now().to_rfc3339()),
            content_hash: Some(content_hash.clone()),
            page_map: Some(page_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            node_map: Some(node_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        };
        ui.ui_provenance = Some(prov);

        info!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Figma import complete"
        );

        Ok(FigmaImportResult {
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

/// Sanity-check a Figma source id (file key) before doing any work.
pub fn validate_file_key(key: &str) -> Result<()> {
    if key.is_empty() {
        return Err(anyhow!("Figma file key is empty"));
    }
    if key.len() < 8 {
        return Err(anyhow!("Figma file key too short: {key:?}"));
    }
    if !key.chars().all(|c| c.is_ascii_alphanumeric()) {
        warn!(key = %key, "Figma file key contains non-alphanumeric characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_file_key_accepts_typical() {
        // Typical Figma file keys are 22-char alphanumeric strings.
        assert!(validate_file_key("aBcD1234EfGh5678IjKl90").is_ok());
    }

    #[test]
    fn validate_file_key_rejects_empty() {
        assert!(validate_file_key("").is_err());
    }

    #[test]
    fn validate_file_key_rejects_too_short() {
        assert!(validate_file_key("abc").is_err());
    }

    #[test]
    fn hash_file_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.fig");
        std::fs::write(&path, b"hello world").unwrap();
        let hash = FigmaImporter::hash_file(&path).unwrap();
        // sha256("hello world")
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }
}
