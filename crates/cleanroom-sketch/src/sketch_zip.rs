//! Stage 1: parse a `.sketch` file (ZIP archive of JSON).
//!
//! ## File layout
//!
//! ```text
//! design.sketch         (ZIP archive)
//! ├── document.json     — top-level document metadata
//! ├── meta.json         — app version + format version
//! ├── pages/
//! │   ├── {uuid-1}.json — page 1 (SketchLayerNode tree)
//! │   ├── {uuid-2}.json — page 2
//! │   └── ...
//! ├── previews/
//! │   └── {uuid}.png    — page preview images
//! ├── fonts/            — embedded fonts (TTF/OTF)
//! └── images/           — image assets
//! ```
//!
//! `document.json` references pages by filename; `meta.json` carries
//! versioning info we record as `UIImportProvenance.source_version`.

use std::collections::BTreeMap;
use std::io::Read;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

use super::sketch_api::{ir_from_page_json, SketchPageJson};
use super::sketch_ir::{SketchIr, SketchIrNode};

/// Parsed `.sketch` file: the top-level document + per-page JSON trees.
#[derive(Debug, Clone)]
pub struct SketchFile {
    /// Top-level document metadata.
    pub document: SketchDocumentJson,
    /// Format / app metadata.
    pub meta: SketchMetaJson,
    /// Per-page id → per-page JSON content.
    pub pages: Vec<ParsedSketchPage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchDocumentJson {
    /// Document name.
    #[serde(default)]
    pub name: String,

    /// Format version (e.g. `"2.0"`).
    #[serde(default)]
    pub version: Option<String>,

    /// Last-modified timestamp.
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,

    /// References to the page files. Each entry has a `id` and a
    /// `filepath` like `"pages/{uuid}.json"`.
    #[serde(default)]
    pub pages: Vec<SketchDocumentPageRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchDocumentPageRef {
    /// Page id (UUID).
    #[serde(rename = "_id", default)]
    pub id: String,

    /// Path inside the ZIP (e.g. `"pages/abc-123.json"`).
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SketchMetaJson {
    /// App that created the file (`"Sketch"`).
    #[serde(default)]
    pub app: String,

    /// App version (e.g. `"98"`).
    #[serde(default, rename = "appVersion")]
    pub app_version: String,

    /// Format version (e.g. `"2.0"`).
    #[serde(default, rename = "version")]
    pub version: String,

    /// Build identifier.
    #[serde(default, rename = "build")]
    pub build: u64,
}

#[derive(Debug, Clone)]
pub struct ParsedSketchPage {
    /// Page id from `document.json`.
    pub id: String,
    /// Page display name (from the page's own JSON `meta.name`).
    pub name: String,
    /// Page JSON tree.
    pub json: SketchPageJson,
}

/// Errors that can occur while parsing a `.sketch` file.
#[derive(Debug, Error)]
pub enum SketchDecodeError {
    /// I/O error reading the ZIP.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// ZIP parsing failed.
    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),

    /// Required entry missing from the archive.
    #[error("required entry missing from .sketch file: {0}")]
    MissingEntry(String),

    /// JSON parsing failed.
    #[error("invalid JSON in {0}: {1}")]
    InvalidJson(String, String),
}

impl SketchFile {
    /// Parse a `.sketch` file from raw bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let cursor = std::io::Cursor::new(bytes);
        let mut zip = zip::ZipArchive::new(cursor)
            .context("opening .sketch file as ZIP")?;

        // Read document.json
        let document: SketchDocumentJson = {
            let s = read_entry(&mut zip, "document.json")?;
            serde_json::from_slice(&s)
                .map_err(|e| anyhow!("invalid JSON in document.json: {e}"))?
        };

        // Read meta.json (optional but expected)
        let meta: SketchMetaJson = match read_entry(&mut zip, "meta.json") {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
            Err(_) => SketchMetaJson::default(),
        };

        // Read each page referenced from document.json
        let mut pages = Vec::new();
        for page_ref in &document.pages {
            // Sketch's "name" field in document.json actually holds the
            // path within the ZIP (e.g. "pages/abc-123.json"). We try
            // that first, then fall back to constructing the conventional
            // path from the page id.
            let candidates = [
                page_ref.name.clone(),
                format!("pages/{}", page_ref.id),
                format!("pages/{}.json", page_ref.id),
            ];

            let mut parsed = false;
            for path in &candidates {
                if let Ok(bytes) = read_entry(&mut zip, path) {
                    match serde_json::from_slice::<SketchPageJson>(&bytes) {
                        Ok(json) => {
                            let name = json
                                .meta
                                .name
                                .clone()
                                .unwrap_or_else(|| page_ref.id.clone());
                            pages.push(ParsedSketchPage {
                                id: page_ref.id.clone(),
                                name,
                                json,
                            });
                            parsed = true;
                            break;
                        }
                        Err(e) => warn!(
                            path = %path,
                            error = %e,
                            "invalid JSON in page; trying next candidate"
                        ),
                    }
                }
            }

            if !parsed {
                warn!(
                    page_id = %page_ref.id,
                    "skipped page: no readable JSON found at any candidate path"
                );
            }
        }

        debug!(
            document_name = %document.name,
            page_count = pages.len(),
            app = %meta.app,
            app_version = %meta.app_version,
            "Parsed .sketch file"
        );

        Ok(Self {
            document,
            meta,
            pages,
        })
    }

    /// Convert the parsed `.sketch` into a [`SketchIr`].
    pub fn into_ir(&self) -> SketchIr {
        let pages = self
            .pages
            .iter()
            .map(|p| ir_from_page_json(&p.id, &p.name, &p.json))
            .collect();

        SketchIr {
            name: self.document.name.clone(),
            last_modified: self.document.last_modified.clone(),
            version: Some(self.meta.version.clone())
                .filter(|s| !s.is_empty())
                .or_else(|| Some(self.meta.app_version.clone()).filter(|s| !s.is_empty())),
            pages,
            styles: BTreeMap::new(),
        }
    }
}

fn read_entry<R: std::io::Read + std::io::Seek>(
    zip: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<Vec<u8>> {
    let mut f = zip
        .by_name(name)
        .map_err(|e| anyhow!("missing entry {name:?}: {e}"))?;
    let mut out = Vec::with_capacity(f.size() as usize);
    f.read_to_end(&mut out)?;
    Ok(out)
}

/// Convenience: parse raw `.sketch` bytes directly into a [`SketchIr`].
///
/// This is the single entry point used by [`crate::SketchImporter::import`]
/// for both Cloud and File sources, because Sketch Cloud's
/// `GET /documents/{id}/download` returns the same `.sketch` ZIP as the
/// on-disk format.
pub fn parse_sketch_zip(bytes: &[u8], fallback_name: &str) -> Result<SketchIr> {
    let file = SketchFile::parse(bytes)?;
    let mut ir = file.into_ir();
    if ir.name.is_empty() {
        ir.name = fallback_name.to_string();
    }
    Ok(ir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal `.sketch` file (ZIP) with one page for testing.
    fn make_minimal_sketch() -> Vec<u8> {
        let document = serde_json::json!({
            "_class": "document",
            "name": "Sample",
            "version": "2.0",
            "lastModified": "2026-06-01T00:00:00Z",
            "pages": [
                { "_id": "page-1", "name": "pages/page-1.json" }
            ]
        });
        let meta = serde_json::json!({
            "app": "Sketch",
            "appVersion": "98",
            "version": "2.0",
            "build": 12345
        });
        let page = serde_json::json!({
            "meta": { "name": "Page 1" },
            "layers": [
                {
                    "do_objectID": "L1",
                    "_class": "MSArtboardGroup",
                    "name": "Artboard 1",
                    "frame": { "x": 0, "y": 0, "width": 320, "height": 480 },
                    "layers": [
                        {
                            "do_objectID": "L2",
                            "_class": "MSTextLayer",
                            "name": "Hello",
                            "frame": { "x": 10, "y": 10, "width": 100, "height": 20 },
                            "layers": []
                        }
                    ]
                }
            ]
        });

        let mut zip_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut zip_bytes);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("document.json", opts).unwrap();
            zip.write_all(serde_json::to_string_pretty(&document).unwrap().as_bytes())
                .unwrap();
            zip.start_file("meta.json", opts).unwrap();
            zip.write_all(serde_json::to_string_pretty(&meta).unwrap().as_bytes())
                .unwrap();
            zip.start_file("pages/page-1.json", opts).unwrap();
            zip.write_all(serde_json::to_string_pretty(&page).unwrap().as_bytes())
                .unwrap();
            zip.finish().unwrap();
        }
        zip_bytes
    }

    #[test]
    fn parses_minimal_sketch() {
        let bytes = make_minimal_sketch();
        let parsed = SketchFile::parse(&bytes).expect("parse should succeed");
        assert_eq!(parsed.document.name, "Sample");
        assert_eq!(parsed.document.version.as_deref(), Some("2.0"));
        assert_eq!(parsed.pages.len(), 1);
        assert_eq!(parsed.pages[0].id, "page-1");
        assert_eq!(parsed.pages[0].name, "Page 1");
    }

    #[test]
    fn into_ir_walks_recursively() {
        let parsed = SketchFile::parse(&make_minimal_sketch()).unwrap();
        let ir = parsed.into_ir();
        assert_eq!(ir.name, "Sample");
        assert_eq!(ir.pages.len(), 1);
        assert_eq!(ir.pages[0].nodes.len(), 1);
        assert_eq!(ir.pages[0].nodes[0].id, "L1");
        assert_eq!(ir.pages[0].nodes[0].children.len(), 1);
        assert_eq!(ir.pages[0].nodes[0].children[0].id, "L2");
    }

    #[test]
    fn rejects_garbage_bytes() {
        // Not a ZIP at all.
        let err = SketchFile::parse(b"not a zip file").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("ZIP") || msg.contains("zip"),
            "expected a ZIP-related error, got: {msg}"
        );
    }

    #[test]
    fn handles_missing_meta_json_gracefully() {
        // A `.sketch` file without meta.json should still parse, with
        // an empty SketchMetaJson default.
        let mut zip_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut zip_bytes);
            let mut zip = zip::ZipWriter::new(cursor);
            let opts = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            let document = serde_json::json!({
                "_class": "document",
                "name": "Minimal",
                "pages": []
            });
            zip.start_file("document.json", opts).unwrap();
            zip.write_all(serde_json::to_string(&document).unwrap().as_bytes())
                .unwrap();
            zip.finish().unwrap();
        }
        let parsed = SketchFile::parse(&zip_bytes).expect("should parse without meta.json");
        assert_eq!(parsed.document.name, "Minimal");
        assert_eq!(parsed.meta.app, ""); // default
    }
}
