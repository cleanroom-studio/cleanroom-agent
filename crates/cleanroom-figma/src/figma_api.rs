//! Stage 1 (REST API): fetch a Figma file as JSON over HTTPS.
//!
//! See https://www.figma.com/developers/api#files for the upstream API.
//!
//! This module provides:
//!
//! - [`FigmaClient`] — minimal async client for `GET /v1/files/{file_key}`.
//! - [`FigmaFileJson`] — typed shape of the response (we declare only the
//!   fields we currently consume; extra fields are preserved via
//!   `#[serde(flatten)]` + `serde_json::Value` where useful).
//! - [`parse_file_json`] — minimal JSON-to-IR adapter used by both the
//!   REST and `.fig` paths.

use std::collections::BTreeMap;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, warn};

use super::figma_ir::{FigmaIr, FigmaIrNode, FigmaIrPage};

const FIGMA_API_BASE: &str = "https://api.figma.com/v1";

/// Async client for the Figma REST API.
///
/// Holds a personal access token. Each call to [`fetch_file_body`] / [`fetch_file`]
/// issues one HTTPS request.
#[derive(Debug, Clone)]
pub struct FigmaClient {
    token: String,
    base: String,
    http: reqwest::Client,
}

impl FigmaClient {
    /// Construct a client with a personal access token.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            base: FIGMA_API_BASE.to_string(),
            http: reqwest::Client::builder()
                .user_agent(concat!("cleanroom-figma/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("reqwest client builder should not fail"),
        }
    }

    /// Override the base URL (used by tests).
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    /// `GET /v1/files/{file_key}` — return the raw response body bytes.
    pub async fn fetch_file_body(&self, file_key: &str) -> Result<Vec<u8>> {
        let url = format!("{}/files/{}", self.base, file_key);
        debug!(%url, "GET Figma file");

        let resp = self
            .http
            .get(&url)
            .header("X-Figma-Token", &self.token)
            .send()
            .await
            .with_context(|| format!("requesting {url}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Figma API returned {}: {}", status, body));
        }

        let bytes = resp.bytes().await?.to_vec();
        Ok(bytes)
    }

    /// `GET /v1/files/{file_key}` — return the parsed JSON.
    pub async fn fetch_file(&self, file_key: &str) -> Result<FigmaFileJson> {
        let body = self.fetch_file_body(file_key).await?;
        let json: FigmaFileJson = serde_json::from_slice(&body)
            .with_context(|| "parsing Figma file JSON")?;
        Ok(json)
    }

    /// `GET /v1/files/{file_key}/variables/local` — return the design tokens.
    pub async fn fetch_variables(&self, file_key: &str) -> Result<Value> {
        let url = format!("{}/files/{}/variables/local", self.base, file_key);
        let resp = self
            .http
            .get(&url)
            .header("X-Figma-Token", &self.token)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            warn!(%status, "Figma variables endpoint returned non-2xx");
            return Ok(Value::Null);
        }
        Ok(resp.json().await?)
    }

    /// Parse raw JSON bytes into a typed [`FigmaFileJson`].
    pub fn parse_file_json_bytes(&self, bytes: &[u8]) -> Result<FigmaFileJson> {
        serde_json::from_slice(bytes).with_context(|| "parsing Figma file JSON")
    }

    /// Convenience: parse raw JSON bytes directly into a [`FigmaIr`].
    ///
    /// Used by the high-level [`crate::FigmaImporter::import`] when the
    /// source is a Figma REST API response. The `_self` parameter is
    /// required because Rust doesn't allow free functions as associated
    /// functions without a receiver in some cases; keeping it as a method
    /// on `FigmaClient` keeps the public API symmetric.
    pub fn parse_file_json(&self, bytes: &[u8]) -> Result<FigmaIr> {
        let json = self.parse_file_json_bytes(bytes)?;
        Ok(ir_from_file_json(&json))
    }
}

/// Typed view of the Figma REST API `/v1/files/{key}` response.
///
/// We only declare the fields we currently consume; unknown fields are
/// silently dropped (`#[serde(default)]` + struct field filtering).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigmaFileJson {
    /// Document name.
    #[serde(default)]
    pub name: String,

    /// Last-modified timestamp (Figma's `lastModified`, as a string).
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,

    /// Version string (e.g. `"1717344000.0"`).
    #[serde(default)]
    pub version: Option<String>,

    /// The document tree. `CANVAS` is the root; children are pages.
    pub document: FigmaDocumentNode,

    /// Styles (colors, text, effects, grids).
    #[serde(default)]
    pub styles: BTreeMap<String, FigmaStyle>,

    /// Component metadata keyed by node id.
    #[serde(default)]
    pub components: BTreeMap<String, FigmaComponentMeta>,
}

/// Figma document node. The `document` field is the root `CANVAS`; its
/// children are pages. Each page's children are the actual frames /
/// components / instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigmaDocumentNode {
    pub id: String,

    #[serde(default)]
    pub name: String,

    #[serde(default, rename = "type")]
    pub type_: Option<String>,

    #[serde(default)]
    pub children: Vec<FigmaDocumentNode>,
}

/// Figma style entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigmaStyle {
    #[serde(default, rename = "style_type")]
    pub style_type: Option<String>,

    #[serde(default)]
    pub name: Option<String>,

    #[serde(default)]
    pub description: Option<String>,

    /// Style-specific payload.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Figma component metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigmaComponentMeta {
    pub key: String,

    #[serde(default)]
    pub name: Option<String>,

    #[serde(default, rename = "description")]
    pub description: Option<String>,
}

// ============================================================================
// JSON → FigmaIr adapter
// ============================================================================

/// Build a [`FigmaIr`] from a parsed [`FigmaFileJson`].
pub fn ir_from_file_json(json: &FigmaFileJson) -> FigmaIr {
    let mut ir = FigmaIr {
        name: json.name.clone(),
        last_modified: json.last_modified.clone(),
        version: json.version.clone(),
        ..Default::default()
    };

    // The root node's children are pages (Figma node type "CANVAS").
    for page_node in &json.document.children {
        let page_id = page_node.id.clone();
        let page_name = page_node.name.clone();

        let mut nodes = Vec::new();
        for child in &page_node.children {
            walk_node(child, &mut nodes);
        }

        ir.pages.push(FigmaIrPage {
            id: page_id,
            name: page_name,
            nodes,
        });
    }

    // Variables are not in the file JSON itself; the importer fetches
    // them via a separate endpoint. We leave `variables` empty here; the
    // orchestrator (see `FigmaImporter::import`) is responsible for
    // merging in variables if needed.

    ir
}

fn walk_node(node: &FigmaDocumentNode, out: &mut Vec<FigmaIrNode>) {
    out.push(FigmaIrNode {
        id: node.id.clone(),
        name: node.name.clone(),
        type_: node.type_.clone().unwrap_or_else(|| "UNKNOWN".to_string()),
        children: node
            .children
            .iter()
            .flat_map(|c| {
                let mut sub = Vec::new();
                walk_node(c, &mut sub);
                sub
            })
            .collect(),
        extra: BTreeMap::new(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_figma_file_json() {
        let body = br#"{
            "name": "Sample",
            "lastModified": "2026-06-01T00:00:00Z",
            "version": "1717200000.0",
            "document": {
                "id": "0:0",
                "name": "Document",
                "type": "DOCUMENT",
                "children": [
                    {
                        "id": "0:1",
                        "name": "Page 1",
                        "type": "CANVAS",
                        "children": [
                            {
                                "id": "1:1",
                                "name": "Frame",
                                "type": "FRAME",
                                "children": []
                            }
                        ]
                    }
                ]
            },
            "styles": {},
            "components": {}
        }"#;

        let json: FigmaFileJson = serde_json::from_slice(body).unwrap();
        let ir = ir_from_file_json(&json);
        assert_eq!(ir.name, "Sample");
        assert_eq!(ir.pages.len(), 1);
        assert_eq!(ir.pages[0].name, "Page 1");
        assert_eq!(ir.pages[0].nodes.len(), 1);
        assert_eq!(ir.pages[0].nodes[0].type_, "FRAME");
    }

    #[test]
    fn missing_optional_fields_default_cleanly() {
        // No `lastModified` / `version` / `styles` / `components`.
        let body = br#"{
            "name": "Minimal",
            "document": {
                "id": "0:0",
                "name": "Document",
                "children": []
            }
        }"#;
        let json: FigmaFileJson = serde_json::from_slice(body).unwrap();
        assert_eq!(json.name, "Minimal");
        assert!(json.last_modified.is_none());
    }
}
