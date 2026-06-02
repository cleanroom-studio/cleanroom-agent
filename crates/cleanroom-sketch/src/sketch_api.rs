//! Stage 1 (REST API): fetch a Sketch Cloud document as a ZIP blob.
//!
//! Sketch Cloud exposes a public REST API for sharing documents. The
//! "Download Document" endpoint returns the same `.sketch` file format
//! as a downloadable ZIP archive, so this client and [`crate::sketch_zip`]
//! share most of the parsing logic.
//!
//! See <https://www.sketch.com/docs/cloud-api/> for the upstream API.
//!
//! ## Authentication
//!
//! The Sketch Cloud API uses a bearer token in the `Authorization` header.
//! Tokens are issued at <https://www.sketch.com/settings/tokens/>.

use std::collections::BTreeMap;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use super::sketch_ir::{SketchIr, SketchIrNode, SketchIrPage};

const SKETCH_API_BASE: &str = "https://api.sketch.com/v1";

/// Async client for the Sketch Cloud REST API.
#[derive(Debug, Clone)]
pub struct SketchCloudClient {
    token: String,
    base: String,
    http: reqwest::Client,
}

impl SketchCloudClient {
    /// Construct a client with a Sketch Cloud access token.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            base: SKETCH_API_BASE.to_string(),
            http: reqwest::Client::builder()
                .user_agent(concat!("cleanroom-sketch/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("reqwest client builder should not fail"),
        }
    }

    /// Override the base URL (used by tests).
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    /// `GET /documents/{id}/download` — returns the raw `.sketch` ZIP bytes.
    pub async fn fetch_document(&self, document_id: &str) -> Result<Vec<u8>> {
        let url = format!("{}/documents/{}/download", self.base, document_id);
        debug!(%url, "GET Sketch document");

        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .with_context(|| format!("requesting {url}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Sketch API returned {}: {}", status, body));
        }

        let bytes = resp.bytes().await?.to_vec();
        Ok(bytes)
    }

    /// `GET /documents/{id}` — return the document metadata (not the full
    /// document tree; the tree is in the ZIP). Useful for displaying
    /// "imported from Sketch X" provenance.
    pub async fn fetch_metadata(&self, document_id: &str) -> Result<SketchDocumentMeta> {
        let url = format!("{}/documents/{}", self.base, document_id);
        let resp = self.http.get(&url).bearer_auth(&self.token).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(anyhow!("Sketch API returned {} for metadata", status));
        }
        let meta: SketchDocumentMeta = resp.json().await?;
        Ok(meta)
    }
}

/// Sketch Cloud document metadata response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchDocumentMeta {
    #[serde(default)]
    pub name: Option<String>,

    #[serde(default)]
    pub version: Option<String>,

    #[serde(default, rename = "updatedAt")]
    pub updated_at: Option<String>,
}

// ============================================================================
// JSON view of the per-page Sketch files inside the ZIP
// ============================================================================
//
// Sketch's per-page JSON has many fields. We only declare the ones we
// currently consume; the IR layer retains unknown fields in
// `SketchIrNode.extra` for future expansion.

/// Top-level shape of a `pages/{page-uuid}.json` file inside a `.sketch`
/// archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchPageJson {
    /// Page metadata.
    #[serde(default)]
    pub meta: SketchPageMeta,

    /// Top-level layers on the page.
    #[serde(default)]
    pub layers: Vec<SketchLayerNode>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SketchPageMeta {
    #[serde(default)]
    pub name: Option<String>,

    #[serde(default, rename = "createdAt")]
    pub created_at: Option<String>,

    #[serde(default, rename = "updatedAt")]
    pub updated_at: Option<String>,
}

/// Recursive layer node from a Sketch page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchLayerNode {
    /// Stable layer id (UUID).
    #[serde(rename = "do_objectID", default)]
    pub do_object_id: String,

    /// Human-readable name.
    #[serde(default)]
    pub name: String,

    /// Class discriminator. Sketch uses class names like
    /// `MSArtboardGroup`, `MSSymbolMaster`, `MSSymbolInstance`,
    /// `MSShapeGroup`, `MSTextLayer`, `MSRectangleShape`, etc.
    #[serde(default, rename = "_class")]
    pub class: Option<String>,

    /// Frame of the layer (x, y, width, height).
    #[serde(default)]
    pub frame: Option<SketchFrame>,

    /// Children layers (for groups / artboards / symbols).
    #[serde(default)]
    pub layers: Vec<SketchLayerNode>,

    /// Any other Sketch-specific fields, preserved verbatim for future use.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SketchFrame {
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
    #[serde(default)]
    pub width: f64,
    #[serde(default)]
    pub height: f64,
}

/// Build a [`SketchIrPage`] from a parsed [`SketchPageJson`].
///
/// The top-level `layers` of the page become the page's `nodes`. Children
/// remain nested inside their parent's `children` field, preserving the
/// Sketch tree structure (as opposed to flattening).
pub fn ir_from_page_json(page_id: &str, page_name: &str, json: &SketchPageJson) -> SketchIrPage {
    let nodes = json.layers.iter().map(ir_node_from_layer).collect();
    SketchIrPage {
        id: page_id.to_string(),
        name: page_name.to_string(),
        nodes,
    }
}

fn ir_node_from_layer(layer: &SketchLayerNode) -> SketchIrNode {
    SketchIrNode {
        id: layer.do_object_id.clone(),
        name: layer.name.clone(),
        type_: layer
            .class
            .clone()
            .unwrap_or_else(|| "UNKNOWN".to_string()),
        children: layer.layers.iter().map(ir_node_from_layer).collect(),
        extra: layer.extra.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_sketch_page_json() {
        let body = br#"{
            "meta": { "name": "Page 1" },
            "layers": [
                {
                    "do_objectID": "AAAA",
                    "_class": "MSArtboardGroup",
                    "name": "Artboard 1",
                    "frame": { "x": 0, "y": 0, "width": 320, "height": 480 },
                    "layers": []
                }
            ]
        }"#;
        let json: SketchPageJson = serde_json::from_slice(body).unwrap();
        assert_eq!(json.layers.len(), 1);
        assert_eq!(json.layers[0].do_object_id, "AAAA");
        assert_eq!(json.layers[0].class.as_deref(), Some("MSArtboardGroup"));
    }

    #[test]
    fn ir_from_page_json_walks_recursively() {
        let body = br#"{
            "meta": {},
            "layers": [
                {
                    "do_objectID": "A",
                    "_class": "MSArtboardGroup",
                    "name": "art",
                    "frame": {},
                    "layers": [
                        {
                            "do_objectID": "B",
                            "_class": "MSTextLayer",
                            "name": "label",
                            "layers": []
                        }
                    ]
                }
            ]
        }"#;
        let json: SketchPageJson = serde_json::from_slice(body).unwrap();
        let page = ir_from_page_json("page-1", "Page 1", &json);
        assert_eq!(page.nodes.len(), 1);
        assert_eq!(page.nodes[0].id, "A");
        assert_eq!(page.nodes[0].children.len(), 1);
        assert_eq!(page.nodes[0].children[0].id, "B");
    }
}
