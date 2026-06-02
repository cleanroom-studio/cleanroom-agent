//! Stage 2: Sketch intermediate representation (SketchIr).
//!
//! The IR is the language-agnostic shape that both the Cloud API path
//! ([`crate::sketch_api`]) and the local `.sketch` path
//! ([`crate::sketch_zip`]) produce. Stage 3 ([`crate::sketch_to_sdef`])
//! consumes the IR and produces S.DEF.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Sketch intermediate representation: a normalized, language-agnostic
/// snapshot of a Sketch document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SketchIr {
    /// File / document name.
    pub name: String,

    /// Sketch `lastModified` (ISO-8601 string).
    pub last_modified: Option<String>,

    /// Format version (e.g. `"2.0"`).
    pub version: Option<String>,

    /// Pages in document order.
    pub pages: Vec<SketchIrPage>,

    /// Styles (layer styles, text styles). Populated from a future
    /// `MSStyle` extraction; for now, empty.
    #[serde(default)]
    pub styles: BTreeMap<String, SketchIrStyle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchIrPage {
    /// Page id (UUID-like).
    pub id: String,
    /// Page display name.
    pub name: String,
    /// Direct child layers.
    pub nodes: Vec<SketchIrNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchIrNode {
    /// Sketch layer id (`do_objectID`).
    pub id: String,
    /// Layer name.
    pub name: String,
    /// Sketch class discriminator (e.g. `"MSArtboardGroup"`,
    /// `"MSSymbolMaster"`, `"MSTextLayer"`).
    pub type_: String,
    /// Children (Sketch layers are trees).
    pub children: Vec<SketchIrNode>,
    /// Catch-all for fields we do not yet model structurally.
    #[serde(default)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchIrStyle {
    pub style_type: String,
    pub name: String,
}

impl SketchIr {
    /// Total number of nodes (recursive) across all pages.
    pub fn node_count(&self) -> usize {
        self.pages
            .iter()
            .map(|p| p.nodes.iter().map(count_node).sum::<usize>())
            .sum()
    }
}

fn count_node(node: &SketchIrNode) -> usize {
    1 + node.children.iter().map(count_node).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_count_is_recursive() {
        let ir = SketchIr {
            pages: vec![SketchIrPage {
                id: "p".into(),
                name: "p".into(),
                nodes: vec![SketchIrNode {
                    id: "a".into(),
                    name: "a".into(),
                    type_: "MSArtboardGroup".into(),
                    children: vec![SketchIrNode {
                        id: "b".into(),
                        name: "b".into(),
                        type_: "MSTextLayer".into(),
                        children: vec![],
                        extra: BTreeMap::new(),
                    }],
                    extra: BTreeMap::new(),
                }],
            }],
            ..Default::default()
        };
        assert_eq!(ir.node_count(), 2);
    }
}
