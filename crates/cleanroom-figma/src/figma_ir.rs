//! Stage 2: Figma intermediate representation (FigmaIr).
//!
//! The IR is the language-agnostic shape that both the REST API path
//! ([`crate::figma_api`]) and the binary `.fig` path
//! ([`crate::figma_fig`]) produce. Stage 3 ([`crate::figma_to_sdef`])
//! consumes the IR and produces S.DEF.
//!
//! The IR intentionally discards fields we do not yet use (e.g. fills,
//! strokes, effects) and stores the rest in `BTreeMap<String, Value>` for
//! later expansion. This keeps the IR small and easy to evolve.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Figma intermediate representation: a normalized, language-agnostic
/// snapshot of a Figma file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FigmaIr {
    /// File / document name.
    pub name: String,

    /// Figma `lastModified` (ISO-8601 string).
    pub last_modified: Option<String>,

    /// Figma file version string.
    pub version: Option<String>,

    /// Pages in document order.
    pub pages: Vec<FigmaIrPage>,

    /// Variables / design tokens (Figma variables API).
    #[serde(default)]
    pub variables: Vec<FigmaIrVariable>,

    /// Styles (color / text / effect / grid).
    #[serde(default)]
    pub styles: BTreeMap<String, FigmaIrStyle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigmaIrPage {
    /// Figma node id for the page (e.g. `"0:1"`).
    pub id: String,
    /// Page name.
    pub name: String,
    /// Direct child nodes (frames, components, etc.).
    pub nodes: Vec<FigmaIrNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigmaIrNode {
    /// Figma node id (e.g. `"1:23"`).
    pub id: String,
    /// Node name.
    pub name: String,
    /// Node type as Figma reports it: `FRAME`, `COMPONENT`, `INSTANCE`,
    /// `TEXT`, `RECTANGLE`, `ELLIPSE`, `VECTOR`, `GROUP`, etc.
    pub type_: String,
    /// Children (Figma nodes are trees).
    pub children: Vec<FigmaIrNode>,
    /// Catch-all for fields we do not yet model structurally.
    #[serde(default)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigmaIrVariable {
    pub id: String,
    pub name: String,
    /// Figma variable type: `COLOR`, `FLOAT`, `STRING`, `BOOLEAN`.
    pub type_: String,
    /// Collection name.
    pub collection: String,
    /// Mode → value mapping.
    pub values_by_mode: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigmaIrStyle {
    pub style_type: String,
    pub name: String,
    pub description: Option<String>,
}

impl FigmaIr {
    /// Total number of nodes (recursive) across all pages.
    pub fn node_count(&self) -> usize {
        self.pages
            .iter()
            .map(|p| p.nodes.iter().map(count_node).sum::<usize>())
            .sum()
    }
}

fn count_node(node: &FigmaIrNode) -> usize {
    1 + node.children.iter().map(count_node).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_count_is_recursive() {
        let ir = FigmaIr {
            pages: vec![FigmaIrPage {
                id: "0:1".into(),
                name: "p".into(),
                nodes: vec![
                    FigmaIrNode {
                        id: "1:1".into(),
                        name: "a".into(),
                        type_: "FRAME".into(),
                        children: vec![FigmaIrNode {
                            id: "1:2".into(),
                            name: "b".into(),
                            type_: "RECTANGLE".into(),
                            children: vec![],
                            extra: BTreeMap::new(),
                        }],
                        extra: BTreeMap::new(),
                    },
                    FigmaIrNode {
                        id: "1:3".into(),
                        name: "c".into(),
                        type_: "TEXT".into(),
                        children: vec![],
                        extra: BTreeMap::new(),
                    },
                ],
            }],
            ..Default::default()
        };
        assert_eq!(ir.node_count(), 3);
    }
}
