//! Stage 3: Map a [`FigmaIr`] into an S.DEF [`UserInterface`].
//!
//! This stage is intentionally simple: it walks the FigmaIr's node tree
//! and converts each node into the corresponding S.DEF `UINode` variant.
//! Per-node type → S.DEF mapping follows the table in
//! `S.DEF/proposals/0000-figma-ui-import.md` §Reference Implementation.

use std::collections::BTreeMap;

use sdef_core::types::ui::{
    UIAxisSizingMode, UIBaseElement, UIComponent, UIComponentType, UIDesignSystem,
    UIDesignTheme, UIDocument, UIFrame, UILayoutAlign, UINavTarget, UINode, UIRectangle,
    UIScreen, UserInterface,
};
use sdef_core::types::ui_figma::UIImportSource;
use tracing::trace;

use super::figma_ir::{FigmaIr, FigmaIrNode, FigmaIrPage};

/// Stateless mapper: FigmaIr → S.DEF UI document.
#[derive(Debug, Default, Clone)]
pub struct FigmaToSdef {
    _private: (),
}

impl FigmaToSdef {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map a FigmaIr into a `(UserInterface, page_map, node_map)` triple.
    ///
    /// `page_map` and `node_map` are written into `ui_provenance` by the
    /// caller (see `FigmaImporter::import`).
    pub fn map(
        &self,
        ir: &FigmaIr,
        _source: UIImportSource,
        _source_id: &str,
    ) -> (UserInterface, BTreeMap<String, String>, BTreeMap<String, String>) {
        let mut page_map: BTreeMap<String, String> = BTreeMap::new();
        let mut node_map: BTreeMap<String, String> = BTreeMap::new();

        // Build a UIDocument where each top-level page becomes a child
        // frame named after the page.
        let mut document_children: Vec<UINode> = Vec::new();

        for (page_idx, page) in ir.pages.iter().enumerate() {
            let page_shard_id = format!("sdef://{}/ui/pages/{}", slugify(&ir.name), page.id);
            page_map.insert(page.id.clone(), page_shard_id.clone());

            // Map each child node.
            let mut children = Vec::new();
            for node in &page.nodes {
                map_node(node, &mut children, &mut node_map, &page_shard_id);
            }

            // Each page becomes a top-level frame in the UIDocument.
            let page_frame = UINode::Frame(UIFrame {
                base: UIBaseElement {
                    id: page.id.clone(),
                    name: Some(page.name.clone()),
                    type_: "frame".to_string(),
                    x: None,
                    y: None,
                    width: None,
                    height: None,
                    reusable: false,
                    theme: None,
                    opacity: None,
                    rotation: None,
        enabled: true,
                    fill: None,
                    stroke: None,
                    effect: None,
                    children: None,
                    sdef_bindings: None,
                    sdef_behaviors: None,
                    sdef_states: None,
                    sdef_accessibility: None,
                    sdef_test_hook: None,
                    sdef_navigation: Some(UINavTarget {
                        target_screen: Some(page.name.clone()),
                        parameters: None,
                    }),
                },
                layout: Some("vertical".to_string()),
                gap: Some(8.0),
                padding: None,
                justify_content: None,
                align_items: None,
                corner_radius: None,
                clip: false,
                slot: None,
                primary_axis_sizing_mode: Some(UIAxisSizingMode::Auto),
                counter_axis_sizing_mode: Some(UIAxisSizingMode::Auto),
                layout_align: None,
                layout_grow: None,
            });
            let _ = page_idx; // silence unused warning if logs change
            let _ = &mut children;
            document_children.push(page_frame);
        }

        // Build the abstract screens (one per Figma page) — this gives the
        // reconstruction agent a semantic-level view of the design.
        let screens: Vec<UIScreen> = ir
            .pages
            .iter()
            .map(|page: &FigmaIrPage| UIScreen {
                id: page.id.clone(),
                name: page.name.clone(),
                route: Some(format!("/{}", slugify(&page.name))),
                purpose: None,
                layout: None,
                components: Some(
                    page.nodes
                        .iter()
                        .map(|n| UIComponent {
                            name: n.name.clone(),
                            type_: n.type_.to_lowercase(),
                            content: None,
                            placeholder: None,
                            props: None,
                            style: None,
                            states: None,
                            children: None,
                            events: None,
                            behaviors: None,
                            bind_to: None,
                        })
                        .collect(),
                ),
                state: None,
                interactions: Some(Vec::new()),
            })
            .collect();

        // Build the design system stub.
        let design_system = UIDesignSystem {
            colors: None,
            typography: None,
            spacing: None,
            border_radius: None,
            shadows: None,
            motion: None,
            themes: Some(
                ir.styles
                    .values()
                    .filter(|s| s.style_type.eq_ignore_ascii_case("FILL"))
                    .map(|s| UIDesignTheme {
                        name: s.name.clone(),
                        overrides: None,
                    })
                    .collect::<Vec<_>>(),
            ),
            variable_modes: None,
            layout_grids: None,
        };

        let document = UIDocument {
            version: Some("2.11".to_string()),
            variables: None,
            themes: None,
            children: document_children,
        };

        let user_interface = UserInterface {
            design_system: Some(design_system),
            document: Some(document),
            screens: Some(screens),
            navigation: None,
            responsive_design: None,
            component_taxonomy: Some(vec![UIComponentType {
                component_id: format!("sdef://{}/ui/component-taxonomy", slugify(&ir.name)),
                type_: Some("figma-import".to_string()),
                data_requirements: None,
                interaction_rules: None,
                variants: None,
                properties: None,
            }]),
            ui_provenance: None, // set by the orchestrator
        };

        trace!(
            pages = ir.pages.len(),
            page_map = page_map.len(),
            node_map = node_map.len(),
            "FigmaToSdef mapping complete"
        );

        (user_interface, page_map, node_map)
    }
}

fn map_node(
    node: &FigmaIrNode,
    out: &mut Vec<UINode>,
    node_map: &mut BTreeMap<String, String>,
    page_shard_id: &str,
) {
    let sdef_id = format!("{}#{}", page_shard_id, slugify(&node.id));
    node_map.insert(node.id.clone(), sdef_id);

    let mut children: Vec<UINode> = Vec::new();
    for child in &node.children {
        map_node(child, &mut children, node_map, page_shard_id);
    }

    let mapped: UINode = match node.type_.as_str() {
        "FRAME" | "COMPONENT" | "COMPONENT_SET" => UINode::Frame(UIFrame {
            base: base_element(&node.id, &node.name, "frame", children, node),
            layout: Some("vertical".to_string()),
            gap: Some(8.0),
            padding: None,
            justify_content: None,
            align_items: None,
            corner_radius: None,
            clip: false,
            slot: None,
            primary_axis_sizing_mode: Some(UIAxisSizingMode::Auto),
            counter_axis_sizing_mode: Some(UIAxisSizingMode::Auto),
            layout_align: None,
            layout_grow: None,
        }),
        "RECTANGLE" => UINode::Rectangle(UIRectangle {
            base: base_element(&node.id, &node.name, "rectangle", children, node),
            corner_radius: None,
        }),
        "TEXT" => {
            // UIText would carry a content field; for now we map TEXT to a
            // UIBaseElement-as-frame so the structural position is preserved.
            UINode::Frame(UIFrame {
                base: base_element(&node.id, &node.name, "text", children, node),
                layout: Some("none".to_string()),
                gap: None,
                padding: None,
                justify_content: None,
                align_items: None,
                corner_radius: None,
                clip: false,
                slot: None,
                primary_axis_sizing_mode: None,
                counter_axis_sizing_mode: None,
                layout_align: Some(UILayoutAlign::Inherit),
                layout_grow: None,
            })
        }
        "INSTANCE" => {
            // Instances are placeholders for now — a follow-up PR will
            // emit `UINode::Ref` with the appropriate `ref` field.
            UINode::Frame(UIFrame {
                base: base_element(&node.id, &node.name, "instance", children, node),
                layout: Some("vertical".to_string()),
                gap: None,
                padding: None,
                justify_content: None,
                align_items: None,
                corner_radius: None,
                clip: false,
                slot: None,
                primary_axis_sizing_mode: Some(UIAxisSizingMode::Auto),
                counter_axis_sizing_mode: Some(UIAxisSizingMode::Auto),
                layout_align: None,
                layout_grow: None,
            })
        }
        // GROUP, ELLIPSE, VECTOR, LINE, STAR, POLYGON, etc. — fall back to a
        // frame for now. A future PR can specialize these.
        _ => UINode::Frame(UIFrame {
            base: base_element(&node.id, &node.name, &node.type_.to_lowercase(), children, node),
            layout: Some("none".to_string()),
            gap: None,
            padding: None,
            justify_content: None,
            align_items: None,
            corner_radius: None,
            clip: false,
            slot: None,
            primary_axis_sizing_mode: None,
            counter_axis_sizing_mode: None,
            layout_align: None,
            layout_grow: None,
        }),
    };

    out.push(mapped);
}

fn base_element(
    id: &str,
    name: &str,
    type_: &str,
    children: Vec<UINode>,
    _node: &FigmaIrNode,
) -> UIBaseElement {
    UIBaseElement {
        id: id.to_string(),
        name: Some(name.to_string()),
        type_: type_.to_string(),
        x: None,
        y: None,
        width: None,
        height: None,
        reusable: matches!(type_, "component" | "component_set"),
        theme: None,
        opacity: None,
        rotation: None,
        enabled: true,
        fill: None,
        stroke: None,
        effect: None,
        children: if children.is_empty() { None } else { Some(children) },
        sdef_bindings: None,
        sdef_behaviors: None,
        sdef_states: None,
        sdef_accessibility: None,
        sdef_test_hook: None,
        sdef_navigation: None,
    }
}

fn slugify(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_empty_ir_yields_empty_document() {
        let ir = FigmaIr::default();
        let (ui, page_map, node_map) = FigmaToSdef::new().map(&ir, UIImportSource::Figma, "x");
        assert!(ui.document.is_some());
        assert!(page_map.is_empty());
        assert!(node_map.is_empty());
        assert!(ui.ui_provenance.is_none());
    }

    #[test]
    fn map_one_page_one_node() {
        let ir = FigmaIr {
            name: "demo".into(),
            pages: vec![FigmaIrPage {
                id: "0:1".into(),
                name: "Page 1".into(),
                nodes: vec![FigmaIrNode {
                    id: "1:2".into(),
                    name: "Frame".into(),
                    type_: "FRAME".into(),
                    children: vec![],
                    extra: Default::default(),
                }],
            }],
            ..Default::default()
        };
        let (ui, page_map, node_map) = FigmaToSdef::new().map(&ir, UIImportSource::Figma, "x");
        assert_eq!(page_map.len(), 1);
        assert!(page_map.contains_key("0:1"));
        assert_eq!(node_map.len(), 1);
        assert!(node_map.contains_key("1:2"));
        let screens = ui.screens.as_ref().unwrap();
        assert_eq!(screens.len(), 1);
        assert_eq!(screens[0].name, "Page 1");
    }
}
