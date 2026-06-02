//! Stage 3: Map a [`SketchIr`] into an S.DEF [`UserInterface`].
//!
//! This stage is intentionally simple: it walks the SketchIr's node tree
//! and converts each node into the corresponding S.DEF `UINode` variant.
//! Per-class → S.DEF mapping follows the table in
//! `S.DEF/proposals/0001-sketch-ui-import.md` §Reference Implementation.

use std::collections::BTreeMap;

use sdef_core::types::ui::{
    UIAxisSizingMode, UIBaseElement, UIComponent, UIComponentType, UIDesignSystem,
    UIDocument, UIFrame, UILayoutAlign, UINavTarget, UINode, UIScreen, UserInterface,
};
use tracing::trace;

use super::sketch_ir::{SketchIr, SketchIrNode, SketchIrPage};

/// Stateless mapper: SketchIr → S.DEF UI document.
#[derive(Debug, Default, Clone)]
pub struct SketchToSdef {
    _private: (),
}

impl SketchToSdef {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map a SketchIr into a `(UserInterface, page_map, node_map)` triple.
    pub fn map(
        &self,
        ir: &SketchIr,
        _source_id: &str,
    ) -> (UserInterface, BTreeMap<String, String>, BTreeMap<String, String>) {
        let mut page_map: BTreeMap<String, String> = BTreeMap::new();
        let mut node_map: BTreeMap<String, String> = BTreeMap::new();

        let mut document_children: Vec<UINode> = Vec::new();

        for page in &ir.pages {
            let page_shard_id = format!("sdef://{}/ui/pages/{}", slugify(&ir.name), page.id);
            page_map.insert(page.id.clone(), page_shard_id.clone());

            let mut children = Vec::new();
            for node in &page.nodes {
                walk_node(node, &mut children, &mut node_map, &page_shard_id);
            }

            // Each page becomes a top-level frame.
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
                    children: if children.is_empty() { None } else { Some(children) },
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
            document_children.push(page_frame);
        }

        // Abstract screens.
        let screens: Vec<UIScreen> = ir
            .pages
            .iter()
            .map(|page: &SketchIrPage| UIScreen {
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

        // Design system stub.
        let design_system = UIDesignSystem {
            colors: None,
            typography: None,
            spacing: None,
            border_radius: None,
            shadows: None,
            motion: None,
            themes: None,
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
                type_: Some("sketch-import".to_string()),
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
            "SketchToSdef mapping complete"
        );

        (user_interface, page_map, node_map)
    }
}

fn walk_node(
    node: &SketchIrNode,
    out: &mut Vec<UINode>,
    node_map: &mut BTreeMap<String, String>,
    page_shard_id: &str,
) {
    let sdef_id = format!("{}#{}", page_shard_id, slugify(&node.id));
    node_map.insert(node.id.clone(), sdef_id);

    let mut children: Vec<UINode> = Vec::new();
    for child in &node.children {
        walk_node(child, &mut children, node_map, page_shard_id);
    }

    let mapped: UINode = match node.type_.as_str() {
        // Sketch's "MSArtboardGroup" ≈ Figma's top-level page frame
        "MSArtboardGroup" | "artboard" => UINode::Frame(UIFrame {
            base: base_element(&node.id, &node.name, "artboard", children, node),
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
        // Sketch's "MSSymbolMaster" ≈ Figma's COMPONENT
        "MSSymbolMaster" | "symbol" => UINode::Frame(UIFrame {
            base: base_element(&node.id, &node.name, "component", children, node),
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
        // Sketch's "MSTextLayer" — map to a UIBaseElement-as-frame for now.
        "MSTextLayer" | "text" => UINode::Frame(UIFrame {
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
        }),
        // Sketch's "MSSymbolInstance" — currently a frame placeholder;
        // a follow-up PR will switch to UINode::Ref with the right `ref`.
        "MSSymbolInstance" | "instance" => UINode::Frame(UIFrame {
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
        }),
        // Fallback for MSShapeGroup / MSRectangleShape / MSShapePath / etc.
        _ => UINode::Frame(UIFrame {
            base: base_element(
                &node.id,
                &node.name,
                &node.type_.to_lowercase(),
                children,
                node,
            ),
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
    _node: &SketchIrNode,
) -> UIBaseElement {
    UIBaseElement {
        id: id.to_string(),
        name: Some(name.to_string()),
        type_: type_.to_string(),
        x: None,
        y: None,
        width: None,
        height: None,
        reusable: matches!(type_, "component" | "symbolmaster"),
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
        let ir = SketchIr::default();
        let (ui, page_map, node_map) = SketchToSdef::new().map(&ir, "x");
        assert!(ui.document.is_some());
        assert!(page_map.is_empty());
        assert!(node_map.is_empty());
    }

    #[test]
    fn map_one_page_one_node() {
        let ir = SketchIr {
            name: "demo".into(),
            pages: vec![SketchIrPage {
                id: "page-1".into(),
                name: "Page 1".into(),
                nodes: vec![SketchIrNode {
                    id: "L1".into(),
                    name: "Artboard 1".into(),
                    type_: "MSArtboardGroup".into(),
                    children: vec![],
                    extra: Default::default(),
                }],
            }],
            ..Default::default()
        };
        let (ui, page_map, node_map) = SketchToSdef::new().map(&ir, "x");
        assert_eq!(page_map.len(), 1);
        assert!(page_map.contains_key("page-1"));
        assert_eq!(node_map.len(), 1);
        assert!(node_map.contains_key("L1"));
        let screens = ui.screens.as_ref().unwrap();
        assert_eq!(screens.len(), 1);
        assert_eq!(screens[0].name, "Page 1");
    }
}
