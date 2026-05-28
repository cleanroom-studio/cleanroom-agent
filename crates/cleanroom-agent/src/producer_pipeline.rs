//! Producer pipeline — full code analysis flow: scan → partition → analyze → S.DEF → persist.

use std::path::Path;
use std::sync::Arc;

use crate::dependency_graph::{DependencyGraph, DepNode, DepNodeType, DepEdgeKind};
use crate::ir_to_sdef::{SdefMapper, MapperConfig};
use crate::module_partitioner::{partition_files, PartitionConfig};
use crate::repo_scanner::{scan_repository, ScanConfig, SourceFile};
use cleanroom_db::{Database, DbError, TaskRepository, TaskType, TaskStatus, Task};
use sdef_core::SoftwareDefinition;
use tracing::{info, instrument};

/// Result of a full pipeline run.
#[derive(Debug)]
pub struct PipelineResult {
    /// Files discovered.
    pub file_count: usize,
    /// Modules identified.
    pub module_count: usize,
    /// Languages found.
    pub languages: Vec<String>,
    /// The generated SoftwareDefinition.
    pub sdef: SoftwareDefinition,
    /// Dependency graph details.
    pub dependency_info: DepInfo,
}

/// Dependency analysis info.
#[derive(Debug)]
pub struct DepInfo {
    pub node_count: usize,
    pub edge_count: usize,
    pub cycle_count: usize,
}

/// Run the complete producer pipeline.
#[instrument(skip(db))]
pub async fn run_analysis_pipeline(
    db: Arc<Database>,
    repo_path: &Path,
    project_name: &str,
    version: &str,
    description: Option<String>,
) -> Result<PipelineResult, DbError> {
    // 1. Scan repository
    info!(path = %repo_path.display(), "Starting repo scan");
    let scan_config = ScanConfig {
        root: repo_path.to_path_buf(),
        ..ScanConfig::default()
    };
    let files = scan_repository(&scan_config);
    info!(count = files.len(), "Scan complete");

    if files.is_empty() {
        return Err(DbError::QueryFailed("No source files found in repository".to_string()));
    }

    // 2. Partition into modules
    let partition_config = PartitionConfig::default();
    let modules = partition_files(files.clone(), &partition_config);
    info!(count = modules.len(), "Modules identified");

    // 3. Build dependency graph
    let dep_graph = build_dependency_graph(&files, &modules);
    let cycles = dep_graph.detect_cycles();
    info!(
        nodes = dep_graph.node_count(),
        edges = dep_graph.edge_count(),
        cycles = cycles.len(),
        "Dependency graph built"
    );

    // 4. Map to S.DEF and persist
    let mapper_config = MapperConfig {
        document_name: project_name.to_string(),
        version: version.to_string(),
        description,
    };
    let mapper = SdefMapper::new(mapper_config, db.clone());
    let (sdef, _) = mapper.map_all(files.clone(), modules.clone()).await?;

    // 5. Compute language summary
    let languages = collect_languages(&files);

    // 6. Update task results
    let dep_info = DepInfo {
        node_count: dep_graph.node_count(),
        edge_count: dep_graph.edge_count(),
        cycle_count: cycles.len(),
    };

    Ok(PipelineResult {
        file_count: files.len(),
        module_count: modules.len(),
        languages,
        sdef,
        dependency_info: dep_info,
    })
}

/// Build a dependency graph from files and modules.
fn build_dependency_graph(files: &[SourceFile], modules: &[crate::module_partitioner::Module]) -> DependencyGraph {
    let mut graph = DependencyGraph::new();

    // Add module nodes
    for module in modules {
        let node = DepNode {
            id: format!("module:{}", module.name),
            name: module.name.clone(),
            node_type: DepNodeType::Module,
        };
        graph.add_node(node);

        // Add file nodes
        for file in &module.files {
            let stem = file.path.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let file_node = DepNode {
                id: format!("file:{}", stem),
                name: stem.clone(),
                node_type: DepNodeType::File,
            };
            graph.add_node(file_node);
            graph.add_edge(
                &format!("module:{}", module.name),
                &format!("file:{}", stem),
                DepEdgeKind::Import,
            );
        }
    }

    // Detect cross-module references via language imports
    // (simplified: files in different modules with same stem may reference each other)
    for i in 0..modules.len() {
        for j in (i + 1)..modules.len() {
            let files_i: Vec<&str> = modules[i].files.iter()
                .filter_map(|f| f.path.file_stem())
                .filter_map(|s| s.to_str())
                .collect();
            let files_j: Vec<&str> = modules[j].files.iter()
                .filter_map(|f| f.path.file_stem())
                .filter_map(|s| s.to_str())
                .collect();

            if files_i.iter().any(|a| files_j.contains(a)) {
                graph.add_edge(
                    &format!("module:{}", modules[i].name),
                    &format!("module:{}", modules[j].name),
                    DepEdgeKind::References,
                );
            }
        }
    }

    graph
}

fn collect_languages(files: &[SourceFile]) -> Vec<String> {
    let mut langs: Vec<String> = files.iter()
        .filter_map(|f| f.language.clone())
        .collect();
    langs.sort();
    langs.dedup();
    langs
}

/// Export a summary of the pipeline run as JSON.
pub fn result_to_json(result: &PipelineResult) -> serde_json::Value {
    serde_json::json!({
        "files_found": result.file_count,
        "modules_identified": result.module_count,
        "languages": result.languages,
        "dependency_graph": {
            "nodes": result.dependency_info.node_count,
            "edges": result.dependency_info.edge_count,
            "cycles": result.dependency_info.cycle_count,
        },
        "sdef": {
            "data_models": result.sdef.data_models.as_ref().map(|v| v.len()).unwrap_or(0),
            "interfaces": result.sdef.contracts.as_ref().and_then(|c| c.interfaces.as_ref()).map(|v| v.len()).unwrap_or(0),
            "functions": result.sdef.behavior.as_ref().and_then(|b| b.functions.as_ref()).map(|v| v.len()).unwrap_or(0),
            "design_decisions": result.sdef.design_decisions.as_ref().map(|v| v.len()).unwrap_or(0),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_test_repo(tmp_dir: &Path) {
        // Create a minimal test repo structure
        std::fs::create_dir_all(tmp_dir.join("src")).unwrap();
        std::fs::create_dir_all(tmp_dir.join("src").join("models")).unwrap();
        std::fs::write(tmp_dir.join("Cargo.toml"), "[package]\nname = \"test\"\nversion = \"0.1.0\"\nedition = \"2021\"\n").unwrap();
        std::fs::write(tmp_dir.join("src").join("main.rs"), "fn main() { println!(\"Hello\"); }\n").unwrap();
        std::fs::write(tmp_dir.join("src").join("lib.rs"), "pub mod models;\n").unwrap();
        std::fs::write(tmp_dir.join("src").join("models").join("user.rs"), "pub struct User { pub id: u64, pub name: String }\n").unwrap();
    }

    fn test_dir(name: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("cleanroom_{}_{}", name, ts))
    }

    #[test]
    fn test_scan_and_partition() {
        let tmp = test_dir("scan_part");
        let _ = std::fs::remove_dir_all(&tmp);
        make_test_repo(&tmp);

        let scan_config = ScanConfig { root: tmp.clone(), ..ScanConfig::default() };
        let files = scan_repository(&scan_config);
        assert!(!files.is_empty());

        let partition_config = PartitionConfig::default();
        let modules = partition_files(files, &partition_config);
        assert!(!modules.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_build_dependency_graph() {
        let tmp = test_dir("dep_graph");
        let _ = std::fs::remove_dir_all(&tmp);
        make_test_repo(&tmp);

        let scan_config = ScanConfig { root: tmp.clone(), ..ScanConfig::default() };
        let files = scan_repository(&scan_config);
        let partition_config = PartitionConfig::default();
        let modules = partition_files(files.clone(), &partition_config);
        
        let graph = build_dependency_graph(&files, &modules);
        assert!(graph.node_count() > 0);
        
        let cycles = graph.detect_cycles();
        assert!(cycles.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_full_pipeline() {
        let tmp = test_dir("full_pipe");
        let _ = std::fs::remove_dir_all(&tmp);
        make_test_repo(&tmp);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let db = Arc::new(cleanroom_db::Database::in_memory().unwrap());
            let result = run_analysis_pipeline(
                db,
                &tmp,
                "test-project",
                "0.1.0",
                Some("A test project".to_string()),
            ).await.unwrap();

            assert!(result.file_count > 0, "Should find files");
            assert!(result.module_count > 0, "Should find modules");
            assert!(!result.languages.is_empty(), "Should detect languages");
            assert!(result.sdef.data_models.is_some(), "Should have data models");

            info!(
                files = result.file_count,
                modules = result.module_count,
                "Pipeline test passed"
            );
        });

        let _ = std::fs::remove_dir_all(&tmp);
    }
}