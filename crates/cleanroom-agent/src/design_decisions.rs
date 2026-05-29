//! Design Decision Inferrer — infers design decisions from codebase analysis.
//!
//! Scans source files for clues about architectural choices:
//! - Framework / library usage (imports, package.json, Cargo.toml, etc.)
//! - Database / storage choices
//! - Architectural patterns (dependency injection, MVC, etc.)
//! - Testing frameworks
//! - Build / CI tools
//!
//! Produces `Vec<DesignDecision>` with confidence scores.

use sdef_core::DesignDecision;

/// Inferred design decision with confidence.
#[derive(Debug, Clone)]
pub struct InferredDecision {
    pub topic: String,
    pub decision: String,
    pub rationale: String,
    pub alternatives: Vec<String>,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// Result of design decision inference.
#[derive(Debug, Clone, Default)]
pub struct InferenceResult {
    pub decisions: Vec<DesignDecision>,
    pub file_count: usize,
    pub match_count: usize,
}

impl InferenceResult {
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }
    pub fn len(&self) -> usize {
        self.decisions.len()
    }
}

// ─── Knowledge Base: Known technology patterns ──────────────────────────

struct Pattern {
    keywords: &'static [&'static str],
    topic: &'static str,
    decision: &'static str,
    rationale: &'static str,
    alternatives: &'static [&'static str],
    weight: f64,
}

static KNOWLEDGE_BASE: &[Pattern] = &[
    Pattern {
        keywords: &["react", "react-dom", "jsx", "tsx"],
        topic: "UI Framework",
        decision: "React",
        rationale: "Component-based UI architecture with virtual DOM",
        alternatives: &["Vue", "Angular", "Svelte"],
        weight: 0.8,
    },
    Pattern {
        keywords: &["vue", "vue-router", "vuex", "pinia"],
        topic: "UI Framework",
        decision: "Vue.js",
        rationale: "Progressive framework with reactive data binding",
        alternatives: &["React", "Angular", "Svelte"],
        weight: 0.8,
    },
    Pattern {
        keywords: &["@angular/core", "@angular/router", "angular"],
        topic: "UI Framework",
        decision: "Angular",
        rationale: "Full-featured framework with dependency injection",
        alternatives: &["React", "Vue", "Svelte"],
        weight: 0.8,
    },
    Pattern {
        keywords: &["express", "koa", "fastify"],
        topic: "Web Framework (Backend)",
        decision: "Express / Koa / Fastify",
        rationale: "Lightweight HTTP server framework for Node.js",
        alternatives: &["NestJS", "Hapi", "Fastify"],
        weight: 0.7,
    },
    Pattern {
        keywords: &["axum", "actix-web", "warp", "rocket", "tide"],
        topic: "Web Framework (Backend)",
        decision: "Rust async web framework",
        rationale: "High-performance async HTTP server",
        alternatives: &["Actix", "Axum", "Warp"],
        weight: 0.7,
    },
    Pattern {
        keywords: &["django", "flask", "fastapi"],
        topic: "Web Framework (Backend)",
        decision: "Python web framework",
        rationale: "Mature Python web framework",
        alternatives: &["Django", "Flask", "FastAPI"],
        weight: 0.7,
    },
    Pattern {
        keywords: &["sqlalchemy", "diesel", "prisma", "typeorm", "sequelize", "mongoose"],
        topic: "Database ORM",
        decision: "ORM-based data access",
        rationale: "Object-relational mapping for database access",
        alternatives: &["Raw SQL", "Query Builder", "ORM"],
        weight: 0.6,
    },
    Pattern {
        keywords: &["postgresql", "postgres", "pg", "psycopg2", "sqlx"],
        topic: "Database",
        decision: "PostgreSQL",
        rationale: "Relational database with strong ACID compliance",
        alternatives: &["MySQL", "SQLite", "MongoDB"],
        weight: 0.7,
    },
    Pattern {
        keywords: &["mysql", "mariadb"],
        topic: "Database",
        decision: "MySQL / MariaDB",
        rationale: "Popular open-source relational database",
        alternatives: &["PostgreSQL", "SQLite", "MongoDB"],
        weight: 0.7,
    },
    Pattern {
        keywords: &["mongodb", "mongoose", "mongo"],
        topic: "Database",
        decision: "MongoDB",
        rationale: "NoSQL document database for flexible schemas",
        alternatives: &["PostgreSQL", "MySQL", "DynamoDB"],
        weight: 0.7,
    },
    Pattern {
        keywords: &["redis", "ioredis"],
        topic: "Cache / Message Queue",
        decision: "Redis",
        rationale: "In-memory data store for caching and pub/sub",
        alternatives: &["Memcached", "RabbitMQ", "Kafka"],
        weight: 0.6,
    },
    Pattern {
        keywords: &["jest", "vitest", "mocha", "jasmine"],
        topic: "Test Framework (JS/TS)",
        decision: "Jest / Vitest / Mocha",
        rationale: "Standard JavaScript testing framework",
        alternatives: &["Jest", "Vitest", "Mocha", "Jasmine"],
        weight: 0.6,
    },
    Pattern {
        keywords: &["pytest", "unittest"],
        topic: "Test Framework (Python)",
        decision: "pytest",
        rationale: "Feature-rich Python testing framework",
        alternatives: &["unittest", "nose", "doctest"],
        weight: 0.6,
    },
    Pattern {
        keywords: &["serde", "serde_json"],
        topic: "Serialization",
        decision: "serde",
        rationale: "Standard Rust serialization framework",
        alternatives: &["bincode", "msgpack", "protobuf"],
        weight: 0.6,
    },
    Pattern {
        keywords: &["tokio", "async-std", "smol"],
        topic: "Async Runtime",
        decision: "Tokio",
        rationale: "Asynchronous runtime for Rust",
        alternatives: &["async-std", "smol", "monoio"],
        weight: 0.7,
    },
    Pattern {
        keywords: &["clap", "structopt"],
        topic: "CLI Framework",
        decision: "Clap",
        rationale: "Command-line argument parsing",
        alternatives: &["clap", "structopt", "argh"],
        weight: 0.6,
    },
    Pattern {
        keywords: &["docker", "dockerfile", "container"],
        topic: "Containerization",
        decision: "Docker",
        rationale: "Container-based deployment and development",
        alternatives: &["Podman", "containerd"],
        weight: 0.7,
    },
    Pattern {
        keywords: &["github.actions", "circleci", "gitlab-ci"],
        topic: "CI/CD",
        decision: "GitHub Actions",
        rationale: "Continuous integration and deployment",
        alternatives: &["CircleCI", "GitLab CI", "Jenkins"],
        weight: 0.6,
    },
    Pattern {
        keywords: &["typescript", "@types/"],
        topic: "Type System",
        decision: "TypeScript",
        rationale: "Static type checking for JavaScript",
        alternatives: &["JavaScript", "Flow"],
        weight: 0.8,
    },
    Pattern {
        keywords: &["rustfmt", "clippy"],
        topic: "Code Quality",
        decision: "Rust linter + formatter",
        rationale: "Automated code style enforcement and linting",
        alternatives: &["rustfmt", "clippy"],
        weight: 0.5,
    },
];

/// Infer design decisions from source file contents.
///
/// `files`: `(relative_path, content)` pairs.
/// `manifest_files`: config files like `Cargo.toml`, `package.json`, etc.
pub fn infer_decisions(files: &[(&str, &str)], _manifest_files: &[(&str, &str)]) -> InferenceResult {
    let mut result = InferenceResult::default();
    let mut decision_map: Vec<InferredDecision> = Vec::new();

    // Scan all files for keywords
    for (path, content) in files {
        result.file_count += 1;
        let lower = content.to_lowercase();
        let path_lower = path.to_lowercase();

        for pattern in KNOWLEDGE_BASE {
            let matches: Vec<&str> = pattern
                .keywords
                .iter()
                .filter(|kw| lower.contains(&kw.to_lowercase()) || path_lower.contains(&kw.to_lowercase()))
                .copied()
                .collect();

            if !matches.is_empty() {
                // Check if we already have this decision
                let existing = decision_map.iter_mut().find(|d: &&mut InferredDecision| d.topic == pattern.topic);
                if let Some(dec) = existing {
                    dec.evidence.push(format!("{}: {}", path, matches.join(", ")));
                    dec.confidence = (dec.confidence + pattern.weight * 0.3).min(1.0);
                    result.match_count += matches.len();
                } else {
                    decision_map.push(InferredDecision {
                        topic: pattern.topic.to_string(),
                        decision: pattern.decision.to_string(),
                        rationale: pattern.rationale.to_string(),
                        alternatives: pattern.alternatives.iter().map(|s| s.to_string()).collect(),
                        confidence: pattern.weight,
                        evidence: vec![format!("{}: {}", path, matches.join(", "))],
                    });
                    result.match_count += matches.len();
                }
            }
        }
    }

    // Convert to DesignDecision and persist to DB if possible
    for inferred in decision_map {
        if inferred.confidence >= 0.4 {
            result.decisions.push(DesignDecision {
                id: format!("dd_{}", slugify(&inferred.topic)),
                topic: inferred.topic.clone(),
                decision: inferred.decision.clone(),
                rationale: format!("{} (confidence: {:.0}%)", inferred.rationale, inferred.confidence * 100.0),
                context: Some(format!(
                    "Inferred from {} pieces of evidence across {} files",
                    inferred.evidence.len(),
                    result.file_count,
                )),
                alternatives: if inferred.alternatives.is_empty() { None } else { Some(inferred.alternatives.clone()) },
                consequences: Some(vec![
                    format!("Adopted {} as the primary {}", inferred.decision, inferred.topic.to_lowercase()),
                ]),
                constraints: None,
            });
        }
    }

    result
}

/// Infer decisions and persist them into the database.
pub fn persist_decisions(
    db: &cleanroom_db::Database,
    document_name: &str,
    decisions: &[DesignDecision],
) -> Result<usize, cleanroom_db::DbError> {
    let conn = db.connection();
    let mut count = 0;
    for dd in decisions {
        conn.execute(
            "INSERT OR IGNORE INTO design_decisions (id, document_name, topic, decision, rationale, context, alternatives_json, consequences_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                dd.id, document_name,
                dd.topic, dd.decision, dd.rationale, dd.context,
                serde_json::to_string(&dd.alternatives).ok(),
                serde_json::to_string(&dd.consequences).ok(),
            ],
        ).map_err(|e| cleanroom_db::DbError::QueryFailed(e.to_string()))?;
        count += 1;
    }
    Ok(count)
}

fn slugify(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_react_decision() {
        let files = &[("src/App.tsx", "import React from 'react'; import { useState } from 'react';")];
        let result = infer_decisions(files, &[]);
        assert!(result.len() >= 1, "Should detect at least one decision");
        let react = result.decisions.iter().find(|d| d.topic == "UI Framework");
        assert!(react.is_some(), "Should detect UI framework decision");
        if let Some(r) = react {
            assert!(r.decision.contains("React"), "Should be React, got: {}", r.decision);
        }
    }

    #[test]
    fn test_infer_database_decision() {
        let files = &[("src/db.rs", "use sqlx::postgres; use postgresql;")];
        let result = infer_decisions(files, &[]);
        let db = result.decisions.iter().find(|d| d.topic == "Database");
        assert!(db.is_some(), "Should detect database decision");
    }

    #[test]
    fn test_infer_multiple_decisions() {
        let files = &[
            ("frontend/App.tsx", "import React from 'react'; import { render } from 'react-dom';"),
            ("backend/main.rs", "use axum::{Router, routing::get}; use tokio; use sqlx::postgres;"),
            ("Cargo.toml", "serde = \"1.0\"\nclap = \"4\"\ntokio = \"1\""),
        ];
        let result = infer_decisions(files, &[]);
        let topics: Vec<&str> = result.decisions.iter().map(|d| d.topic.as_str()).collect();
        assert!(topics.contains(&"UI Framework"), "Should detect React: {:?}", topics);
        assert!(topics.contains(&"Web Framework (Backend)") || topics.contains(&"Async Runtime"), "Should detect framework: {:?}", topics);
        assert!(result.len() >= 3, "Should detect multiple decisions, got {}: {:?}", result.len(), topics);
    }

    #[test]
    fn test_empty_codebase() {
        let result = infer_decisions(&[], &[]);
        assert!(result.is_empty(), "Empty codebase should have no decisions");
    }

    #[test]
    fn test_persist_decisions() {
        use cleanroom_db::Database;
        let db = Database::in_memory().expect("Create DB");
        let conn = db.connection();
        conn.execute(
            "INSERT INTO sdef_documents (name) VALUES ('test-doc')",
            [],
        ).expect("Insert doc");
        drop(conn);

        let decisions = vec![DesignDecision {
            id: "dd_test".to_string(),
            topic: "Test".to_string(),
            decision: "Use testing framework".to_string(),
            rationale: "Testing is important".to_string(),
            context: None,
            alternatives: None,
            consequences: None,
            constraints: None,
        }];
        let count = persist_decisions(&db, "test-doc", &decisions).expect("Persist");
        assert_eq!(count, 1, "Should persist 1 decision");
    }
}
