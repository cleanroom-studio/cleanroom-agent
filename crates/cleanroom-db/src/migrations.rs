//! Database migrations management.

use rusqlite::Connection;
use std::fs;
use std::path::Path;
use tracing::info;

use super::error::{DbError, DbResult};

/// Schema migrations directory.
const MIGRATIONS_DIR: &str = "migrations";

/// Get all migration files in order.
fn get_migrations(migrations_path: &Path) -> DbResult<Vec<(String, String)>> {
    let mut migrations = Vec::new();

    let entries = fs::read_dir(migrations_path)
        .map_err(|e| DbError::MigrationFailed(format!("Failed to read migrations dir: {}", e)))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "sql") {
            let filename = path.file_name().unwrap().to_string_lossy().to_string();
            let content = fs::read_to_string(&path)
                .map_err(|e| DbError::MigrationFailed(format!("Failed to read {}: {}", filename, e)))?;
            migrations.push((filename, content));
        }
    }

    migrations.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(migrations)
}

/// Run pending migrations.
pub fn run_pending(conn: &Connection) -> DbResult<()> {
    // Ensure schema_migrations table exists
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )
    .map_err(|e| DbError::QueryFailed(e.to_string()))?;

    // Get applied migrations
    let mut stmt = conn
        .prepare("SELECT version FROM schema_migrations")
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
    let applied: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| DbError::QueryFailed(e.to_string()))?
        .filter_map(|r| r.ok())
        .collect();

    // Get pending migrations
    let exe_path = std::env::current_exe().ok();
    let search_paths = if let Some(ref exe) = exe_path {
        vec![
            exe.parent().unwrap().join(MIGRATIONS_DIR),
            Path::new(MIGRATIONS_DIR).to_path_buf(),
            Path::new("../migrations").to_path_buf(),
        ]
    } else {
        vec![
            Path::new(MIGRATIONS_DIR).to_path_buf(),
            Path::new("../migrations").to_path_buf(),
        ]
    };

    let mut pending_migrations = None;
    for path in &search_paths {
        if path.exists() {
            pending_migrations = Some(get_migrations(path)?);
            break;
        }
    }

    let migrations = pending_migrations.ok_or_else(|| {
        DbError::MigrationFailed("Could not find migrations directory".to_string())
    })?;

    for (filename, content) in migrations {
        if !applied.contains(&filename) {
            info!(migration = %filename, "Applying migration");
            conn.execute_batch(&content)
                .map_err(|e| DbError::MigrationFailed(format!("Failed to apply {}: {}", filename, e)))?;

            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (?1)",
                rusqlite::params![filename],
            )
            .map_err(|e| DbError::QueryFailed(e.to_string()))?;

            info!(migration = %filename, "Migration applied successfully");
        }
    }

    Ok(())
}