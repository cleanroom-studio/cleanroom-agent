//! Database connection management.

use rusqlite::{Connection, OpenFlags};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{info, instrument, warn};

use super::error::{DbError, DbResult};
use super::migrations;

/// Backup scheduling configuration.
#[derive(Debug, Clone)]
pub struct BackupConfig {
    /// Backup interval (e.g., Duration::from_secs(3600) for hourly).
    pub interval: Duration,
    /// Directory to store backups.
    pub backup_dir: PathBuf,
    /// Filename template. Supports {timestamp} placeholder.
    pub filename_template: String,
    /// Maximum number of backups to keep (0 = unlimited).
    pub max_backups: usize,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(3600), // hourly
            backup_dir: PathBuf::from("."),
            filename_template: "state-{timestamp}.db".to_string(),
            max_backups: 24, // keep 24 hourly backups
        }
    }
}

/// Thread-safe database connection wrapper.
#[derive(Debug, Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    /// Open or create a database at the given path.
    #[instrument(skip_all, fields(path = %path.display()))]
    pub fn open(path: &Path) -> DbResult<Self> {
        Self::open_with_migrations_from(path, None)
    }

    /// Like [`Self::open`], but with an explicit migrations directory.
    ///
    /// Pass `Some(dir)` to point at a non-default migrations path (useful for
    /// tests where the binary's CWD isn't the workspace root). `None` keeps
    /// the CWD-based default search behavior.
    #[instrument(skip_all, fields(path = %path.display()))]
    pub fn open_with_migrations_from(path: &Path, migrations_dir: Option<&Path>) -> DbResult<Self> {
        let flags = OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_FULL_MUTEX;

        let conn = Connection::open_with_flags(path, flags)
            .map_err(|e| DbError::ConnectionFailed(e.to_string()))?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.configure()?;
        match migrations_dir {
            Some(dir) => db.run_migrations_at(dir)?,
            None => db.run_migrations()?,
        }
        info!("Database opened successfully");
        Ok(db)
    }

    /// Configure PRAGMAs for optimal performance and safety.
    fn configure(&self) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            PRAGMA strict = ON;
            PRAGMA synchronous = FULL;
            PRAGMA cache_size = -50000;
            PRAGMA temp_store = MEMORY;
            PRAGMA busy_timeout = 5000;
            "#,
        )
        .map_err(|e| DbError::QueryFailed(e.to_string()))?;
        Ok(())
    }

    /// Run pending migrations.
    fn run_migrations(&self) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        migrations::run_pending(&conn)?;
        drop(conn);
        Ok(())
    }

    /// Run pending migrations from an explicit directory.
    ///
    /// `Database::open` is a hot path that runs migrations at CWD-relative
    /// paths. Tests that don't sit in the workspace root should call this
    /// method (or the underlying [`migrations::run_pending_at`]) with the
    /// workspace `migrations/` directory derived from
    /// `env!("CARGO_MANIFEST_DIR")`.
    pub fn run_migrations_at(&self, migrations_dir: &Path) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        migrations::run_pending_at(&conn, migrations_dir)?;
        drop(conn);
        Ok(())
    }

    /// Execute a transaction.
    #[instrument(skip_all)]
    pub fn transaction<T, F>(&self, f: F) -> DbResult<T>
    where
        F: FnOnce(&Connection) -> DbResult<T>,
    {
        let conn = self.conn.lock().unwrap();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| DbError::TransactionError(e.to_string()))?;

        let result = f(&tx);

        match result {
            Ok(value) => {
                tx.commit()
                    .map_err(|e| DbError::TransactionError(e.to_string()))?;
                Ok(value)
            }
            Err(e) => {
                let _: Result<(), _> = tx.rollback();
                Err(e)
            }
        }
    }

    /// Get the underlying connection (read-only reference).
    pub fn connection(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap()
    }

    /// Get an Arc reference to the connection.
    pub fn connection_arc(&self) -> Arc<Mutex<Connection>> {
        Arc::clone(&self.conn)
    }

    /// Open a database using embedded schema (no filesystem migrations needed).
    ///
    /// Useful for testing or environments without access to the migrations directory.
    /// Applies both the initial schema and sdef storage schema.
    pub fn open_embedded(path: &Path) -> DbResult<Self> {
        let flags = OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_FULL_MUTEX;

        let conn = Connection::open_with_flags(path, flags)
            .map_err(|e| DbError::ConnectionFailed(e.to_string()))?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.configure()?;
        let conn = db.conn.lock().unwrap();
        let combined = format!(
            "{}\n{}\n{}\n{}",
            crate::embedded_schema::INITIAL_SCHEMA_SQL,
            include_str!("../../../migrations/002_sdef_storage.sql"),
            include_str!("../../../migrations/003_unique_constraints.sql"),
            include_str!("../../../migrations/004_fts_extended.sql"),
        );
        conn.execute_batch(&combined)
            .map_err(|e| DbError::MigrationFailed(e.to_string()))?;
        drop(conn);
        info!("Database opened successfully (embedded schema)");
        Ok(db)
    }

    /// Create an in-memory database for testing.
    pub fn in_memory() -> DbResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| DbError::ConnectionFailed(e.to_string()))?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.configure()?;
        // Apply embedded schema directly for tests
        let conn = db.conn.lock().unwrap();
        conn.execute_batch(crate::embedded_schema::INITIAL_SCHEMA_SQL)
            .map_err(|e| DbError::MigrationFailed(e.to_string()))?;
        drop(conn);
        Ok(db)
    }

    /// Create a backup of the database.
    #[instrument(skip_all)]
    pub fn backup(&self, dest_path: &Path) -> DbResult<()> {
        let conn = self.conn.lock().unwrap();
        let mut dest = Connection::open(dest_path)
            .map_err(|e| DbError::ConnectionFailed(e.to_string()))?;

        let backup = rusqlite::backup::Backup::new(&conn, &mut dest)
            .map_err(|e| DbError::QueryFailed(format!("Backup init failed: {}", e)))?;

        backup.run_to_completion(5, std::time::Duration::from_millis(250), None)
            .map_err(|e| DbError::QueryFailed(format!("Backup failed: {}", e)))?;

        info!(path = %dest_path.display(), "Backup created successfully");
        Ok(())
    }

    /// Start a background scheduled backup task.
    ///
    /// Spawns a tokio task that creates backups at the configured interval.
    /// The task prunes old backups to keep at most `config.max_backups`.
    /// Returns a shutdown handle that can be used to stop the backup loop.
    pub fn start_scheduled_backup(
        db_clone: Self,
        config: BackupConfig,
    ) -> tokio::sync::oneshot::Sender<()> {
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        tokio::spawn(async move {
            // Ensure backup directory exists
            let _ = tokio::task::spawn_blocking({
                let dir = config.backup_dir.clone();
                move || std::fs::create_dir_all(&dir)
            }).await;

            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => {
                        info!("Scheduled backup task shutting down");
                        break;
                    }
                    _ = tokio::time::sleep(config.interval) => {
                        // Generate backup filename with timestamp
                        let timestamp = chrono::Utc::now()
                            .format("%Y%m%d-%H%M%S")
                            .to_string();
                        let filename = config.filename_template
                            .replace("{timestamp}", &timestamp);
                        let backup_path = config.backup_dir.join(&filename);

                        info!(path = %backup_path.display(), "Starting scheduled backup");

                        if let Err(e) = db_clone.backup(&backup_path) {
                            warn!(error = %e, "Scheduled backup failed");
                        } else {
                            // Prune old backups
                            if config.max_backups > 0 {
                                prune_backups(&config.backup_dir, config.max_backups);
                            }
                        }
                    }
                }
            }
        });

        shutdown_tx
    }
}

/// Keep only the N most recent backup files.
fn prune_backups(backup_dir: &Path, max_keep: usize) {
    let mut backups: Vec<_> = match std::fs::read_dir(backup_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "db"))
            .map(|e| (e.path(), std::fs::metadata(e.path()).and_then(|m| m.modified()).ok()))
            .collect(),
        Err(_) => return,
    };

    // Sort by modification time (oldest first)
    backups.sort_by(|a, b| a.1.cmp(&b.1));

    // Remove oldest backups beyond the limit
    if backups.len() > max_keep {
        for (path, _) in backups.drain(..backups.len() - max_keep) {
            if let Err(e) = std::fs::remove_file(&path) {
                warn!(path = %path.display(), error = %e, "Failed to prune old backup");
            } else {
                info!(path = %path.display(), "Pruned old backup");
            }
        }
    }
}

/// Verify SQLite database integrity using `PRAGMA integrity_check`.
///
/// Runs a full database consistency check. Returns `Ok(())` if the
/// database is healthy, or an error describing the corruption.
///
/// Call this on startup to detect corruption before beginning work.
///
/// # Example
///
/// ```rust,ignore
/// let db = Database::open(path)?;
/// Database::verify_integrity(db.connection())?;
/// ```
pub fn verify_database_integrity(conn: &rusqlite::Connection) -> Result<(), crate::DbError> {
    let result: String = conn
        .query_row("PRAGMA integrity_check", [], |row| row.get(0))
        .map_err(|e| crate::DbError::QueryFailed(e.to_string()))?;

    if result != "ok" {
        return Err(crate::DbError::QueryFailed(format!(
            "Database integrity check failed: {}",
            result
        )));
    }

    tracing::info!("Database integrity check passed");
    Ok(())
}

/// Recover database from the most recent backup if corruption is detected.
///
/// Searches `<db_file_dir>/backups/` for backup files (`.db` extension),
/// copies the most recent one to `db_path`, and reopens the connection.
///
/// Returns an error if no backup exists.
///
/// # Example
///
/// ```rust,ignore
/// let db = Database::open("state.db")?;
/// if let Err(_) = Database::verify_integrity(db.connection()) {
///     recover_from_backup(Path::new("state.db"))?;
///     db.reopen()?;
/// }
/// ```
pub fn recover_from_backup(db_path: &std::path::Path) -> Result<(), crate::DbError> {
    let backup_dir = db_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("backups");

    if !backup_dir.exists() {
        return Err(crate::DbError::NotFound {
            resource: "backup directory",
            field: "path",
            value: backup_dir.display().to_string(),
        });
    }

    // Find the most recent backup
    let mut backups: Vec<_> = std::fs::read_dir(&backup_dir)
        .map_err(|e| crate::DbError::QueryFailed(format!("read_dir failed: {}", e)))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "db"))
        .filter_map(|e| {
            let path = e.path();
            let modified = std::fs::metadata(&path)
                .and_then(|m| m.modified())
                .ok()?;
            Some((path, modified))
        })
        .collect();

    backups.sort_by(|a, b| b.1.cmp(&a.1)); // newest first

    if let Some((latest_path, _)) = backups.first() {
        tracing::info!(
            path = %latest_path.display(),
            "Restoring database from backup"
        );
        std::fs::copy(latest_path, db_path)
            .map_err(|e| crate::DbError::QueryFailed(format!(
                "Failed to restore backup '{}' to '{}': {}",
                latest_path.display(),
                db_path.display(),
                e
            )))?;

        // Re-verify after restore
        let temp_conn = rusqlite::Connection::open(db_path)
            .map_err(|e| crate::DbError::ConnectionFailed(format!("{}", e)))?;
        verify_database_integrity(&temp_conn)?;

        Ok(())
    } else {
        Err(crate::DbError::NotFound {
            resource: "backup",
            field: "path",
            value: backup_dir.display().to_string(),
        })
    }
}