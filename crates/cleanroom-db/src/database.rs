//! Database connection management.

use rusqlite::{Connection, OpenFlags};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{info, instrument};

use super::error::{DbError, DbResult};
use super::migrations;

/// Thread-safe database connection wrapper.
#[derive(Debug, Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    /// Open or create a database at the given path.
    #[instrument(skip_all, fields(path = %path.display()))]
    pub fn open(path: &Path) -> DbResult<Self> {
        let flags = OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_FULL_MUTEX;

        let conn = Connection::open_with_flags(path, flags)
            .map_err(|e| DbError::ConnectionFailed(e.to_string()))?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.configure()?;
        db.run_migrations()?;
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

    /// Create an in-memory database for testing.
    #[cfg(test)]
    pub fn in_memory() -> DbResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| DbError::ConnectionFailed(e.to_string()))?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.configure()?;
        db.run_migrations()?;
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
}