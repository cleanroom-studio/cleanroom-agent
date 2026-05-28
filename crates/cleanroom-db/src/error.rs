//! Database error types.

use thiserror::Error;

/// Database operation errors.
#[derive(Debug, Error)]
pub enum DbError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Query execution failed: {0}")]
    QueryFailed(String),

    #[error("Migration failed: {0}")]
    MigrationFailed(String),

    #[error("Not found: {resource} with {field} = {value}")]
    NotFound {
        resource: &'static str,
        field: &'static str,
        value: String,
    },

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Transaction error: {0}")]
    TransactionError(String),

    #[error("Invalid status transition from {from} to {to}")]
    InvalidStatusTransition { from: &'static str, to: &'static str },

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Result type alias for database operations.
pub type DbResult<T> = Result<T, DbError>;