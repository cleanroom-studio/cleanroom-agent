//! Progress tracking utilities for the Cleanroom Agent CLI.
//!
//! Provides progress bars, spinners, and output formatting helpers to give
//! users visual feedback during long-running operations like code analysis
//! and generation.
//!
//! # Components
//!
//! - [`ProgressBar`] — Determinate progress with percentage and ETA
//! - [`ProgressSpinner`] — Indeterminate spinner for unknown-duration tasks
//! - [`MultiProgress`] — Manager for parallel progress tracking
//! - Formatting helpers — Key-value pairs, sections, and bullet lists
//!
//! # Example
//!
//! ```rust,ignore
//! use cleanroom_cli::progress::{create_progress_bar, create_spinner, format_kv};
//!
//! let pb = create_progress_bar(100, "Processing files...");
//! for i in 0..100 {
//!     pb.set_position(i);
//!     std::thread::sleep(std::time::Duration::from_millis(20));
//! }
//! pb.finish_with_message("Done");
//!
//! let spinner = create_spinner("Analyzing...");
//! // ... do work ...
//! spinner.finish_with_message("Complete");
//! ```

#![allow(dead_code)]

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Duration;

/// Create a styled progress bar for tracking pipeline progress.
pub fn create_progress_bar(total: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message(msg.to_string());
    pb
}

/// Create a spinner for indeterminate tasks.
pub fn create_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a multi-progress manager for parallel task tracking.
pub fn create_multi_progress() -> MultiProgress {
    MultiProgress::new()
}

/// Format a key-value pair for human-readable output.
pub fn format_kv(key: &str, value: impl std::fmt::Display) -> String {
    format!("  {:<30} {}", key, value)
}

/// Format a section header.
pub fn format_section(title: &str) -> String {
    format!("\n─── {} ───", title)
}

/// Pretty-print a list of items with indentation.
pub fn format_list(items: &[impl std::fmt::Display]) -> String {
    items.iter().map(|item| format!("    • {}", item)).collect::<Vec<_>>().join("\n")
}
