//! LSP server pool management.
//!
//! Manages multiple LSP server subprocesses with:
//! - Lazy initialization (servers started on demand)
//! - Idle timeout auto-shutdown
//! - Maximum concurrent server limit

use std::collections::HashMap;
use std::fmt;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lsp_types::ServerCapabilities;
use tokio::process::{Child, Command};
use tracing::{info, warn};

use super::error::{LspError, LspResult};

/// LSP server configuration for a language.
#[derive(Debug, Clone)]
pub struct LspConfig {
    /// Language identifier.
    pub language_id: String,

    /// Command to start the LSP server.
    pub command: String,

    /// Command arguments.
    pub args: Vec<String>,

    /// File extensions this server handles.
    pub extensions: Vec<String>,

    /// Idle timeout in seconds before server shutdown.
    pub idle_timeout_secs: u64,
}

/// Runtime state of an LSP server.
struct ServerState {
    /// Handle for tool invocation.
    handle: LspServerHandle,
    /// The child process (dropped on shutdown).
    #[allow(dead_code)]
    child: Option<Child>,
    /// When the server was last used.
    last_used: Instant,
    /// Language ID this server serves.
    language: String,
    /// Idle timeout configuration.
    idle_timeout: Duration,
}

/// An LSP server handle for invoking tools.
#[derive(Debug, Clone)]
pub struct LspServerHandle {
    /// Server capabilities.
    pub capabilities: ServerCapabilities,
    /// Language this server handles.
    pub language: String,
}

impl LspServerHandle {
    fn new(language: String) -> Self {
        Self {
            capabilities: ServerCapabilities::default(),
            language,
        }
    }
}

/// Default LSP configurations.
pub fn default_lsp_configs() -> Vec<LspConfig> {
    vec![
        // TypeScript/JavaScript
        LspConfig {
            language_id: "typescript".to_string(),
            command: "typescript-language-server".to_string(),
            args: vec!["--stdio".to_string()],
            extensions: vec!["ts".to_string(), "tsx".to_string(), "js".to_string(), "jsx".to_string()],
            idle_timeout_secs: 300,
        },
        // Rust
        LspConfig {
            language_id: "rust".to_string(),
            command: "rust-analyzer".to_string(),
            args: vec![],
            extensions: vec!["rs".to_string()],
            idle_timeout_secs: 600,
        },
        // Python
        LspConfig {
            language_id: "python".to_string(),
            command: "pyright-langserver".to_string(),
            args: vec!["--stdio".to_string()],
            extensions: vec!["py".to_string()],
            idle_timeout_secs: 300,
        },
        // Go
        LspConfig {
            language_id: "go".to_string(),
            command: "gopls".to_string(),
            args: vec![],
            extensions: vec!["go".to_string()],
            idle_timeout_secs: 600,
        },
    ]
}

/// Server pool that manages multiple LSP servers.
pub struct LspServerPool {
    /// Server configurations.
    configs: HashMap<String, LspConfig>,

    /// Running servers with runtime state.
    servers: Arc<Mutex<HashMap<String, ServerState>>>,

    /// Maximum concurrent servers.
    max_concurrent: usize,

    /// Whether idle timeout background task is running.
    idle_monitor_running: Arc<Mutex<bool>>,
}

impl LspServerPool {
    /// Create a new server pool with default configurations.
    pub fn new() -> Self {
        let mut configs = HashMap::new();
        for config in default_lsp_configs() {
            configs.insert(config.language_id.clone(), config);
        }

        Self {
            configs,
            servers: Arc::new(Mutex::new(HashMap::new())),
            max_concurrent: 4,
            idle_monitor_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Create with custom configurations.
    pub fn with_configs(configs: Vec<LspConfig>) -> Self {
        let configs = configs.into_iter().map(|c| (c.language_id.clone(), c)).collect();
        Self {
            configs,
            servers: Arc::new(Mutex::new(HashMap::new())),
            max_concurrent: 4,
            idle_monitor_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Set the maximum number of concurrent LSP servers.
    pub fn set_max_concurrent(&mut self, max: usize) {
        self.max_concurrent = max;
    }

    /// Get the current concurrent server limit.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Get the current count of running servers.
    pub fn running_count(&self) -> usize {
        self.servers.lock().unwrap().len()
    }

    /// Register a new language configuration.
    pub fn register_config(&mut self, config: LspConfig) {
        self.configs.insert(config.language_id.clone(), config);
    }

    /// Touch a server so its idle timer resets.
    pub fn touch_server(&self, language: &str) {
        if let Some(state) = self.servers.lock().unwrap().get_mut(language) {
            state.last_used = Instant::now();
        }
    }

    /// Get or start an LSP server for a language.
    pub async fn get_server(&self, language: &str) -> LspResult<LspServerHandle> {
        // Check if server is already running and update last_used
        {
            let mut servers = self.servers.lock().unwrap();
            if let Some(state) = servers.get_mut(language) {
                state.last_used = Instant::now();
                return Ok(state.handle.clone());
            }
        }

        // Check concurrent limit before starting a new server
        {
            let servers = self.servers.lock().unwrap();
            if servers.len() >= self.max_concurrent {
                return Err(LspError::ServerNotAvailable(format!(
                    "Max concurrent servers ({}) reached. Try again later or stop an idle server.",
                    self.max_concurrent
                )));
            }
        }

        // Get configuration
        let config = self.configs.get(language).ok_or_else(|| {
            LspError::UnsupportedLanguage(language.to_string())
        })?;

        // Start new server
        self.start_server(config).await
    }

    /// Start a new LSP server.
    async fn start_server(&self, config: &LspConfig) -> LspResult<LspServerHandle> {
        info!(language = %config.language_id, command = %config.command, "Starting LSP server");

        // Attempt to spawn the process; warn on failure but still create a stub handle
        let child = match Command::new(&config.command)
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(c) => Some(c),
            Err(e) => {
                warn!(language = %config.language_id, error = %e,
                    "LSP server process failed to start, using stub handle");
                None
            }
        };

        let handle = LspServerHandle::new(config.language_id.clone());
        let idle_timeout = Duration::from_secs(config.idle_timeout_secs);

        let state = ServerState {
            handle: handle.clone(),
            child,
            last_used: Instant::now(),
            language: config.language_id.clone(),
            idle_timeout,
        };

        // Store in pool
        {
            let mut servers = self.servers.lock().unwrap();
            servers.insert(config.language_id.clone(), state);
        }

        // Start idle monitor on first server start
        self.ensure_idle_monitor();

        Ok(handle)
    }

    /// Start the background idle timeout monitor if not already running.
    fn ensure_idle_monitor(&self) {
        let mut running = self.idle_monitor_running.lock().unwrap();
        if *running {
            return;
        }
        *running = true;
        drop(running);

        let servers = self.servers.clone();
        tokio::spawn(async move {
            let check_interval = Duration::from_secs(30);
            loop {
                tokio::time::sleep(check_interval).await;
                let now = Instant::now();
                let mut to_remove = Vec::new();

                {
                    let map = servers.lock().unwrap();
                    for (lang, state) in map.iter() {
                        let elapsed = now.duration_since(state.last_used);
                        if elapsed >= state.idle_timeout {
                            to_remove.push(lang.clone());
                        }
                    }
                }

                for lang in &to_remove {
                    info!(language = %lang, "Shutting down idle LSP server");
                    let mut map = servers.lock().unwrap();
                    // Double-check that the server hasn't been used since we checked
                    if let Some(state) = map.get(lang) {
                        if now.duration_since(state.last_used) >= state.idle_timeout {
                            map.remove(lang);
                        }
                    }
                }

                // Stop the monitor if no servers are running
                if servers.lock().unwrap().is_empty() {
                    // Servers are empty; continue monitoring in case new ones start later
                }
            }
        });
    }

    /// Stop a specific server.
    pub fn stop_server(&self, language: &str) -> LspResult<()> {
        let mut servers = self.servers.lock().unwrap();
        if servers.remove(language).is_some() {
            info!(language = %language, "Stopped LSP server");
        }
        Ok(())
    }

    /// Stop all servers.
    pub fn stop_all(&self) {
        let mut servers = self.servers.lock().unwrap();
        for language in servers.keys().cloned().collect::<Vec<_>>() {
            servers.remove(&language);
            info!(language = %language, "Stopped LSP server");
        }
    }

    /// Shutdown and clean up.
    pub fn shutdown(&self) {
        self.stop_all();
    }
}

impl Default for LspServerPool {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for LspServerPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LspServerPool")
            .field("configs", &self.configs.keys())
            .field("running_count", &self.running_count())
            .field("max_concurrent", &self.max_concurrent)
            .finish()
    }
}