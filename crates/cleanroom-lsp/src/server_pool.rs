//! LSP server pool management.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::{Arc, Mutex};

use lsp_server::{Connection, IoThreads};
use lsp_types::ServerCapabilities;
use tokio::process::Command;
use tracing::info;

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

/// An LSP server handle.
pub struct LspServerHandle {
    /// Server capabilities.
    capabilities: ServerCapabilities,
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
    
    /// Running servers.
    servers: Arc<Mutex<HashMap<String, LspServerHandle>>>,
    
    /// Maximum concurrent servers.
    max_concurrent: usize,
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
        }
    }
    
    /// Create with custom configurations.
    pub fn with_configs(configs: Vec<LspConfig>) -> Self {
        let configs = configs.into_iter().map(|c| (c.language_id.clone(), c)).collect();
        Self {
            configs,
            servers: Arc::new(Mutex::new(HashMap::new())),
            max_concurrent: 4,
        }
    }
    
    /// Register a new language configuration.
    pub fn register_config(&mut self, config: LspConfig) {
        self.configs.insert(config.language_id.clone(), config);
    }
    
    /// Get or start an LSP server for a language.
    pub async fn get_server(&self, language: &str) -> LspResult<LspServerHandle> {
        // Check if server is already running
        {
            let servers = self.servers.lock().unwrap();
            if let Some(handle) = servers.get(language) {
                return Ok(handle.clone());
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
        
        // Start the process
        let _child = Command::new(&config.command)
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| LspError::ServerStartFailed(e.to_string()))?;
        
        let handle = LspServerHandle {
            capabilities: ServerCapabilities::default(),
        };
        
        // Store in pool
        {
            let mut servers = self.servers.lock().unwrap();
            servers.insert(config.language_id.clone(), handle.clone());
        }
        
        Ok(handle)
    }
    
    /// Stop a server.
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

impl Clone for LspServerHandle {
    fn clone(&self) -> Self {
        Self {
            capabilities: self.capabilities.clone(),
        }
    }
}