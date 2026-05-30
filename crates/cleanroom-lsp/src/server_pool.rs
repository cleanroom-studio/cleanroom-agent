//! LSP server pool management.
//!
//! Manages multiple LSP server subprocesses with:
//! - Lazy initialization (servers started on demand via `LspClient`)
//! - Idle timeout auto-shutdown
//! - Maximum concurrent server limit

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lsp_types::ServerCapabilities;
use tracing::{info, warn};

use super::client::LspClient;
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
    /// Handle for tool invocation (wraps LspClient).
    handle: LspServerHandle,
    /// When the server was last used.
    last_used: Instant,
    /// Language ID this server serves.
    language: String,
    /// Idle timeout configuration.
    idle_timeout: Duration,
}

/// An LSP server handle for invoking tools.
///
/// Provides methods that delegate to the underlying `LspClient`.
#[derive(Clone)]
pub struct LspServerHandle {
    /// Server capabilities.
    pub capabilities: ServerCapabilities,
    /// Language this server handles.
    pub language: String,
    /// Shared state for the running client.
    inner: Arc<Mutex<Option<LspClient>>>,
}

impl LspServerHandle {
    /// Create a new stub handle (for fallback when server can't start).
    fn new_stub(language: String) -> Self {
        Self {
            capabilities: ServerCapabilities::default(),
            language,
            inner: Arc::new(Mutex::new(None)),
        }
    }

    /// Create a handle backed by a running LspClient.
    fn new(client: LspClient) -> Self {
        let language = client.language.clone();
        Self {
            capabilities: ServerCapabilities::default(),
            language,
            inner: Arc::new(Mutex::new(Some(client))),
        }
    }

    /// Check if the underlying LSP client is available.
    pub fn is_connected(&self) -> bool {
        self.inner.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Open a document in the LSP server.
    pub fn did_open(&self, file_path: &str, text: &str, language_id: &str) -> LspResult<()> {
        let guard = self.inner.lock().map_err(|e| LspError::CommunicationError(e.to_string()))?;
        match guard.as_ref() {
            Some(client) => client.did_open(file_path, text, language_id),
            None => Err(LspError::ServerNotAvailable("LSP client not initialized".to_string())),
        }
    }

    /// Get document symbols.
    pub fn document_symbols(&self, file_path: &str) -> LspResult<Vec<super::file_analysis::DocumentSymbol>> {
        let guard = self.inner.lock().map_err(|e| LspError::CommunicationError(e.to_string()))?;
        match guard.as_ref() {
            Some(client) => client.document_symbols(file_path),
            None => Err(LspError::ServerNotAvailable("LSP client not initialized".to_string())),
        }
    }

    /// Get type info at a position via hover.
    pub fn hover(&self, file_path: &str, line: u32, character: u32) -> LspResult<Option<super::file_analysis::TypeInfo>> {
        let guard = self.inner.lock().map_err(|e| LspError::CommunicationError(e.to_string()))?;
        match guard.as_ref() {
            Some(client) => client.hover(file_path, line, character),
            None => Err(LspError::ServerNotAvailable("LSP client not initialized".to_string())),
        }
    }

    /// Find references at a position.
    pub fn find_references(&self, file_path: &str, line: u32, character: u32) -> LspResult<Vec<lsp_types::Location>> {
        let guard = self.inner.lock().map_err(|e| LspError::CommunicationError(e.to_string()))?;
        match guard.as_ref() {
            Some(client) => client.find_references(file_path, line, character),
            None => Err(LspError::ServerNotAvailable("LSP client not initialized".to_string())),
        }
    }

    /// Get diagnostics for a document.
    pub fn diagnostics(&self, file_path: &str) -> LspResult<Vec<super::file_analysis::Diagnostic>> {
        let guard = self.inner.lock().map_err(|e| LspError::CommunicationError(e.to_string()))?;
        match guard.as_ref() {
            Some(client) => client.diagnostics(file_path),
            None => Err(LspError::ServerNotAvailable("LSP client not initialized".to_string())),
        }
    }

    /// Get type hierarchy (supertypes).
    pub fn type_hierarchy(&self, file_path: &str, line: u32, character: u32) -> LspResult<super::file_analysis::TypeHierarchy> {
        let guard = self.inner.lock().map_err(|e| LspError::CommunicationError(e.to_string()))?;
        match guard.as_ref() {
            Some(client) => client.type_hierarchy(file_path, line, character),
            None => Err(LspError::ServerNotAvailable("LSP client not initialized".to_string())),
        }
    }

    /// Full file analysis.
    pub fn analyze_file(&self, file_path: &str, language_id: &str) -> LspResult<super::file_analysis::FileAnalysis> {
        let guard = self.inner.lock().map_err(|e| LspError::CommunicationError(e.to_string()))?;
        match guard.as_ref() {
            Some(client) => client.analyze_file(file_path, language_id),
            None => Err(LspError::ServerNotAvailable("LSP client not initialized".to_string())),
        }
    }
}

impl fmt::Debug for LspServerHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LspServerHandle")
            .field("language", &self.language)
            .field("connected", &self.is_connected())
            .finish()
    }
}

/// Default LSP configurations covering all major languages.
/// Matches the OpenCode supported LSP list.
pub fn default_lsp_configs() -> Vec<LspConfig> {
    vec![
        // ── Compiled Languages ──
        LspConfig { language_id: "rust".into(), command: "rust-analyzer".into(), args: vec![], extensions: vec!["rs".into()], idle_timeout_secs: 600 },
        LspConfig { language_id: "c".into(), command: "clangd".into(), args: vec![], extensions: vec!["c".into(), "h".into(), "cpp".into(), "hpp".into(), "cc".into(), "cxx".into()], idle_timeout_secs: 600 },
        LspConfig { language_id: "go".into(), command: "gopls".into(), args: vec![], extensions: vec!["go".into()], idle_timeout_secs: 600 },
        LspConfig { language_id: "java".into(), command: "jdtls".into(), args: vec![], extensions: vec!["java".into()], idle_timeout_secs: 600 },

        // ── .NET ──
        LspConfig { language_id: "csharp".into(), command: "csharp-ls".into(), args: vec!["--stdio".into()], extensions: vec!["cs".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "fsharp".into(), command: "fsautocomplete".into(), args: vec![], extensions: vec!["fs".into(), "fsi".into(), "fsx".into()], idle_timeout_secs: 300 },

        // ── Apple ──
        LspConfig { language_id: "swift".into(), command: "sourcekit-lsp".into(), args: vec![], extensions: vec!["swift".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "objc".into(), command: "clangd".into(), args: vec![], extensions: vec!["m".into(), "mm".into()], idle_timeout_secs: 300 },

        // ── JVM ──
        LspConfig { language_id: "kotlin".into(), command: "kotlin-language-server".into(), args: vec![], extensions: vec!["kt".into(), "kts".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "scala".into(), command: "metals".into(), args: vec![], extensions: vec!["scala".into(), "sc".into()], idle_timeout_secs: 300 },

        // ── Scripting / Dynamic ──
        LspConfig { language_id: "python".into(), command: "pyright-langserver".into(), args: vec!["--stdio".into()], extensions: vec!["py".into(), "pyi".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "ruby".into(), command: "ruby-lsp".into(), args: vec![], extensions: vec!["rb".into(), "rake".into(), "gemspec".into(), "ru".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "php".into(), command: "intelephense".into(), args: vec!["--stdio".into()], extensions: vec!["php".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "lua".into(), command: "lua-language-server".into(), args: vec![], extensions: vec!["lua".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "elixir".into(), command: "elixir-ls".into(), args: vec![], extensions: vec!["ex".into(), "exs".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "clojure".into(), command: "clojure-lsp".into(), args: vec![], extensions: vec!["clj".into(), "cljs".into(), "cljc".into(), "edn".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "dart".into(), command: "dart".into(), args: vec!["language-server".into()], extensions: vec!["dart".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "haskell".into(), command: "haskell-language-server-wrapper".into(), args: vec![], extensions: vec!["hs".into(), "lhs".into()], idle_timeout_secs: 600 },
        LspConfig { language_id: "ocaml".into(), command: "ocamllsp".into(), args: vec![], extensions: vec!["ml".into(), "mli".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "gleam".into(), command: "gleam".into(), args: vec!["lsp".into()], extensions: vec!["gleam".into()], idle_timeout_secs: 300 },

        // ── TypeScript / Web ──
        LspConfig { language_id: "typescript".into(), command: "typescript-language-server".into(), args: vec!["--stdio".into()], extensions: vec!["ts".into(), "tsx".into(), "js".into(), "jsx".into(), "mjs".into(), "cjs".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "deno".into(), command: "deno".into(), args: vec!["lsp".into()], extensions: vec!["ts".into(), "tsx".into(), "js".into(), "jsx".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "vue".into(), command: "vue-language-server".into(), args: vec!["--stdio".into()], extensions: vec!["vue".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "svelte".into(), command: "svelteserver".into(), args: vec!["--stdio".into()], extensions: vec!["svelte".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "astro".into(), command: "astro-ls".into(), args: vec!["--stdio".into()], extensions: vec!["astro".into()], idle_timeout_secs: 300 },

        // ── Linting ──
        LspConfig { language_id: "eslint".into(), command: "vscode-eslint-language-server".into(), args: vec!["--stdio".into()], extensions: vec!["ts".into(), "tsx".into(), "js".into(), "jsx".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "oxlint".into(), command: "oxlint".into(), args: vec!["--lsp".into()], extensions: vec!["ts".into(), "tsx".into(), "js".into(), "jsx".into()], idle_timeout_secs: 300 },

        // ── Config / Data ──
        LspConfig { language_id: "json".into(), command: "vscode-json-languageserver".into(), args: vec!["--stdio".into()], extensions: vec!["json".into(), "jsonc".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "yaml".into(), command: "yaml-language-server".into(), args: vec!["--stdio".into()], extensions: vec!["yaml".into(), "yml".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "toml".into(), command: "taplo".into(), args: vec!["lsp".into(), "stdio".into()], extensions: vec!["toml".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "sql".into(), command: "sql-language-server".into(), args: vec!["up".into(), "--method".into(), "stdio".into()], extensions: vec!["sql".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "prisma".into(), command: "prisma-language-server".into(), args: vec!["--stdio".into()], extensions: vec!["prisma".into()], idle_timeout_secs: 300 },

        // ── Infrastructure ──
        LspConfig { language_id: "shell".into(), command: "bash-language-server".into(), args: vec!["start".into()], extensions: vec!["sh".into(), "bash".into(), "zsh".into(), "ksh".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "docker".into(), command: "docker-langserver".into(), args: vec!["--stdio".into()], extensions: vec!["dockerfile".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "terraform".into(), command: "terraform-ls".into(), args: vec!["serve".into()], extensions: vec!["tf".into(), "tfvars".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "nix".into(), command: "nixd".into(), args: vec![], extensions: vec!["nix".into()], idle_timeout_secs: 300 },

        // ── Emerging Languages ──
        LspConfig { language_id: "zig".into(), command: "zls".into(), args: vec![], extensions: vec!["zig".into(), "zon".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "julia".into(), command: "julia".into(), args: vec!["--startup-file=no".into(), "--history-file=no".into(), "-e".into(), "using LanguageServer; runserver()".into()], extensions: vec!["jl".into()], idle_timeout_secs: 300 },
        LspConfig { language_id: "typst".into(), command: "tinymist".into(), args: vec![], extensions: vec!["typ".into(), "typc".into()], idle_timeout_secs: 300 },
    ]
}

/// Server pool that manages multiple LSP servers.
pub struct LspServerPool {
    configs: HashMap<String, LspConfig>,
    servers: Arc<Mutex<HashMap<String, ServerState>>>,
    max_concurrent: usize,
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
        // Check if server is already running
        {
            let mut servers = self.servers.lock().unwrap();
            if let Some(state) = servers.get_mut(language) {
                state.last_used = Instant::now();
                return Ok(state.handle.clone());
            }
        }

        // Check concurrent limit
        {
            let servers = self.servers.lock().unwrap();
            if servers.len() >= self.max_concurrent {
                return Err(LspError::ServerNotAvailable(format!(
                    "Max concurrent servers ({}) reached", self.max_concurrent
                )));
            }
        }

        let config = self.configs.get(language).ok_or_else(|| {
            LspError::UnsupportedLanguage(language.to_string())
        })?;

        self.start_server(config).await
    }

    /// Start a new LSP server.
    async fn start_server(&self, config: &LspConfig) -> LspResult<LspServerHandle> {
        info!(language = %config.language_id, command = %config.command, "Starting LSP server");

        // Try to spawn and initialize the LSP client
        let handle = match LspClient::spawn(
            &config.command,
            &config.args,
            &config.language_id,
            ".",
        ) {
            Ok(client) => {
                info!(language = %config.language_id, "LSP client initialized successfully");
                LspServerHandle::new(client)
            }
            Err(e) => {
                warn!(language = %config.language_id, error = %e,
                    "LSP server process failed to start/spawn, using stub handle");
                LspServerHandle::new_stub(config.language_id.clone())
            }
        };

        let idle_timeout = Duration::from_secs(config.idle_timeout_secs);

        let state = ServerState {
            handle: handle.clone(),
            last_used: Instant::now(),
            language: config.language_id.clone(),
            idle_timeout,
        };

        {
            let mut servers = self.servers.lock().unwrap();
            servers.insert(config.language_id.clone(), state);
        }

        self.ensure_idle_monitor();
        Ok(handle)
    }

    /// Start the background idle timeout monitor.
    fn ensure_idle_monitor(&self) {
        let mut running = self.idle_monitor_running.lock().unwrap();
        if *running { return; }
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
                        if now.duration_since(state.last_used) >= state.idle_timeout {
                            to_remove.push(lang.clone());
                        }
                    }
                }

                for lang in &to_remove {
                    info!(language = %lang, "Shutting down idle LSP server");
                    let mut map = servers.lock().unwrap();
                    if let Some(state) = map.get(lang) {
                        if now.duration_since(state.last_used) >= state.idle_timeout {
                            map.remove(lang);
                        }
                    }
                }

                if servers.lock().unwrap().is_empty() {
                    // Continue monitoring in case new servers start
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
    fn default() -> Self { Self::new() }
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
