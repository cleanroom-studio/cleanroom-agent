//! Language detection utilities.

use std::collections::HashMap;

/// Known file extensions mapped to languages.
/// Covers all OpenCode-supported LSPs plus common file types.
const LANGUAGE_EXTENSIONS: &[(&str, &str)] = &[
    // Rust
    ("rs", "rust"),
    // C/C++
    ("c", "c"),
    ("h", "c"),
    ("cc", "c++"),
    ("cpp", "c++"),
    ("cxx", "c++"),
    ("c++", "c++"),
    ("hpp", "c++"),
    ("hh", "c++"),
    ("hxx", "c++"),
    ("h++", "c++"),
    // Go
    ("go", "go"),
    // Java
    ("java", "java"),
    // Kotlin
    ("kt", "kotlin"),
    ("kts", "kotlin"),
    // Scala
    ("scala", "scala"),
    ("sc", "scala"),
    // C#
    ("cs", "csharp"),
    // F#
    ("fs", "fsharp"),
    ("fsi", "fsharp"),
    ("fsx", "fsharp"),
    ("fsscript", "fsharp"),
    // Swift
    ("swift", "swift"),
    ("objc", "objc"),
    ("objcpp", "objc"),
    // TypeScript/JavaScript
    ("ts", "typescript"),
    ("tsx", "typescript"),
    ("js", "javascript"),
    ("jsx", "javascript"),
    ("mjs", "javascript"),
    ("cjs", "javascript"),
    ("mts", "typescript"),
    ("cts", "typescript"),
    // Python
    ("py", "python"),
    ("pyi", "python"),
    ("pyx", "python"),
    // Ruby
    ("rb", "ruby"),
    ("rake", "ruby"),
    ("gemspec", "ruby"),
    ("ru", "ruby"),
    // PHP
    ("php", "php"),
    // Lua
    ("lua", "lua"),
    // Dart
    ("dart", "dart"),
    // Elixir
    ("ex", "elixir"),
    ("exs", "elixir"),
    // Clojure
    ("clj", "clojure"),
    ("cljs", "clojure"),
    ("cljc", "clojure"),
    ("edn", "clojure"),
    // Haskell
    ("hs", "haskell"),
    ("lhs", "haskell"),
    // OCaml
    ("ml", "ocaml"),
    ("mli", "ocaml"),
    // Gleam
    ("gleam", "gleam"),
    // Shell
    ("sh", "shell"),
    ("bash", "shell"),
    ("zsh", "shell"),
    ("ksh", "shell"),
    // SQL
    ("sql", "sql"),
    // HTML
    ("html", "html"),
    ("htm", "html"),
    // CSS
    ("css", "css"),
    ("scss", "scss"),
    ("sass", "sass"),
    ("less", "less"),
    // Vue
    ("vue", "vue"),
    // Svelte
    ("svelte", "svelte"),
    // Astro
    ("astro", "astro"),
    // YAML
    ("yaml", "yaml"),
    ("yml", "yaml"),
    // JSON
    ("json", "json"),
    ("jsonc", "json"),
    // TOML
    ("toml", "toml"),
    // Markdown
    ("md", "markdown"),
    ("mdx", "markdown"),
    // XML
    ("xml", "xml"),
    // Docker
    ("dockerfile", "dockerfile"),
    // Infrastructure
    ("tf", "terraform"),
    ("tfvars", "terraform"),
    ("nix", "nix"),
    // Proto
    ("proto", "protobuf"),
    // Prisma
    ("prisma", "prisma"),
    // Typst
    ("typ", "typst"),
    ("typc", "typst"),
    // Zig
    ("zig", "zig"),
    ("zon", "zig"),
    // Julia
    ("jl", "julia"),
];

lazy_static::lazy_static! {
    static ref EXTENSION_MAP: HashMap<&'static str, &'static str> = {
        let mut map = HashMap::new();
        for (ext, lang) in LANGUAGE_EXTENSIONS {
            map.insert(*ext, *lang);
        }
        map
    };
}

/// Detect language from file extension.
pub fn detect_language(file_path: &str) -> Option<&'static str> {
    let extension = std::path::Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    extension.and_then(|ext| EXTENSION_MAP.get(ext.as_str()).copied())
}

/// Get all supported languages.
pub fn supported_languages() -> Vec<&'static str> {
    let mut languages: Vec<&str> = EXTENSION_MAP.values().copied().collect();
    languages.sort();
    languages.dedup();
    languages
}

/// Check if a language is supported.
pub fn is_language_supported(language: &str) -> bool {
    supported_languages().contains(&language)
}