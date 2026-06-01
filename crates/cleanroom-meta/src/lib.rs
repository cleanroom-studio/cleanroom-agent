//! # cleanroom-meta
//!
//! Facade crate for the Cleanroom Meta agent framework. The
//! `#[derive(MetaHooks)]`, `#[derive(MetaOutput)]`, and `#[meta_agent]` proc
//! macros emit absolute paths like `::cleanroom_meta::async_trait` and
//! `::cleanroom_meta::core::agent::MetaHooks`. This crate exists to give those
//! paths a single resolution target.
//!
//! Layout:
//! - `cleanroom_meta::async_trait` — re-export of the `async_trait` crate
//!   (consumers don't need to depend on it directly).
//! - `cleanroom_meta::core` — re-export of `cleanroom-meta-core`.
//! - `cleanroom_meta::llm` — re-export of `cleanroom-meta-llm`.
//! - `cleanroom_meta::protocol` — re-export of `cleanroom-meta-protocol`.
//! - `cleanroom_meta::derive` — re-export of `cleanroom-meta-derive`.
//!
//! Consumers (e.g. `cleanroom-agent`) typically depend on this crate plus
//! the individual subcrates they need.

// Re-export the `async_trait` proc-macro attribute itself (not the whole crate)
// so that `::cleanroom_meta::async_trait` resolves to the macro, matching
// the path emitted by `#[derive(MetaHooks)]` and friends in
// `cleanroom-meta-derive`. The earlier form `pub use ::async_trait;`
// re-exported the whole crate as a module, which made the path resolve to
// a module rather than the attribute item and produced
// `cannot find async_trait in cleanroom_meta` at proc-macro expansion time.
pub use ::async_trait::async_trait;

pub mod core {
    pub use ::cleanroom_meta_core::*;
}

pub mod llm {
    pub use ::cleanroom_meta_llm::*;
}

pub mod protocol {
    pub use ::cleanroom_meta_protocol::*;
}

pub mod derive {
    pub use ::cleanroom_meta_derive::*;
}
