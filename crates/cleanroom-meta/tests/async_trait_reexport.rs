//! Regression test for the `cleanroom_meta::async_trait` re-export.
//!
//! The `#[derive(MetaHooks)]` / `#[meta_agent]` proc macros in
//! `cleanroom-meta-derive` emit absolute paths like
//! `#[::cleanroom_meta::async_trait]` (see
//! `cleanroom-meta-derive/src/lib.rs`). For that path to resolve, the
//! `cleanroom-meta` facade must re-export the `async_trait` *proc-macro
//! function itself* — not the whole `async_trait` crate as a module.
//!
//! Earlier the facade had:
//! ```ignore
//! pub use ::async_trait;
//! ```
//! which re-exported the crate as a module, so `::cleanroom_meta::async_trait`
//! resolved to a module rather than the attribute item. Proc-macro
//! attribute paths need to resolve to a single item, not a module, so the
//! derive expansion failed at compile time with:
//!   `cannot find async_trait in cleanroom_meta`
//!
//! The fix is:
//! ```ignore
//! pub use ::async_trait::async_trait;
//! ```
//! which re-exports the macro function itself.
//!
//! This test reproduces the exact attribute path the derive macros emit.
//! If the re-export is the wrong shape, this file will not compile.

#[::cleanroom_meta::async_trait]
pub trait Greet {
    async fn greet(&self) -> &'static str;
}

pub struct Hello;

#[::cleanroom_meta::async_trait]
impl Greet for Hello {
    async fn greet(&self) -> &'static str {
        "hello"
    }
}

#[test]
fn facade_async_trait_path_resolves_to_proc_macro() {
    // The mere fact that this file compiled is the actual regression
    // check — both the trait and the impl use the absolute path
    // `#[::cleanroom_meta::async_trait]` that the derive macros emit.
    // The body below only exists so `cargo test` has a function to
    // discover and report; it also confirms the macro-generated
    // `&dyn Greet` vtable is type-correct end-to-end.
    let hello: &dyn Greet = &Hello;
    let _ = hello;
}
