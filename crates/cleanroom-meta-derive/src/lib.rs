extern crate proc_macro;
use agent::{MetaAgentParser, output::MetaOutputParser};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};
use tool::{ToolParser, input::InputParser};

mod agent;
mod tool;

#[proc_macro_derive(ToolInput, attributes(input))]
pub fn input(input: TokenStream) -> TokenStream {
    InputParser::default().parse(input)
}

#[proc_macro_derive(MetaOutput, attributes(output, strict))]
pub fn agent_output(input: TokenStream) -> TokenStream {
    MetaOutputParser::default().parse(input)
}

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    ToolParser::default().parse(attr, item)
}

#[proc_macro_attribute]
pub fn meta_agent(attr: TokenStream, item: TokenStream) -> TokenStream {
    MetaAgentParser::default().parse(attr, item)
}

#[proc_macro_derive(MetaHooks)]
pub fn derive_meta_hooks(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    // Correctly handle generics
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        // bring async_trait in via absolute path to avoid needing use in consumer crate
        #[::cleanroom_meta::async_trait]
        impl #impl_generics ::cleanroom_meta::core::agent::MetaHooks for #name #ty_generics #where_clause {}
    };

    TokenStream::from(expanded)
}
