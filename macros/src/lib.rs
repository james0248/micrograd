use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

#[proc_macro_attribute]
pub fn compact(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let visibility = &input_fn.vis;
    let inputs = &input_fn.sig.inputs;
    let output = &input_fn.sig.output;
    let block = &input_fn.block;

    let expanded = quote! {
        #visibility fn #fn_name(#inputs) #output {
            let full_name = std::any::type_name::<Self>();
            let base_name = full_name.split("::").last().unwrap();

            tangent::__private::push_path(base_name);
            let result = {
                #block
            };
            tangent::__private::pop_path();

            result
        }
    };

    TokenStream::from(expanded)
}
