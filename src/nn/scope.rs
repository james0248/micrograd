use std::cell::RefCell;
use std::collections::HashMap;

use crate::{nn::tree::TensorTree, tensor::Tensor};

use rand::rngs::StdRng;

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = RefCell::new(None);
}

enum RunMode {
    Init,
    Apply,
}

pub struct Scope {
    run_mode: RunMode,
    params: HashMap<Box<str>, TensorTree>,
    prng_key: StdRng,
}

pub fn param<F>(name: &str, init_fn: F, shape: Vec<usize>) -> Tensor
where
    F: Fn(&mut StdRng, Vec<usize>) -> Tensor,
{
    SCOPE.with(|cell| {
        let mut maybe_scope = cell.borrow_mut();
        let scope = maybe_scope
            .as_mut()
            .expect("Called param outside of init() or apply()");

        match scope.run_mode {
            RunMode::Init => {
                let param = init_fn(&mut scope.prng_key, shape);
                scope
                    .params
                    .insert(Box::from(name), TensorTree::Leaf(param.clone()));

                param
            }

            RunMode::Apply => {
                let param = scope
                    .params
                    .get(name)
                    .unwrap_or_else(|| panic!("Parameter {} not found in scope", name));

                match param {
                    TensorTree::Leaf(tensor) => tensor.clone(),
                    _ => panic!("Parameter {} is not a leaf", name),
                }
            }
        }
    })
}
