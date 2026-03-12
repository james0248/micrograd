use std::collections::HashMap;

use crate::{nn::tree::TensorTree, tensor::Tensor};

use rand::rngs::StdRng;

enum RunMode {
    Init,
    Apply,
}

pub struct Scope {
    run_mode: RunMode,
    params: HashMap<Box<str>, TensorTree>,
    prng_key: StdRng,
}

impl Scope {
    pub fn params<F>(&mut self, name: &str, init_fn: F, shape: Vec<usize>) -> Tensor
    where
        F: Fn(&mut StdRng, Vec<usize>) -> Tensor,
    {
        match self.run_mode {
            RunMode::Init => {
                let param = init_fn(&mut self.prng_key, shape);
                self.params
                    .insert(Box::from(name), TensorTree::Leaf(param.clone()));

                param
            }

            RunMode::Apply => {
                let param = self.params.get(name).expect("Parameter not found");

                match param {
                    TensorTree::Leaf(tensor) => tensor.clone(),
                    _ => panic!("Parameter {} is not a leaf", name),
                }
            }
        }
    }
}
