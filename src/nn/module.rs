use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{nn::tree::TensorTree, tensor::Tensor};

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;

    fn init(&self, rng: &mut StdRng, dummy_input: &Tensor) -> TensorTree {
        crate::nn::scope::enter_init_scope(StdRng::from_rng(rng));
        self.forward(dummy_input);
        crate::nn::scope::exit_scope()
    }

    fn apply(&self, params: TensorTree, input: &Tensor) -> Tensor {
        crate::nn::scope::enter_apply_scope(params);
        let output = self.forward(input);
        crate::nn::scope::exit_scope();

        output
    }
}
