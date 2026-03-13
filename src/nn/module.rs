use rand::rngs::StdRng;

use crate::{
    nn::scope::{enter_apply_scope, enter_init_scope, exit_scope, pop_path, push_path},
    nn::tree::TensorTree,
    tensor::Tensor,
};

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;

    fn call(&self, input: &Tensor) -> Tensor {
        let full_name = std::any::type_name::<Self>();
        let base_name = full_name.split("::").last().unwrap();

        push_path(base_name);
        let output = self.forward(input);
        pop_path();

        output
    }

    fn init(&self, rng: &mut StdRng, dummy_input: &Tensor) -> TensorTree {
        enter_init_scope(rng.clone());
        self.call(dummy_input);
        exit_scope()
    }

    // Apply the module to an input tensor using the given parameters
    fn apply(&self, params: TensorTree, input: &Tensor) -> Tensor {
        enter_apply_scope(params);
        let output = self.call(input);
        exit_scope();

        output
    }
}
