use rand::Rng;

use crate::{nn::module::Module, nn::scope::param, tensor::Tensor};

pub struct Dense {
    pub out_features: usize,
    pub use_bias: bool,
}

impl Dense {
    pub fn new(out_features: usize, use_bias: bool) -> Self {
        assert!(out_features > 0, "out_features must be > 0");

        Self {
            out_features,
            use_bias,
        }
    }
}

impl Module for Dense {
    fn forward(&self, input: &Tensor) -> Tensor {
        let in_features = input.shape().last().unwrap();

        let kernel = param(
            "kernel",
            |rng, shape| Tensor::zeros(shape),
            vec![*in_features, self.out_features],
        );
        let out = input.matmul(&kernel);

        if self.use_bias {
            let bias = param(
                "bias",
                |rng, shape| Tensor::zeros(shape),
                vec![self.out_features],
            );
            out.add(&bias);
        }

        out
    }
}
