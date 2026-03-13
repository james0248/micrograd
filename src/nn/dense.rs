use crate::{
    nn::{
        init::{self, Initializer},
        module::Module,
        scope::param,
    },
    tensor::Tensor,
};

pub struct Dense {
    pub out_features: usize,
    pub use_bias: bool,

    kernel_initializer: Initializer,
    bias_initializer: Initializer,
}

impl Dense {
    pub fn new(out_features: usize, use_bias: bool) -> Self {
        assert!(out_features > 0, "out_features must be > 0");

        Self {
            out_features,
            use_bias,
            kernel_initializer: init::lecun_normal,
            bias_initializer: init::zeros,
        }
    }

    pub fn with_kernel_init(mut self, initializer: Initializer) -> Self {
        self.kernel_initializer = initializer;
        self
    }
}

impl Module for Dense {
    fn forward(&self, input: &Tensor) -> Tensor {
        let in_features = input.shape().last().unwrap();

        let kernel = param(
            "kernel",
            self.kernel_initializer,
            vec![*in_features, self.out_features],
        );
        let mut out = input.matmul(&kernel);

        if self.use_bias {
            let bias = param("bias", self.bias_initializer, vec![self.out_features]);
            out = out.add(&bias);
        }

        out
    }
}
