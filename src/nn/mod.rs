pub mod dense;
pub mod init;
pub mod module;
pub mod scope;
pub mod tree;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::checkpoint::{MLP_CHECKPOINT_VERSION, load_mlp_checkpoint, save_mlp_checkpoint};
use crate::optim::Parameterized;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug, Clone, Default)]
pub struct Mlp {
    pub layers: Vec<Linear>,
}

impl Linear {
    pub fn new(nin: usize, nout: usize, rng: &mut StdRng) -> Self {
        assert!(nin > 0, "nin must be > 0");
        assert!(nout > 0, "nout must be > 0");

        let scale = (1.0f32 / nin as f32).sqrt();
        let mut w = Vec::with_capacity(nin * nout);
        for _ in 0..(nin * nout) {
            w.push(rng.random_range(-scale..scale));
        }

        let b = vec![0.0; nout];

        Self {
            weight: Tensor::from_vec(w, vec![nin, nout]),
            bias: Tensor::from_vec(b, vec![nout]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add(&self.bias)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl Mlp {
    pub fn new(dims: &[usize], seed: u64) -> Self {
        assert!(
            dims.len() >= 2,
            "mlp dims must include input and at least one output size"
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let layers = dims
            .windows(2)
            .map(|pair| Linear::new(pair[0], pair[1], &mut rng))
            .collect();

        Self { layers }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_with_params(&self.parameters(), x)
    }

    pub fn forward_with_params(&self, params: &[Tensor], x: &Tensor) -> Tensor {
        self.validate_param_shapes(params, "parameter");

        let mut out = x.clone();
        let last_idx = self.layers.len().saturating_sub(1);
        for (idx, _layer) in self.layers.iter().enumerate() {
            let weight = &params[idx * 2];
            let bias = &params[idx * 2 + 1];

            out = out.matmul(weight).add(bias);
            if idx < last_idx {
                out = out.relu();
            }
        }

        out
    }

    fn validate_param_shapes(&self, tensors: &[Tensor], label: &str) {
        let expected = self.layers.len() * 2;
        assert_eq!(
            tensors.len(),
            expected,
            "{label} count mismatch: expected {expected}, got {}",
            tensors.len()
        );
        for (idx, layer) in self.layers.iter().enumerate() {
            assert_eq!(
                tensors[idx * 2].shape(),
                layer.weight.shape(),
                "{label} shape mismatch for weight at layer {idx}: expected {:?}, got {:?}",
                layer.weight.shape(),
                tensors[idx * 2].shape()
            );
            assert_eq!(
                tensors[idx * 2 + 1].shape(),
                layer.bias.shape(),
                "{label} shape mismatch for bias at layer {idx}: expected {:?}, got {:?}",
                layer.bias.shape(),
                tensors[idx * 2 + 1].shape()
            );
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for layer in &self.layers {
            out.extend(layer.parameters());
        }
        out
    }

    pub fn dims(&self) -> Vec<usize> {
        if self.layers.is_empty() {
            return Vec::new();
        }

        let mut dims = Vec::with_capacity(self.layers.len() + 1);
        for (idx, layer) in self.layers.iter().enumerate() {
            let shape = layer.weight.shape();
            assert_eq!(
                shape.len(),
                2,
                "linear layer weight must be rank-2, got shape={shape:?}"
            );
            if idx == 0 {
                dims.push(shape[0]);
            } else {
                let expected_nin = *dims.last().expect("dims must contain previous output size");
                assert_eq!(
                    shape[0], expected_nin,
                    "layer input/output mismatch at layer {idx}: expected nin={expected_nin}, got nin={}",
                    shape[0]
                );
            }
            dims.push(shape[1]);
        }
        dims
    }

    pub fn save_weights<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), String> {
        let dims = self.dims();
        let params: Vec<Vec<f32>> = self.parameters().iter().map(Tensor::to_vec).collect();
        save_mlp_checkpoint(path, &dims, &params)
    }

    pub fn load_weights<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), String> {
        let checkpoint = load_mlp_checkpoint(path)?;

        if checkpoint.version != MLP_CHECKPOINT_VERSION {
            return Err(format!(
                "unsupported checkpoint version: expected {MLP_CHECKPOINT_VERSION}, got {}",
                checkpoint.version
            ));
        }

        let dims = self.dims();
        if checkpoint.dims != dims {
            return Err(format!(
                "checkpoint dims mismatch: expected {:?}, got {:?}",
                dims, checkpoint.dims
            ));
        }

        let params = self.parameters();
        if checkpoint.params.len() != params.len() {
            return Err(format!(
                "checkpoint parameter count mismatch: expected {}, got {}",
                params.len(),
                checkpoint.params.len()
            ));
        }

        for (idx, (ckpt_param, model_param)) in
            checkpoint.params.iter().zip(params.iter()).enumerate()
        {
            if ckpt_param.len() != model_param.numel() {
                return Err(format!(
                    "checkpoint parameter size mismatch at index {idx}: expected {}, got {}",
                    model_param.numel(),
                    ckpt_param.len()
                ));
            }
        }

        let mut checkpoint_params = checkpoint.params.into_iter();
        for layer in &mut self.layers {
            let weight = checkpoint_params
                .next()
                .expect("validated checkpoint must contain layer weight");
            let bias = checkpoint_params
                .next()
                .expect("validated checkpoint must contain layer bias");
            layer.weight = Tensor::from_vec(weight, layer.weight.shape().to_vec());
            layer.bias = Tensor::from_vec(bias, layer.bias.shape().to_vec());
        }

        Ok(())
    }
}

impl Parameterized for Mlp {
    fn parameters(&self) -> Vec<Tensor> {
        Mlp::parameters(self)
    }

    fn apply_gradients(&mut self, grads: &[Tensor], scale: f32) {
        self.validate_param_shapes(grads, "gradient");

        let scale_tensor = Tensor::from_vec(vec![scale], vec![1]);
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let weight_grad = &grads[idx * 2];
            let bias_grad = &grads[idx * 2 + 1];

            layer.weight = layer.weight.sub(&weight_grad.mul(&scale_tensor));
            layer.bias = layer.bias.sub(&bias_grad.mul(&scale_tensor));
        }
    }
}
