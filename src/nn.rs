use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::checkpoint::{MLP_CHECKPOINT_VERSION, load_mlp_checkpoint, save_mlp_checkpoint};
use crate::engine::Tensor;

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
            w.push(rng.gen_range(-scale..scale));
        }

        let b = vec![0.0; nout];

        Self {
            weight: Tensor::parameter(w, vec![nin, nout]),
            bias: Tensor::parameter(b, vec![nout]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add(&self.bias)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight, self.bias]
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
        let mut out = *x;
        let last_idx = self.layers.len().saturating_sub(1);
        for (idx, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out);
            if idx < last_idx {
                out = out.relu();
            }
        }
        out
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
        let params: Vec<Vec<f32>> = self.parameters().iter().map(Tensor::data).collect();
        save_mlp_checkpoint(path, &dims, &params)
    }

    pub fn load_weights<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), String> {
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

        for (ckpt_param, model_param) in checkpoint.params.iter().zip(params.iter()) {
            model_param.set_data(ckpt_param);
        }

        Ok(())
    }
}
