use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
        x.matmul(&self.weight).add_row_bias(&self.bias)
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
}
