use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::engine_v1::Value;

#[derive(Debug, Clone, Default)]
pub struct Neuron {
    pub weights: Vec<Value>,
    pub bias: Value,
}

#[derive(Debug, Clone, Default)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

#[derive(Debug, Clone, Default)]
pub struct Mlp {
    pub layers: Vec<Layer>,
}

impl Neuron {
    pub fn new(nin: usize, rng: &mut StdRng) -> Self {
        let weights = (0..nin)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0)))
            .collect();

        Self {
            weights,
            bias: Value::new(0.0),
        }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        assert_eq!(
            x.len(),
            self.weights.len(),
            "input length must match neuron weight count"
        );

        let mut out = self.bias.clone();
        for (w, xi) in self.weights.iter().zip(x.iter()) {
            let wx = w * xi;
            out = &out + &wx;
        }
        out
    }

    pub fn parameters(&self) -> (Vec<Value>, Vec<Value>) {
        (self.weights.clone(), vec![self.bias.clone()])
    }
}

impl Layer {
    pub fn new(nin: usize, nout: usize, rng: &mut StdRng) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, rng)).collect();
        Self { neurons }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn parameters(&self) -> (Vec<Value>, Vec<Value>) {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for neuron in &self.neurons {
            let (neuron_weights, neuron_biases) = neuron.parameters();
            weights.extend(neuron_weights);
            biases.extend(neuron_biases);
        }

        (weights, biases)
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
            .map(|pair| Layer::new(pair[0], pair[1], &mut rng))
            .collect();

        Self { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut activations: Vec<Value> = x.to_vec();
        let last_idx = self.layers.len().saturating_sub(1);

        for (idx, layer) in self.layers.iter().enumerate() {
            let mut out = layer.forward(&activations);
            if idx < last_idx {
                out = out.into_iter().map(|v| v.tanh()).collect();
            }
            activations = out;
        }

        activations
    }

    pub fn parameters(&self) -> (Vec<Value>, Vec<Value>) {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for layer in &self.layers {
            let (layer_weights, layer_biases) = layer.parameters();
            weights.extend(layer_weights);
            biases.extend(layer_biases);
        }

        (weights, biases)
    }
}
