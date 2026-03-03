use crate::engine::Value;

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
