use crate::optim::{Optimizer, Parameterized};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Sgd {
    lr: f32,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        assert!(lr > 0.0, "learning rate must be > 0");
        Self { lr }
    }

    pub fn set_lr(&mut self, lr: f32) {
        assert!(lr > 0.0, "learning rate must be > 0");
        self.lr = lr;
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }
}

impl Optimizer for Sgd {
    fn step<M: Parameterized>(&mut self, model: &mut M, grads: &[Tensor]) {
        model.apply_gradients(grads, self.lr);
    }
}
