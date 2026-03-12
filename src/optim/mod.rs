use crate::tensor::Tensor;

mod sgd;

pub use sgd::Sgd;

pub trait Parameterized {
    fn parameters(&self) -> Vec<Tensor>;
    fn apply_gradients(&mut self, grads: &[Tensor], scale: f32);
}

pub trait Optimizer {
    fn step<M: Parameterized>(&mut self, model: &mut M, grads: &[Tensor]);
}
