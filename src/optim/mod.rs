mod sgd;

pub use sgd::Sgd;

pub trait Optimizer {
    fn zero_grad(&mut self);
    fn step(&mut self);
}
