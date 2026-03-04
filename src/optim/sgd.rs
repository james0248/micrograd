use crate::optim::Optimizer;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Sgd {
    params: Vec<Tensor>,
    lr: f32,
}

impl Sgd {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        assert!(lr > 0.0, "learning rate must be > 0");
        Self { params, lr }
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
    fn zero_grad(&mut self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn step(&mut self) {
        for p in &self.params {
            let mut data = p.data();
            let grad = p.grad();
            assert_eq!(data.len(), grad.len(), "parameter/grad size mismatch");
            for i in 0..data.len() {
                data[i] -= self.lr * grad[i];
            }
            p.set_data(&data);
        }
    }
}
