#[derive(Debug, Clone, Default)]
pub struct Value {
    data: f64,
    grad: f64,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self { data, grad: 0.0 }
    }

    pub fn data(&self) -> f64 {
        self.data
    }

    pub fn grad(&self) -> f64 {
        self.grad
    }
}
