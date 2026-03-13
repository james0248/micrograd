use crate::tensor::Tensor;
use rand::RngExt;
use rand::rngs::StdRng;
use rand_distr::Normal;

pub type Initializer = fn(&mut StdRng, Vec<usize>) -> Tensor;

pub fn zeros(_rng: &mut StdRng, shape: Vec<usize>) -> Tensor {
    let size = shape.iter().product();
    Tensor::from_vec(vec![0.0; size], shape)
}

pub fn lecun_normal(rng: &mut StdRng, shape: Vec<usize>) -> Tensor {
    let in_dim = shape[0] as f32;
    let scale = 1.0 / in_dim.sqrt();

    let size: usize = shape.iter().product();
    let dist = Normal::new(0.0, scale).unwrap();
    let data: Vec<f32> = (0..size).map(|_| rng.sample(&dist)).collect();
    Tensor::from_vec(data, shape)
}
