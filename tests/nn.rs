use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use micrograd::engine::{Tensor, no_grad, reset_state, with_grad};
use micrograd::losses::cross_entropy_with_logits;
use micrograd::nn::Mlp;
use micrograd::optim::{Optimizer, Sgd};

fn tiny_dataset() -> Vec<([f32; 2], u8)> {
    vec![
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
        ([1.0, 1.0], 1),
    ]
}

fn build_batch(rows: &[([f32; 2], u8)]) -> (Vec<f32>, Vec<u8>) {
    let mut x = Vec::with_capacity(rows.len() * 2);
    let mut y = Vec::with_capacity(rows.len());
    for (xy, label) in rows {
        x.push(xy[0]);
        x.push(xy[1]);
        y.push(*label);
    }
    (x, y)
}

fn mean_loss(model: &Mlp, rows: &[([f32; 2], u8)]) -> f32 {
    no_grad(|| {
        let (x, y) = build_batch(rows);
        let xb = Tensor::from_vec(x, vec![rows.len(), 2]);
        let logits = model.forward(&xb);
        let loss = cross_entropy_with_logits(&logits, &y);
        loss.data()[0]
    })
}

#[test]
fn mlp_tensor_forward_shape_is_batch_by_output() {
    reset_state();
    let model = Mlp::new(&[784, 128, 10], 7);
    with_grad(|| {
        let x = Tensor::from_vec(vec![0.0; 8 * 784], vec![8, 784]);
        let out = model.forward(&x);
        assert_eq!(out.shape(), vec![8, 10]);
    });
}

#[test]
fn mlp_tensor_parameter_count_matches_dims() {
    reset_state();
    let model = Mlp::new(&[784, 128, 10], 7);
    let params = model.parameters();
    assert_eq!(params.len(), 4);
    let total_numel: usize = params.iter().map(Tensor::numel).sum();
    assert_eq!(total_numel, 784 * 128 + 128 + 128 * 10 + 10);
}

#[test]
fn tensor_training_smoke_loss_decreases() {
    reset_state();
    let model = Mlp::new(&[2, 8, 2], 42);
    let mut optimizer = Sgd::new(model.parameters(), 0.1);
    let mut rng = StdRng::seed_from_u64(11);
    let mut rows = tiny_dataset();
    let initial = mean_loss(&model, &rows);

    for _ in 0..200 {
        rows.shuffle(&mut rng);
        optimizer.zero_grad();

        let (x, y) = build_batch(&rows);
        with_grad(|| {
            let xb = Tensor::from_vec(x, vec![rows.len(), 2]);
            let logits = model.forward(&xb);
            let loss = cross_entropy_with_logits(&logits, &y);
            loss.backward();
        });

        optimizer.step();
    }

    let final_loss = mean_loss(&model, &rows);
    assert!(
        final_loss < initial,
        "expected final loss < initial loss, got initial={initial}, final={final_loss}"
    );
}
