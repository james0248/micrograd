use tangent::nn::Mlp;
use tangent::optim::{Optimizer, Sgd};
use tangent::tensor::Tensor;

fn snapshot_params(model: &Mlp) -> Vec<Vec<f32>> {
    model.parameters().iter().map(Tensor::to_vec).collect()
}

#[test]
fn sgd_step_updates_model_parameter_data() {
    let mut model = Mlp::new(&[2, 3], 7);
    let before = snapshot_params(&model);
    let grads = vec![
        Tensor::from_vec(vec![0.5, -1.0, 2.0, 0.25, 1.5, -0.75], vec![2, 3]),
        Tensor::from_vec(vec![1.0, -0.5, 0.25], vec![3]),
    ];

    let mut opt = Sgd::new(0.1);
    opt.step(&mut model, &grads);

    let after = snapshot_params(&model);
    for ((before_param, after_param), grad) in before.iter().zip(after.iter()).zip(grads.iter()) {
        for ((before_value, after_value), grad_value) in before_param
            .iter()
            .zip(after_param.iter())
            .zip(grad.to_vec().iter())
        {
            let expected = before_value - 0.1 * grad_value;
            assert!((after_value - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn sgd_set_lr_is_applied_on_next_step() {
    let mut model = Mlp::new(&[1, 1], 11);
    let before = snapshot_params(&model);
    let grads = vec![
        Tensor::from_vec(vec![1.0], vec![1, 1]),
        Tensor::from_vec(vec![1.0], vec![1]),
    ];

    let mut opt = Sgd::new(0.1);
    opt.set_lr(0.5);
    opt.step(&mut model, &grads);

    let after = snapshot_params(&model);
    assert!((after[0][0] - (before[0][0] - 0.5)).abs() < 1e-6);
    assert!((after[1][0] - (before[1][0] - 0.5)).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "gradient count mismatch")]
fn sgd_step_rejects_gradient_count_mismatch() {
    let mut model = Mlp::new(&[2, 3], 5);
    let grads = vec![Tensor::from_vec(vec![1.0; 6], vec![2, 3])];

    let mut opt = Sgd::new(0.1);
    opt.step(&mut model, &grads);
}

#[test]
#[should_panic(expected = "gradient shape mismatch")]
fn sgd_step_rejects_gradient_shape_mismatch() {
    let mut model = Mlp::new(&[2, 3], 5);
    let grads = vec![
        Tensor::from_vec(vec![1.0; 3], vec![3, 1]),
        Tensor::from_vec(vec![1.0; 3], vec![3]),
    ];

    let mut opt = Sgd::new(0.1);
    opt.step(&mut model, &grads);
}
