use tangent::engine::{Tensor, reset_state};
use tangent::optim::{Optimizer, Sgd};

#[test]
fn sgd_step_updates_parameter_data() {
    reset_state();
    let p = Tensor::parameter(vec![1.0, -2.0, 3.0], vec![3]);
    p.set_grad(&[0.5, -1.0, 2.0]);

    let mut opt = Sgd::new(vec![p], 0.1);
    opt.step();

    let updated = p.data();
    assert_eq!(updated.len(), 3);
    assert!((updated[0] - 0.95).abs() < 1e-6);
    assert!((updated[1] - -1.9).abs() < 1e-6);
    assert!((updated[2] - 2.8).abs() < 1e-6);
}

#[test]
fn sgd_zero_grad_clears_all_parameter_grads() {
    reset_state();
    let a = Tensor::parameter(vec![1.0, 2.0], vec![2]);
    let b = Tensor::parameter(vec![3.0], vec![1]);
    a.set_grad(&[1.0, -2.0]);
    b.set_grad(&[5.0]);

    let mut opt = Sgd::new(vec![a, b], 0.01);
    opt.zero_grad();

    assert_eq!(a.grad(), vec![0.0, 0.0]);
    assert_eq!(b.grad(), vec![0.0]);
}

#[test]
fn sgd_set_lr_is_applied_on_next_step() {
    reset_state();
    let p = Tensor::parameter(vec![1.0], vec![1]);
    p.set_grad(&[1.0]);

    let mut opt = Sgd::new(vec![p], 0.1);
    opt.set_lr(0.5);
    opt.step();
    assert!((p.data()[0] - 0.5).abs() < 1e-6);
}
