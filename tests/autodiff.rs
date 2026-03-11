use tangent::{autodiff, tensor::Tensor};

fn assert_close(actual: f32, expected: f32, eps: f32) {
    assert!(
        (actual - expected).abs() <= eps,
        "expected {expected:.8}, got {actual:.8} (eps={eps})"
    );
}

fn assert_vec_close(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "vector length mismatch: expected {}, got {}",
        expected.len(),
        actual.len()
    );
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_close(actual, expected, eps);
    }
}

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = err.downcast_ref::<&str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = err.downcast_ref::<String>() {
        return msg.clone();
    }
    "<non-string panic>".to_string()
}

#[test]
fn tensor_from_vec_keeps_shape_and_data() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(x.shape(), vec![2, 2]);
    assert_eq!(x.data(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn eager_elementwise_ops_require_same_shapes() {
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let y = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

    assert_eq!(x.add(&y).data(), vec![4.0, 6.0]);
    assert_eq!(x.sub(&y).data(), vec![-2.0, -2.0]);
    assert_eq!(x.mul(&y).data(), vec![3.0, 8.0]);
    assert_eq!(y.div(&x).data(), vec![3.0, 2.0]);
}

#[test]
fn eager_unary_and_reduction_ops_are_correct() {
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let exp = x.exp();
    assert_vec_close(&exp.data(), &[1.0f32.exp(), 2.0f32.exp()], 1e-6);

    let log = exp.log();
    assert_vec_close(&log.data(), &[1.0, 2.0], 1e-6);

    let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(y.sum_all().shape(), vec![1]);
    assert_vec_close(&y.sum_all().data(), &[10.0], 1e-6);
    assert_vec_close(&y.mean_all().data(), &[2.5], 1e-6);
}

#[test]
fn shape_mismatch_panics_with_explicit_error() {
    let result = std::panic::catch_unwind(|| {
        let _ = autodiff::value_and_grad(
            |xs| xs[0].add(&xs[1]).sum_all(),
            &[
                Tensor::from_vec(vec![1.0, 2.0], vec![2]),
                Tensor::from_vec(vec![1.0], vec![1]),
            ],
        );
    });
    let msg = panic_message(result.expect_err("expected shape mismatch panic"));
    assert!(msg.contains("same-shape elementwise op"));
}

#[test]
fn nonscalar_output_panics_with_explicit_error() {
    let result = std::panic::catch_unwind(|| {
        let _ = autodiff::value_and_grad(
            |xs| xs[0].add(&xs[1]),
            &[
                Tensor::from_vec(vec![1.0, 2.0], vec![2]),
                Tensor::from_vec(vec![3.0, 4.0], vec![2]),
            ],
        );
    });
    let msg = panic_message(result.expect_err("expected nonscalar output panic"));
    assert!(msg.contains("require a scalar output tensor"));
}

#[test]
fn add_grad_returns_ones_for_both_inputs() {
    let grads = autodiff::grad(
        |xs| xs[0].add(&xs[1]).sum_all(),
        &[
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
        ],
    );

    assert_eq!(grads.len(), 2);
    assert_vec_close(&grads[0].data(), &[1.0, 1.0], 1e-6);
    assert_vec_close(&grads[1].data(), &[1.0, 1.0], 1e-6);
}

#[test]
fn sub_grad_negates_rhs() {
    let grads = autodiff::grad(
        |xs| xs[0].sub(&xs[1]).sum_all(),
        &[
            Tensor::from_vec(vec![4.0, 5.0], vec![2]),
            Tensor::from_vec(vec![1.5, 2.5], vec![2]),
        ],
    );

    assert_vec_close(&grads[0].data(), &[1.0, 1.0], 1e-6);
    assert_vec_close(&grads[1].data(), &[-1.0, -1.0], 1e-6);
}

#[test]
fn mul_grad_matches_other_operand() {
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let y = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
    let grads = autodiff::grad(|xs| xs[0].mul(&xs[1]).sum_all(), &[x, y]);

    assert_vec_close(&grads[0].data(), &[3.0, 4.0], 1e-6);
    assert_vec_close(&grads[1].data(), &[1.0, 2.0], 1e-6);
}

#[test]
fn div_grad_matches_analytic_result() {
    let x = Tensor::from_vec(vec![2.0, 6.0], vec![2]);
    let y = Tensor::from_vec(vec![4.0, 3.0], vec![2]);
    let grads = autodiff::grad(|xs| xs[0].div(&xs[1]).sum_all(), &[x, y]);

    assert_vec_close(&grads[0].data(), &[0.25, 1.0 / 3.0], 1e-6);
    assert_vec_close(&grads[1].data(), &[-0.125, -2.0 / 3.0], 1e-6);
}

#[test]
fn exp_grad_matches_exp_output() {
    let x = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
    let (value, grads) = autodiff::value_and_grad(|xs| xs[0].exp().sum_all(), &[x]);

    assert_vec_close(&value.data(), &[1.0 + 1.0f32.exp()], 1e-6);
    assert_vec_close(&grads[0].data(), &[1.0, 1.0f32.exp()], 1e-6);
}

#[test]
fn log_grad_is_reciprocal() {
    let grads = autodiff::grad(
        |xs| xs[0].log().sum_all(),
        &[Tensor::from_vec(vec![2.0, 4.0], vec![2])],
    );

    assert_vec_close(&grads[0].data(), &[0.5, 0.25], 1e-6);
}

#[test]
fn mean_all_grad_scales_by_numel() {
    let grads = autodiff::grad(
        |xs| xs[0].mean_all(),
        &[Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
    );

    assert_vec_close(&grads[0].data(), &[0.25, 0.25, 0.25, 0.25], 1e-6);
}

#[test]
fn reused_input_accumulates_cotangents() {
    let grads = autodiff::grad(
        |xs| xs[0].mul(&xs[0]).sum_all(),
        &[Tensor::from_vec(vec![2.0, 3.0], vec![2])],
    );

    assert_vec_close(&grads[0].data(), &[4.0, 6.0], 1e-6);
}

fn finite_difference<F>(f: F, inputs: &[Tensor], input_index: usize, eps: f32) -> Vec<f32>
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let mut out = Vec::new();
    let base_inputs: Vec<Tensor> = inputs.to_vec();
    let target = base_inputs[input_index].data();
    let shape = base_inputs[input_index].shape();

    for idx in 0..target.len() {
        let mut plus_data = base_inputs[input_index].data().to_vec();
        plus_data[idx] += eps;
        let mut plus_inputs = base_inputs.clone();
        plus_inputs[input_index] = Tensor::from_vec(plus_data, shape.to_vec());

        let mut minus_data = base_inputs[input_index].data().to_vec();
        minus_data[idx] -= eps;
        let mut minus_inputs = base_inputs.clone();
        minus_inputs[input_index] = Tensor::from_vec(minus_data, shape.to_vec());

        let plus = f(&plus_inputs).data()[0];
        let minus = f(&minus_inputs).data()[0];
        out.push((plus - minus) / (2.0 * eps));
    }

    out
}

#[test]
fn finite_difference_matches_value_and_grad_on_composite_function() {
    let inputs = vec![
        Tensor::from_vec(vec![1.2, 0.7], vec![2]),
        Tensor::from_vec(vec![2.0, 1.5], vec![2]),
    ];
    let f = |xs: &[Tensor]| xs[0].mul(&xs[0]).add(&xs[1]).log().mean_all();
    let (_value, grads) = autodiff::value_and_grad(f, &inputs);

    let fd_x = finite_difference(f, &inputs, 0, 1e-3);
    let fd_y = finite_difference(f, &inputs, 1, 1e-3);

    assert_vec_close(&grads[0].data(), &fd_x, 2e-3);
    assert_vec_close(&grads[1].data(), &fd_y, 2e-3);
}
