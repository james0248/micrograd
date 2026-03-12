use tangent::{
    autodiff,
    engine::{self, Tensor as EngineTensor},
    tensor::Tensor,
};

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

fn legacy_sum_all(mut tensor: EngineTensor) -> EngineTensor {
    while tensor.numel() != 1 {
        tensor = tensor.sum(0, false);
    }
    tensor
}

fn legacy_value_and_grad<F>(input_data: &[(&[f32], &[usize])], f: F) -> (Vec<f32>, Vec<Vec<f32>>)
where
    F: Fn(&[EngineTensor]) -> EngineTensor,
{
    engine::reset_state();
    engine::with_grad(|| {
        let inputs: Vec<EngineTensor> = input_data
            .iter()
            .map(|(data, shape)| EngineTensor::from_vec(data.to_vec(), shape.to_vec()))
            .collect();
        let value = f(&inputs);
        let output = value.data();
        value.backward();
        let grads = inputs.iter().map(|tensor| tensor.grad()).collect();
        (output, grads)
    })
}

#[derive(Clone)]
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    fn next_f32(&mut self, min: f32, max: f32) -> f32 {
        let unit = self.next_u32() as f32 / u32::MAX as f32;
        min + (max - min) * unit
    }
}

fn random_shape(rng: &mut Lcg) -> Vec<usize> {
    match rng.next_u32() % 3 {
        0 => vec![2],
        1 => vec![3],
        _ => vec![2, 2],
    }
}

fn random_tensor(rng: &mut Lcg, shape: &[usize], min: f32, max: f32) -> Tensor {
    let len: usize = shape.iter().product();
    let data = (0..len).map(|_| rng.next_f32(min, max)).collect();
    Tensor::from_vec(data, shape.to_vec())
}

fn eval_public_stress_case(case_id: usize, xs: &[Tensor]) -> Tensor {
    match case_id {
        0 => xs[0].add(&xs[1]).sum_all(),
        1 => xs[0].sub(&xs[1]).mean_all(),
        2 => xs[0].mul(&xs[1]).sum_all(),
        3 => xs[0].div(&xs[1]).mean_all(),
        4 => xs[0].exp().add(&xs[1]).mean_all(),
        5 => xs[0].log().add(&xs[1].log()).sum_all(),
        6 => xs[0].mul(&xs[0]).add(&xs[1].exp()).mean_all(),
        7 => xs[0]
            .mul(&xs[0])
            .add(&xs[1].exp())
            .div(&xs[1])
            .log()
            .mean_all(),
        _ => panic!("unknown public stress case {case_id}"),
    }
}

fn eval_legacy_stress_case(case_id: usize, xs: &[EngineTensor]) -> EngineTensor {
    match case_id {
        0 => legacy_sum_all(xs[0].add(&xs[1])),
        1 => xs[0].sub(&xs[1]).mean(),
        2 => legacy_sum_all(xs[0].mul(&xs[1])),
        3 => xs[0].div(&xs[1]).mean(),
        4 => xs[0].exp().add(&xs[1]).mean(),
        5 => legacy_sum_all(xs[0].log().add(&xs[1].log())),
        6 => xs[0].mul(&xs[0]).add(&xs[1].exp()).mean(),
        7 => xs[0].mul(&xs[0]).add(&xs[1].exp()).div(&xs[1]).log().mean(),
        _ => panic!("unknown legacy stress case {case_id}"),
    }
}

#[test]
fn tensor_from_vec_keeps_shape_and_values() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(x.shape(), vec![2, 2]);
    assert_eq!(x.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn eager_elementwise_ops_support_broadcasting() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = Tensor::from_vec(vec![10.0, 20.0], vec![2]);

    assert_eq!(x.add(&y).to_vec(), vec![11.0, 22.0, 13.0, 24.0]);
    assert_eq!(x.sub(&y).to_vec(), vec![-9.0, -18.0, -7.0, -16.0]);
    assert_eq!(x.mul(&y).to_vec(), vec![10.0, 40.0, 30.0, 80.0]);
    assert_eq!(x.div(&y).to_vec(), vec![0.1, 0.1, 0.3, 0.2]);
}

#[test]
fn eager_unary_and_reduction_ops_are_correct() {
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let exp = x.exp();
    assert_vec_close(&exp.to_vec(), &[1.0f32.exp(), 2.0f32.exp()], 1e-6);

    let log = exp.log();
    assert_vec_close(&log.to_vec(), &[1.0, 2.0], 1e-6);

    let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(y.sum_all().shape(), vec![1]);
    assert_vec_close(&y.sum_all().to_vec(), &[10.0], 1e-6);
    assert_vec_close(&y.mean_all().to_vec(), &[2.5], 1e-6);
}

#[test]
fn eager_axis_reductions_match_expected_shapes_and_values() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let s_row_keep = x.sum(1, true);
    assert_eq!(s_row_keep.shape(), vec![2, 1]);
    assert_eq!(s_row_keep.to_vec(), vec![6.0, 15.0]);

    let s_row = x.sum(1, false);
    assert_eq!(s_row.shape(), vec![2]);
    assert_eq!(s_row.to_vec(), vec![6.0, 15.0]);

    let s_col = x.sum(0, false);
    assert_eq!(s_col.shape(), vec![3]);
    assert_eq!(s_col.to_vec(), vec![5.0, 7.0, 9.0]);

    let m_row_keep = x.max(1, true);
    assert_eq!(m_row_keep.shape(), vec![2, 1]);
    assert_eq!(m_row_keep.to_vec(), vec![3.0, 6.0]);

    let m_row = x.max(1, false);
    assert_eq!(m_row.shape(), vec![2]);
    assert_eq!(m_row.to_vec(), vec![3.0, 6.0]);

    let m_col = x.max(0, false);
    assert_eq!(m_col.shape(), vec![3]);
    assert_eq!(m_col.to_vec(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn reductions_and_relu_work_on_transposed_views() {
    let x = Tensor::from_vec(vec![-1.0, 2.0, 3.0, -4.0, 5.0, 6.0], vec![2, 3]).transpose(0, 1);

    assert_eq!(x.sum(1, false).to_vec(), vec![-5.0, 7.0, 9.0]);
    assert_eq!(x.max(1, false).to_vec(), vec![-1.0, 5.0, 6.0]);
    assert_eq!(x.relu().to_vec(), vec![0.0, 0.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn matmul_supports_rectangular_transposed_and_batched_inputs() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let w = Tensor::from_vec(vec![1.0, 2.0, 0.0, 1.0, 2.0, 3.0], vec![3, 2]);
    let y = x.matmul(&w);
    assert_eq!(y.shape(), vec![2, 2]);
    assert_eq!(y.to_vec(), vec![7.0, 13.0, 16.0, 31.0]);

    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = a.transpose(0, 1).matmul(&b);
    assert_eq!(out.shape(), vec![3, 2]);
    assert_eq!(out.to_vec(), vec![13.0, 18.0, 17.0, 24.0, 21.0, 30.0]);

    let batch_a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
    );
    let batch_b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]);
    let batch_out = batch_a.matmul(&batch_b);
    assert_eq!(batch_out.shape(), vec![2, 2, 2]);
    assert_eq!(
        batch_out.to_vec(),
        vec![4.0, 5.0, 10.0, 11.0, 16.0, 17.0, 22.0, 23.0]
    );
}

#[test]
fn stage2_forward_ops_match_legacy_engine_on_overlapping_behavior() {
    engine::reset_state();

    let legacy_a = EngineTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let legacy_b = EngineTensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
    let legacy_add = legacy_a.add(&legacy_b);
    let legacy_sum = legacy_a.transpose(0, 1).sum(1, false);
    let legacy_max = legacy_a.transpose(0, 1).max(1, false);
    let legacy_relu = EngineTensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).relu();
    let legacy_matmul = legacy_a.transpose(0, 1).matmul(&EngineTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
    ));

    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
    let add = a.add(&b);
    let sum = a.transpose(0, 1).sum(1, false);
    let max = a.transpose(0, 1).max(1, false);
    let relu = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).relu();
    let matmul = a
        .transpose(0, 1)
        .matmul(&Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));

    assert_vec_close(&add.to_vec(), &legacy_add.data(), 1e-6);
    assert_vec_close(&sum.to_vec(), &legacy_sum.data(), 1e-6);
    assert_vec_close(&max.to_vec(), &legacy_max.data(), 1e-6);
    assert_vec_close(&relu.to_vec(), &legacy_relu.data(), 1e-6);
    assert_vec_close(&matmul.to_vec(), &legacy_matmul.data(), 1e-6);
}

#[test]
fn transpose_is_a_view_and_to_vec_materializes_logical_order() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let xt = x.transpose(0, 1);

    assert_eq!(xt.shape(), vec![3, 2]);
    assert_eq!(xt.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(x.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn transpose_composes_with_existing_ops() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = x.transpose(0, 1);
    let z = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).transpose(0, 1);

    assert_vec_close(&y.exp().log().to_vec(), &[1.0, 3.0, 2.0, 4.0], 1e-6);
    assert_vec_close(&y.add(&z).to_vec(), &[11.0, 33.0, 22.0, 44.0], 1e-6);
    assert_vec_close(&y.sum_all().to_vec(), &[10.0], 1e-6);
    assert_vec_close(&y.mean_all().to_vec(), &[2.5], 1e-6);
}

#[test]
fn transpose_backward_matches_legacy_engine() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let (new_value, grads) = autodiff::value_and_grad(|xs| xs[0].transpose(0, 1).sum_all(), &[x]);

    let (legacy_value, legacy_grads) =
        legacy_value_and_grad(&[(&[1.0, 2.0, 3.0, 4.0], &[2, 2])], |xs| {
            legacy_sum_all(xs[0].transpose(0, 1))
        });

    assert_vec_close(&new_value.to_vec(), &legacy_value, 1e-6);
    assert_vec_close(&grads[0].to_vec(), &legacy_grads[0], 1e-6);
}

#[test]
fn to_vec_panics_while_tracing() {
    let result = std::panic::catch_unwind(|| {
        let _ = autodiff::value_and_grad(
            |xs| {
                let _ = xs[0].to_vec();
                xs[0].sum_all()
            },
            &[Tensor::from_vec(vec![1.0, 2.0], vec![2])],
        );
    });
    let msg = panic_message(result.expect_err("expected to_vec panic while tracing"));
    assert!(msg.contains("Tensor::to_vec() is unavailable while tracing"));
}

#[test]
fn invalid_transpose_dims_panic() {
    let result = std::panic::catch_unwind(|| {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _ = x.transpose(0, 2);
    });
    let msg = panic_message(result.expect_err("expected invalid transpose panic"));
    assert!(msg.contains("transpose dims out of bounds"));
}

#[test]
fn incompatible_broadcast_panics_with_explicit_error() {
    let result = std::panic::catch_unwind(|| {
        let _ = Tensor::from_vec(vec![1.0, 2.0], vec![2])
            .add(&Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]));
    });
    let msg = panic_message(result.expect_err("expected shape mismatch panic"));
    assert!(msg.contains("broadcast shape mismatch"));
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
fn autodiff_rejects_broadcasted_add_until_broadcast_gradients_exist() {
    let result = std::panic::catch_unwind(|| {
        let _ = autodiff::value_and_grad(
            |xs| xs[0].add(&xs[1]).sum_all(),
            &[
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                Tensor::from_vec(vec![10.0, 20.0], vec![2]),
            ],
        );
    });
    let msg = panic_message(result.expect_err("expected broadcast autodiff panic"));
    assert!(msg.contains("autodiff does not support broadcasted add yet"));
}

#[test]
fn autodiff_rejects_stage2_forward_ops_until_stage4() {
    let relu = std::panic::catch_unwind(|| {
        let _ = autodiff::value_and_grad(
            |xs| xs[0].relu().sum_all(),
            &[Tensor::from_vec(vec![-1.0, 2.0], vec![2])],
        );
    });
    assert!(
        panic_message(relu.expect_err("expected relu autodiff panic"))
            .contains("autodiff does not support relu yet")
    );

    let sum = std::panic::catch_unwind(|| {
        let _ = autodiff::value_and_grad(
            |xs| xs[0].sum(1, false).sum_all(),
            &[Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
        );
    });
    assert!(
        panic_message(sum.expect_err("expected sum autodiff panic"))
            .contains("autodiff does not support sum yet")
    );

    let max = std::panic::catch_unwind(|| {
        let _ = autodiff::value_and_grad(
            |xs| xs[0].max(1, false).sum_all(),
            &[Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
        );
    });
    assert!(
        panic_message(max.expect_err("expected max autodiff panic"))
            .contains("autodiff does not support max yet")
    );

    let matmul = std::panic::catch_unwind(|| {
        let _ = autodiff::value_and_grad(
            |xs| xs[0].matmul(&xs[1]).sum_all(),
            &[
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
            ],
        );
    });
    assert!(
        panic_message(matmul.expect_err("expected matmul autodiff panic"))
            .contains("autodiff does not support matmul yet")
    );
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
    assert_vec_close(&grads[0].to_vec(), &[1.0, 1.0], 1e-6);
    assert_vec_close(&grads[1].to_vec(), &[1.0, 1.0], 1e-6);
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

    assert_vec_close(&grads[0].to_vec(), &[1.0, 1.0], 1e-6);
    assert_vec_close(&grads[1].to_vec(), &[-1.0, -1.0], 1e-6);
}

#[test]
fn mul_grad_matches_other_operand() {
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let y = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
    let grads = autodiff::grad(|xs| xs[0].mul(&xs[1]).sum_all(), &[x, y]);

    assert_vec_close(&grads[0].to_vec(), &[3.0, 4.0], 1e-6);
    assert_vec_close(&grads[1].to_vec(), &[1.0, 2.0], 1e-6);
}

#[test]
fn div_grad_matches_analytic_result() {
    let x = Tensor::from_vec(vec![2.0, 6.0], vec![2]);
    let y = Tensor::from_vec(vec![4.0, 3.0], vec![2]);
    let grads = autodiff::grad(|xs| xs[0].div(&xs[1]).sum_all(), &[x, y]);

    assert_vec_close(&grads[0].to_vec(), &[0.25, 1.0 / 3.0], 1e-6);
    assert_vec_close(&grads[1].to_vec(), &[-0.125, -2.0 / 3.0], 1e-6);
}

#[test]
fn exp_grad_matches_exp_output() {
    let x = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
    let (value, grads) = autodiff::value_and_grad(|xs| xs[0].exp().sum_all(), &[x]);

    assert_vec_close(&value.to_vec(), &[1.0 + 1.0f32.exp()], 1e-6);
    assert_vec_close(&grads[0].to_vec(), &[1.0, 1.0f32.exp()], 1e-6);
}

#[test]
fn log_grad_is_reciprocal() {
    let grads = autodiff::grad(
        |xs| xs[0].log().sum_all(),
        &[Tensor::from_vec(vec![2.0, 4.0], vec![2])],
    );

    assert_vec_close(&grads[0].to_vec(), &[0.5, 0.25], 1e-6);
}

#[test]
fn mean_all_grad_scales_by_numel() {
    let grads = autodiff::grad(
        |xs| xs[0].mean_all(),
        &[Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
    );

    assert_vec_close(&grads[0].to_vec(), &[0.25, 0.25, 0.25, 0.25], 1e-6);
}

#[test]
fn reused_input_accumulates_cotangents() {
    let grads = autodiff::grad(
        |xs| xs[0].mul(&xs[0]).sum_all(),
        &[Tensor::from_vec(vec![2.0, 3.0], vec![2])],
    );

    assert_vec_close(&grads[0].to_vec(), &[4.0, 6.0], 1e-6);
}

fn finite_difference<F>(f: F, inputs: &[Tensor], input_index: usize, eps: f32) -> Vec<f32>
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let mut out = Vec::new();
    let base_inputs: Vec<Tensor> = inputs.to_vec();
    let target = base_inputs[input_index].to_vec();
    let shape = base_inputs[input_index].shape();

    for idx in 0..target.len() {
        let mut plus_data = base_inputs[input_index].to_vec();
        plus_data[idx] += eps;
        let mut plus_inputs = base_inputs.clone();
        plus_inputs[input_index] = Tensor::from_vec(plus_data, shape.to_vec());

        let mut minus_data = base_inputs[input_index].to_vec();
        minus_data[idx] -= eps;
        let mut minus_inputs = base_inputs.clone();
        minus_inputs[input_index] = Tensor::from_vec(minus_data, shape.to_vec());

        let plus = f(&plus_inputs).to_vec()[0];
        let minus = f(&minus_inputs).to_vec()[0];
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

    assert_vec_close(&grads[0].to_vec(), &fd_x, 2e-3);
    assert_vec_close(&grads[1].to_vec(), &fd_y, 2e-3);
}

#[test]
fn autodiff_matches_tape_engine_on_binary_mean_composition() {
    let input_specs = [(&[1.5, 2.5][..], &[2][..]), (&[0.5, 4.0][..], &[2][..])];
    let public_inputs = vec![
        Tensor::from_vec(vec![1.5, 2.5], vec![2]),
        Tensor::from_vec(vec![0.5, 4.0], vec![2]),
    ];

    let (new_value, new_grads) = autodiff::value_and_grad(
        |xs| xs[0].add(&xs[1]).sub(&xs[1]).mean_all(),
        &public_inputs,
    );
    let (legacy_value, legacy_grads) =
        legacy_value_and_grad(&input_specs, |xs| xs[0].add(&xs[1]).sub(&xs[1]).mean());

    assert_vec_close(&new_value.to_vec(), &legacy_value, 1e-6);
    assert_eq!(new_grads.len(), legacy_grads.len());
    for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
        assert_vec_close(&actual.to_vec(), expected, 1e-6);
    }
}

#[test]
fn autodiff_matches_tape_engine_on_mul_div_log_mean_composition() {
    let input_specs = [(&[1.2, 0.7][..], &[2][..]), (&[2.0, 1.5][..], &[2][..])];
    let public_inputs = vec![
        Tensor::from_vec(vec![1.2, 0.7], vec![2]),
        Tensor::from_vec(vec![2.0, 1.5], vec![2]),
    ];

    let (new_value, new_grads) = autodiff::value_and_grad(
        |xs| xs[0].mul(&xs[0]).add(&xs[1]).div(&xs[1]).log().mean_all(),
        &public_inputs,
    );
    let (legacy_value, legacy_grads) = legacy_value_and_grad(&input_specs, |xs| {
        xs[0].mul(&xs[0]).add(&xs[1]).div(&xs[1]).log().mean()
    });

    assert_vec_close(&new_value.to_vec(), &legacy_value, 1e-6);
    assert_eq!(new_grads.len(), legacy_grads.len());
    for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
        assert_vec_close(&actual.to_vec(), expected, 1e-6);
    }
}

#[test]
fn autodiff_matches_tape_engine_on_exp_and_reused_input_gradients() {
    let input_specs = [(&[0.5, 1.5][..], &[2][..])];
    let public_inputs = vec![Tensor::from_vec(vec![0.5, 1.5], vec![2])];

    let (new_value, new_grads) = autodiff::value_and_grad(
        |xs| xs[0].exp().add(&xs[0].mul(&xs[0])).mean_all(),
        &public_inputs,
    );
    let (legacy_value, legacy_grads) = legacy_value_and_grad(&input_specs, |xs| {
        xs[0].exp().add(&xs[0].mul(&xs[0])).mean()
    });

    assert_vec_close(&new_value.to_vec(), &legacy_value, 1e-6);
    assert_eq!(new_grads.len(), legacy_grads.len());
    for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
        assert_vec_close(&actual.to_vec(), expected, 1e-6);
    }
}

#[test]
fn autodiff_matches_tape_engine_on_sum_all_for_matrix_input() {
    let input_specs = [(&[1.0, 2.0, 3.0, 4.0][..], &[2, 2][..])];
    let public_inputs = vec![Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];

    let (new_value, new_grads) =
        autodiff::value_and_grad(|xs| xs[0].mul(&xs[0]).sum_all(), &public_inputs);
    let (legacy_value, legacy_grads) =
        legacy_value_and_grad(&input_specs, |xs| legacy_sum_all(xs[0].mul(&xs[0])));

    assert_vec_close(&new_value.to_vec(), &legacy_value, 1e-6);
    assert_eq!(new_grads.len(), legacy_grads.len());
    for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
        assert_vec_close(&actual.to_vec(), expected, 1e-6);
    }
}

#[test]
fn randomized_autodiff_matches_tape_engine_on_overlapping_scalar_suite() {
    let mut rng = Lcg::new(0xfeed_beef);

    for case_id in 0..8 {
        for _ in 0..20 {
            let shape = random_shape(&mut rng);
            let x = random_tensor(&mut rng, &shape, 0.5, 2.0);
            let y = random_tensor(&mut rng, &shape, 0.5, 2.0);
            let public_inputs = vec![x.clone(), y.clone()];
            let x_data = x.to_vec();
            let y_data = y.to_vec();
            let input_specs = [(&x_data[..], &shape[..]), (&y_data[..], &shape[..])];

            let (new_value, new_grads) =
                autodiff::value_and_grad(|xs| eval_public_stress_case(case_id, xs), &public_inputs);
            let (legacy_value, legacy_grads) =
                legacy_value_and_grad(&input_specs, |xs| eval_legacy_stress_case(case_id, xs));

            assert_vec_close(&new_value.to_vec(), &legacy_value, 1e-5);
            assert_eq!(new_grads.len(), legacy_grads.len());
            for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
                assert_vec_close(&actual.to_vec(), expected, 1e-5);
            }
        }
    }
}

#[test]
fn randomized_finite_difference_matches_public_autodiff_on_composite_suite() {
    let mut rng = Lcg::new(0x0ddc_0ffe);

    for case_id in 0..4 {
        for _ in 0..10 {
            let shape = random_shape(&mut rng);
            let inputs = vec![
                random_tensor(&mut rng, &shape, 0.5, 2.0),
                random_tensor(&mut rng, &shape, 0.5, 2.0),
            ];

            let f = |xs: &[Tensor]| match case_id {
                0 => xs[0].mul(&xs[0]).add(&xs[1]).log().mean_all(),
                1 => xs[0].div(&xs[1]).add(&xs[1].exp()).mean_all(),
                2 => xs[0].log().mul(&xs[1]).sum_all(),
                3 => xs[0].exp().add(&xs[1].log()).mean_all(),
                _ => panic!("unknown finite-difference stress case {case_id}"),
            };

            let (_value, grads) = autodiff::value_and_grad(f, &inputs);
            let fd_x = finite_difference(f, &inputs, 0, 1e-3);
            let fd_y = finite_difference(f, &inputs, 1, 1e-3);

            assert_vec_close(&grads[0].to_vec(), &fd_x, 3e-3);
            assert_vec_close(&grads[1].to_vec(), &fd_y, 3e-3);
        }
    }
}
