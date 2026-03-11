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

    assert_vec_close(new_value.data(), &legacy_value, 1e-6);
    assert_eq!(new_grads.len(), legacy_grads.len());
    for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
        assert_vec_close(actual.data(), expected, 1e-6);
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

    assert_vec_close(new_value.data(), &legacy_value, 1e-6);
    assert_eq!(new_grads.len(), legacy_grads.len());
    for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
        assert_vec_close(actual.data(), expected, 1e-6);
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

    assert_vec_close(new_value.data(), &legacy_value, 1e-6);
    assert_eq!(new_grads.len(), legacy_grads.len());
    for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
        assert_vec_close(actual.data(), expected, 1e-6);
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

    assert_vec_close(new_value.data(), &legacy_value, 1e-6);
    assert_eq!(new_grads.len(), legacy_grads.len());
    for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
        assert_vec_close(actual.data(), expected, 1e-6);
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
            let input_specs = [(&x.data()[..], &shape[..]), (&y.data()[..], &shape[..])];

            let (new_value, new_grads) =
                autodiff::value_and_grad(|xs| eval_public_stress_case(case_id, xs), &public_inputs);
            let (legacy_value, legacy_grads) =
                legacy_value_and_grad(&input_specs, |xs| eval_legacy_stress_case(case_id, xs));

            assert_vec_close(new_value.data(), &legacy_value, 1e-5);
            assert_eq!(new_grads.len(), legacy_grads.len());
            for (actual, expected) in new_grads.iter().zip(legacy_grads.iter()) {
                assert_vec_close(actual.data(), expected, 1e-5);
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

            assert_vec_close(grads[0].data(), &fd_x, 3e-3);
            assert_vec_close(grads[1].data(), &fd_y, 3e-3);
        }
    }
}
