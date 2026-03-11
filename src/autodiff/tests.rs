use crate::tensor::{DenseTensor, Tensor};

use super::jvp::linearize;
use super::transpose::transpose_linearized;
use super::{
    Operation, execute_trace, grad, record_trace, value_and_grad, value_and_grad_direct,
    value_and_grad_transposed,
};

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = err.downcast_ref::<&str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = err.downcast_ref::<String>() {
        return msg.clone();
    }
    "<non-string panic>".to_string()
}

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

fn assert_dense_close(actual: &DenseTensor, expected: &[f32], shape: &[usize], eps: f32) {
    assert_eq!(actual.shape, shape);
    assert_vec_close(&actual.data, expected, eps);
}

fn add_dense(lhs: &DenseTensor, rhs: &DenseTensor) -> DenseTensor {
    DenseTensor::from_vec(
        lhs.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&x, &y)| x + y)
            .collect(),
        lhs.shape.clone(),
    )
}

fn scale_dense(input: &DenseTensor, scale: f32) -> DenseTensor {
    DenseTensor::from_vec(
        input.data.iter().map(|&x| x * scale).collect(),
        input.shape.clone(),
    )
}

fn directional_finite_difference<F>(
    f: F,
    primals: &[DenseTensor],
    tangents: &[DenseTensor],
    eps: f32,
) -> DenseTensor
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let plus_inputs: Vec<Tensor> = primals
        .iter()
        .zip(tangents.iter())
        .map(|(primal, tangent)| {
            let data = primal
                .data
                .iter()
                .zip(tangent.data.iter())
                .map(|(&x, &dx)| x + eps * dx)
                .collect();
            Tensor::from_vec(data, primal.shape.clone())
        })
        .collect();
    let minus_inputs: Vec<Tensor> = primals
        .iter()
        .zip(tangents.iter())
        .map(|(primal, tangent)| {
            let data = primal
                .data
                .iter()
                .zip(tangent.data.iter())
                .map(|(&x, &dx)| x - eps * dx)
                .collect();
            Tensor::from_vec(data, primal.shape.clone())
        })
        .collect();

    let plus = f(&plus_inputs);
    let minus = f(&minus_inputs);
    let data = plus
        .data()
        .iter()
        .zip(minus.data().iter())
        .map(|(&p, &m)| (p - m) / (2.0 * eps))
        .collect();
    DenseTensor::from_vec(data, plus.shape().to_vec())
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor, eps: f32) {
    assert_eq!(actual.shape(), expected.shape());
    assert_vec_close(actual.data(), expected.data(), eps);
}

fn assert_value_and_grad_close(
    actual: &(Tensor, Vec<Tensor>),
    expected: &(Tensor, Vec<Tensor>),
    eps: f32,
) {
    assert_tensor_close(&actual.0, &expected.0, eps);
    assert_eq!(actual.1.len(), expected.1.len());
    for (actual_grad, expected_grad) in actual.1.iter().zip(expected.1.iter()) {
        assert_tensor_close(actual_grad, expected_grad, eps);
    }
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

fn random_positive_dense(rng: &mut Lcg, shape: &[usize], min: f32, max: f32) -> DenseTensor {
    let len = shape.iter().product();
    let data = (0..len).map(|_| rng.next_f32(min, max)).collect();
    DenseTensor::from_vec(data, shape.to_vec())
}

fn public_inputs_from_dense(inputs: &[DenseTensor]) -> Vec<Tensor> {
    inputs
        .iter()
        .map(|tensor| Tensor::from_vec(tensor.data.clone(), tensor.shape.clone()))
        .collect()
}

fn scalar_stress_case(case_id: usize, xs: &[Tensor]) -> Tensor {
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
        _ => panic!("unknown scalar stress case {case_id}"),
    }
}

fn nonscalar_stress_case(case_id: usize, xs: &[Tensor]) -> Tensor {
    match case_id {
        0 => xs[0].add(&xs[1]),
        1 => xs[0].sub(&xs[1]).exp(),
        2 => xs[0].mul(&xs[1]).log(),
        3 => xs[0].div(&xs[1]),
        4 => xs[0].mul(&xs[0]).add(&xs[1].exp()),
        _ => panic!("unknown nonscalar stress case {case_id}"),
    }
}

#[test]
fn trace_emits_stable_instruction_order_and_specs() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0], vec![2]),
        DenseTensor::from_vec(vec![3.0, 4.0], vec![2]),
    ];
    let trace = record_trace(|xs| xs[0].add(&xs[1]).mul(&xs[0]).log().sum_all(), &inputs);

    assert_eq!(trace.inputs.len(), 2);
    assert_eq!(trace.outputs.len(), 1);
    assert_eq!(trace.instructions.len(), 4);
    assert_eq!(trace.instructions[0].op, Operation::Add);
    assert_eq!(trace.instructions[1].op, Operation::Mul);
    assert_eq!(trace.instructions[2].op, Operation::Log);
    assert_eq!(trace.instructions[3].op, Operation::SumAll);
    assert_eq!(trace.instructions[0].inputs.len(), 2);
    assert_eq!(trace.instructions[0].spec.shape, vec![2]);
    assert_eq!(trace.instructions[3].spec.shape, vec![1]);
}

#[test]
fn execute_trace_matches_eager_execution_for_recorded_trace() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0], vec![2]),
        DenseTensor::from_vec(vec![0.5, 1.5], vec![2]),
    ];
    let trace = record_trace(|xs| xs[0].mul(&xs[1]).add(&xs[0].exp()).mean_all(), &inputs);
    let interpreted = execute_trace(&trace, &inputs);
    let eager = Tensor::from_vec(vec![1.0, 2.0], vec![2])
        .mul(&Tensor::from_vec(vec![0.5, 1.5], vec![2]))
        .add(&Tensor::from_vec(vec![1.0, 2.0], vec![2]).exp())
        .mean_all();

    assert_eq!(interpreted.len(), 1);
    assert_vec_close(&interpreted[0].data, &eager.data(), 1e-6);
}

#[test]
fn record_trace_allows_nonscalar_output_but_build_vjp_rejects_it() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0], vec![2]),
        DenseTensor::from_vec(vec![3.0, 4.0], vec![2]),
    ];
    let trace = record_trace(|xs| xs[0].add(&xs[1]), &inputs);

    assert_eq!(trace.outputs.len(), 1);

    let result = std::panic::catch_unwind(|| super::vjp::build_vjp(&trace));
    let msg = panic_message(result.expect_err("expected non-scalar VJP panic"));
    assert!(msg.contains("require a scalar output tensor"));
}

#[test]
fn linearize_add_matches_analytic_jvp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0], vec![2]),
        DenseTensor::from_vec(vec![3.0, 4.0], vec![2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![0.5, 1.5], vec![2]),
        DenseTensor::from_vec(vec![2.0, -1.0], vec![2]),
    ];

    let (output, linearized) = linearize(|xs| xs[0].add(&xs[1]), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[4.0, 6.0], &[2], 1e-6);
    assert_dense_close(&tangent, &[2.5, 0.5], &[2], 1e-6);
}

#[test]
fn linearize_sub_matches_analytic_jvp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![5.0, 7.0], vec![2]),
        DenseTensor::from_vec(vec![2.0, 3.0], vec![2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![1.0, -2.0], vec![2]),
        DenseTensor::from_vec(vec![0.5, 4.0], vec![2]),
    ];

    let (_output, linearized) = linearize(|xs| xs[0].sub(&xs[1]), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&tangent, &[0.5, -6.0], &[2], 1e-6);
}

#[test]
fn linearize_mul_matches_analytic_jvp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![2.0, 3.0], vec![2]),
        DenseTensor::from_vec(vec![4.0, 5.0], vec![2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![1.0, 0.5], vec![2]),
        DenseTensor::from_vec(vec![0.25, 2.0], vec![2]),
    ];

    let (_output, linearized) = linearize(|xs| xs[0].mul(&xs[1]), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&tangent, &[4.5, 8.5], &[2], 1e-6);
}

#[test]
fn linearize_div_matches_analytic_jvp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![6.0, 8.0], vec![2]),
        DenseTensor::from_vec(vec![3.0, 4.0], vec![2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![1.5, 2.0], vec![2]),
        DenseTensor::from_vec(vec![0.5, 1.0], vec![2]),
    ];

    let (_output, linearized) = linearize(|xs| xs[0].div(&xs[1]), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&tangent, &[0.16666667, 0.0], &[2], 1e-6);
}

#[test]
fn linearize_exp_matches_analytic_jvp() {
    let inputs = vec![DenseTensor::from_vec(vec![0.0, 1.0], vec![2])];
    let tangents = vec![DenseTensor::from_vec(vec![2.0, -0.5], vec![2])];

    let (_output, linearized) = linearize(|xs| xs[0].exp(), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&tangent, &[2.0, -0.5 * 1.0f32.exp()], &[2], 1e-6);
}

#[test]
fn linearize_log_matches_analytic_jvp() {
    let inputs = vec![DenseTensor::from_vec(vec![2.0, 4.0], vec![2])];
    let tangents = vec![DenseTensor::from_vec(vec![1.0, -2.0], vec![2])];

    let (_output, linearized) = linearize(|xs| xs[0].log(), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&tangent, &[0.5, -0.5], &[2], 1e-6);
}

#[test]
fn linearize_sum_all_matches_analytic_jvp() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])];
    let tangents = vec![DenseTensor::from_vec(vec![0.5, -1.0, 2.5], vec![3])];

    let (_output, linearized) = linearize(|xs| xs[0].sum_all(), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&tangent, &[2.0], &[1], 1e-6);
}

#[test]
fn linearize_mean_all_matches_analytic_jvp() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
    let tangents = vec![DenseTensor::from_vec(vec![2.0, 0.0, -2.0, 4.0], vec![2, 2])];

    let (_output, linearized) = linearize(|xs| xs[0].mean_all(), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&tangent, &[1.0], &[1], 1e-6);
}

#[test]
fn linearize_supports_nonscalar_outputs() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0], vec![2]),
        DenseTensor::from_vec(vec![3.0, 4.0], vec![2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![1.0, 1.0], vec![2]),
        DenseTensor::from_vec(vec![2.0, 3.0], vec![2]),
    ];

    let (output, linearized) = linearize(|xs| xs[0].add(&xs[1]).log(), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[4.0f32.ln(), 6.0f32.ln()], &[2], 1e-6);
    assert_dense_close(&tangent, &[0.75, 2.0 / 3.0], &[2], 1e-6);
}

#[test]
fn linearize_captures_forward_coefficients_as_residual_inputs() {
    let inputs = vec![
        DenseTensor::from_vec(vec![2.0, 3.0], vec![2]),
        DenseTensor::from_vec(vec![4.0, 5.0], vec![2]),
    ];

    let (_output, linearized) = linearize(|xs| xs[0].mul(&xs[1]), &inputs);

    assert_eq!(linearized.tangent_input_specs.len(), 2);
    assert_eq!(linearized.trace.inputs.len(), 4);
    assert_eq!(linearized.residuals.len(), 2);
    assert_eq!(linearized.trace.consts.len(), 0);
    assert_eq!(linearized.trace.instructions.len(), 3);
    assert_eq!(linearized.trace.instructions[0].op, Operation::Mul);
    assert_eq!(linearized.trace.instructions[1].op, Operation::Mul);
    assert_eq!(linearized.trace.instructions[2].op, Operation::Add);
}

#[test]
fn linearize_matches_directional_finite_difference_on_composite_function() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.2, 0.7], vec![2]),
        DenseTensor::from_vec(vec![2.0, 1.5], vec![2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![0.25, -0.5], vec![2]),
        DenseTensor::from_vec(vec![1.0, 0.75], vec![2]),
    ];
    let f = |xs: &[Tensor]| xs[0].mul(&xs[0]).add(&xs[1]).log();

    let (_output, linearized) = linearize(f, &inputs);
    let tangent = linearized.apply_dense(&tangents);
    let fd = directional_finite_difference(f, &inputs, &tangents, 1e-3);

    assert_vec_close(&tangent.data, &fd.data, 2e-3);
}

#[test]
fn linearize_tangent_map_is_linear() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0], vec![2])];
    let v1 = DenseTensor::from_vec(vec![0.5, -1.0], vec![2]);
    let v2 = DenseTensor::from_vec(vec![1.5, 2.0], vec![2]);
    let scalar = 2.5;

    let (_output, linearized) = linearize(|xs| xs[0].mul(&xs[0]).add(&xs[0].exp()), &inputs);

    let lhs_add = linearized.apply_dense(&[add_dense(&v1, &v2)]);
    let rhs_add = add_dense(
        &linearized.apply_dense(std::slice::from_ref(&v1)),
        &linearized.apply_dense(std::slice::from_ref(&v2)),
    );
    assert_vec_close(&lhs_add.data, &rhs_add.data, 1e-6);

    let lhs_scale = linearized.apply_dense(&[scale_dense(&v1, scalar)]);
    let rhs_scale = scale_dense(&linearized.apply_dense(std::slice::from_ref(&v1)), scalar);
    assert_vec_close(&lhs_scale.data, &rhs_scale.data, 1e-6);
}

#[test]
fn linearize_handles_jvp_plus_constant_operands() {
    let inputs = vec![DenseTensor::from_vec(vec![2.0, 3.0], vec![2])];
    let tangents = vec![DenseTensor::from_vec(vec![1.5, -2.0], vec![2])];

    let (_output, linearized) = linearize(
        |xs| xs[0].mul(&Tensor::from_vec(vec![3.0, 4.0], vec![2])),
        &inputs,
    );
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&tangent, &[4.5, -8.0], &[2], 1e-6);
}

#[test]
fn linearize_constant_outputs_produce_zero_tangents() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0], vec![2])];
    let tangents = vec![DenseTensor::from_vec(vec![3.0, 4.0], vec![2])];

    let (output, linearized) = linearize(|_xs| Tensor::from_vec(vec![5.0, 6.0], vec![2]), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[5.0, 6.0], &[2], 1e-6);
    assert_dense_close(&tangent, &[0.0, 0.0], &[2], 1e-6);
}

#[test]
fn transpose_pullback_add_matches_analytic_vjp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0], vec![2]),
        DenseTensor::from_vec(vec![3.0, 4.0], vec![2]),
    ];
    let output_cotangent = DenseTensor::from_vec(vec![2.0, -1.0], vec![2]);

    let (_output, linearized) = linearize(|xs| xs[0].add(&xs[1]), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_eq!(grads.len(), 2);
    assert_dense_close(&grads[0], &[2.0, -1.0], &[2], 1e-6);
    assert_dense_close(&grads[1], &[2.0, -1.0], &[2], 1e-6);
}

#[test]
fn transpose_pullback_sub_matches_analytic_vjp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![5.0, 7.0], vec![2]),
        DenseTensor::from_vec(vec![2.0, 3.0], vec![2]),
    ];
    let output_cotangent = DenseTensor::from_vec(vec![1.5, -0.5], vec![2]);

    let (_output, linearized) = linearize(|xs| xs[0].sub(&xs[1]), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[1.5, -0.5], &[2], 1e-6);
    assert_dense_close(&grads[1], &[-1.5, 0.5], &[2], 1e-6);
}

#[test]
fn transpose_pullback_mul_matches_analytic_vjp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![2.0, 3.0], vec![2]),
        DenseTensor::from_vec(vec![4.0, 5.0], vec![2]),
    ];
    let output_cotangent = DenseTensor::from_vec(vec![0.5, -2.0], vec![2]);

    let (_output, linearized) = linearize(|xs| xs[0].mul(&xs[1]), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[2.0, -10.0], &[2], 1e-6);
    assert_dense_close(&grads[1], &[1.0, -6.0], &[2], 1e-6);
}

#[test]
fn transpose_pullback_div_matches_analytic_vjp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![6.0, 8.0], vec![2]),
        DenseTensor::from_vec(vec![3.0, 4.0], vec![2]),
    ];
    let output_cotangent = DenseTensor::from_vec(vec![2.0, -1.0], vec![2]);

    let (_output, linearized) = linearize(|xs| xs[0].div(&xs[1]), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[0.6666667, -0.25], &[2], 1e-6);
    assert_dense_close(&grads[1], &[-1.3333334, 0.5], &[2], 1e-6);
}

#[test]
fn transpose_pullback_exp_matches_analytic_vjp() {
    let inputs = vec![DenseTensor::from_vec(vec![0.0, 1.0], vec![2])];
    let output_cotangent = DenseTensor::from_vec(vec![2.0, -0.5], vec![2]);

    let (_output, linearized) = linearize(|xs| xs[0].exp(), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[2.0, -0.5 * 1.0f32.exp()], &[2], 1e-6);
}

#[test]
fn transpose_pullback_log_matches_analytic_vjp() {
    let inputs = vec![DenseTensor::from_vec(vec![2.0, 4.0], vec![2])];
    let output_cotangent = DenseTensor::from_vec(vec![1.0, -2.0], vec![2]);

    let (_output, linearized) = linearize(|xs| xs[0].log(), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[0.5, -0.5], &[2], 1e-6);
}

#[test]
fn transpose_pullback_sum_all_matches_analytic_vjp() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])];
    let output_cotangent = DenseTensor::from_vec(vec![2.5], vec![1]);

    let (_output, linearized) = linearize(|xs| xs[0].sum_all(), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[2.5, 2.5, 2.5], &[3], 1e-6);
}

#[test]
fn transpose_pullback_mean_all_matches_analytic_vjp() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
    let output_cotangent = DenseTensor::from_vec(vec![4.0], vec![1]);

    let (_output, linearized) = linearize(|xs| xs[0].mean_all(), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[1.0, 1.0, 1.0, 1.0], &[2, 2], 1e-6);
}

#[test]
fn transpose_pullback_matches_direct_vjp_on_scalar_output() {
    let public_inputs = vec![
        Tensor::from_vec(vec![1.2, 0.7], vec![2]),
        Tensor::from_vec(vec![2.0, 1.5], vec![2]),
    ];
    let f = |xs: &[Tensor]| xs[0].mul(&xs[0]).add(&xs[1]).log().sum_all();

    let direct = value_and_grad_direct(&f, &public_inputs);
    let transposed = value_and_grad_transposed(&f, &public_inputs);

    assert_value_and_grad_close(&transposed, &direct, 1e-6);
}

#[test]
fn transpose_pullback_constant_outputs_produce_zero_input_cotangents() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0], vec![2])];

    let (_output, linearized) = linearize(|_xs| Tensor::from_vec(vec![5.0, 6.0], vec![2]), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&DenseTensor::from_vec(vec![3.0, 4.0], vec![2]));

    assert_eq!(grads.len(), 1);
    assert_dense_close(&grads[0], &[0.0, 0.0], &[2], 1e-6);
}

#[test]
fn direct_and_transposed_paths_match_on_reused_input_accumulation() {
    let inputs = vec![Tensor::from_vec(vec![2.0, 3.0], vec![2])];
    let f = |xs: &[Tensor]| xs[0].mul(&xs[0]).sum_all();

    let direct = value_and_grad_direct(&f, &inputs);
    let transposed = value_and_grad_transposed(&f, &inputs);

    assert_value_and_grad_close(&transposed, &direct, 1e-6);
}

#[test]
fn direct_and_transposed_paths_match_on_mixed_nonlinear_composition() {
    let inputs = vec![
        Tensor::from_vec(vec![1.2, 0.7], vec![2]),
        Tensor::from_vec(vec![2.0, 1.5], vec![2]),
    ];
    let f = |xs: &[Tensor]| {
        xs[0]
            .mul(&xs[0])
            .add(&xs[1].exp())
            .div(&xs[1])
            .log()
            .mean_all()
    };

    let direct = value_and_grad_direct(&f, &inputs);
    let transposed = value_and_grad_transposed(&f, &inputs);

    assert_value_and_grad_close(&transposed, &direct, 1e-6);
}

#[test]
fn public_value_and_grad_matches_direct_baseline() {
    let inputs = vec![
        Tensor::from_vec(vec![1.5, 2.5], vec![2]),
        Tensor::from_vec(vec![0.5, 1.0], vec![2]),
    ];
    let f = |xs: &[Tensor]| xs[0].div(&xs[1]).add(&xs[1].log()).sum_all();

    let direct = value_and_grad_direct(&f, &inputs);
    let public = value_and_grad(f, &inputs);

    assert_value_and_grad_close(&public, &direct, 1e-6);
}

#[test]
fn public_grad_matches_value_and_grad_gradients() {
    let inputs = vec![
        Tensor::from_vec(vec![1.5, 2.5], vec![2]),
        Tensor::from_vec(vec![0.5, 1.0], vec![2]),
    ];
    let f = |xs: &[Tensor]| xs[0].div(&xs[1]).add(&xs[1].log()).sum_all();

    let grads = grad(&f, &inputs);
    let (_, expected_grads) = value_and_grad(f, &inputs);

    assert_eq!(grads.len(), expected_grads.len());
    for (actual, expected) in grads.iter().zip(expected_grads.iter()) {
        assert_tensor_close(actual, expected, 1e-6);
    }
}

#[test]
fn randomized_direct_and_transposed_paths_match_on_scalar_stress_suite() {
    let mut rng = Lcg::new(0x5eed_cafe);

    for case_id in 0..8 {
        for _ in 0..24 {
            let shape = random_shape(&mut rng);
            let inputs = vec![
                Tensor::from_vec(
                    random_positive_dense(&mut rng, &shape, 0.5, 2.0).data,
                    shape.clone(),
                ),
                Tensor::from_vec(
                    random_positive_dense(&mut rng, &shape, 0.5, 2.0).data,
                    shape.clone(),
                ),
            ];

            let direct = value_and_grad_direct(&|xs| scalar_stress_case(case_id, xs), &inputs);
            let transposed =
                value_and_grad_transposed(&|xs| scalar_stress_case(case_id, xs), &inputs);

            assert_value_and_grad_close(&transposed, &direct, 1e-5);
        }
    }
}

#[test]
fn randomized_pullback_matches_scalarized_direct_vjp_for_nonscalar_outputs() {
    let mut rng = Lcg::new(0x1234_5678);

    for case_id in 0..5 {
        for _ in 0..16 {
            let shape = random_shape(&mut rng);
            let concrete_inputs = vec![
                random_positive_dense(&mut rng, &shape, 0.5, 2.0),
                random_positive_dense(&mut rng, &shape, 0.5, 2.0),
            ];
            let public_inputs = public_inputs_from_dense(&concrete_inputs);
            let output_cotangent = random_positive_dense(&mut rng, &shape, -1.5, 1.5);

            let (_output, linearized) =
                linearize(|xs| nonscalar_stress_case(case_id, xs), &concrete_inputs);
            let pullback = transpose_linearized(&linearized);
            let grads = pullback.apply_dense(&output_cotangent);

            let cotangent_tensor = Tensor::from_vec(
                output_cotangent.data.clone(),
                output_cotangent.shape.clone(),
            );
            let scalarized_direct = value_and_grad_direct(
                &|xs| {
                    nonscalar_stress_case(case_id, xs)
                        .mul(&cotangent_tensor)
                        .sum_all()
                },
                &public_inputs,
            );

            assert_eq!(grads.len(), scalarized_direct.1.len());
            for (actual, expected) in grads.iter().zip(scalarized_direct.1.iter()) {
                assert_vec_close(&actual.data, expected.data(), 1e-5);
            }
        }
    }
}
