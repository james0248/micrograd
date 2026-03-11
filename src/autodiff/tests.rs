use crate::tensor::{DenseTensor, Tensor};

use super::jvp::linearize;
use super::{Operation, execute_trace, record_trace};

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
