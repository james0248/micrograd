use crate::tensor::{DenseTensor, Tensor};

use super::jvp::linearize;
use super::transpose::transpose_linearized;
use super::{Operation, grad, value_and_grad};

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
    assert_vec_close(&actual.to_vec(), expected, eps);
}

fn add_dense(lhs: &DenseTensor, rhs: &DenseTensor) -> DenseTensor {
    let lhs_data = lhs.to_vec();
    let rhs_data = rhs.to_vec();
    DenseTensor::from_vec(
        lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&x, &y)| x + y)
            .collect(),
        lhs.shape.clone(),
    )
}

fn scale_dense(input: &DenseTensor, scale: f32) -> DenseTensor {
    let data = input.to_vec();
    DenseTensor::from_vec(
        data.iter().map(|&x| x * scale).collect(),
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
            let primal_data = primal.to_vec();
            let tangent_data = tangent.to_vec();
            let data = primal_data
                .iter()
                .zip(tangent_data.iter())
                .map(|(&x, &dx)| x + eps * dx)
                .collect();
            Tensor::from_vec(data, primal.shape.clone())
        })
        .collect();
    let minus_inputs: Vec<Tensor> = primals
        .iter()
        .zip(tangents.iter())
        .map(|(primal, tangent)| {
            let primal_data = primal.to_vec();
            let tangent_data = tangent.to_vec();
            let data = primal_data
                .iter()
                .zip(tangent_data.iter())
                .map(|(&x, &dx)| x - eps * dx)
                .collect();
            Tensor::from_vec(data, primal.shape.clone())
        })
        .collect();

    let plus = f(&plus_inputs);
    let minus = f(&minus_inputs);
    let plus_data = plus.to_vec();
    let minus_data = minus.to_vec();
    let data = plus_data
        .iter()
        .zip(minus_data.iter())
        .map(|(&p, &m)| (p - m) / (2.0 * eps))
        .collect();
    DenseTensor::from_vec(data, plus.shape().to_vec())
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor, eps: f32) {
    assert_eq!(actual.shape(), expected.shape());
    assert_vec_close(&actual.to_vec(), &expected.to_vec(), eps);
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
fn linearize_transpose_matches_analytic_jvp() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
    let tangents = vec![DenseTensor::from_vec(vec![0.5, -1.0, 2.0, 1.5], vec![2, 2])];

    let (output, linearized) = linearize(|xs| xs[0].transpose(0, 1), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[1.0, 3.0, 2.0, 4.0], &[2, 2], 1e-6);
    assert_dense_close(&tangent, &[0.5, 2.0, -1.0, 1.5], &[2, 2], 1e-6);
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

    assert_vec_close(&tangent.to_vec(), &fd.to_vec(), 2e-3);
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
    assert_vec_close(&lhs_add.to_vec(), &rhs_add.to_vec(), 1e-6);

    let lhs_scale = linearized.apply_dense(&[scale_dense(&v1, scalar)]);
    let rhs_scale = scale_dense(&linearized.apply_dense(std::slice::from_ref(&v1)), scalar);
    assert_vec_close(&lhs_scale.to_vec(), &rhs_scale.to_vec(), 1e-6);
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
fn transpose_pullback_transpose_matches_analytic_vjp() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
    let output_cotangent = DenseTensor::from_vec(vec![2.0, -1.0, 0.5, 3.0], vec![2, 2]);

    let (_output, linearized) = linearize(|xs| xs[0].transpose(0, 1), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_eq!(grads.len(), 1);
    assert_dense_close(&grads[0], &[2.0, 0.5, -1.0, 3.0], &[2, 2], 1e-6);
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
