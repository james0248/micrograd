use crate::tensor::{
    DenseTensor, Tensor, TensorSpec, expand_to_shape, max_axis_weights, sum_to_shape,
};

use super::interpreter::execute_trace;
use super::jvp::linearize;
use super::trace::Recorder;
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
fn execute_trace_supports_stage3_batched_ops() {
    let mut recorder = Recorder::new_empty();
    let x = recorder.add_input(TensorSpec::new(vec![2, 2]));
    let y = recorder.add_input(TensorSpec::new(vec![2]));
    let w = recorder.add_input(TensorSpec::new(vec![2, 2]));

    let add = recorder.add_instruction(
        Operation::Add,
        vec![x.var, y.var],
        TensorSpec::new(vec![2, 2]),
    );
    let relu =
        recorder.add_instruction(Operation::Relu, vec![add.var], TensorSpec::new(vec![2, 2]));
    let sum = recorder.add_instruction(
        Operation::Sum {
            axis: 1,
            keepdim: false,
        },
        vec![relu.var],
        TensorSpec::new(vec![2]),
    );
    let matmul = recorder.add_instruction(
        Operation::MatMul,
        vec![x.var, w.var],
        TensorSpec::new(vec![2, 2]),
    );
    let max = recorder.add_instruction(
        Operation::Max {
            axis: 1,
            keepdim: false,
        },
        vec![matmul.var],
        TensorSpec::new(vec![2]),
    );
    let trace = recorder.into_trace(vec![sum.var, max.var]);

    let x_val = DenseTensor::from_vec(vec![-1.0, 2.0, 3.0, -4.0], vec![2, 2]);
    let y_val = DenseTensor::from_vec(vec![10.0, -1.0], vec![2]);
    let w_val = DenseTensor::from_vec(vec![1.0, 2.0, 0.0, 1.0], vec![2, 2]);
    let outputs = execute_trace(&trace, &[x_val.clone(), y_val.clone(), w_val.clone()]);

    let expected_sum = Tensor::from_vec(x_val.to_vec(), vec![2, 2])
        .add(&Tensor::from_vec(y_val.to_vec(), vec![2]))
        .relu()
        .sum(1, false);
    let expected_max = Tensor::from_vec(x_val.to_vec(), vec![2, 2])
        .matmul(&Tensor::from_vec(w_val.to_vec(), vec![2, 2]))
        .max(1, false);

    assert_dense_close(&outputs[0], &expected_sum.to_vec(), &[2], 1e-6);
    assert_dense_close(&outputs[1], &expected_max.to_vec(), &[2], 1e-6);
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
fn linearize_broadcast_add_matches_expected_jvp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        DenseTensor::from_vec(vec![10.0, 20.0], vec![2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![0.5, -1.0, 1.5, 2.0], vec![2, 2]),
        DenseTensor::from_vec(vec![2.0, 3.0], vec![2]),
    ];

    let (output, linearized) = linearize(|xs| xs[0].add(&xs[1]), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[11.0, 22.0, 13.0, 24.0], &[2, 2], 1e-6);
    assert_dense_close(&tangent, &[2.5, 2.0, 3.5, 5.0], &[2, 2], 1e-6);
}

#[test]
fn linearize_broadcast_mul_matches_expected_jvp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        DenseTensor::from_vec(vec![10.0, 20.0], vec![2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![0.5, -1.0, 1.5, 2.0], vec![2, 2]),
        DenseTensor::from_vec(vec![2.0, 3.0], vec![2]),
    ];

    let (output, linearized) = linearize(|xs| xs[0].mul(&xs[1]), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[10.0, 40.0, 30.0, 80.0], &[2, 2], 1e-6);
    assert_dense_close(&tangent, &[7.0, -14.0, 21.0, 52.0], &[2, 2], 1e-6);
}

#[test]
fn linearize_broadcast_div_matches_expected_jvp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]),
        DenseTensor::from_vec(vec![2.0, 4.0], vec![2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![1.0, -2.0, 3.0, 4.0], vec![2, 2]),
        DenseTensor::from_vec(vec![0.5, 1.0], vec![2]),
    ];

    let (output, linearized) = linearize(|xs| xs[0].div(&xs[1]), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[5.0, 5.0, 15.0, 10.0], &[2, 2], 1e-6);
    assert_dense_close(&tangent, &[-0.75, -1.75, -2.25, -1.5], &[2, 2], 1e-6);
}

#[test]
fn linearize_sum_axis_matches_expected_jvp() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
    let tangents = vec![DenseTensor::from_vec(vec![0.5, -1.0, 2.0, 1.5], vec![2, 2])];

    let (output, linearized) = linearize(|xs| xs[0].sum(1, false), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[3.0, 7.0], &[2], 1e-6);
    assert_dense_close(&tangent, &[-0.5, 3.5], &[2], 1e-6);
}

#[test]
fn linearize_relu_matches_expected_jvp() {
    let inputs = vec![DenseTensor::from_vec(
        vec![-1.0, 2.0, 3.0, -4.0],
        vec![2, 2],
    )];
    let tangents = vec![DenseTensor::from_vec(vec![0.5, -1.0, 2.0, 1.5], vec![2, 2])];

    let (output, linearized) = linearize(|xs| xs[0].relu(), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[0.0, 2.0, 3.0, 0.0], &[2, 2], 1e-6);
    assert_dense_close(&tangent, &[0.0, -1.0, 2.0, 0.0], &[2, 2], 1e-6);
}

#[test]
fn linearize_matmul_matches_expected_jvp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        DenseTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
    ];
    let tangents = vec![
        DenseTensor::from_vec(vec![0.5, -1.0, 1.5, 2.0], vec![2, 2]),
        DenseTensor::from_vec(vec![1.0, 0.0, -0.5, 2.0], vec![2, 2]),
    ];

    let (output, linearized) = linearize(|xs| xs[0].matmul(&xs[1]), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[19.0, 22.0, 43.0, 50.0], &[2, 2], 1e-6);
    assert_dense_close(&tangent, &[-4.5, -1.0, 22.5, 33.0], &[2, 2], 1e-6);
}

#[test]
fn linearize_max_matches_expected_jvp_with_split_ties() {
    let inputs = vec![DenseTensor::from_vec(vec![2.0, 2.0, 1.0, 3.0], vec![2, 2])];
    let tangents = vec![DenseTensor::from_vec(vec![4.0, 2.0, -1.0, 6.0], vec![2, 2])];

    let (output, linearized) = linearize(|xs| xs[0].max(1, false), &inputs);
    let tangent = linearized.apply_dense(&tangents);

    assert_dense_close(&output, &[2.0, 3.0], &[2], 1e-6);
    assert_dense_close(&tangent, &[3.0, 6.0], &[2], 1e-6);
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
fn sum_to_shape_reduces_broadcasted_axes() {
    let input = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let reduced = sum_to_shape(&input, &[2]);
    assert_dense_close(&reduced, &[4.0, 6.0], &[2], 1e-6);

    let input = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let reduced = sum_to_shape(&input, &[1, 2]);
    assert_dense_close(&reduced, &[4.0, 6.0], &[1, 2], 1e-6);
}

#[test]
fn expand_to_shape_reconstructs_reduction_shape() {
    let input = DenseTensor::from_vec(vec![3.0, 7.0], vec![2]);
    let expanded = expand_to_shape(&input, &[2, 2], &[1]);
    assert_dense_close(&expanded, &[3.0, 3.0, 7.0, 7.0], &[2, 2], 1e-6);

    let keepdim_input = DenseTensor::from_vec(vec![3.0, 7.0], vec![2, 1]);
    let keepdim_expanded = expand_to_shape(&keepdim_input, &[2, 2], &[]);
    assert_dense_close(&keepdim_expanded, &[3.0, 3.0, 7.0, 7.0], &[2, 2], 1e-6);
}

#[test]
fn max_axis_weights_split_ties_evenly() {
    let input = DenseTensor::from_vec(vec![2.0, 2.0, 1.0, 3.0], vec![2, 2]);
    let weights = max_axis_weights(&input, 1, false);
    assert_dense_close(&weights, &[0.5, 0.5, 0.0, 1.0], &[2, 2], 1e-6);
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
fn transpose_pullback_broadcast_add_reduces_to_operand_shapes() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        DenseTensor::from_vec(vec![10.0, 20.0], vec![2]),
    ];
    let output_cotangent = DenseTensor::from_vec(vec![2.0, -1.0, 0.5, 3.0], vec![2, 2]);

    let (_output, linearized) = linearize(|xs| xs[0].add(&xs[1]), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[2.0, -1.0, 0.5, 3.0], &[2, 2], 1e-6);
    assert_dense_close(&grads[1], &[2.5, 2.0], &[2], 1e-6);
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
fn transpose_pullback_sum_axis_matches_analytic_vjp() {
    let inputs = vec![DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])];
    let output_cotangent = DenseTensor::from_vec(vec![2.0, -1.5], vec![2]);

    let (_output, linearized) = linearize(|xs| xs[0].sum(1, false), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[2.0, 2.0, -1.5, -1.5], &[2, 2], 1e-6);
}

#[test]
fn transpose_pullback_relu_matches_analytic_vjp() {
    let inputs = vec![DenseTensor::from_vec(
        vec![-1.0, 2.0, 3.0, -4.0],
        vec![2, 2],
    )];
    let output_cotangent = DenseTensor::from_vec(vec![0.5, -1.0, 2.0, 1.5], vec![2, 2]);

    let (_output, linearized) = linearize(|xs| xs[0].relu(), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[0.0, -1.0, 2.0, 0.0], &[2, 2], 1e-6);
}

#[test]
fn transpose_pullback_matmul_matches_analytic_vjp() {
    let inputs = vec![
        DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        DenseTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
    ];
    let output_cotangent = DenseTensor::from_vec(vec![1.0, -0.5, 2.0, 3.0], vec![2, 2]);

    let (_output, linearized) = linearize(|xs| xs[0].matmul(&xs[1]), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[2.0, 3.0, 28.0, 38.0], &[2, 2], 1e-6);
    assert_dense_close(&grads[1], &[7.0, 8.5, 10.0, 11.0], &[2, 2], 1e-6);
}

#[test]
fn transpose_pullback_max_matches_split_tie_policy() {
    let inputs = vec![DenseTensor::from_vec(vec![2.0, 2.0, 1.0, 3.0], vec![2, 2])];
    let output_cotangent = DenseTensor::from_vec(vec![4.0, -2.0], vec![2]);

    let (_output, linearized) = linearize(|xs| xs[0].max(1, false), &inputs);
    let pullback = transpose_linearized(&linearized);
    let grads = pullback.apply_dense(&output_cotangent);

    assert_dense_close(&grads[0], &[2.0, 2.0, 0.0, -2.0], &[2, 2], 1e-6);
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
