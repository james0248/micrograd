mod interpreter;
mod ir;
mod jvp;
mod trace;
mod transpose;
mod vjp;

#[cfg(test)]
mod tests;

use crate::tensor::{DenseTensor, Tensor};

use interpreter::concrete_inputs;
pub(crate) use interpreter::execute_trace;
pub(crate) use ir::Operation;
use jvp::linearize;
pub(crate) use jvp::{jvp_binary, jvp_unary};
pub(crate) use trace::{record_trace, trace_binary, trace_unary};
use transpose::transpose_linearized;
use vjp::build_vjp;

pub fn grad<F>(f: F, inputs: &[Tensor]) -> Vec<Tensor>
where
    F: Fn(&[Tensor]) -> Tensor,
{
    value_and_grad(f, inputs).1
}

pub fn value_and_grad<F>(f: F, inputs: &[Tensor]) -> (Tensor, Vec<Tensor>)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let result = value_and_grad_transposed(&f, inputs);

    #[cfg(debug_assertions)]
    {
        let baseline = value_and_grad_direct(&f, inputs);
        assert_value_and_grad_close(&result, &baseline, 1e-6);
    }

    result
}

fn value_and_grad_transposed<F>(f: &F, inputs: &[Tensor]) -> (Tensor, Vec<Tensor>)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let concrete_inputs = concrete_inputs(inputs);
    let (output, linearized) = linearize(f, &concrete_inputs);
    assert_scalar_output(&output);
    let pullback = transpose_linearized(&linearized);
    let seed = DenseTensor::filled(output.shape.clone(), 1.0);
    let grads = pullback
        .apply_dense(&seed)
        .into_iter()
        .map(Tensor::from_concrete)
        .collect();
    (Tensor::from_concrete(output), grads)
}

fn value_and_grad_direct<F>(f: &F, inputs: &[Tensor]) -> (Tensor, Vec<Tensor>)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let concrete_inputs = concrete_inputs(inputs);
    let forward_trace = record_trace(f, &concrete_inputs);
    let output = execute_trace(&forward_trace, &concrete_inputs)
        .into_iter()
        .next()
        .expect("forward trace must produce exactly one output");
    let backward_trace = build_vjp(&forward_trace);
    let grads = execute_trace(&backward_trace, &concrete_inputs)
        .into_iter()
        .map(Tensor::from_concrete)
        .collect();
    (Tensor::from_concrete(output), grads)
}

fn assert_scalar_output(output: &DenseTensor) {
    assert!(
        output.spec().numel() == 1,
        "autodiff::grad/value_and_grad require a scalar output tensor, got shape {:?}",
        output.shape
    );
}

#[cfg(debug_assertions)]
fn assert_value_and_grad_close(
    actual: &(Tensor, Vec<Tensor>),
    expected: &(Tensor, Vec<Tensor>),
    eps: f32,
) {
    assert_tensor_close(&actual.0, &expected.0, eps, "value");
    assert_eq!(
        actual.1.len(),
        expected.1.len(),
        "gradient count mismatch: expected {}, got {}",
        expected.1.len(),
        actual.1.len()
    );
    for (index, (actual_grad, expected_grad)) in actual.1.iter().zip(expected.1.iter()).enumerate()
    {
        assert_tensor_close(actual_grad, expected_grad, eps, &format!("grad[{index}]"));
    }
}

#[cfg(debug_assertions)]
fn assert_tensor_close(actual: &Tensor, expected: &Tensor, eps: f32, label: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label} shape mismatch: expected {:?}, got {:?}",
        expected.shape(),
        actual.shape()
    );
    assert_eq!(
        actual.data().len(),
        expected.data().len(),
        "{label} data length mismatch: expected {}, got {}",
        expected.data().len(),
        actual.data().len()
    );
    for (&actual, &expected) in actual.data().iter().zip(expected.data().iter()) {
        assert!(
            (actual - expected).abs() <= eps,
            "{label} mismatch: expected {expected:.8}, got {actual:.8} (eps={eps})"
        );
    }
}
