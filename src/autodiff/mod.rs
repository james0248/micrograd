mod interpreter;
mod ir;
mod jvp;
mod trace;
mod transpose;

#[cfg(test)]
mod tests;

use crate::tensor::{DenseTensor, Tensor};

use interpreter::concrete_inputs;
pub(crate) use ir::Operation;
use jvp::linearize;
pub(crate) use jvp::{jvp_binary, jvp_unary};
use transpose::transpose_linearized;

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
    let concrete_inputs = concrete_inputs(inputs);
    let (output, linearized) = linearize(&f, &concrete_inputs);
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

fn assert_scalar_output(output: &DenseTensor) {
    assert!(
        output.spec().numel() == 1,
        "autodiff::grad/value_and_grad require a scalar output tensor, got shape {:?}",
        output.shape
    );
}
