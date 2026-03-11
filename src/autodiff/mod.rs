mod interpreter;
mod ir;
mod jvp;
mod trace;
mod vjp;

#[cfg(test)]
mod tests;

use crate::tensor::Tensor;

use interpreter::concrete_inputs;
pub(crate) use interpreter::execute_trace;
pub(crate) use ir::Operation;
pub(crate) use jvp::{jvp_binary, jvp_unary};
pub(crate) use trace::{record_trace, trace_binary, trace_unary};
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
