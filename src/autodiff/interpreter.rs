use crate::tensor::{
    DenseTensor, Tensor, TensorInner, elementwise_binary, expand_to_shape, matmul, max_axis,
    mean_all, relu, sum_all, sum_axis, sum_to_shape, unary_map,
};

use super::ir::{Operation, Trace};

pub(crate) fn execute_trace(trace: &Trace, inputs: &[DenseTensor]) -> Vec<DenseTensor> {
    assert_eq!(
        inputs.len(),
        trace.inputs.len(),
        "trace input count mismatch: expected {}, got {}",
        trace.inputs.len(),
        inputs.len()
    );

    let mut values: Vec<Option<DenseTensor>> = vec![None; trace.next_var];

    for (binding, input) in trace.inputs.iter().zip(inputs.iter()) {
        assert_eq!(
            input.shape, binding.spec.shape,
            "trace input shape mismatch: expected {:?}, got {:?}",
            binding.spec.shape, input.shape
        );
        values[binding.var] = Some(input.clone());
    }

    for constant in &trace.consts {
        values[constant.var] = Some(constant.tensor.clone());
    }

    for instruction in &trace.instructions {
        let out = match &instruction.op {
            Operation::Add => {
                let lhs = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("add lhs value must exist");
                let rhs = values[instruction.inputs[1]]
                    .as_ref()
                    .expect("add rhs value must exist");
                elementwise_binary(lhs, rhs, |x, y| x + y)
            }
            Operation::Sub => {
                let lhs = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("sub lhs value must exist");
                let rhs = values[instruction.inputs[1]]
                    .as_ref()
                    .expect("sub rhs value must exist");
                elementwise_binary(lhs, rhs, |x, y| x - y)
            }
            Operation::Mul => {
                let lhs = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("mul lhs value must exist");
                let rhs = values[instruction.inputs[1]]
                    .as_ref()
                    .expect("mul rhs value must exist");
                elementwise_binary(lhs, rhs, |x, y| x * y)
            }
            Operation::Div => {
                let lhs = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("div lhs value must exist");
                let rhs = values[instruction.inputs[1]]
                    .as_ref()
                    .expect("div rhs value must exist");
                elementwise_binary(lhs, rhs, |x, y| x / y)
            }
            Operation::Exp => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("exp input value must exist");
                unary_map(input, |x| x.exp())
            }
            Operation::Log => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("log input value must exist");
                unary_map(input, |x| x.ln())
            }
            Operation::Sum { axis, keepdim } => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("sum input value must exist");
                sum_axis(input, *axis, *keepdim)
            }
            Operation::Max { axis, keepdim } => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("max input value must exist");
                max_axis(input, *axis, *keepdim)
            }
            Operation::Relu => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("relu input value must exist");
                relu(input)
            }
            Operation::MatMul => {
                let lhs = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("matmul lhs value must exist");
                let rhs = values[instruction.inputs[1]]
                    .as_ref()
                    .expect("matmul rhs value must exist");
                matmul(lhs, rhs)
            }
            Operation::SumAll => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("sum_all input value must exist");
                sum_all(input)
            }
            Operation::MeanAll => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("mean_all input value must exist");
                mean_all(input)
            }
            Operation::Transpose { dim0, dim1 } => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("transpose input value must exist");
                input.transpose(*dim0, *dim1)
            }
            Operation::SumToShape { shape } => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("sum_to_shape input value must exist");
                sum_to_shape(input, shape)
            }
            Operation::ExpandToShape {
                shape,
                inserted_axes,
            } => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("expand_to_shape input value must exist");
                expand_to_shape(input, shape, inserted_axes)
            }
            Operation::ExpandScalar { shape } => {
                let input = values[instruction.inputs[0]]
                    .as_ref()
                    .expect("expand_scalar input value must exist");
                assert_eq!(
                    input.shape,
                    [1],
                    "ExpandScalar requires a scalar input, got shape {:?}",
                    input.shape
                );
                DenseTensor::filled(shape.clone(), input.value_at(&[0]))
            }
        };

        assert_eq!(
            out.shape, instruction.spec.shape,
            "interpreter shape mismatch for {:?}: expected {:?}, got {:?}",
            instruction.op, instruction.spec.shape, out.shape
        );
        values[instruction.out] = Some(out);
    }

    trace
        .outputs
        .iter()
        .map(|&var| values[var].clone().expect("trace output value must exist"))
        .collect()
}

pub(crate) fn concrete_inputs(inputs: &[Tensor]) -> Vec<DenseTensor> {
    inputs
        .iter()
        .map(|tensor| match &tensor.inner {
            TensorInner::Concrete(value) => value.clone(),
            TensorInner::Jvp(_) => {
                panic!("autodiff inputs must be concrete tensors")
            }
        })
        .collect()
}
