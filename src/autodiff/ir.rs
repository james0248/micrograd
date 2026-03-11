use crate::tensor::{DenseTensor, TensorSpec, ValueId};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Log,
    SumAll,
    MeanAll,
    ExpandScalar { shape: Vec<usize> },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InputBinding {
    pub(crate) var: ValueId,
    pub(crate) spec: TensorSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConstBinding {
    pub(crate) var: ValueId,
    pub(crate) tensor: DenseTensor,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Instruction {
    pub(crate) out: ValueId,
    pub(crate) op: Operation,
    pub(crate) inputs: Vec<ValueId>,
    pub(crate) spec: TensorSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Trace {
    pub(crate) inputs: Vec<InputBinding>,
    pub(crate) consts: Vec<ConstBinding>,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) outputs: Vec<ValueId>,
    pub(crate) next_var: usize,
}

pub(crate) fn build_spec_table(trace: &Trace) -> Vec<Option<TensorSpec>> {
    let mut specs = vec![None; trace.next_var];
    for input in &trace.inputs {
        specs[input.var] = Some(input.spec.clone());
    }
    for constant in &trace.consts {
        specs[constant.var] = Some(constant.tensor.spec());
    }
    for instruction in &trace.instructions {
        specs[instruction.out] = Some(instruction.spec.clone());
    }
    specs
}
