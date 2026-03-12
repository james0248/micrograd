use crate::tensor::{DenseTensor, TensorSpec, TracedTensor, ValueId};

use super::ir::{ConstBinding, InputBinding, Instruction, Operation, Trace};

#[derive(Debug, Clone)]
pub(crate) struct Recorder {
    pub(crate) inputs: Vec<InputBinding>,
    pub(crate) consts: Vec<ConstBinding>,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) next_var: usize,
}

impl Recorder {
    pub(crate) fn new_empty() -> Self {
        Self {
            inputs: Vec::new(),
            consts: Vec::new(),
            instructions: Vec::new(),
            next_var: 0,
        }
    }

    pub(crate) fn into_trace(self, outputs: Vec<ValueId>) -> Trace {
        Trace {
            inputs: self.inputs,
            consts: self.consts,
            instructions: self.instructions,
            outputs,
            next_var: self.next_var,
        }
    }

    fn alloc_value(&mut self) -> ValueId {
        let var = self.next_var;
        self.next_var += 1;
        var
    }

    pub(crate) fn add_input(&mut self, spec: TensorSpec) -> TracedTensor {
        let var = self.alloc_value();
        self.inputs.push(InputBinding {
            var,
            spec: spec.clone(),
        });
        TracedTensor { var, spec }
    }

    pub(crate) fn add_const_tensor(&mut self, tensor: DenseTensor) -> ValueId {
        let var = self.alloc_value();
        self.consts.push(ConstBinding { var, tensor });
        var
    }

    pub(crate) fn add_instruction(
        &mut self,
        op: Operation,
        inputs: Vec<ValueId>,
        spec: TensorSpec,
    ) -> TracedTensor {
        let out = self.alloc_value();
        self.instructions.push(Instruction {
            out,
            op,
            inputs,
            spec: spec.clone(),
        });
        TracedTensor { var: out, spec }
    }
}

pub(super) fn accumulate_cotangent(
    recorder: &mut Recorder,
    cotangents: &mut [Option<ValueId>],
    target: ValueId,
    contrib: ValueId,
    spec: &TensorSpec,
) {
    if let Some(existing) = cotangents[target] {
        let sum = recorder
            .add_instruction(Operation::Add, vec![existing, contrib], spec.clone())
            .var;
        cotangents[target] = Some(sum);
    } else {
        cotangents[target] = Some(contrib);
    }
}

pub(super) fn negate_value(recorder: &mut Recorder, input: ValueId, spec: &TensorSpec) -> ValueId {
    let minus_one = recorder.add_const_tensor(DenseTensor::filled(spec.shape.clone(), -1.0));
    recorder
        .add_instruction(Operation::Mul, vec![input, minus_one], spec.clone())
        .var
}
