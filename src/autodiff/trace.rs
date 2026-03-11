use std::cell::RefCell;

use crate::tensor::{DenseTensor, Tensor, TensorSpec, TracedTensor, ValueId};

use super::ir::{ConstBinding, InputBinding, Instruction, Operation, Trace};

thread_local! {
    static TRACE_STACK: RefCell<Vec<Recorder>> = const { RefCell::new(Vec::new()) };
}

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

    pub(crate) fn from_trace(trace: &Trace) -> Self {
        Self {
            inputs: trace.inputs.clone(),
            consts: trace.consts.clone(),
            instructions: trace.instructions.clone(),
            next_var: trace.next_var,
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

    fn new(inputs: &[DenseTensor]) -> (Self, Vec<Tensor>) {
        let mut recorder = Self::new_empty();
        recorder.inputs.reserve(inputs.len());
        let mut traced_inputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let spec = input.spec();
            let traced = recorder.add_input(spec);
            traced_inputs.push(Tensor::from_traced(traced.var, traced.spec));
        }
        (recorder, traced_inputs)
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

    pub(crate) fn lift_tensor(&mut self, tensor: &Tensor) -> TracedTensor {
        if let Some(traced) = tensor.as_traced() {
            return traced.clone();
        }
        let concrete = tensor.expect_concrete("trace-lift").clone();
        let var = self.alloc_value();
        let spec = concrete.spec();
        self.consts.push(ConstBinding {
            var,
            tensor: concrete,
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

    fn finalize(mut self, output: Tensor) -> Trace {
        let output = self.lift_tensor(&output);
        Trace {
            inputs: self.inputs,
            consts: self.consts,
            instructions: self.instructions,
            outputs: vec![output.var],
            next_var: self.next_var,
        }
    }
}

pub(crate) fn trace_binary(lhs: &Tensor, rhs: &Tensor, op: Operation) -> Option<Tensor> {
    with_recorder(|recorder| {
        let lhs = recorder.lift_tensor(lhs);
        let rhs = recorder.lift_tensor(rhs);
        let spec = abstract_eval_binary(&op, &lhs.spec, &rhs.spec);
        Tensor::from_traced(
            recorder
                .add_instruction(op, vec![lhs.var, rhs.var], spec.clone())
                .var,
            spec,
        )
    })
}

pub(crate) fn trace_unary(input: &Tensor, op: Operation) -> Option<Tensor> {
    with_recorder(|recorder| {
        let input = recorder.lift_tensor(input);
        let spec = abstract_eval_unary(&op, &input.spec);
        Tensor::from_traced(
            recorder
                .add_instruction(op, vec![input.var], spec.clone())
                .var,
            spec,
        )
    })
}

pub(crate) fn record_trace<F>(f: F, inputs: &[DenseTensor]) -> Trace
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let (recorder, traced_inputs) = Recorder::new(inputs);
    let (recorder, output) = with_recorder_scope(recorder, || f(&traced_inputs));
    let trace = recorder.finalize(output);
    assert_eq!(
        trace.outputs.len(),
        1,
        "autodiff trace must produce exactly one output"
    );
    trace
}

fn with_recorder<R>(f: impl FnOnce(&mut Recorder) -> R) -> Option<R> {
    TRACE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let recorder = stack.last_mut()?;
        Some(f(recorder))
    })
}

fn push_recorder(recorder: Recorder) {
    TRACE_STACK.with(|stack| stack.borrow_mut().push(recorder));
}

fn pop_recorder() -> Recorder {
    TRACE_STACK.with(|stack| stack.borrow_mut().pop().expect("recorder stack underflow"))
}

fn clear_recorder_on_unwind() {
    TRACE_STACK.with(|stack| {
        let _ = stack.borrow_mut().pop();
    });
}

fn with_recorder_scope<F, T>(recorder: Recorder, f: F) -> (Recorder, T)
where
    F: FnOnce() -> T,
{
    struct PopOnDrop(bool);
    impl Drop for PopOnDrop {
        fn drop(&mut self) {
            if self.0 {
                clear_recorder_on_unwind();
            }
        }
    }

    push_recorder(recorder);
    let mut guard = PopOnDrop(true);
    let out = f();
    guard.0 = false;
    let recorder = pop_recorder();
    (recorder, out)
}

fn abstract_eval_binary(op: &Operation, lhs: &TensorSpec, rhs: &TensorSpec) -> TensorSpec {
    assert!(
        matches!(
            op,
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div
        ),
        "abstract_eval_binary only supports binary elementwise operations"
    );
    assert_eq!(
        lhs.shape, rhs.shape,
        "same-shape elementwise op {:?} requires equal shapes, got {:?} and {:?}",
        op, lhs.shape, rhs.shape
    );
    TensorSpec::new(lhs.shape.clone())
}

fn abstract_eval_unary(op: &Operation, input: &TensorSpec) -> TensorSpec {
    match op {
        Operation::Exp | Operation::Log => TensorSpec::new(input.shape.clone()),
        Operation::SumAll | Operation::MeanAll => TensorSpec::new(vec![1]),
        _ => panic!("abstract_eval_unary does not support operation {:?}", op),
    }
}
