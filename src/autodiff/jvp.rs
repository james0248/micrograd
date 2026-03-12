use std::cell::RefCell;

use crate::tensor::{
    DenseTensor, Tensor, TensorInner, TensorSpec, TracedTensor, ValueId, elementwise_binary,
    matmul, max_axis, max_axis_weights, mean_all, relu, sum_all, sum_axis, unary_map,
};

use super::Operation;
use super::interpreter::execute_trace;
use super::ir::Trace;
use super::trace::Recorder;

thread_local! {
    static JVP_STACK: RefCell<Vec<JvpRecorder>> = const { RefCell::new(Vec::new()) };
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct Linearized {
    pub(crate) trace: Trace,
    pub(crate) tangent_input_specs: Vec<TensorSpec>,
    pub(crate) residuals: Vec<DenseTensor>,
    pub(crate) output_spec: TensorSpec,
}

#[cfg_attr(not(test), allow(dead_code))]
impl Linearized {
    pub(crate) fn apply_dense(&self, tangents: &[DenseTensor]) -> DenseTensor {
        assert_eq!(
            tangents.len(),
            self.tangent_input_specs.len(),
            "linearized tangent input count mismatch: expected {}, got {}",
            self.tangent_input_specs.len(),
            tangents.len()
        );
        for (input, spec) in tangents.iter().zip(self.tangent_input_specs.iter()) {
            assert_eq!(
                input.shape, spec.shape,
                "linearized tangent input shape mismatch: expected {:?}, got {:?}",
                spec.shape, input.shape
            );
        }

        let mut inputs = tangents.to_vec();
        inputs.extend(self.residuals.iter().cloned());

        let output = execute_trace(&self.trace, &inputs)
            .into_iter()
            .next()
            .expect("linearized tangent trace must produce exactly one output");
        assert_eq!(
            output.shape, self.output_spec.shape,
            "linearized output tangent shape mismatch: expected {:?}, got {:?}",
            self.output_spec.shape, output.shape
        );
        output
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn linearize<F>(f: F, primals: &[DenseTensor]) -> (DenseTensor, Linearized)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let (recorder, jvp_inputs) = JvpRecorder::new(primals);
    let (recorder, output) = with_jvp_scope(recorder, || f(&jvp_inputs));
    recorder.into_linearized(output)
}

pub(crate) fn jvp_binary(lhs: &Tensor, rhs: &Tensor, op: Operation) -> Option<Tensor> {
    if lhs.as_jvp().is_none() && rhs.as_jvp().is_none() {
        return None;
    }

    let output = with_jvp_recorder(|recorder| {
        let lhs = jvp_operand(recorder, lhs);
        let rhs = jvp_operand(recorder, rhs);
        let primal = eager_binary(&op, &lhs.primal, &rhs.primal);
        let spec = primal.spec();
        let tangent = match op {
            Operation::Add => {
                recorder
                    .add_instruction(
                        Operation::Add,
                        vec![lhs.tangent.var, rhs.tangent.var],
                        spec.clone(),
                    )
                    .var
            }
            Operation::Sub => {
                recorder
                    .add_instruction(
                        Operation::Sub,
                        vec![lhs.tangent.var, rhs.tangent.var],
                        spec.clone(),
                    )
                    .var
            }
            Operation::Mul => {
                let lhs_residual = recorder.add_residual(lhs.primal.clone()).var;
                let rhs_residual = recorder.add_residual(rhs.primal.clone()).var;
                let lhs_term = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![lhs.tangent.var, rhs_residual],
                        spec.clone(),
                    )
                    .var;
                let rhs_term = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![lhs_residual, rhs.tangent.var],
                        spec.clone(),
                    )
                    .var;
                recorder
                    .add_instruction(Operation::Add, vec![lhs_term, rhs_term], spec.clone())
                    .var
            }
            Operation::Div => {
                let lhs_coeff = elementwise_binary(&lhs.primal, &rhs.primal, |_x, y| 1.0 / y);
                let rhs_coeff = elementwise_binary(&lhs.primal, &rhs.primal, |x, y| -x / (y * y));
                let lhs_residual = recorder.add_residual(lhs_coeff).var;
                let rhs_residual = recorder.add_residual(rhs_coeff).var;
                let lhs_term = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![lhs.tangent.var, lhs_residual],
                        spec.clone(),
                    )
                    .var;
                let rhs_term = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![rhs.tangent.var, rhs_residual],
                        spec.clone(),
                    )
                    .var;
                recorder
                    .add_instruction(Operation::Add, vec![lhs_term, rhs_term], spec.clone())
                    .var
            }
            Operation::MatMul => {
                let lhs_residual = recorder.add_residual(lhs.primal.clone()).var;
                let rhs_residual = recorder.add_residual(rhs.primal.clone()).var;
                let lhs_term = recorder
                    .add_instruction(
                        Operation::MatMul,
                        vec![lhs.tangent.var, rhs_residual],
                        spec.clone(),
                    )
                    .var;
                let rhs_term = recorder
                    .add_instruction(
                        Operation::MatMul,
                        vec![lhs_residual, rhs.tangent.var],
                        spec.clone(),
                    )
                    .var;
                recorder
                    .add_instruction(Operation::Add, vec![lhs_term, rhs_term], spec.clone())
                    .var
            }
            _ => panic!("jvp_binary does not support operation {:?}", op),
        };
        Tensor::from_jvp(primal, TracedTensor { var: tangent, spec })
    })
    .expect("JVP tensors require an active linearize scope");

    Some(output)
}

pub(crate) fn jvp_unary(input: &Tensor, op: Operation) -> Option<Tensor> {
    let Some(jvp) = input.as_jvp() else {
        return None;
    };

    let output = with_jvp_recorder(|recorder| {
        let primal = eager_unary(&op, &jvp.primal);
        let spec = primal.spec();
        let tangent = match op {
            Operation::Exp => {
                let primal_residual = recorder.add_residual(primal.clone()).var;
                recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![jvp.tangent.var, primal_residual],
                        spec.clone(),
                    )
                    .var
            }
            Operation::Log => {
                let reciprocal = recorder
                    .add_residual(unary_map(&jvp.primal, |x| 1.0 / x))
                    .var;
                recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![jvp.tangent.var, reciprocal],
                        spec.clone(),
                    )
                    .var
            }
            Operation::Sum { axis, keepdim } => {
                recorder
                    .add_instruction(
                        Operation::Sum { axis, keepdim },
                        vec![jvp.tangent.var],
                        spec.clone(),
                    )
                    .var
            }
            Operation::Relu => {
                let mask = recorder
                    .add_residual(unary_map(&jvp.primal, |x| if x > 0.0 { 1.0 } else { 0.0 }))
                    .var;
                recorder
                    .add_instruction(Operation::Mul, vec![jvp.tangent.var, mask], spec.clone())
                    .var
            }
            Operation::Max { axis, keepdim } => {
                let weights = recorder
                    .add_residual(max_axis_weights(&jvp.primal, axis, keepdim))
                    .var;
                let weighted = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![jvp.tangent.var, weights],
                        jvp.primal.spec(),
                    )
                    .var;
                recorder
                    .add_instruction(
                        Operation::Sum { axis, keepdim },
                        vec![weighted],
                        spec.clone(),
                    )
                    .var
            }
            Operation::SumAll => {
                recorder
                    .add_instruction(Operation::SumAll, vec![jvp.tangent.var], spec.clone())
                    .var
            }
            Operation::MeanAll => {
                recorder
                    .add_instruction(Operation::MeanAll, vec![jvp.tangent.var], spec.clone())
                    .var
            }
            Operation::Transpose { dim0, dim1 } => {
                recorder
                    .add_instruction(
                        Operation::Transpose { dim0, dim1 },
                        vec![jvp.tangent.var],
                        spec.clone(),
                    )
                    .var
            }
            _ => panic!("jvp_unary does not support operation {:?}", op),
        };
        Tensor::from_jvp(primal, TracedTensor { var: tangent, spec })
    })
    .expect("JVP tensors require an active linearize scope");

    Some(output)
}

#[derive(Debug, Clone)]
struct JvpOperand {
    primal: DenseTensor,
    tangent: TracedTensor,
}

#[derive(Debug, Clone)]
struct JvpRecorder {
    recorder: Recorder,
    tangent_input_specs: Vec<TensorSpec>,
    residuals: Vec<DenseTensor>,
}

impl JvpRecorder {
    fn new(primals: &[DenseTensor]) -> (Self, Vec<Tensor>) {
        let mut recorder = Recorder::new_empty();
        let mut jvp_inputs = Vec::with_capacity(primals.len());
        let mut tangent_input_specs = Vec::with_capacity(primals.len());

        for primal in primals {
            let spec = primal.spec();
            let tangent = recorder.add_input(spec.clone());
            tangent_input_specs.push(spec.clone());
            jvp_inputs.push(Tensor::from_jvp(primal.clone(), tangent));
        }

        (
            Self {
                recorder,
                tangent_input_specs,
                residuals: Vec::new(),
            },
            jvp_inputs,
        )
    }

    fn add_residual(&mut self, tensor: DenseTensor) -> TracedTensor {
        let spec = tensor.spec();
        let residual = self.recorder.add_input(spec.clone());
        self.residuals.push(tensor);
        residual
    }

    fn add_const_tensor(&mut self, tensor: DenseTensor) -> ValueId {
        self.recorder.add_const_tensor(tensor)
    }

    fn add_instruction(
        &mut self,
        op: Operation,
        inputs: Vec<ValueId>,
        spec: TensorSpec,
    ) -> TracedTensor {
        self.recorder.add_instruction(op, inputs, spec)
    }

    fn into_linearized(mut self, output: Tensor) -> (DenseTensor, Linearized) {
        let (output_primal, output_tangent, output_spec) = match output.inner {
            TensorInner::Jvp(jvp) => {
                let output_spec = jvp.primal.spec();
                (jvp.primal, jvp.tangent.var, output_spec)
            }
            TensorInner::Concrete(primal) => {
                let output_spec = primal.spec();
                let zero = self
                    .recorder
                    .add_const_tensor(DenseTensor::zeros(output_spec.shape.clone()));
                (primal, zero, output_spec)
            }
        };

        let linearized = Linearized {
            trace: self.recorder.into_trace(vec![output_tangent]),
            tangent_input_specs: self.tangent_input_specs,
            residuals: self.residuals,
            output_spec: output_spec.clone(),
        };
        (output_primal, linearized)
    }
}

fn jvp_operand(recorder: &mut JvpRecorder, tensor: &Tensor) -> JvpOperand {
    match &tensor.inner {
        TensorInner::Concrete(primal) => {
            let spec = primal.spec();
            let zero = recorder.add_const_tensor(DenseTensor::zeros(spec.shape.clone()));
            JvpOperand {
                primal: primal.clone(),
                tangent: TracedTensor { var: zero, spec },
            }
        }
        TensorInner::Jvp(jvp) => JvpOperand {
            primal: jvp.primal.clone(),
            tangent: jvp.tangent.clone(),
        },
    }
}

fn eager_binary(op: &Operation, lhs: &DenseTensor, rhs: &DenseTensor) -> DenseTensor {
    match op {
        Operation::Add => elementwise_binary(lhs, rhs, |x, y| x + y),
        Operation::Sub => elementwise_binary(lhs, rhs, |x, y| x - y),
        Operation::Mul => elementwise_binary(lhs, rhs, |x, y| x * y),
        Operation::Div => elementwise_binary(lhs, rhs, |x, y| x / y),
        Operation::MatMul => matmul(lhs, rhs),
        _ => panic!("eager_binary does not support operation {:?}", op),
    }
}

fn eager_unary(op: &Operation, input: &DenseTensor) -> DenseTensor {
    match op {
        Operation::Exp => unary_map(input, |x| x.exp()),
        Operation::Log => unary_map(input, |x| x.ln()),
        Operation::Sum { axis, keepdim } => sum_axis(input, *axis, *keepdim),
        Operation::Max { axis, keepdim } => max_axis(input, *axis, *keepdim),
        Operation::Relu => relu(input),
        Operation::SumAll => sum_all(input),
        Operation::MeanAll => mean_all(input),
        Operation::Transpose { dim0, dim1 } => input.transpose(*dim0, *dim1),
        _ => panic!("eager_unary does not support operation {:?}", op),
    }
}

fn with_jvp_recorder<R>(f: impl FnOnce(&mut JvpRecorder) -> R) -> Option<R> {
    JVP_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let recorder = stack.last_mut()?;
        Some(f(recorder))
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn push_jvp_recorder(recorder: JvpRecorder) {
    JVP_STACK.with(|stack| stack.borrow_mut().push(recorder));
}

#[cfg_attr(not(test), allow(dead_code))]
fn pop_jvp_recorder() -> JvpRecorder {
    JVP_STACK.with(|stack| {
        stack
            .borrow_mut()
            .pop()
            .expect("JVP recorder stack underflow")
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn clear_jvp_recorder_on_unwind() {
    JVP_STACK.with(|stack| {
        let _ = stack.borrow_mut().pop();
    });
}

#[cfg_attr(not(test), allow(dead_code))]
fn with_jvp_scope<F, T>(recorder: JvpRecorder, f: F) -> (JvpRecorder, T)
where
    F: FnOnce() -> T,
{
    struct PopOnDrop(bool);

    impl Drop for PopOnDrop {
        fn drop(&mut self) {
            if self.0 {
                clear_jvp_recorder_on_unwind();
            }
        }
    }

    push_jvp_recorder(recorder);
    let mut guard = PopOnDrop(true);
    let out = f();
    guard.0 = false;
    let recorder = pop_jvp_recorder();
    (recorder, out)
}
