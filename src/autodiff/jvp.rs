use std::cell::RefCell;

use crate::tensor::{
    DenseTensor, Tensor, TensorInner, TensorSpec, TracedTensor, elementwise_binary, unary_map,
};

use super::Operation;
use super::interpreter::execute_trace;
use super::ir::Trace;
use super::trace::Recorder;

thread_local! {
    static JVP_STACK: RefCell<Vec<Recorder>> = const { RefCell::new(Vec::new()) };
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct Linearized {
    tangent_trace: Trace,
    input_specs: Vec<TensorSpec>,
    output_spec: TensorSpec,
}

#[cfg_attr(not(test), allow(dead_code))]
impl Linearized {
    pub(crate) fn apply_dense(&self, tangents: &[DenseTensor]) -> DenseTensor {
        assert_eq!(
            tangents.len(),
            self.input_specs.len(),
            "linearized tangent input count mismatch: expected {}, got {}",
            self.input_specs.len(),
            tangents.len()
        );
        for (input, spec) in tangents.iter().zip(self.input_specs.iter()) {
            assert_eq!(
                input.shape, spec.shape,
                "linearized tangent input shape mismatch: expected {:?}, got {:?}",
                spec.shape, input.shape
            );
        }

        let output = execute_trace(&self.tangent_trace, tangents)
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
    let mut recorder = Recorder::new_empty();
    let mut jvp_inputs = Vec::with_capacity(primals.len());
    let mut input_specs = Vec::with_capacity(primals.len());

    for primal in primals {
        let spec = primal.spec();
        let tangent = recorder.add_input(spec.clone());
        input_specs.push(spec.clone());
        jvp_inputs.push(Tensor::from_jvp(primal.clone(), tangent));
    }

    let (mut recorder, output) = with_jvp_scope(recorder, || f(&jvp_inputs));
    let (output_primal, output_tangent, output_spec) = match output.inner {
        TensorInner::Jvp(jvp) => {
            let output_spec = jvp.primal.spec();
            (jvp.primal, jvp.tangent.var, output_spec)
        }
        TensorInner::Concrete(primal) => {
            let output_spec = primal.spec();
            let zero = recorder.add_const_tensor(DenseTensor::zeros(output_spec.shape.clone()));
            (primal, zero, output_spec)
        }
        TensorInner::Traced(_) => panic!("linearize output must not be a regular traced tensor"),
    };

    let linearized = Linearized {
        tangent_trace: recorder.into_trace(vec![output_tangent]),
        input_specs,
        output_spec: output_spec.clone(),
    };
    (output_primal, linearized)
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
                let lhs_const = recorder.add_const_tensor(lhs.primal.clone());
                let rhs_const = recorder.add_const_tensor(rhs.primal.clone());
                let lhs_term = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![lhs.tangent.var, rhs_const],
                        spec.clone(),
                    )
                    .var;
                let rhs_term = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![lhs_const, rhs.tangent.var],
                        spec.clone(),
                    )
                    .var;
                recorder
                    .add_instruction(Operation::Add, vec![lhs_term, rhs_term], spec.clone())
                    .var
            }
            Operation::Div => {
                let lhs_const = recorder.add_const_tensor(lhs.primal.clone());
                let rhs_const = recorder.add_const_tensor(rhs.primal.clone());
                let lhs_term = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![lhs.tangent.var, rhs_const],
                        spec.clone(),
                    )
                    .var;
                let rhs_term = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![lhs_const, rhs.tangent.var],
                        spec.clone(),
                    )
                    .var;
                let numerator = recorder
                    .add_instruction(Operation::Sub, vec![lhs_term, rhs_term], spec.clone())
                    .var;
                let denominator = recorder
                    .add_instruction(Operation::Mul, vec![rhs_const, rhs_const], spec.clone())
                    .var;
                recorder
                    .add_instruction(Operation::Div, vec![numerator, denominator], spec.clone())
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
                let primal_const = recorder.add_const_tensor(primal.clone());
                recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![primal_const, jvp.tangent.var],
                        spec.clone(),
                    )
                    .var
            }
            Operation::Log => {
                let input_const = recorder.add_const_tensor(jvp.primal.clone());
                recorder
                    .add_instruction(
                        Operation::Div,
                        vec![jvp.tangent.var, input_const],
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

fn jvp_operand(recorder: &mut Recorder, tensor: &Tensor) -> JvpOperand {
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
        TensorInner::Traced(_) => {
            panic!("JVP tracing does not support regular traced tensors as operands")
        }
    }
}

fn eager_binary(op: &Operation, lhs: &DenseTensor, rhs: &DenseTensor) -> DenseTensor {
    match op {
        Operation::Add => elementwise_binary(lhs, rhs, "add", |x, y| x + y),
        Operation::Sub => elementwise_binary(lhs, rhs, "sub", |x, y| x - y),
        Operation::Mul => elementwise_binary(lhs, rhs, "mul", |x, y| x * y),
        Operation::Div => elementwise_binary(lhs, rhs, "div", |x, y| x / y),
        _ => panic!("eager_binary does not support operation {:?}", op),
    }
}

fn eager_unary(op: &Operation, input: &DenseTensor) -> DenseTensor {
    match op {
        Operation::Exp => unary_map(input, |x| x.exp()),
        Operation::Log => unary_map(input, |x| x.ln()),
        Operation::SumAll => DenseTensor::from_vec(vec![input.data.iter().copied().sum()], vec![1]),
        Operation::MeanAll => {
            let sum: f32 = input.data.iter().copied().sum();
            DenseTensor::from_vec(vec![sum / input.data.len() as f32], vec![1])
        }
        _ => panic!("eager_unary does not support operation {:?}", op),
    }
}

fn with_jvp_recorder<R>(f: impl FnOnce(&mut Recorder) -> R) -> Option<R> {
    JVP_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let recorder = stack.last_mut()?;
        Some(f(recorder))
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn push_jvp_recorder(recorder: Recorder) {
    JVP_STACK.with(|stack| stack.borrow_mut().push(recorder));
}

#[cfg_attr(not(test), allow(dead_code))]
fn pop_jvp_recorder() -> Recorder {
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
fn with_jvp_scope<F, T>(recorder: Recorder, f: F) -> (Recorder, T)
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
