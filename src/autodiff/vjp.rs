use std::collections::HashMap;

use crate::tensor::{DenseTensor, TensorSpec, ValueId};

use super::ir::{Operation, Trace, build_spec_table};
use super::trace::Recorder;

pub(crate) fn build_vjp(forward: &Trace) -> Trace {
    let specs = build_spec_table(forward);
    let output = *forward
        .outputs
        .first()
        .expect("forward trace must have one output");
    let output_spec = specs[output]
        .as_ref()
        .expect("forward output must have a tensor spec");
    assert!(
        output_spec.numel() == 1,
        "autodiff::grad/value_and_grad require a scalar output tensor, got shape {:?}",
        output_spec.shape
    );

    let mut recorder = Recorder::from_trace(forward);
    let mut cotangents: HashMap<ValueId, ValueId> = HashMap::new();
    let seed = recorder.add_const_tensor(DenseTensor::filled(output_spec.shape.clone(), 1.0));
    cotangents.insert(output, seed);

    for instruction in forward.instructions.iter().rev() {
        let Some(out_ct) = cotangents.get(&instruction.out).copied() else {
            continue;
        };

        match &instruction.op {
            Operation::Add => {
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[0],
                    out_ct,
                    specs[instruction.inputs[0]]
                        .as_ref()
                        .expect("add lhs spec must exist"),
                );
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[1],
                    out_ct,
                    specs[instruction.inputs[1]]
                        .as_ref()
                        .expect("add rhs spec must exist"),
                );
            }
            Operation::Sub => {
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[0],
                    out_ct,
                    specs[instruction.inputs[0]]
                        .as_ref()
                        .expect("sub lhs spec must exist"),
                );
                let rhs_spec = specs[instruction.inputs[1]]
                    .as_ref()
                    .expect("sub rhs spec must exist");
                let neg = negate_value(&mut recorder, out_ct, rhs_spec);
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[1],
                    neg,
                    rhs_spec,
                );
            }
            Operation::Mul => {
                let lhs_spec = specs[instruction.inputs[0]]
                    .as_ref()
                    .expect("mul lhs spec must exist");
                let rhs_spec = specs[instruction.inputs[1]]
                    .as_ref()
                    .expect("mul rhs spec must exist");
                let d_lhs = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![out_ct, instruction.inputs[1]],
                        lhs_spec.clone(),
                    )
                    .var;
                let d_rhs = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![out_ct, instruction.inputs[0]],
                        rhs_spec.clone(),
                    )
                    .var;
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[0],
                    d_lhs,
                    lhs_spec,
                );
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[1],
                    d_rhs,
                    rhs_spec,
                );
            }
            Operation::Div => {
                let lhs_spec = specs[instruction.inputs[0]]
                    .as_ref()
                    .expect("div lhs spec must exist");
                let rhs_spec = specs[instruction.inputs[1]]
                    .as_ref()
                    .expect("div rhs spec must exist");
                let d_lhs = recorder
                    .add_instruction(
                        Operation::Div,
                        vec![out_ct, instruction.inputs[1]],
                        lhs_spec.clone(),
                    )
                    .var;
                let numerator = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![out_ct, instruction.inputs[0]],
                        rhs_spec.clone(),
                    )
                    .var;
                let denom = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![instruction.inputs[1], instruction.inputs[1]],
                        rhs_spec.clone(),
                    )
                    .var;
                let fraction = recorder
                    .add_instruction(Operation::Div, vec![numerator, denom], rhs_spec.clone())
                    .var;
                let d_rhs = negate_value(&mut recorder, fraction, rhs_spec);
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[0],
                    d_lhs,
                    lhs_spec,
                );
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[1],
                    d_rhs,
                    rhs_spec,
                );
            }
            Operation::Exp => {
                let input_spec = specs[instruction.inputs[0]]
                    .as_ref()
                    .expect("exp input spec must exist");
                let d_input = recorder
                    .add_instruction(
                        Operation::Mul,
                        vec![out_ct, instruction.out],
                        input_spec.clone(),
                    )
                    .var;
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[0],
                    d_input,
                    input_spec,
                );
            }
            Operation::Log => {
                let input_spec = specs[instruction.inputs[0]]
                    .as_ref()
                    .expect("log input spec must exist");
                let d_input = recorder
                    .add_instruction(
                        Operation::Div,
                        vec![out_ct, instruction.inputs[0]],
                        input_spec.clone(),
                    )
                    .var;
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[0],
                    d_input,
                    input_spec,
                );
            }
            Operation::SumAll => {
                let input_spec = specs[instruction.inputs[0]]
                    .as_ref()
                    .expect("sum_all input spec must exist");
                let expanded = recorder
                    .add_instruction(
                        Operation::ExpandScalar {
                            shape: input_spec.shape.clone(),
                        },
                        vec![out_ct],
                        input_spec.clone(),
                    )
                    .var;
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[0],
                    expanded,
                    input_spec,
                );
            }
            Operation::MeanAll => {
                let input_spec = specs[instruction.inputs[0]]
                    .as_ref()
                    .expect("mean_all input spec must exist");
                let expanded = recorder
                    .add_instruction(
                        Operation::ExpandScalar {
                            shape: input_spec.shape.clone(),
                        },
                        vec![out_ct],
                        input_spec.clone(),
                    )
                    .var;
                let scale = recorder.add_const_tensor(DenseTensor::filled(
                    input_spec.shape.clone(),
                    1.0 / input_spec.numel() as f32,
                ));
                let d_input = recorder
                    .add_instruction(Operation::Mul, vec![expanded, scale], input_spec.clone())
                    .var;
                accumulate_cotangent(
                    &mut recorder,
                    &mut cotangents,
                    instruction.inputs[0],
                    d_input,
                    input_spec,
                );
            }
            Operation::ExpandScalar { .. } => {
                panic!("ExpandScalar is an internal primitive and should not appear in forward IR")
            }
        }
    }

    let outputs = forward
        .inputs
        .iter()
        .map(|input| {
            cotangents.get(&input.var).copied().unwrap_or_else(|| {
                recorder.add_const_tensor(DenseTensor::zeros(input.spec.shape.clone()))
            })
        })
        .collect();

    recorder.into_trace(outputs)
}

fn accumulate_cotangent(
    recorder: &mut Recorder,
    cotangents: &mut HashMap<ValueId, ValueId>,
    target: ValueId,
    contrib: ValueId,
    spec: &TensorSpec,
) {
    if let Some(existing) = cotangents.get(&target).copied() {
        let sum = recorder
            .add_instruction(Operation::Add, vec![existing, contrib], spec.clone())
            .var;
        cotangents.insert(target, sum);
    } else {
        cotangents.insert(target, contrib);
    }
}

fn negate_value(recorder: &mut Recorder, input: ValueId, spec: &TensorSpec) -> ValueId {
    let minus_one = recorder.add_const_tensor(DenseTensor::filled(spec.shape.clone(), -1.0));
    recorder
        .add_instruction(Operation::Mul, vec![input, minus_one], spec.clone())
        .var
}
