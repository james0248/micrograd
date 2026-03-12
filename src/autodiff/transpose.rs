use std::collections::HashMap;

use crate::tensor::{DenseTensor, TensorSpec, ValueId, matmul_shape};

use super::interpreter::execute_trace;
use super::ir::{Operation, build_spec_table};
use super::jvp::Linearized;
use super::trace::{Recorder, accumulate_cotangent, negate_value};

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct Pullback {
    pub(crate) trace: super::ir::Trace,
    pub(crate) residuals: Vec<DenseTensor>,
    pub(crate) input_cotangent_specs: Vec<TensorSpec>,
    pub(crate) output_cotangent_spec: TensorSpec,
}

#[cfg_attr(not(test), allow(dead_code))]
impl Pullback {
    pub(crate) fn apply_dense(&self, output_cotangent: &DenseTensor) -> Vec<DenseTensor> {
        assert_eq!(
            output_cotangent.shape, self.output_cotangent_spec.shape,
            "pullback output cotangent shape mismatch: expected {:?}, got {:?}",
            self.output_cotangent_spec.shape, output_cotangent.shape
        );

        let mut inputs = Vec::with_capacity(1 + self.residuals.len());
        inputs.push(output_cotangent.clone());
        inputs.extend(self.residuals.iter().cloned());

        let outputs = execute_trace(&self.trace, &inputs);
        assert_eq!(
            outputs.len(),
            self.input_cotangent_specs.len(),
            "pullback output count mismatch: expected {}, got {}",
            self.input_cotangent_specs.len(),
            outputs.len()
        );
        for (output, spec) in outputs.iter().zip(self.input_cotangent_specs.iter()) {
            assert_eq!(
                output.shape, spec.shape,
                "pullback input cotangent shape mismatch: expected {:?}, got {:?}",
                spec.shape, output.shape
            );
        }
        outputs
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn transpose_linearized(linearized: &Linearized) -> Pullback {
    let trace = &linearized.trace;
    let specs = build_spec_table(trace);
    let depends_on_tangent = tangent_dependency_table(linearized);
    let output = *trace
        .outputs
        .first()
        .expect("linearized trace must produce exactly one output");

    let mut recorder = Recorder::new_empty();
    let output_cotangent = recorder.add_input(linearized.output_spec.clone());
    let mut static_var_map = HashMap::new();

    for input in trace
        .inputs
        .iter()
        .skip(linearized.tangent_input_specs.len())
    {
        let mapped = recorder.add_input(input.spec.clone());
        static_var_map.insert(input.var, mapped.var);
    }
    for constant in &trace.consts {
        let mapped = recorder.add_const_tensor(constant.tensor.clone());
        static_var_map.insert(constant.var, mapped);
    }

    let mut cotangents: Vec<Option<ValueId>> = vec![None; trace.next_var];
    cotangents[output] = Some(output_cotangent.var);

    for instruction in trace.instructions.iter().rev() {
        let Some(out_cotangent) = cotangents[instruction.out] else {
            continue;
        };

        match &instruction.op {
            Operation::Add => {
                let lhs_spec = specs[instruction.inputs[0]]
                    .as_ref()
                    .expect("add lhs spec must exist");
                let rhs_spec = specs[instruction.inputs[1]]
                    .as_ref()
                    .expect("add rhs spec must exist");
                if depends_on_tangent[instruction.inputs[0]] {
                    let d_lhs = sum_to_shape_if_needed(
                        &mut recorder,
                        out_cotangent,
                        &instruction.spec,
                        lhs_spec,
                    );
                    accumulate_cotangent(
                        &mut recorder,
                        &mut cotangents,
                        instruction.inputs[0],
                        d_lhs,
                        lhs_spec,
                    );
                }
                if depends_on_tangent[instruction.inputs[1]] {
                    let d_rhs = sum_to_shape_if_needed(
                        &mut recorder,
                        out_cotangent,
                        &instruction.spec,
                        rhs_spec,
                    );
                    accumulate_cotangent(
                        &mut recorder,
                        &mut cotangents,
                        instruction.inputs[1],
                        d_rhs,
                        rhs_spec,
                    );
                }
            }
            Operation::Sub => {
                let lhs_spec = specs[instruction.inputs[0]]
                    .as_ref()
                    .expect("sub lhs spec must exist");
                let rhs_spec = specs[instruction.inputs[1]]
                    .as_ref()
                    .expect("sub rhs spec must exist");
                if depends_on_tangent[instruction.inputs[0]] {
                    let d_lhs = sum_to_shape_if_needed(
                        &mut recorder,
                        out_cotangent,
                        &instruction.spec,
                        lhs_spec,
                    );
                    accumulate_cotangent(
                        &mut recorder,
                        &mut cotangents,
                        instruction.inputs[0],
                        d_lhs,
                        lhs_spec,
                    );
                }
                if depends_on_tangent[instruction.inputs[1]] {
                    let reduced = sum_to_shape_if_needed(
                        &mut recorder,
                        out_cotangent,
                        &instruction.spec,
                        rhs_spec,
                    );
                    let neg = negate_value(&mut recorder, reduced, rhs_spec);
                    accumulate_cotangent(
                        &mut recorder,
                        &mut cotangents,
                        instruction.inputs[1],
                        neg,
                        rhs_spec,
                    );
                }
            }
            Operation::Mul => {
                let lhs_dep = depends_on_tangent[instruction.inputs[0]];
                let rhs_dep = depends_on_tangent[instruction.inputs[1]];
                assert!(
                    lhs_dep ^ rhs_dep,
                    "linearized mul must have exactly one tangent-dependent operand"
                );

                if lhs_dep {
                    let coeff = mapped_static_var(
                        &static_var_map,
                        instruction.inputs[1],
                        "mul rhs coefficient",
                    );
                    let lhs_spec = specs[instruction.inputs[0]]
                        .as_ref()
                        .expect("mul lhs spec must exist");
                    let raw_spec = TensorSpec::new(instruction.spec.shape.clone());
                    let d_lhs = recorder
                        .add_instruction(
                            Operation::Mul,
                            vec![out_cotangent, coeff],
                            raw_spec.clone(),
                        )
                        .var;
                    let d_lhs = sum_to_shape_if_needed(&mut recorder, d_lhs, &raw_spec, lhs_spec);
                    accumulate_cotangent(
                        &mut recorder,
                        &mut cotangents,
                        instruction.inputs[0],
                        d_lhs,
                        lhs_spec,
                    );
                } else {
                    let coeff = mapped_static_var(
                        &static_var_map,
                        instruction.inputs[0],
                        "mul lhs coefficient",
                    );
                    let rhs_spec = specs[instruction.inputs[1]]
                        .as_ref()
                        .expect("mul rhs spec must exist");
                    let raw_spec = TensorSpec::new(instruction.spec.shape.clone());
                    let d_rhs = recorder
                        .add_instruction(
                            Operation::Mul,
                            vec![coeff, out_cotangent],
                            raw_spec.clone(),
                        )
                        .var;
                    let d_rhs = sum_to_shape_if_needed(&mut recorder, d_rhs, &raw_spec, rhs_spec);
                    accumulate_cotangent(
                        &mut recorder,
                        &mut cotangents,
                        instruction.inputs[1],
                        d_rhs,
                        rhs_spec,
                    );
                }
            }
            Operation::Div | Operation::Exp | Operation::Log => {
                unreachable!("validated by tangent_dependency_table")
            }
            Operation::Sum { axis, keepdim } => {
                if depends_on_tangent[instruction.inputs[0]] {
                    let input_spec = specs[instruction.inputs[0]]
                        .as_ref()
                        .expect("sum input spec must exist");
                    let inserted_axes =
                        if *keepdim || input_spec.shape.len() == instruction.spec.shape.len() {
                            Vec::new()
                        } else {
                            vec![*axis]
                        };
                    let d_input = recorder
                        .add_instruction(
                            Operation::ExpandToShape {
                                shape: input_spec.shape.clone(),
                                inserted_axes,
                            },
                            vec![out_cotangent],
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
            }
            Operation::Max { .. } => {
                panic!(
                    "linearized trace must not contain Max; capture max tie weights as a residual"
                );
            }
            Operation::Relu => {
                panic!(
                    "linearized trace must not contain Relu; capture the derivative mask as a residual"
                );
            }
            Operation::MatMul => {
                let lhs_dep = depends_on_tangent[instruction.inputs[0]];
                let rhs_dep = depends_on_tangent[instruction.inputs[1]];
                assert!(
                    lhs_dep ^ rhs_dep,
                    "linearized matmul must have exactly one tangent-dependent operand"
                );

                if lhs_dep {
                    let lhs_spec = specs[instruction.inputs[0]]
                        .as_ref()
                        .expect("matmul lhs spec must exist");
                    let rhs_spec = specs[instruction.inputs[1]]
                        .as_ref()
                        .expect("matmul rhs spec must exist");
                    let coeff = mapped_static_var(
                        &static_var_map,
                        instruction.inputs[1],
                        "matmul rhs coefficient",
                    );
                    let transposed_rhs_spec = transpose_spec(
                        rhs_spec,
                        rhs_spec.shape.len() - 2,
                        rhs_spec.shape.len() - 1,
                    );
                    let transposed_rhs = recorder
                        .add_instruction(
                            Operation::Transpose {
                                dim0: rhs_spec.shape.len() - 2,
                                dim1: rhs_spec.shape.len() - 1,
                            },
                            vec![coeff],
                            transposed_rhs_spec.clone(),
                        )
                        .var;
                    let raw_spec = TensorSpec::new(matmul_shape(
                        &instruction.spec.shape,
                        &transposed_rhs_spec.shape,
                    ));
                    let d_lhs = recorder
                        .add_instruction(
                            Operation::MatMul,
                            vec![out_cotangent, transposed_rhs],
                            raw_spec.clone(),
                        )
                        .var;
                    let d_lhs = sum_to_shape_if_needed(&mut recorder, d_lhs, &raw_spec, lhs_spec);
                    accumulate_cotangent(
                        &mut recorder,
                        &mut cotangents,
                        instruction.inputs[0],
                        d_lhs,
                        lhs_spec,
                    );
                } else {
                    let lhs_spec = specs[instruction.inputs[0]]
                        .as_ref()
                        .expect("matmul lhs spec must exist");
                    let rhs_spec = specs[instruction.inputs[1]]
                        .as_ref()
                        .expect("matmul rhs spec must exist");
                    let coeff = mapped_static_var(
                        &static_var_map,
                        instruction.inputs[0],
                        "matmul lhs coefficient",
                    );
                    let transposed_lhs_spec = transpose_spec(
                        lhs_spec,
                        lhs_spec.shape.len() - 2,
                        lhs_spec.shape.len() - 1,
                    );
                    let transposed_lhs = recorder
                        .add_instruction(
                            Operation::Transpose {
                                dim0: lhs_spec.shape.len() - 2,
                                dim1: lhs_spec.shape.len() - 1,
                            },
                            vec![coeff],
                            transposed_lhs_spec.clone(),
                        )
                        .var;
                    let raw_spec = TensorSpec::new(matmul_shape(
                        &transposed_lhs_spec.shape,
                        &instruction.spec.shape,
                    ));
                    let d_rhs = recorder
                        .add_instruction(
                            Operation::MatMul,
                            vec![transposed_lhs, out_cotangent],
                            raw_spec.clone(),
                        )
                        .var;
                    let d_rhs = sum_to_shape_if_needed(&mut recorder, d_rhs, &raw_spec, rhs_spec);
                    accumulate_cotangent(
                        &mut recorder,
                        &mut cotangents,
                        instruction.inputs[1],
                        d_rhs,
                        rhs_spec,
                    );
                }
            }
            Operation::SumAll => {
                if depends_on_tangent[instruction.inputs[0]] {
                    let input_spec = specs[instruction.inputs[0]]
                        .as_ref()
                        .expect("sum_all input spec must exist");
                    let expanded = recorder
                        .add_instruction(
                            Operation::ExpandScalar {
                                shape: input_spec.shape.clone(),
                            },
                            vec![out_cotangent],
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
            }
            Operation::MeanAll => {
                if depends_on_tangent[instruction.inputs[0]] {
                    let input_spec = specs[instruction.inputs[0]]
                        .as_ref()
                        .expect("mean_all input spec must exist");
                    let expanded = recorder
                        .add_instruction(
                            Operation::ExpandScalar {
                                shape: input_spec.shape.clone(),
                            },
                            vec![out_cotangent],
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
            }
            Operation::Transpose { dim0, dim1 } => {
                if depends_on_tangent[instruction.inputs[0]] {
                    let input_spec = specs[instruction.inputs[0]]
                        .as_ref()
                        .expect("transpose input spec must exist");
                    let d_input = recorder
                        .add_instruction(
                            Operation::Transpose {
                                dim0: *dim0,
                                dim1: *dim1,
                            },
                            vec![out_cotangent],
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
            }
            Operation::SumToShape { .. } | Operation::ExpandToShape { .. } => {
                unreachable!("validated by tangent_dependency_table")
            }
            Operation::ExpandScalar { .. } => {
                unreachable!("validated by tangent_dependency_table")
            }
        }
    }

    let outputs = trace
        .inputs
        .iter()
        .take(linearized.tangent_input_specs.len())
        .map(|input| {
            cotangents[input.var].unwrap_or_else(|| {
                recorder.add_const_tensor(DenseTensor::zeros(input.spec.shape.clone()))
            })
        })
        .collect();

    Pullback {
        trace: recorder.into_trace(outputs),
        residuals: linearized.residuals.clone(),
        input_cotangent_specs: linearized.tangent_input_specs.clone(),
        output_cotangent_spec: linearized.output_spec.clone(),
    }
}

fn mapped_static_var(
    static_var_map: &HashMap<ValueId, ValueId>,
    old_var: ValueId,
    label: &str,
) -> ValueId {
    static_var_map
        .get(&old_var)
        .copied()
        .unwrap_or_else(|| panic!("{label} must map to a residual or literal constant"))
}

fn sum_to_shape_if_needed(
    recorder: &mut Recorder,
    input: ValueId,
    input_spec: &TensorSpec,
    target_spec: &TensorSpec,
) -> ValueId {
    if input_spec.shape == target_spec.shape {
        input
    } else {
        recorder
            .add_instruction(
                Operation::SumToShape {
                    shape: target_spec.shape.clone(),
                },
                vec![input],
                target_spec.clone(),
            )
            .var
    }
}

fn transpose_spec(spec: &TensorSpec, dim0: usize, dim1: usize) -> TensorSpec {
    let mut shape = spec.shape.clone();
    shape.swap(dim0, dim1);
    TensorSpec::new(shape)
}

fn tangent_dependency_table(linearized: &Linearized) -> Vec<bool> {
    let trace = &linearized.trace;
    let mut depends_on_tangent = vec![false; trace.next_var];

    for input in trace
        .inputs
        .iter()
        .take(linearized.tangent_input_specs.len())
    {
        depends_on_tangent[input.var] = true;
    }
    for input in trace
        .inputs
        .iter()
        .skip(linearized.tangent_input_specs.len())
    {
        depends_on_tangent[input.var] = false;
    }
    for constant in &trace.consts {
        depends_on_tangent[constant.var] = false;
    }

    for instruction in &trace.instructions {
        let depends = match &instruction.op {
            Operation::Add | Operation::Sub => {
                depends_on_tangent[instruction.inputs[0]]
                    || depends_on_tangent[instruction.inputs[1]]
            }
            Operation::Mul | Operation::MatMul => {
                let lhs = depends_on_tangent[instruction.inputs[0]];
                let rhs = depends_on_tangent[instruction.inputs[1]];
                assert!(
                    !(lhs && rhs),
                    "linearized {:?} must not multiply two tangent-dependent values",
                    instruction.op
                );
                lhs || rhs
            }
            Operation::Div => {
                panic!(
                    "linearized trace must not contain Div; capture division coefficients as residuals"
                )
            }
            Operation::Exp | Operation::Log => {
                panic!(
                    "linearized trace must not contain {:?}; capture nonlinear coefficients as residuals",
                    instruction.op
                )
            }
            Operation::Sum { .. }
            | Operation::SumAll
            | Operation::MeanAll
            | Operation::Transpose { .. } => depends_on_tangent[instruction.inputs[0]],
            Operation::Max { .. } => {
                panic!(
                    "linearized trace must not contain Max; capture max tie weights as a residual"
                )
            }
            Operation::Relu => {
                panic!(
                    "linearized trace must not contain Relu; capture the derivative mask as a residual"
                )
            }
            Operation::SumToShape { .. } | Operation::ExpandToShape { .. } => {
                panic!(
                    "SumToShape and ExpandToShape are internal backward ops and must not appear in linearized traces"
                )
            }
            Operation::ExpandScalar { .. } => {
                panic!(
                    "ExpandScalar is an internal backward op and must not appear in linearized traces"
                )
            }
        };
        depends_on_tangent[instruction.out] = depends;
    }

    depends_on_tangent
}
