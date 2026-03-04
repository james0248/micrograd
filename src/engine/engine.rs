use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use super::Op;
use super::kernels::{matmul_backward_da, matmul_backward_db};
use super::shape::{
    broadcast_shape, broadcast_strides_for, contiguous_strides, for_each_index, numel,
    offset_from_coords, reduced_offset_from_input_coords, reduced_shape,
};
use super::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Handle {
    Param {
        idx: usize,
    },
    Temp {
        generation: u64,
        idx: usize,
        id: usize,
    },
}

#[derive(Debug, Clone)]
pub(super) enum Parents {
    None,
    Unary(Tensor),
    Binary(Tensor, Tensor),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ContextKind {
    WithGrad,
    NoGrad,
}

impl ContextKind {
    fn recording(self) -> bool {
        matches!(self, ContextKind::WithGrad)
    }
}

#[derive(Debug, Clone)]
struct ParamNode {
    id: usize,
    data: Vec<f32>,
    grad: Vec<f32>,
    shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct TempNode {
    id: usize,
    generation: u64,
    data: Vec<f32>,
    grad: Vec<f32>,
    shape: Vec<usize>,
    op: Option<Op>,
    parents: Parents,
}

#[derive(Debug, Clone)]
struct Tape {
    generation: u64,
    kind: ContextKind,
    nodes: Vec<TempNode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TempWork {
    generation: u64,
    idx: usize,
    id: usize,
}

impl TempWork {
    fn from_handle(handle: Handle) -> Option<Self> {
        match handle {
            Handle::Temp {
                generation,
                idx,
                id,
            } => Some(Self {
                generation,
                idx,
                id,
            }),
            Handle::Param { .. } => None,
        }
    }

    fn to_tensor(self) -> Tensor {
        Tensor {
            handle: Handle::Temp {
                generation: self.generation,
                idx: self.idx,
                id: self.id,
            },
        }
    }
}

impl Ord for TempWork {
    fn cmp(&self, other: &Self) -> Ordering {
        self.generation
            .cmp(&other.generation)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for TempWork {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub(super) struct Engine {
    next_id: usize,
    next_generation: u64,
    params: Vec<ParamNode>,
    tapes: Vec<Tape>,
}

impl Engine {
    pub(super) fn new() -> Self {
        Self {
            next_id: 1,
            next_generation: 1,
            params: Vec::new(),
            tapes: vec![Tape {
                generation: 0,
                kind: ContextKind::WithGrad,
                nodes: Vec::new(),
            }],
        }
    }

    fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn alloc_generation(&mut self) -> u64 {
        let generation = self.next_generation;
        self.next_generation += 1;
        generation
    }

    pub(super) fn context_depth(&self) -> usize {
        self.tapes.len().saturating_sub(1)
    }

    fn active_tape(&self) -> &Tape {
        self.tapes
            .last()
            .expect("tensor engine must always keep default tape")
    }

    fn active_tape_mut(&mut self) -> &mut Tape {
        self.tapes
            .last_mut()
            .expect("tensor engine must always keep default tape")
    }

    pub(super) fn active_generation(&self) -> u64 {
        self.active_tape().generation
    }

    fn active_kind(&self) -> ContextKind {
        self.active_tape().kind
    }

    pub(super) fn is_with_grad_active(&self) -> bool {
        self.active_kind().recording()
    }

    fn find_tape_pos(&self, generation: u64) -> Option<usize> {
        self.tapes
            .iter()
            .position(|tape| tape.generation == generation)
    }

    pub(super) fn enter_context(&mut self, kind: ContextKind) {
        let generation = self.alloc_generation();
        self.tapes.push(Tape {
            generation,
            kind,
            nodes: Vec::new(),
        });
    }

    pub(super) fn exit_context(&mut self, expected: ContextKind) {
        assert!(
            self.tapes.len() > 1,
            "context stack underflow while exiting grad context"
        );
        let tape = self.tapes.pop().expect("context stack underflow");
        assert_eq!(
            tape.kind, expected,
            "context stack mismatch while exiting context"
        );
    }

    pub(super) fn clear_graph(&mut self) {
        assert_eq!(
            self.context_depth(),
            0,
            "engine::clear_graph() is only allowed when no extra scope is active"
        );
        self.tapes[0].nodes.clear();
    }

    fn find_tape_checked(&self, generation: u64, idx: usize, id: usize) -> usize {
        let tape_pos = self.find_tape_pos(generation).unwrap_or_else(|| {
            panic!(
                "stale Tensor handle: tape generation {} is not active",
                generation
            )
        });

        let node = self.tapes[tape_pos].nodes.get(idx).unwrap_or_else(|| {
            panic!(
                "stale Tensor handle: invalid temp index {} for generation {}",
                idx, generation
            )
        });

        assert_eq!(
            node.generation, generation,
            "stale Tensor handle: generation mismatch (value={}, node={})",
            generation, node.generation
        );
        assert_eq!(
            node.id, id,
            "stale Tensor handle: id mismatch (value={}, node={})",
            id, node.id
        );
        tape_pos
    }

    fn temp(&self, generation: u64, idx: usize, id: usize) -> &TempNode {
        let tape_pos = self.find_tape_checked(generation, idx, id);
        &self.tapes[tape_pos].nodes[idx]
    }

    fn temp_mut(&mut self, generation: u64, idx: usize, id: usize) -> &mut TempNode {
        let tape_pos = self.find_tape_checked(generation, idx, id);
        &mut self.tapes[tape_pos].nodes[idx]
    }

    fn param(&self, idx: usize) -> &ParamNode {
        self.params
            .get(idx)
            .unwrap_or_else(|| panic!("invalid parameter index {idx}"))
    }

    fn param_mut(&mut self, idx: usize) -> &mut ParamNode {
        self.params
            .get_mut(idx)
            .unwrap_or_else(|| panic!("invalid parameter index {idx}"))
    }

    fn validate_parent_for_active_tape(&self, parent: Tensor) {
        if let Handle::Temp {
            generation,
            idx,
            id,
        } = parent.handle
        {
            let parent_pos = self.find_tape_checked(generation, idx, id);
            assert!(
                parent_pos <= self.tapes.len() - 1,
                "parent tape must be current or ancestor"
            );
        }
    }

    fn validate_parents_for_active_tape(&self, parents: &Parents) {
        match parents {
            Parents::None => {}
            Parents::Unary(a) => self.validate_parent_for_active_tape(*a),
            Parents::Binary(a, b) => {
                self.validate_parent_for_active_tape(*a);
                self.validate_parent_for_active_tape(*b);
            }
        }
    }

    pub(super) fn create_temp_leaf(&mut self, data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        assert_eq!(
            data.len(),
            numel(&shape),
            "data length ({}) must match shape numel ({})",
            data.len(),
            numel(&shape)
        );

        let generation = self.active_generation();
        let id = self.alloc_id();
        let grad = vec![0.0; data.len()];

        let tape = self.active_tape_mut();
        let idx = tape.nodes.len();
        tape.nodes.push(TempNode {
            id,
            generation,
            data,
            grad,
            shape,
            op: None,
            parents: Parents::None,
        });

        Tensor {
            handle: Handle::Temp {
                generation,
                idx,
                id,
            },
        }
    }

    pub(super) fn create_parameter(&mut self, data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        assert_eq!(
            data.len(),
            numel(&shape),
            "data length ({}) must match shape numel ({})",
            data.len(),
            numel(&shape)
        );

        let idx = self.params.len();
        let id = self.alloc_id();
        let grad = vec![0.0; data.len()];
        self.params.push(ParamNode {
            id,
            data,
            grad,
            shape,
        });
        Tensor {
            handle: Handle::Param { idx },
        }
    }

    pub(super) fn create_from_op(
        &mut self,
        data: Vec<f32>,
        shape: Vec<usize>,
        op: Op,
        parents: Parents,
    ) -> Tensor {
        assert_eq!(
            data.len(),
            numel(&shape),
            "data length ({}) must match shape numel ({})",
            data.len(),
            numel(&shape)
        );
        self.validate_parents_for_active_tape(&parents);

        let generation = self.active_generation();
        let recording = self.active_kind().recording();
        let id = self.alloc_id();

        let (op, parents) = if recording {
            (Some(op), parents)
        } else {
            (None, Parents::None)
        };

        let grad = vec![0.0; data.len()];
        let tape = self.active_tape_mut();
        let idx = tape.nodes.len();
        tape.nodes.push(TempNode {
            id,
            generation,
            data,
            grad,
            shape,
            op,
            parents,
        });

        Tensor {
            handle: Handle::Temp {
                generation,
                idx,
                id,
            },
        }
    }

    pub(super) fn data_of(&self, value: Tensor) -> Vec<f32> {
        match value.handle {
            Handle::Param { idx } => self.param(idx).data.clone(),
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).data.clone(),
        }
    }

    pub(super) fn grad_of(&self, value: Tensor) -> Vec<f32> {
        match value.handle {
            Handle::Param { idx } => self.param(idx).grad.clone(),
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).grad.clone(),
        }
    }

    pub(super) fn shape_of(&self, value: Tensor) -> Vec<usize> {
        match value.handle {
            Handle::Param { idx } => self.param(idx).shape.clone(),
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).shape.clone(),
        }
    }

    pub(super) fn numel_of(&self, value: Tensor) -> usize {
        numel(&self.shape_of(value))
    }

    pub(super) fn id_of(&self, value: Tensor) -> usize {
        match value.handle {
            Handle::Param { idx } => self.param(idx).id,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).id,
        }
    }

    pub(super) fn set_data(&mut self, value: Tensor, data: &[f32]) {
        match value.handle {
            Handle::Param { idx } => {
                let shape = self.param(idx).shape.clone();
                assert_eq!(
                    data.len(),
                    numel(&shape),
                    "set_data length mismatch: expected {}, got {}",
                    numel(&shape),
                    data.len()
                );
                self.param_mut(idx).data.clone_from_slice(data);
            }
            Handle::Temp {
                generation,
                idx,
                id,
            } => {
                let shape = self.temp(generation, idx, id).shape.clone();
                assert_eq!(
                    data.len(),
                    numel(&shape),
                    "set_data length mismatch: expected {}, got {}",
                    numel(&shape),
                    data.len()
                );
                self.temp_mut(generation, idx, id)
                    .data
                    .clone_from_slice(data);
            }
        }
    }

    pub(super) fn set_grad(&mut self, value: Tensor, grad: &[f32]) {
        match value.handle {
            Handle::Param { idx } => {
                let shape = self.param(idx).shape.clone();
                assert_eq!(
                    grad.len(),
                    numel(&shape),
                    "set_grad length mismatch: expected {}, got {}",
                    numel(&shape),
                    grad.len()
                );
                self.param_mut(idx).grad.clone_from_slice(grad);
            }
            Handle::Temp {
                generation,
                idx,
                id,
            } => {
                let shape = self.temp(generation, idx, id).shape.clone();
                assert_eq!(
                    grad.len(),
                    numel(&shape),
                    "set_grad length mismatch: expected {}, got {}",
                    numel(&shape),
                    grad.len()
                );
                self.temp_mut(generation, idx, id)
                    .grad
                    .clone_from_slice(grad);
            }
        }
    }

    pub(super) fn add_grad(&mut self, value: Tensor, delta: &[f32]) {
        match value.handle {
            Handle::Param { idx } => {
                let expected = self.param(idx).grad.len();
                assert_eq!(
                    delta.len(),
                    expected,
                    "add_grad length mismatch: expected {}, got {}",
                    expected,
                    delta.len()
                );
                let grad = &mut self.param_mut(idx).grad;
                for (g, d) in grad.iter_mut().zip(delta.iter()) {
                    *g += *d;
                }
            }
            Handle::Temp {
                generation,
                idx,
                id,
            } => {
                let expected = self.temp(generation, idx, id).grad.len();
                assert_eq!(
                    delta.len(),
                    expected,
                    "add_grad length mismatch: expected {}, got {}",
                    expected,
                    delta.len()
                );
                let grad = &mut self.temp_mut(generation, idx, id).grad;
                for (g, d) in grad.iter_mut().zip(delta.iter()) {
                    *g += *d;
                }
            }
        }
    }

    pub(super) fn zero_grad(&mut self, value: Tensor) {
        match value.handle {
            Handle::Param { idx } => {
                self.param_mut(idx).grad.fill(0.0);
            }
            Handle::Temp {
                generation,
                idx,
                id,
            } => {
                self.temp_mut(generation, idx, id).grad.fill(0.0);
            }
        }
    }

    pub(super) fn op_of(&self, value: Tensor) -> Option<Op> {
        match value.handle {
            Handle::Param { .. } => None,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).op.clone(),
        }
    }

    pub(super) fn parents_of(&self, value: Tensor) -> Parents {
        match value.handle {
            Handle::Param { .. } => Parents::None,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).parents.clone(),
        }
    }

    pub(super) fn backward_step(&mut self, node: Tensor) {
        let op = self.op_of(node);
        let parents = self.parents_of(node);
        let out_grad = self.grad_of(node);
        let out_data = self.data_of(node);
        let out_shape = self.shape_of(node);

        match (op, parents) {
            (Some(Op::MatMul), Parents::Binary(a, b)) => {
                let a_shape = self.shape_of(a);
                let b_shape = self.shape_of(b);
                assert!(
                    a_shape.len() >= 2 && b_shape.len() >= 2,
                    "matmul backward expects rank >=2: left={:?}, right={:?}",
                    a_shape,
                    b_shape
                );

                let m = a_shape[a_shape.len() - 2];
                let k = a_shape[a_shape.len() - 1];
                let k_b = b_shape[b_shape.len() - 2];
                let n = b_shape[b_shape.len() - 1];
                assert_eq!(k, k_b, "matmul backward shape mismatch");

                let a_batch = &a_shape[..a_shape.len() - 2];
                let b_batch = &b_shape[..b_shape.len() - 2];
                let batch_shape = broadcast_shape(a_batch, b_batch);
                let mut expected_out_shape = batch_shape.clone();
                expected_out_shape.push(m);
                expected_out_shape.push(n);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "matmul backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let a_data = self.data_of(a);
                let b_data = self.data_of(b);
                let a_strides = contiguous_strides(&a_shape);
                let b_strides = contiguous_strides(&b_shape);
                let a_batch_strides =
                    broadcast_strides_for(a_batch, &a_strides[..a_batch.len()], &batch_shape);
                let b_batch_strides =
                    broadcast_strides_for(b_batch, &b_strides[..b_batch.len()], &batch_shape);
                let batch_strides = contiguous_strides(&batch_shape);

                let a_block = m * k;
                let b_block = k * n;
                let out_block = m * n;
                let mut da_total = vec![0.0; numel(&a_shape)];
                let mut db_total = vec![0.0; numel(&b_shape)];

                for_each_index(&batch_shape, |batch_coords| {
                    let a_off = offset_from_coords(batch_coords, &a_batch_strides);
                    let b_off = offset_from_coords(batch_coords, &b_batch_strides);
                    let batch_off = offset_from_coords(batch_coords, &batch_strides);
                    let out_off = batch_off * out_block;

                    let g_block = &out_grad[out_off..out_off + out_block];
                    let a_block_data = &a_data[a_off..a_off + a_block];
                    let b_block_data = &b_data[b_off..b_off + b_block];

                    let da_block = matmul_backward_da(g_block, b_block_data, m, n, k);
                    let db_block = matmul_backward_db(a_block_data, g_block, m, k, n);

                    for i in 0..a_block {
                        da_total[a_off + i] += da_block[i];
                    }
                    for i in 0..b_block {
                        db_total[b_off + i] += db_block[i];
                    }
                });

                self.add_grad(a, &da_total);
                self.add_grad(b, &db_total);
            }
            (Some(Op::Add), Parents::Binary(a, b)) => {
                let a_shape = self.shape_of(a);
                let b_shape = self.shape_of(b);
                let expected_out_shape = broadcast_shape(&a_shape, &b_shape);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "add backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let a_strides = contiguous_strides(&a_shape);
                let b_strides = contiguous_strides(&b_shape);
                let a_bstrides = broadcast_strides_for(&a_shape, &a_strides, &out_shape);
                let b_bstrides = broadcast_strides_for(&b_shape, &b_strides, &out_shape);
                let mut da = vec![0.0; numel(&a_shape)];
                let mut db = vec![0.0; numel(&b_shape)];

                let mut out_i = 0usize;
                for_each_index(&out_shape, |coords| {
                    let ai = offset_from_coords(coords, &a_bstrides);
                    let bi = offset_from_coords(coords, &b_bstrides);
                    let g = out_grad[out_i];
                    da[ai] += g;
                    db[bi] += g;
                    out_i += 1;
                });

                self.add_grad(a, &da);
                self.add_grad(b, &db);
            }
            (Some(Op::Sub), Parents::Binary(a, b)) => {
                let a_shape = self.shape_of(a);
                let b_shape = self.shape_of(b);
                let expected_out_shape = broadcast_shape(&a_shape, &b_shape);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "sub backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let a_strides = contiguous_strides(&a_shape);
                let b_strides = contiguous_strides(&b_shape);
                let a_bstrides = broadcast_strides_for(&a_shape, &a_strides, &out_shape);
                let b_bstrides = broadcast_strides_for(&b_shape, &b_strides, &out_shape);
                let mut da = vec![0.0; numel(&a_shape)];
                let mut db = vec![0.0; numel(&b_shape)];

                let mut out_i = 0usize;
                for_each_index(&out_shape, |coords| {
                    let ai = offset_from_coords(coords, &a_bstrides);
                    let bi = offset_from_coords(coords, &b_bstrides);
                    let g = out_grad[out_i];
                    da[ai] += g;
                    db[bi] -= g;
                    out_i += 1;
                });

                self.add_grad(a, &da);
                self.add_grad(b, &db);
            }
            (Some(Op::Mul), Parents::Binary(a, b)) => {
                let a_shape = self.shape_of(a);
                let b_shape = self.shape_of(b);
                let expected_out_shape = broadcast_shape(&a_shape, &b_shape);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "mul backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let a_data = self.data_of(a);
                let b_data = self.data_of(b);
                let a_strides = contiguous_strides(&a_shape);
                let b_strides = contiguous_strides(&b_shape);
                let a_bstrides = broadcast_strides_for(&a_shape, &a_strides, &out_shape);
                let b_bstrides = broadcast_strides_for(&b_shape, &b_strides, &out_shape);
                let mut da = vec![0.0; numel(&a_shape)];
                let mut db = vec![0.0; numel(&b_shape)];

                let mut out_i = 0usize;
                for_each_index(&out_shape, |coords| {
                    let ai = offset_from_coords(coords, &a_bstrides);
                    let bi = offset_from_coords(coords, &b_bstrides);
                    let g = out_grad[out_i];
                    da[ai] += g * b_data[bi];
                    db[bi] += g * a_data[ai];
                    out_i += 1;
                });

                self.add_grad(a, &da);
                self.add_grad(b, &db);
            }
            (Some(Op::Div), Parents::Binary(a, b)) => {
                let a_shape = self.shape_of(a);
                let b_shape = self.shape_of(b);
                let expected_out_shape = broadcast_shape(&a_shape, &b_shape);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "div backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let a_data = self.data_of(a);
                let b_data = self.data_of(b);
                let a_strides = contiguous_strides(&a_shape);
                let b_strides = contiguous_strides(&b_shape);
                let a_bstrides = broadcast_strides_for(&a_shape, &a_strides, &out_shape);
                let b_bstrides = broadcast_strides_for(&b_shape, &b_strides, &out_shape);
                let mut da = vec![0.0; numel(&a_shape)];
                let mut db = vec![0.0; numel(&b_shape)];

                let mut out_i = 0usize;
                for_each_index(&out_shape, |coords| {
                    let ai = offset_from_coords(coords, &a_bstrides);
                    let bi = offset_from_coords(coords, &b_bstrides);
                    let g = out_grad[out_i];
                    let denom = b_data[bi];
                    da[ai] += g / denom;
                    db[bi] += -g * a_data[ai] / (denom * denom);
                    out_i += 1;
                });

                self.add_grad(a, &da);
                self.add_grad(b, &db);
            }
            (Some(Op::Exp), Parents::Unary(a)) => {
                assert_eq!(out_data.len(), out_grad.len(), "exp backward size mismatch");
                let mut da = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() {
                    da[i] = out_grad[i] * out_data[i];
                }
                self.add_grad(a, &da);
            }
            (Some(Op::Log), Parents::Unary(a)) => {
                let a_data = self.data_of(a);
                assert_eq!(a_data.len(), out_grad.len(), "log backward size mismatch");
                let mut da = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() {
                    da[i] = out_grad[i] / a_data[i];
                }
                self.add_grad(a, &da);
            }
            (Some(Op::Sum { axis, keepdim }), Parents::Unary(a)) => {
                let a_shape = self.shape_of(a);
                let expected_out_shape = reduced_shape(&a_shape, axis, keepdim);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "sum backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );
                let out_strides = contiguous_strides(&out_shape);
                let mut da = vec![0.0; numel(&a_shape)];
                let mut a_i = 0usize;

                for_each_index(&a_shape, |coords| {
                    let out_i = reduced_offset_from_input_coords(
                        coords,
                        &out_shape,
                        &out_strides,
                        axis,
                        keepdim,
                    );
                    da[a_i] = out_grad[out_i];
                    a_i += 1;
                });

                self.add_grad(a, &da);
            }
            (Some(Op::Max { axis, keepdim }), Parents::Unary(a)) => {
                let a_shape = self.shape_of(a);
                let expected_out_shape = reduced_shape(&a_shape, axis, keepdim);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "max backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let out_strides = contiguous_strides(&out_shape);
                let a_data = self.data_of(a);
                let mut counts = vec![0usize; out_grad.len()];
                let mut a_i = 0usize;

                for_each_index(&a_shape, |coords| {
                    let out_i = reduced_offset_from_input_coords(
                        coords,
                        &out_shape,
                        &out_strides,
                        axis,
                        keepdim,
                    );
                    if a_data[a_i] == out_data[out_i] {
                        counts[out_i] += 1;
                    }
                    a_i += 1;
                });

                let mut da = vec![0.0; numel(&a_shape)];
                let mut a_i = 0usize;
                for_each_index(&a_shape, |coords| {
                    let out_i = reduced_offset_from_input_coords(
                        coords,
                        &out_shape,
                        &out_strides,
                        axis,
                        keepdim,
                    );
                    if a_data[a_i] == out_data[out_i] {
                        let count = counts[out_i];
                        assert!(count > 0, "max backward encountered zero tie count");
                        da[a_i] += out_grad[out_i] / count as f32;
                    }
                    a_i += 1;
                });

                self.add_grad(a, &da);
            }
            (Some(Op::Relu), Parents::Unary(a)) => {
                let a_data = self.data_of(a);
                assert_eq!(a_data.len(), out_grad.len(), "relu backward size mismatch");
                let mut da = vec![0.0; a_data.len()];
                for i in 0..a_data.len() {
                    da[i] = if a_data[i] > 0.0 { out_grad[i] } else { 0.0 };
                }
                self.add_grad(a, &da);
            }
            (Some(Op::Mean), Parents::Unary(a)) => {
                let n = self.numel_of(a);
                let upstream = out_grad
                    .first()
                    .copied()
                    .expect("mean backward requires scalar grad");
                let each = upstream / n as f32;
                let da = vec![each; n];
                self.add_grad(a, &da);
            }
            _ => {}
        }
    }

    pub(super) fn assert_backward_allowed(&self, root: Tensor) {
        if let Handle::Temp {
            generation,
            idx,
            id,
        } = root.handle
        {
            let tape_pos = self.find_tape_checked(generation, idx, id);
            assert!(
                self.tapes[tape_pos].kind.recording(),
                "cannot call backward on no_grad tensor from generation {}",
                generation
            );
        }

        let n = self.numel_of(root);
        assert_eq!(
            n, 1,
            "backward root must be scalar (numel=1), got numel={}",
            n
        );
    }

    pub(super) fn collect_reachable_heap_order(&self, root: Tensor) -> Vec<Tensor> {
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        let mut heap = BinaryHeap::new();

        let Some(root_work) = TempWork::from_handle(root.handle) else {
            return order;
        };
        heap.push(root_work);
        seen.insert(root);

        let push_parent =
            |parent: Tensor, seen: &mut HashSet<Tensor>, heap: &mut BinaryHeap<TempWork>| {
                if seen.insert(parent) {
                    if let Some(work) = TempWork::from_handle(parent.handle) {
                        heap.push(work);
                    }
                }
            };

        while let Some(work) = heap.pop() {
            let node = work.to_tensor();
            order.push(node);

            match self.parents_of(node) {
                Parents::None => {}
                Parents::Unary(a) => push_parent(a, &mut seen, &mut heap),
                Parents::Binary(a, b) => {
                    push_parent(a, &mut seen, &mut heap);
                    push_parent(b, &mut seen, &mut heap);
                }
            }
        }
        order
    }

    pub(super) fn param_count(&self) -> usize {
        self.params.len()
    }

    pub(super) fn temp_count(&self) -> usize {
        self.tapes.iter().map(|tape| tape.nodes.len()).sum()
    }
}
