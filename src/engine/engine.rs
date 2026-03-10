use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use super::Op;
use super::kernels::{MatRef, matmul};
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
pub(super) struct NodeLayout {
    pub(super) buffer_id: usize,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
    pub(super) offset: usize,
}

#[derive(Debug, Clone)]
struct ParamNode {
    id: usize,
    layout: NodeLayout,
    grad: Vec<f32>,
}

#[derive(Debug, Clone)]
struct TempNode {
    id: usize,
    generation: u64,
    layout: NodeLayout,
    grad: Vec<f32>,
    op: Option<Op>,
    parents: Parents,
}

#[derive(Debug, Clone)]
struct Tape {
    generation: u64,
    kind: ContextKind,
    nodes: Vec<TempNode>,
}

#[derive(Debug, Clone)]
struct BufferSlot {
    data: Vec<f32>,
    ref_count: usize,
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
    buffers: Vec<Option<BufferSlot>>,
    free_buffer_ids: Vec<usize>,
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
            buffers: Vec::new(),
            free_buffer_ids: Vec::new(),
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

    fn alloc_buffer(&mut self, data: Vec<f32>) -> usize {
        if let Some(id) = self.free_buffer_ids.pop() {
            self.buffers[id] = Some(BufferSlot { data, ref_count: 1 });
            id
        } else {
            let id = self.buffers.len();
            self.buffers.push(Some(BufferSlot { data, ref_count: 1 }));
            id
        }
    }

    fn buffer_slot(&self, buffer_id: usize) -> &BufferSlot {
        self.buffers
            .get(buffer_id)
            .and_then(Option::as_ref)
            .unwrap_or_else(|| panic!("invalid buffer id {buffer_id}"))
    }

    fn buffer_slot_mut(&mut self, buffer_id: usize) -> &mut BufferSlot {
        self.buffers
            .get_mut(buffer_id)
            .and_then(Option::as_mut)
            .unwrap_or_else(|| panic!("invalid buffer id {buffer_id}"))
    }

    fn retain_buffer(&mut self, buffer_id: usize) {
        let slot = self.buffer_slot_mut(buffer_id);
        slot.ref_count += 1;
    }

    fn release_buffer(&mut self, buffer_id: usize) {
        let slot = self.buffer_slot_mut(buffer_id);
        assert!(slot.ref_count > 0, "buffer ref_count underflow");
        slot.ref_count -= 1;
        if slot.ref_count == 0 {
            self.buffers[buffer_id] = None;
            self.free_buffer_ids.push(buffer_id);
        }
    }

    fn last_index(shape: &[usize], strides: &[usize], offset: usize) -> usize {
        if shape.is_empty() || numel(shape) == 0 {
            return offset;
        }

        let mut out = offset;
        for (&dim, &stride) in shape.iter().zip(strides.iter()) {
            if dim > 0 {
                let term = (dim - 1)
                    .checked_mul(stride)
                    .expect("layout stride overflow");
                out = out.checked_add(term).expect("layout offset overflow");
            }
        }
        out
    }

    fn validate_layout(layout: &NodeLayout, data_len: usize) {
        assert_eq!(
            layout.shape.len(),
            layout.strides.len(),
            "layout rank mismatch: shape={:?}, strides={:?}",
            layout.shape,
            layout.strides
        );

        if numel(&layout.shape) == 0 {
            assert!(
                layout.offset <= data_len,
                "layout offset out of bounds: offset={}, len={}",
                layout.offset,
                data_len
            );
            return;
        }

        let last = Self::last_index(&layout.shape, &layout.strides, layout.offset);
        assert!(
            last < data_len,
            "layout out of bounds: last_index={}, len={}",
            last,
            data_len
        );
    }

    fn read_layout_data(&self, layout: &NodeLayout) -> Vec<f32> {
        let slot = self.buffer_slot(layout.buffer_id);
        Self::validate_layout(layout, slot.data.len());
        if layout.offset == 0
            && Self::is_standard_contiguous(&layout.shape, &layout.strides)
            && numel(&layout.shape) == slot.data.len()
        {
            return slot.data.clone();
        }
        let mut out = vec![0.0; numel(&layout.shape)];
        let mut out_i = 0usize;

        for_each_index(&layout.shape, |coords| {
            let src_i = layout.offset + offset_from_coords(coords, &layout.strides);
            out[out_i] = slot.data[src_i];
            out_i += 1;
        });

        out
    }

    fn write_layout_data(&mut self, layout: &NodeLayout, data: &[f32]) {
        let expected = numel(&layout.shape);
        assert_eq!(
            data.len(),
            expected,
            "set_data length mismatch: expected {}, got {}",
            expected,
            data.len()
        );

        let shape = layout.shape.clone();
        let strides = layout.strides.clone();
        let offset = layout.offset;
        let src = data;

        let slot = self.buffer_slot_mut(layout.buffer_id);
        Self::validate_layout(layout, slot.data.len());
        if offset == 0
            && Self::is_standard_contiguous(&shape, &strides)
            && numel(&shape) == slot.data.len()
        {
            slot.data.clone_from_slice(src);
            return;
        }

        let mut src_i = 0usize;
        for_each_index(&shape, |coords| {
            let dst_i = offset + offset_from_coords(coords, &strides);
            slot.data[dst_i] = src[src_i];
            src_i += 1;
        });
    }

    fn is_standard_contiguous(shape: &[usize], strides: &[usize]) -> bool {
        if shape.len() != strides.len() {
            return false;
        }
        if shape.is_empty() {
            return true;
        }

        let mut expected = 1usize;
        for axis in (0..shape.len()).rev() {
            if strides[axis] != expected {
                return false;
            }
            expected = expected
                .checked_mul(shape[axis])
                .expect("shape product overflow while validating contiguity");
        }
        true
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
        for node in tape.nodes {
            self.release_buffer(node.layout.buffer_id);
        }
    }

    pub(super) fn clear_graph(&mut self) {
        assert_eq!(
            self.context_depth(),
            0,
            "engine::clear_graph() is only allowed when no extra scope is active"
        );
        let nodes = std::mem::take(&mut self.tapes[0].nodes);
        for node in nodes {
            self.release_buffer(node.layout.buffer_id);
        }
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

    pub(super) fn layout_of(&self, value: Tensor) -> NodeLayout {
        match value.handle {
            Handle::Param { idx } => self.param(idx).layout.clone(),
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).layout.clone(),
        }
    }

    pub(super) fn buffer_of(&self, buffer_id: usize) -> &[f32] {
        &self.buffer_slot(buffer_id).data
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
        let buffer_id = self.alloc_buffer(data);
        let layout = NodeLayout {
            buffer_id,
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
        };
        let grad = vec![0.0; numel(&shape)];

        let tape = self.active_tape_mut();
        let idx = tape.nodes.len();
        tape.nodes.push(TempNode {
            id,
            generation,
            layout,
            grad,
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
        let buffer_id = self.alloc_buffer(data);
        let layout = NodeLayout {
            buffer_id,
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
        };
        let grad = vec![0.0; numel(&shape)];

        self.params.push(ParamNode { id, layout, grad });
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

        let buffer_id = self.alloc_buffer(data);
        let layout = NodeLayout {
            buffer_id,
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
        };
        let grad = if recording {
            vec![0.0; numel(&shape)]
        } else {
            Vec::new()
        };
        let tape = self.active_tape_mut();
        let idx = tape.nodes.len();
        tape.nodes.push(TempNode {
            id,
            generation,
            layout,
            grad,
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

    pub(super) fn create_transpose_view(
        &mut self,
        input: Tensor,
        dim0: usize,
        dim1: usize,
    ) -> Tensor {
        self.validate_parent_for_active_tape(input);
        let generation = self.active_generation();
        let recording = self.active_kind().recording();
        let id = self.alloc_id();

        let mut layout = self.layout_of(input);
        let rank = layout.shape.len();
        assert!(
            dim0 < rank && dim1 < rank,
            "transpose dims out of bounds: dim0={}, dim1={}, rank={}",
            dim0,
            dim1,
            rank
        );
        layout.shape.swap(dim0, dim1);
        layout.strides.swap(dim0, dim1);
        self.retain_buffer(layout.buffer_id);

        let (op, parents) = if recording {
            (Some(Op::Transpose { dim0, dim1 }), Parents::Unary(input))
        } else {
            (None, Parents::None)
        };

        let grad = if recording {
            vec![0.0; numel(&layout.shape)]
        } else {
            Vec::new()
        };
        let tape = self.active_tape_mut();
        let idx = tape.nodes.len();
        tape.nodes.push(TempNode {
            id,
            generation,
            layout,
            grad,
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
        let layout = self.layout_of(value);
        self.read_layout_data(&layout)
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
        self.layout_of(value).shape
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
        let layout = self.layout_of(value);
        self.write_layout_data(&layout, data);
    }

    pub(super) fn sgd_step(&mut self, value: Tensor, lr: f32) {
        let layout = self.layout_of(value);
        let grad = self.grad_of(value);
        let shape = layout.shape.clone();
        assert_eq!(grad.len(), numel(&shape), "sgd_step grad/shape mismatch");

        let slot = self.buffer_slot_mut(layout.buffer_id);
        if layout.offset == 0
            && Self::is_standard_contiguous(&shape, &layout.strides)
            && slot.data.len() == grad.len()
        {
            for (d, g) in slot.data.iter_mut().zip(grad.iter()) {
                *d -= lr * *g;
            }
            return;
        }

        let strides = layout.strides.clone();
        let offset = layout.offset;
        let mut gi = 0usize;
        for_each_index(&shape, |coords| {
            let di = offset + offset_from_coords(coords, &strides);
            slot.data[di] -= lr * grad[gi];
            gi += 1;
        });
    }

    pub(super) fn set_grad(&mut self, value: Tensor, grad: &[f32]) {
        match value.handle {
            Handle::Param { idx } => {
                let shape = self.param(idx).layout.shape.clone();
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
                let shape = self.temp(generation, idx, id).layout.shape.clone();
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

    fn op_of(&self, value: Tensor) -> Option<Op> {
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
        let out_layout = self.layout_of(node);
        let out_shape = out_layout.shape.clone();

        match (op, parents) {
            (Some(Op::MatMul), Parents::Binary(a, b)) => {
                let a_layout = self.layout_of(a);
                let b_layout = self.layout_of(b);
                let a_shape = a_layout.shape.clone();
                let b_shape = b_layout.shape.clone();
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

                let a_data = self.buffer_of(a_layout.buffer_id);
                let b_data = self.buffer_of(b_layout.buffer_id);
                let a_batch_strides = broadcast_strides_for(
                    a_batch,
                    &a_layout.strides[..a_batch.len()],
                    &batch_shape,
                );
                let b_batch_strides = broadcast_strides_for(
                    b_batch,
                    &b_layout.strides[..b_batch.len()],
                    &batch_shape,
                );
                let batch_strides = contiguous_strides(&batch_shape);

                let a_block = m * k;
                let b_block = k * n;
                let out_block = m * n;
                let mut da_total = vec![0.0; numel(&a_shape)];
                let mut db_total = vec![0.0; numel(&b_shape)];
                let a_grad_strides = contiguous_strides(&a_shape);
                let b_grad_strides = contiguous_strides(&b_shape);
                let a_grad_batch_strides =
                    broadcast_strides_for(a_batch, &a_grad_strides[..a_batch.len()], &batch_shape);
                let b_grad_batch_strides =
                    broadcast_strides_for(b_batch, &b_grad_strides[..b_batch.len()], &batch_shape);

                let d_out_ref = MatRef {
                    rows: m,
                    cols: n,
                    row_stride: n,
                    col_stride: 1,
                    offset: 0,
                };

                for_each_index(&batch_shape, |batch_coords| {
                    let a_off =
                        a_layout.offset + offset_from_coords(batch_coords, &a_batch_strides);
                    let b_off =
                        b_layout.offset + offset_from_coords(batch_coords, &b_batch_strides);
                    let batch_off = offset_from_coords(batch_coords, &batch_strides);
                    let a_grad_off = offset_from_coords(batch_coords, &a_grad_batch_strides);
                    let b_grad_off = offset_from_coords(batch_coords, &b_grad_batch_strides);
                    let out_off = batch_off * out_block;

                    let g_block = &out_grad[out_off..out_off + out_block];
                    let da_block = matmul(
                        g_block,
                        d_out_ref,
                        b_data,
                        MatRef {
                            rows: n,
                            cols: k,
                            row_stride: b_layout.strides[b_layout.strides.len() - 1],
                            col_stride: b_layout.strides[b_layout.strides.len() - 2],
                            offset: b_off,
                        },
                    );
                    let db_block = matmul(
                        a_data,
                        MatRef {
                            rows: k,
                            cols: m,
                            row_stride: a_layout.strides[a_layout.strides.len() - 1],
                            col_stride: a_layout.strides[a_layout.strides.len() - 2],
                            offset: a_off,
                        },
                        g_block,
                        d_out_ref,
                    );

                    for i in 0..a_block {
                        da_total[a_grad_off + i] += da_block[i];
                    }
                    for i in 0..b_block {
                        db_total[b_grad_off + i] += db_block[i];
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
                let a_layout = self.layout_of(a);
                let b_layout = self.layout_of(b);
                let a_shape = a_layout.shape.clone();
                let b_shape = b_layout.shape.clone();
                let expected_out_shape = broadcast_shape(&a_shape, &b_shape);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "mul backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let a_data = self.buffer_of(a_layout.buffer_id);
                let b_data = self.buffer_of(b_layout.buffer_id);
                let a_grad_strides = contiguous_strides(&a_shape);
                let b_grad_strides = contiguous_strides(&b_shape);
                let a_grad_bstrides = broadcast_strides_for(&a_shape, &a_grad_strides, &out_shape);
                let b_grad_bstrides = broadcast_strides_for(&b_shape, &b_grad_strides, &out_shape);
                let a_phys_bstrides =
                    broadcast_strides_for(&a_shape, &a_layout.strides, &out_shape);
                let b_phys_bstrides =
                    broadcast_strides_for(&b_shape, &b_layout.strides, &out_shape);
                let mut da = vec![0.0; numel(&a_shape)];
                let mut db = vec![0.0; numel(&b_shape)];

                let mut out_i = 0usize;
                for_each_index(&out_shape, |coords| {
                    let ai = offset_from_coords(coords, &a_grad_bstrides);
                    let bi = offset_from_coords(coords, &b_grad_bstrides);
                    let ai_phys = a_layout.offset + offset_from_coords(coords, &a_phys_bstrides);
                    let bi_phys = b_layout.offset + offset_from_coords(coords, &b_phys_bstrides);
                    let g = out_grad[out_i];
                    da[ai] += g * b_data[bi_phys];
                    db[bi] += g * a_data[ai_phys];
                    out_i += 1;
                });

                self.add_grad(a, &da);
                self.add_grad(b, &db);
            }
            (Some(Op::Div), Parents::Binary(a, b)) => {
                let a_layout = self.layout_of(a);
                let b_layout = self.layout_of(b);
                let a_shape = a_layout.shape.clone();
                let b_shape = b_layout.shape.clone();
                let expected_out_shape = broadcast_shape(&a_shape, &b_shape);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "div backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let a_data = self.buffer_of(a_layout.buffer_id);
                let b_data = self.buffer_of(b_layout.buffer_id);
                let a_grad_strides = contiguous_strides(&a_shape);
                let b_grad_strides = contiguous_strides(&b_shape);
                let a_grad_bstrides = broadcast_strides_for(&a_shape, &a_grad_strides, &out_shape);
                let b_grad_bstrides = broadcast_strides_for(&b_shape, &b_grad_strides, &out_shape);
                let a_phys_bstrides =
                    broadcast_strides_for(&a_shape, &a_layout.strides, &out_shape);
                let b_phys_bstrides =
                    broadcast_strides_for(&b_shape, &b_layout.strides, &out_shape);
                let mut da = vec![0.0; numel(&a_shape)];
                let mut db = vec![0.0; numel(&b_shape)];

                let mut out_i = 0usize;
                for_each_index(&out_shape, |coords| {
                    let ai = offset_from_coords(coords, &a_grad_bstrides);
                    let bi = offset_from_coords(coords, &b_grad_bstrides);
                    let ai_phys = a_layout.offset + offset_from_coords(coords, &a_phys_bstrides);
                    let bi_phys = b_layout.offset + offset_from_coords(coords, &b_phys_bstrides);
                    let g = out_grad[out_i];
                    let denom = b_data[bi_phys];
                    da[ai] += g / denom;
                    db[bi] += -g * a_data[ai_phys] / (denom * denom);
                    out_i += 1;
                });

                self.add_grad(a, &da);
                self.add_grad(b, &db);
            }
            (Some(Op::Exp), Parents::Unary(a)) => {
                let out_data = self.buffer_of(out_layout.buffer_id);
                let mut da = vec![0.0; out_grad.len()];
                let mut i = 0usize;
                for_each_index(&out_shape, |coords| {
                    let out_i = out_layout.offset + offset_from_coords(coords, &out_layout.strides);
                    da[i] = out_grad[i] * out_data[out_i];
                    i += 1;
                });
                self.add_grad(a, &da);
            }
            (Some(Op::Log), Parents::Unary(a)) => {
                let a_layout = self.layout_of(a);
                let a_data = self.buffer_of(a_layout.buffer_id);
                assert_eq!(
                    a_layout.shape, out_shape,
                    "log backward shape mismatch: input={:?}, output={:?}",
                    a_layout.shape, out_shape
                );
                let mut da = vec![0.0; out_grad.len()];
                let mut i = 0usize;
                for_each_index(&out_shape, |coords| {
                    let a_i = a_layout.offset + offset_from_coords(coords, &a_layout.strides);
                    da[i] = out_grad[i] / a_data[a_i];
                    i += 1;
                });
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
                let a_layout = self.layout_of(a);
                let a_shape = a_layout.shape.clone();
                let expected_out_shape = reduced_shape(&a_shape, axis, keepdim);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "max backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let out_grad_strides = contiguous_strides(&out_shape);
                let a_data = self.buffer_of(a_layout.buffer_id);
                let out_data = self.buffer_of(out_layout.buffer_id);
                let a_grad_strides = contiguous_strides(&a_shape);
                let mut counts = vec![0usize; out_grad.len()];

                for_each_index(&a_shape, |coords| {
                    let out_grad_i = reduced_offset_from_input_coords(
                        coords,
                        &out_shape,
                        &out_grad_strides,
                        axis,
                        keepdim,
                    );
                    let out_data_i = reduced_offset_from_input_coords(
                        coords,
                        &out_shape,
                        &out_layout.strides,
                        axis,
                        keepdim,
                    ) + out_layout.offset;
                    let a_i = a_layout.offset + offset_from_coords(coords, &a_layout.strides);
                    if a_data[a_i] == out_data[out_data_i] {
                        counts[out_grad_i] += 1;
                    }
                });

                let mut da = vec![0.0; numel(&a_shape)];
                for_each_index(&a_shape, |coords| {
                    let out_grad_i = reduced_offset_from_input_coords(
                        coords,
                        &out_shape,
                        &out_grad_strides,
                        axis,
                        keepdim,
                    );
                    let out_data_i = reduced_offset_from_input_coords(
                        coords,
                        &out_shape,
                        &out_layout.strides,
                        axis,
                        keepdim,
                    ) + out_layout.offset;
                    let a_i = a_layout.offset + offset_from_coords(coords, &a_layout.strides);
                    if a_data[a_i] == out_data[out_data_i] {
                        let count = counts[out_grad_i];
                        assert!(count > 0, "max backward encountered zero tie count");
                        let a_grad_i = offset_from_coords(coords, &a_grad_strides);
                        da[a_grad_i] += out_grad[out_grad_i] / count as f32;
                    }
                });

                self.add_grad(a, &da);
            }
            (Some(Op::Transpose { dim0, dim1 }), Parents::Unary(a)) => {
                let a_shape = self.shape_of(a);
                let mut expected_out_shape = a_shape.clone();
                expected_out_shape.swap(dim0, dim1);
                assert_eq!(
                    out_shape, expected_out_shape,
                    "transpose backward output shape mismatch: expected {:?}, got {:?}",
                    expected_out_shape, out_shape
                );

                let out_strides = contiguous_strides(&out_shape);
                let mut da = vec![0.0; numel(&a_shape)];
                let mut a_i = 0usize;
                let mut out_coords = vec![0usize; a_shape.len()];
                for_each_index(&a_shape, |coords| {
                    out_coords.copy_from_slice(coords);
                    out_coords.swap(dim0, dim1);
                    let out_i = offset_from_coords(&out_coords, &out_strides);
                    da[a_i] = out_grad[out_i];
                    a_i += 1;
                });
                self.add_grad(a, &da);
            }
            (Some(Op::Relu), Parents::Unary(a)) => {
                let a_layout = self.layout_of(a);
                let a_data = self.buffer_of(a_layout.buffer_id);
                assert_eq!(
                    a_layout.shape, out_shape,
                    "relu backward shape mismatch: input={:?}, output={:?}",
                    a_layout.shape, out_shape
                );
                let mut da = vec![0.0; out_grad.len()];
                let mut i = 0usize;
                for_each_index(&out_shape, |coords| {
                    let a_i = a_layout.offset + offset_from_coords(coords, &a_layout.strides);
                    da[i] = if a_data[a_i] > 0.0 { out_grad[i] } else { 0.0 };
                    i += 1;
                });
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
}
