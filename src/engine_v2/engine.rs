use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use super::Op;
use super::value::Value;

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

#[derive(Debug, Clone, Copy)]
pub(super) enum Parents {
    None,
    Unary(Value),
    Binary(Value, Value),
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

#[derive(Debug, Clone, Copy)]
struct ParamNode {
    id: usize,
    data: f64,
    grad: f64,
}

#[derive(Debug, Clone, Copy)]
struct TempNode {
    id: usize,
    generation: u64,
    data: f64,
    grad: f64,
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

    fn to_value(self) -> Value {
        Value {
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

macro_rules! handle_dispatch {
    // Immutable getter: fn name(&self, value) -> ReturnType via .field
    (get $name:ident -> $ret:ty, $field:ident) => {
        pub(super) fn $name(&self, value: Value) -> $ret {
            match value.handle {
                Handle::Param { idx } => self.param(idx).$field,
                Handle::Temp {
                    generation,
                    idx,
                    id,
                } => self.temp(generation, idx, id).$field,
            }
        }
    };
    // Mutable setter: fn name(&mut self, value, val) via .field = val
    (set $name:ident ($val:ident : $vt:ty), $field:ident) => {
        pub(super) fn $name(&mut self, value: Value, $val: $vt) {
            match value.handle {
                Handle::Param { idx } => self.param_mut(idx).$field = $val,
                Handle::Temp {
                    generation,
                    idx,
                    id,
                } => self.temp_mut(generation, idx, id).$field = $val,
            }
        }
    };
    // Mutable add: fn name(&mut self, value, delta) via .field += delta
    (add $name:ident ($val:ident : $vt:ty), $field:ident) => {
        pub(super) fn $name(&mut self, value: Value, $val: $vt) {
            match value.handle {
                Handle::Param { idx } => self.param_mut(idx).$field += $val,
                Handle::Temp {
                    generation,
                    idx,
                    id,
                } => self.temp_mut(generation, idx, id).$field += $val,
            }
        }
    };
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
            .expect("engine must always keep default tape")
    }

    fn active_tape_mut(&mut self) -> &mut Tape {
        self.tapes
            .last_mut()
            .expect("engine must always keep default tape")
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
            "context stack underflow while exiting context"
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
            "clear_graph() is only allowed when no extra scope is active"
        );
        self.tapes[0].nodes.clear();
    }

    fn find_tape_checked(&self, generation: u64, idx: usize, id: usize) -> usize {
        let tape_pos = self.find_tape_pos(generation).unwrap_or_else(|| {
            panic!(
                "stale Value handle: tape generation {} is not active",
                generation
            )
        });

        let node = self.tapes[tape_pos].nodes.get(idx).unwrap_or_else(|| {
            panic!(
                "stale Value handle: invalid temp index {} for generation {}",
                idx, generation
            )
        });

        assert_eq!(
            node.generation, generation,
            "stale Value handle: generation mismatch (value={}, node={})",
            generation, node.generation
        );
        assert_eq!(
            node.id, id,
            "stale Value handle: id mismatch (value={}, node={})",
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

    fn validate_parent_for_active_tape(&self, parent: Value) {
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

    fn validate_parents_for_active_tape(&self, parents: Parents) {
        match parents {
            Parents::None => {}
            Parents::Unary(a) => self.validate_parent_for_active_tape(a),
            Parents::Binary(a, b) => {
                self.validate_parent_for_active_tape(a);
                self.validate_parent_for_active_tape(b);
            }
        }
    }

    pub(super) fn create_temp_leaf(&mut self, data: f64) -> Value {
        let generation = self.active_generation();
        let id = self.alloc_id();

        let tape = self.active_tape_mut();
        let idx = tape.nodes.len();
        tape.nodes.push(TempNode {
            id,
            generation,
            data,
            grad: 0.0,
            op: None,
            parents: Parents::None,
        });

        Value {
            handle: Handle::Temp {
                generation,
                idx,
                id,
            },
        }
    }

    pub(super) fn create_parameter(&mut self, data: f64) -> Value {
        let idx = self.params.len();
        let id = self.alloc_id();
        self.params.push(ParamNode {
            id,
            data,
            grad: 0.0,
        });
        Value {
            handle: Handle::Param { idx },
        }
    }

    pub(super) fn create_from_op(&mut self, data: f64, op: Op, parents: Parents) -> Value {
        self.validate_parents_for_active_tape(parents);

        let generation = self.active_generation();
        let recording = self.active_kind().recording();
        let id = self.alloc_id();

        let (op, parents) = if recording {
            (Some(op), parents)
        } else {
            (None, Parents::None)
        };

        let tape = self.active_tape_mut();
        let idx = tape.nodes.len();
        tape.nodes.push(TempNode {
            id,
            generation,
            data,
            grad: 0.0,
            op,
            parents,
        });

        Value {
            handle: Handle::Temp {
                generation,
                idx,
                id,
            },
        }
    }

    handle_dispatch!(get data_of -> f64, data);
    handle_dispatch!(get grad_of -> f64, grad);
    handle_dispatch!(set set_data(data: f64), data);
    handle_dispatch!(set set_grad(grad: f64), grad);
    handle_dispatch!(add add_grad(delta: f64), grad);
    handle_dispatch!(get id_of -> usize, id);

    pub(super) fn op_of(&self, value: Value) -> Option<Op> {
        match value.handle {
            Handle::Param { .. } => None,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).op,
        }
    }

    pub(super) fn parents_of(&self, value: Value) -> Parents {
        match value.handle {
            Handle::Param { .. } => Parents::None,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).parents,
        }
    }

    pub(super) fn backward_step(&mut self, node: Value) {
        let op = self.op_of(node);
        let parents = self.parents_of(node);
        let out_grad = self.grad_of(node);
        let out_data = self.data_of(node);

        match (op, parents) {
            (Some(Op::Add), Parents::Binary(a, b)) => {
                self.add_grad(a, out_grad);
                self.add_grad(b, out_grad);
            }
            (Some(Op::Mul), Parents::Binary(a, b)) => {
                let a_data = self.data_of(a);
                let b_data = self.data_of(b);
                self.add_grad(a, b_data * out_grad);
                self.add_grad(b, a_data * out_grad);
            }
            (Some(Op::Sub), Parents::Binary(a, b)) => {
                self.add_grad(a, out_grad);
                self.add_grad(b, -out_grad);
            }
            (Some(Op::Div), Parents::Binary(a, b)) => {
                let a_data = self.data_of(a);
                let b_data = self.data_of(b);
                self.add_grad(a, (1.0 / b_data) * out_grad);
                self.add_grad(b, (-a_data / (b_data * b_data)) * out_grad);
            }
            (Some(Op::Neg), Parents::Unary(a)) => {
                self.add_grad(a, -out_grad);
            }
            (Some(Op::Pow), Parents::Binary(base, exponent)) => {
                let base_data = self.data_of(base);
                let exponent_data = self.data_of(exponent);
                let base_grad = exponent_data * base_data.powf(exponent_data - 1.0) * out_grad;
                let exponent_grad = out_data * base_data.ln() * out_grad;
                self.add_grad(base, base_grad);
                self.add_grad(exponent, exponent_grad);
            }
            (Some(Op::Tanh), Parents::Unary(a)) => {
                self.add_grad(a, (1.0 - out_data * out_data) * out_grad);
            }
            (Some(Op::Exp), Parents::Unary(a)) => {
                self.add_grad(a, out_data * out_grad);
            }
            (Some(Op::Log), Parents::Unary(a)) => {
                self.add_grad(a, (1.0 / self.data_of(a)) * out_grad);
            }
            (Some(Op::Relu), Parents::Unary(a)) => {
                let local = if self.data_of(a) > 0.0 { 1.0 } else { 0.0 };
                self.add_grad(a, local * out_grad);
            }
            _ => {}
        }
    }

    pub(super) fn assert_backward_allowed(&self, root: Value) {
        if let Handle::Temp {
            generation,
            idx,
            id,
        } = root.handle
        {
            let tape_pos = self.find_tape_checked(generation, idx, id);
            assert!(
                self.tapes[tape_pos].kind.recording(),
                "cannot call backward on no_grad value from generation {}",
                generation
            );
        }
    }

    pub(super) fn collect_reachable_heap_order(&self, root: Value) -> Vec<Value> {
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        let mut heap = BinaryHeap::new();

        let Some(root_work) = TempWork::from_handle(root.handle) else {
            return order;
        };
        heap.push(root_work);
        seen.insert(root);

        let push_parent =
            |parent: Value, seen: &mut HashSet<Value>, heap: &mut BinaryHeap<TempWork>| {
                if seen.insert(parent) {
                    if let Some(w) = TempWork::from_handle(parent.handle) {
                        heap.push(w);
                    }
                }
            };

        while let Some(work) = heap.pop() {
            let node = work.to_value();
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
        self.tapes.iter().map(|t| t.nodes.len()).sum()
    }
}
