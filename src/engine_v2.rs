use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Mutex;

thread_local! {
    static ENGINE: Mutex<Engine> = Mutex::new(Engine::new());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Mul,
    Sub,
    Div,
    Neg,
    Pow,
    Tanh,
    Exp,
    Log,
    Relu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Handle {
    Param {
        idx: usize,
    },
    Temp {
        generation: u64,
        idx: usize,
        id: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value {
    handle: Handle,
}

#[derive(Debug, Clone, Copy)]
enum Parents {
    None,
    Unary(Value),
    Binary(Value, Value),
}

impl Parents {
    fn as_vec(self) -> Vec<Value> {
        match self {
            Parents::None => Vec::new(),
            Parents::Unary(a) => vec![a],
            Parents::Binary(a, b) => vec![a, b],
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextKind {
    WithGrad,
    NoGrad,
}

#[derive(Debug, Clone, Copy)]
struct ContextFrame {
    kind: ContextKind,
    generation: u64,
    mark: usize,
}

#[derive(Debug, Clone)]
struct Engine {
    next_id: usize,
    next_generation: u64,
    params: Vec<ParamNode>,
    temps: Vec<TempNode>,
    contexts: Vec<ContextFrame>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineStats {
    pub param_count: usize,
    pub temp_count: usize,
    pub generation: u64,
    pub context_depth: usize,
    pub with_grad_active: bool,
}

impl Default for Value {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Engine {
    fn new() -> Self {
        Self {
            next_id: 1,
            next_generation: 1,
            params: Vec::new(),
            temps: Vec::new(),
            contexts: Vec::new(),
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

    fn enter_with_grad(&mut self) {
        assert!(
            self.contexts.is_empty(),
            "with_grad(...) cannot be nested or entered while another context is active"
        );

        let generation = self.alloc_generation();
        let mark = self.temps.len();
        self.contexts.push(ContextFrame {
            kind: ContextKind::WithGrad,
            generation,
            mark,
        });
    }

    fn enter_no_grad(&mut self) {
        let generation = self
            .contexts
            .last()
            .map(|frame| frame.generation)
            .unwrap_or_else(|| self.alloc_generation());

        let mark = self.temps.len();
        self.contexts.push(ContextFrame {
            kind: ContextKind::NoGrad,
            generation,
            mark,
        });
    }

    fn exit_context(&mut self, expected: ContextKind) {
        let frame = self
            .contexts
            .pop()
            .expect("context stack underflow while exiting context");
        assert_eq!(
            frame.kind, expected,
            "context stack mismatch while exiting context"
        );
        self.temps.truncate(frame.mark);
    }

    fn ensure_graph_context(&self) {
        assert!(
            !self.contexts.is_empty(),
            "graph operation requires an active context; use with_grad(...) or no_grad(...)"
        );
    }

    fn current_generation(&self) -> u64 {
        self.ensure_graph_context();
        self.contexts
            .last()
            .map(|frame| frame.generation)
            .expect("graph operation requires an active context")
    }

    fn is_recording(&self) -> bool {
        matches!(
            self.contexts.last(),
            Some(ContextFrame {
                kind: ContextKind::WithGrad,
                ..
            })
        )
    }

    fn is_with_grad_active(&self) -> bool {
        self.is_recording()
    }

    fn clear_graph(&mut self) {
        assert!(
            self.contexts.is_empty(),
            "clear_graph() is only allowed when no context is active"
        );
        self.temps.clear();
    }

    fn ensure_temp_live(&self, generation: u64, idx: usize, id: usize) {
        let node = self
            .temps
            .get(idx)
            .unwrap_or_else(|| panic!("stale Value handle: invalid temp index {idx}"));

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
    }

    fn temp(&self, generation: u64, idx: usize, id: usize) -> &TempNode {
        self.ensure_temp_live(generation, idx, id);
        &self.temps[idx]
    }

    fn temp_mut(&mut self, generation: u64, idx: usize, id: usize) -> &mut TempNode {
        self.ensure_temp_live(generation, idx, id);
        &mut self.temps[idx]
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

    fn create_temp_leaf(&mut self, data: f64) -> Value {
        self.ensure_graph_context();
        let idx = self.temps.len();
        let id = self.alloc_id();

        self.temps.push(TempNode {
            id,
            generation: self.current_generation(),
            data,
            grad: 0.0,
            op: None,
            parents: Parents::None,
        });

        Value {
            handle: Handle::Temp {
                generation: self.current_generation(),
                idx,
                id,
            },
        }
    }

    fn create_parameter(&mut self, data: f64) -> Value {
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

    fn create_from_op(&mut self, data: f64, op: Op, parents: Parents) -> Value {
        self.ensure_graph_context();
        let idx = self.temps.len();
        let id = self.alloc_id();
        let generation = self.current_generation();
        let (op, parents) = if self.is_recording() {
            (Some(op), parents)
        } else {
            (None, Parents::None)
        };

        self.temps.push(TempNode {
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

    fn data_of(&self, value: Value) -> f64 {
        match value.handle {
            Handle::Param { idx } => self.param(idx).data,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).data,
        }
    }

    fn grad_of(&self, value: Value) -> f64 {
        match value.handle {
            Handle::Param { idx } => self.param(idx).grad,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).grad,
        }
    }

    fn set_data(&mut self, value: Value, data: f64) {
        match value.handle {
            Handle::Param { idx } => self.param_mut(idx).data = data,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp_mut(generation, idx, id).data = data,
        }
    }

    fn set_grad(&mut self, value: Value, grad: f64) {
        match value.handle {
            Handle::Param { idx } => self.param_mut(idx).grad = grad,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp_mut(generation, idx, id).grad = grad,
        }
    }

    fn add_grad(&mut self, value: Value, delta: f64) {
        match value.handle {
            Handle::Param { idx } => self.param_mut(idx).grad += delta,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp_mut(generation, idx, id).grad += delta,
        }
    }

    fn id_of(&self, value: Value) -> usize {
        match value.handle {
            Handle::Param { idx } => self.param(idx).id,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).id,
        }
    }

    fn op_of(&self, value: Value) -> Option<Op> {
        match value.handle {
            Handle::Param { .. } => None,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).op,
        }
    }

    fn parents_of(&self, value: Value) -> Parents {
        match value.handle {
            Handle::Param { .. } => Parents::None,
            Handle::Temp {
                generation,
                idx,
                id,
            } => self.temp(generation, idx, id).parents,
        }
    }

    fn backward_step(&mut self, node: Value) {
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

    fn assert_backward_allowed(&self, root: Value) {
        assert!(
            self.is_with_grad_active(),
            "backward requires an active with_grad(...) context"
        );

        if let Handle::Temp {
            generation,
            idx,
            id,
        } = root.handle
        {
            self.ensure_temp_live(generation, idx, id);
            assert_eq!(
                generation,
                self.current_generation(),
                "cannot call backward on a value from a different context generation"
            );
        }
    }

    fn build_topo(&self, node: Value, visited: &mut HashSet<Value>, topo: &mut Vec<Value>) {
        if !visited.insert(node) {
            return;
        }

        match self.parents_of(node) {
            Parents::None => {}
            Parents::Unary(a) => self.build_topo(a, visited, topo),
            Parents::Binary(a, b) => {
                self.build_topo(a, visited, topo);
                self.build_topo(b, visited, topo);
            }
        }

        topo.push(node);
    }
}

fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> R {
    ENGINE.with(|engine| {
        let mut guard = engine.lock().unwrap_or_else(|poison| poison.into_inner());
        f(&mut guard)
    })
}

struct ContextGuard {
    kind: ContextKind,
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        with_engine(|engine| engine.exit_context(self.kind));
    }
}

pub fn with_grad<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    with_engine(|engine| engine.enter_with_grad());
    let guard = ContextGuard {
        kind: ContextKind::WithGrad,
    };
    let out = f();
    drop(guard);
    out
}

pub fn no_grad<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    with_engine(|engine| engine.enter_no_grad());
    let guard = ContextGuard {
        kind: ContextKind::NoGrad,
    };
    let out = f();
    drop(guard);
    out
}

pub fn clear_graph() {
    with_engine(|engine| engine.clear_graph());
}

pub fn stats() -> EngineStats {
    with_engine(|engine| EngineStats {
        param_count: engine.params.len(),
        temp_count: engine.temps.len(),
        generation: engine.next_generation.saturating_sub(1),
        context_depth: engine.contexts.len(),
        with_grad_active: engine.is_with_grad_active(),
    })
}

pub fn reset_state() {
    with_engine(|engine| {
        *engine = Engine::new();
    });
}

impl Value {
    pub fn new(data: f64) -> Self {
        with_engine(|engine| engine.create_temp_leaf(data))
    }

    pub fn parameter(data: f64) -> Self {
        with_engine(|engine| engine.create_parameter(data))
    }

    pub fn id(&self) -> usize {
        with_engine(|engine| engine.id_of(*self))
    }

    pub fn data(&self) -> f64 {
        with_engine(|engine| engine.data_of(*self))
    }

    pub fn grad(&self) -> f64 {
        with_engine(|engine| engine.grad_of(*self))
    }

    pub fn set_data(&self, data: f64) {
        with_engine(|engine| engine.set_data(*self, data));
    }

    pub fn set_grad(&self, grad: f64) {
        with_engine(|engine| engine.set_grad(*self, grad));
    }

    pub fn add_grad(&self, delta: f64) {
        with_engine(|engine| engine.add_grad(*self, delta));
    }

    pub fn zero_grad(&self) {
        self.set_grad(0.0);
    }

    pub fn op(&self) -> Option<Op> {
        with_engine(|engine| engine.op_of(*self))
    }

    pub fn parents(&self) -> Vec<Value> {
        with_engine(|engine| engine.parents_of(*self).as_vec())
    }

    pub fn is_leaf(&self) -> bool {
        self.parents().is_empty()
    }

    pub fn add(&self, other: &Value) -> Value {
        with_engine(|engine| {
            let out = engine.data_of(*self) + engine.data_of(*other);
            engine.create_from_op(out, Op::Add, Parents::Binary(*self, *other))
        })
    }

    pub fn mul(&self, other: &Value) -> Value {
        with_engine(|engine| {
            let out = engine.data_of(*self) * engine.data_of(*other);
            engine.create_from_op(out, Op::Mul, Parents::Binary(*self, *other))
        })
    }

    pub fn sub(&self, other: &Value) -> Value {
        with_engine(|engine| {
            let out = engine.data_of(*self) - engine.data_of(*other);
            engine.create_from_op(out, Op::Sub, Parents::Binary(*self, *other))
        })
    }

    pub fn div(&self, other: &Value) -> Value {
        with_engine(|engine| {
            let out = engine.data_of(*self) / engine.data_of(*other);
            engine.create_from_op(out, Op::Div, Parents::Binary(*self, *other))
        })
    }

    pub fn neg(&self) -> Value {
        with_engine(|engine| {
            let out = -engine.data_of(*self);
            engine.create_from_op(out, Op::Neg, Parents::Unary(*self))
        })
    }

    pub fn pow(&self, other: &Value) -> Value {
        with_engine(|engine| {
            let out = engine.data_of(*self).powf(engine.data_of(*other));
            engine.create_from_op(out, Op::Pow, Parents::Binary(*self, *other))
        })
    }

    pub fn tanh(&self) -> Value {
        with_engine(|engine| {
            let out = engine.data_of(*self).tanh();
            engine.create_from_op(out, Op::Tanh, Parents::Unary(*self))
        })
    }

    pub fn exp(&self) -> Value {
        with_engine(|engine| {
            let out = engine.data_of(*self).exp();
            engine.create_from_op(out, Op::Exp, Parents::Unary(*self))
        })
    }

    pub fn log(&self) -> Value {
        with_engine(|engine| {
            let out = engine.data_of(*self).ln();
            engine.create_from_op(out, Op::Log, Parents::Unary(*self))
        })
    }

    pub fn relu(&self) -> Value {
        with_engine(|engine| {
            let x = engine.data_of(*self);
            let out = if x > 0.0 { x } else { 0.0 };
            engine.create_from_op(out, Op::Relu, Parents::Unary(*self))
        })
    }

    pub fn backward(&self) {
        self.backward_with_options(false);
    }

    pub fn backward_with_options(&self, retain_graph: bool) {
        with_engine(|engine| {
            let _ = retain_graph;
            engine.assert_backward_allowed(*self);

            let mut topo = Vec::new();
            let mut visited = HashSet::new();
            engine.build_topo(*self, &mut visited, &mut topo);

            engine.set_grad(*self, 1.0);
            for node in topo.into_iter().rev() {
                engine.backward_step(node);
            }
        });
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value::add(&self, &rhs)
    }
}

impl Add for &Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value::add(self, rhs)
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        Value::sub(&self, &rhs)
    }
}

impl Sub for &Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        Value::sub(self, rhs)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value::mul(&self, &rhs)
    }
}

impl Mul for &Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value::mul(self, rhs)
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        Value::div(&self, &rhs)
    }
}

impl Div for &Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        Value::div(self, rhs)
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value::neg(&self)
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value::neg(self)
    }
}
