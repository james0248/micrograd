use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_VALUE_ID: AtomicUsize = AtomicUsize::new(1);

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

#[derive(Debug)]
struct ValueInner {
    id: usize,
    data: f64,
    grad: f64,
    op: Option<Op>,
    parents: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Default for Value {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self::new_inner(data, None, Vec::new())
    }

    #[allow(dead_code)]
    pub(crate) fn from_op(data: f64, op: Op, parents: Vec<Value>) -> Self {
        Self::new_inner(data, Some(op), parents)
    }

    fn new_inner(data: f64, op: Option<Op>, parents: Vec<Value>) -> Self {
        Self(Rc::new(RefCell::new(ValueInner {
            id: NEXT_VALUE_ID.fetch_add(1, Ordering::Relaxed),
            data,
            grad: 0.0,
            op,
            parents,
        })))
    }

    pub fn id(&self) -> usize {
        self.0.borrow().id
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn add_grad(&self, delta: f64) {
        self.0.borrow_mut().grad += delta;
    }

    pub fn zero_grad(&self) {
        self.set_grad(0.0);
    }

    pub fn op(&self) -> Option<Op> {
        self.0.borrow().op
    }

    pub fn parents(&self) -> Vec<Value> {
        self.0.borrow().parents.clone()
    }

    pub fn is_leaf(&self) -> bool {
        self.0.borrow().parents.is_empty()
    }

    pub fn add(&self, other: &Value) -> Value {
        Value::from_op(
            self.data() + other.data(),
            Op::Add,
            vec![self.clone(), other.clone()],
        )
    }

    pub fn mul(&self, other: &Value) -> Value {
        Value::from_op(
            self.data() * other.data(),
            Op::Mul,
            vec![self.clone(), other.clone()],
        )
    }

    pub fn sub(&self, other: &Value) -> Value {
        Value::from_op(
            self.data() - other.data(),
            Op::Sub,
            vec![self.clone(), other.clone()],
        )
    }

    pub fn div(&self, other: &Value) -> Value {
        Value::from_op(
            self.data() / other.data(),
            Op::Div,
            vec![self.clone(), other.clone()],
        )
    }

    pub fn neg(&self) -> Value {
        Value::from_op(-self.data(), Op::Neg, vec![self.clone()])
    }

    pub fn pow(&self, other: &Value) -> Value {
        Value::from_op(
            self.data().powf(other.data()),
            Op::Pow,
            vec![self.clone(), other.clone()],
        )
    }

    pub fn tanh(&self) -> Value {
        Value::from_op(self.data().tanh(), Op::Tanh, vec![self.clone()])
    }

    pub fn exp(&self) -> Value {
        Value::from_op(self.data().exp(), Op::Exp, vec![self.clone()])
    }

    pub fn log(&self) -> Value {
        Value::from_op(self.data().ln(), Op::Log, vec![self.clone()])
    }

    pub fn relu(&self) -> Value {
        let x = self.data();
        let out = if x > 0.0 { x } else { 0.0 };
        Value::from_op(out, Op::Relu, vec![self.clone()])
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo(&mut visited, &mut topo);

        self.set_grad(1.0);
        for node in topo.into_iter().rev() {
            node.backward_step();
        }
    }

    fn build_topo(&self, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
        if !visited.insert(self.id()) {
            return;
        }

        for parent in self.parents() {
            parent.build_topo(visited, topo);
        }
        topo.push(self.clone());
    }

    fn backward_step(&self) {
        let (op, parents, out_grad, out_data) = {
            let inner = self.0.borrow();
            (inner.op, inner.parents.clone(), inner.grad, inner.data)
        };

        match op {
            Some(Op::Add) => {
                if let [a, b] = parents.as_slice() {
                    a.add_grad(out_grad);
                    b.add_grad(out_grad);
                }
            }
            Some(Op::Mul) => {
                if let [a, b] = parents.as_slice() {
                    let a_data = a.data();
                    let b_data = b.data();
                    a.add_grad(b_data * out_grad);
                    b.add_grad(a_data * out_grad);
                }
            }
            Some(Op::Sub) => {
                if let [a, b] = parents.as_slice() {
                    a.add_grad(out_grad);
                    b.add_grad(-out_grad);
                }
            }
            Some(Op::Div) => {
                if let [a, b] = parents.as_slice() {
                    let a_data = a.data();
                    let b_data = b.data();
                    a.add_grad((1.0 / b_data) * out_grad);
                    b.add_grad((-a_data / (b_data * b_data)) * out_grad);
                }
            }
            Some(Op::Neg) => {
                if let [a] = parents.as_slice() {
                    a.add_grad(-out_grad);
                }
            }
            Some(Op::Pow) => {
                if let [base, exponent] = parents.as_slice() {
                    let base_data = base.data();
                    let exponent_data = exponent.data();
                    let base_grad = exponent_data * base_data.powf(exponent_data - 1.0) * out_grad;
                    let exponent_grad = out_data * base_data.ln() * out_grad;
                    base.add_grad(base_grad);
                    exponent.add_grad(exponent_grad);
                }
            }
            Some(Op::Tanh) => {
                if let [a] = parents.as_slice() {
                    a.add_grad((1.0 - out_data * out_data) * out_grad);
                }
            }
            Some(Op::Exp) => {
                if let [a] = parents.as_slice() {
                    a.add_grad(out_data * out_grad);
                }
            }
            Some(Op::Log) => {
                if let [a] = parents.as_slice() {
                    a.add_grad((1.0 / a.data()) * out_grad);
                }
            }
            Some(Op::Relu) => {
                if let [a] = parents.as_slice() {
                    let local = if a.data() > 0.0 { 1.0 } else { 0.0 };
                    a.add_grad(local * out_grad);
                }
            }
            None => {}
        }
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
