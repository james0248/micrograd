use std::cell::RefCell;
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

    pub fn relu(&self) -> Value {
        let x = self.data();
        let out = if x > 0.0 { x } else { 0.0 };
        Value::from_op(out, Op::Relu, vec![self.clone()])
    }
}

#[cfg(test)]
mod tests {
    use super::{Op, Value};

    fn assert_close(actual: f64, expected: f64, epsilon: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= epsilon,
            "expected {expected}, got {actual}, diff={diff}, epsilon={epsilon}"
        );
    }

    #[test]
    fn new_initializes_leaf_state() {
        let v = Value::new(3.5);
        assert_eq!(v.data(), 3.5);
        assert_eq!(v.grad(), 0.0);
        assert_eq!(v.op(), None);
        assert!(v.parents().is_empty());
        assert!(v.is_leaf());
    }

    #[test]
    fn mutators_update_state() {
        let v = Value::new(2.0);

        v.set_data(2.5);
        assert_eq!(v.data(), 2.5);

        v.set_grad(1.0);
        assert_eq!(v.grad(), 1.0);

        v.add_grad(0.75);
        assert_eq!(v.grad(), 1.75);

        v.zero_grad();
        assert_eq!(v.grad(), 0.0);
    }

    #[test]
    fn clone_shares_underlying_node() {
        let original = Value::new(4.0);
        let alias = original.clone();

        alias.add_grad(2.0);
        alias.set_data(10.0);

        assert_eq!(original.grad(), 2.0);
        assert_eq!(original.data(), 10.0);
    }

    #[test]
    fn from_op_builds_non_leaf_metadata() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let out = Value::from_op(6.0, Op::Mul, vec![a.clone(), b.clone()]);

        assert_eq!(out.data(), 6.0);
        assert_eq!(out.op(), Some(Op::Mul));
        assert!(!out.is_leaf());

        let parents = out.parents();
        assert_eq!(parents.len(), 2);
        assert_eq!(parents[0].id(), a.id());
        assert_eq!(parents[1].id(), b.id());
    }

    #[test]
    fn ids_are_unique_for_new_nodes() {
        let a = Value::new(0.0);
        let b = Value::new(0.0);
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn add_mul_sub_div_neg_forward_values_and_metadata() {
        let a = Value::new(4.0);
        let b = Value::new(2.0);

        let add = a.add(&b);
        assert_eq!(add.data(), 6.0);
        assert_eq!(add.op(), Some(Op::Add));
        assert_eq!(add.parents().len(), 2);
        assert_eq!(add.parents()[0].id(), a.id());
        assert_eq!(add.parents()[1].id(), b.id());

        let mul = a.mul(&b);
        assert_eq!(mul.data(), 8.0);
        assert_eq!(mul.op(), Some(Op::Mul));

        let sub = a.sub(&b);
        assert_eq!(sub.data(), 2.0);
        assert_eq!(sub.op(), Some(Op::Sub));

        let div = a.div(&b);
        assert_eq!(div.data(), 2.0);
        assert_eq!(div.op(), Some(Op::Div));

        let neg = a.neg();
        assert_eq!(neg.data(), -4.0);
        assert_eq!(neg.op(), Some(Op::Neg));
        assert_eq!(neg.parents().len(), 1);
        assert_eq!(neg.parents()[0].id(), a.id());
    }

    #[test]
    fn pow_is_value_based() {
        let a = Value::new(3.0);
        let exponent = Value::new(2.0);
        let out = a.pow(&exponent);

        assert_eq!(out.data(), 9.0);
        assert_eq!(out.op(), Some(Op::Pow));
        assert_eq!(out.parents().len(), 2);
        assert_eq!(out.parents()[0].id(), a.id());
        assert_eq!(out.parents()[1].id(), exponent.id());
    }

    #[test]
    fn tanh_and_exp_forward_values() {
        let a = Value::new(0.0);
        let tanh = a.tanh();
        assert_close(tanh.data(), 0.0, 1e-12);
        assert_eq!(tanh.op(), Some(Op::Tanh));

        let b = Value::new(1.0);
        let exp = b.exp();
        assert_close(exp.data(), std::f64::consts::E, 1e-12);
        assert_eq!(exp.op(), Some(Op::Exp));
    }

    #[test]
    fn relu_handles_negative_zero_and_positive() {
        let neg = Value::new(-3.0).relu();
        assert_eq!(neg.data(), 0.0);
        assert_eq!(neg.op(), Some(Op::Relu));

        let zero = Value::new(0.0).relu();
        assert_eq!(zero.data(), 0.0);
        assert_eq!(zero.op(), Some(Op::Relu));

        let pos = Value::new(2.5).relu();
        assert_eq!(pos.data(), 2.5);
        assert_eq!(pos.op(), Some(Op::Relu));
    }

    #[test]
    fn composed_expression_matches_expected_forward_value() {
        let x = Value::new(2.0);
        let y = Value::new(-3.0);
        let z = Value::new(10.0);

        let q = x.mul(&y);
        let n = q.add(&z);
        let out = n.tanh();

        assert_close(out.data(), 4.0f64.tanh(), 1e-12);
        assert_eq!(out.op(), Some(Op::Tanh));
        assert_eq!(out.parents().len(), 1);
        assert_eq!(out.parents()[0].id(), n.id());
    }
}
