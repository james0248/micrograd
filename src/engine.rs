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
}

#[cfg(test)]
mod tests {
    use super::{Op, Value};

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
}
