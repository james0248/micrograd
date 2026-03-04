use std::ops::{Add, Div, Mul, Neg, Sub};

use super::Op;
use super::engine::{Handle, Parents};
use super::with_engine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value {
    pub(super) handle: Handle,
}

impl Default for Value {
    fn default() -> Self {
        Self::new(0.0)
    }
}

macro_rules! binary_op {
    ($op:expr, $lhs:expr, $rhs:expr, $f:expr) => {
        with_engine(|engine| {
            let out = ($f)(engine.data_of($lhs), engine.data_of($rhs));
            engine.create_from_op(out, $op, Parents::Binary($lhs, $rhs))
        })
    };
}

macro_rules! unary_op {
    ($op:expr, $val:expr, $f:expr) => {
        with_engine(|engine| {
            let out = ($f)(engine.data_of($val));
            engine.create_from_op(out, $op, Parents::Unary($val))
        })
    };
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident) => {
        impl $trait for Value {
            type Output = Value;
            fn $method(self, rhs: Self) -> Self::Output {
                Value::$method(&self, &rhs)
            }
        }
        impl $trait for &Value {
            type Output = Value;
            fn $method(self, rhs: Self) -> Self::Output {
                Value::$method(self, rhs)
            }
        }
    };
}

macro_rules! impl_unary_op {
    ($trait:ident, $method:ident) => {
        impl $trait for Value {
            type Output = Value;
            fn $method(self) -> Self::Output {
                Value::$method(&self)
            }
        }
        impl $trait for &Value {
            type Output = Value;
            fn $method(self) -> Self::Output {
                Value::$method(self)
            }
        }
    };
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
        with_engine(|engine| match engine.parents_of(*self) {
            Parents::None => Vec::new(),
            Parents::Unary(a) => vec![a],
            Parents::Binary(a, b) => vec![a, b],
        })
    }

    pub fn is_leaf(&self) -> bool {
        with_engine(|engine| matches!(engine.parents_of(*self), Parents::None))
    }

    pub fn add(&self, other: &Value) -> Value {
        binary_op!(Op::Add, *self, *other, |a, b| a + b)
    }

    pub fn mul(&self, other: &Value) -> Value {
        binary_op!(Op::Mul, *self, *other, |a, b| a * b)
    }

    pub fn sub(&self, other: &Value) -> Value {
        binary_op!(Op::Sub, *self, *other, |a, b| a - b)
    }

    pub fn div(&self, other: &Value) -> Value {
        binary_op!(Op::Div, *self, *other, |a, b| a / b)
    }

    pub fn neg(&self) -> Value {
        unary_op!(Op::Neg, *self, |x: f64| -x)
    }

    pub fn pow(&self, other: &Value) -> Value {
        binary_op!(Op::Pow, *self, *other, |a: f64, b| a.powf(b))
    }

    pub fn tanh(&self) -> Value {
        unary_op!(Op::Tanh, *self, |x: f64| x.tanh())
    }

    pub fn exp(&self) -> Value {
        unary_op!(Op::Exp, *self, |x: f64| x.exp())
    }

    pub fn log(&self) -> Value {
        unary_op!(Op::Log, *self, |x: f64| x.ln())
    }

    pub fn relu(&self) -> Value {
        unary_op!(Op::Relu, *self, |x: f64| if x > 0.0 { x } else { 0.0 })
    }

    pub fn backward(&self) {
        with_engine(|engine| {
            engine.assert_backward_allowed(*self);

            let order = engine.collect_reachable_heap_order(*self);

            engine.set_grad(*self, 1.0);
            for node in order {
                engine.backward_step(node);
            }
        });
    }

    /// Accepted for API compatibility; `_retain_graph` is not yet functional.
    pub fn backward_with_options(&self, _retain_graph: bool) {
        self.backward();
    }
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);
impl_unary_op!(Neg, neg);
