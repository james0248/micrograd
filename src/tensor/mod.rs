mod dense;
mod traced;

use crate::autodiff::{Operation, jvp_binary, jvp_unary};

pub(crate) use dense::{DenseTensor, elementwise_binary, unary_map};
pub(crate) use traced::{JvpTensor, TensorSpec, TracedTensor, ValueId};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TensorInner {
    Concrete(DenseTensor),
    Jvp(JvpTensor),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub(crate) inner: TensorInner,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            inner: TensorInner::Concrete(DenseTensor::from_vec(data, shape)),
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self {
            inner: TensorInner::Concrete(DenseTensor::zeros(shape)),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match &self.inner {
            TensorInner::Concrete(tensor) => &tensor.shape,
            TensorInner::Jvp(tensor) => &tensor.primal.shape,
        }
    }

    pub fn data(&self) -> &[f32] {
        match &self.inner {
            TensorInner::Concrete(tensor) => &tensor.data,
            TensorInner::Jvp(_) => {
                panic!("tangent::tensor::Tensor::data() is unavailable while tracing")
            }
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        if let Some(out) = jvp_binary(self, other, Operation::Add) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("add"),
            other.expect_concrete("add"),
            "add",
            |x, y| x + y,
        ))
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        if let Some(out) = jvp_binary(self, other, Operation::Sub) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("sub"),
            other.expect_concrete("sub"),
            "sub",
            |x, y| x - y,
        ))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        if let Some(out) = jvp_binary(self, other, Operation::Mul) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("mul"),
            other.expect_concrete("mul"),
            "mul",
            |x, y| x * y,
        ))
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        if let Some(out) = jvp_binary(self, other, Operation::Div) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("div"),
            other.expect_concrete("div"),
            "div",
            |x, y| x / y,
        ))
    }

    pub fn exp(&self) -> Tensor {
        if let Some(out) = jvp_unary(self, Operation::Exp) {
            return out;
        }
        Tensor::from_concrete(unary_map(self.expect_concrete("exp"), |x| x.exp()))
    }

    pub fn log(&self) -> Tensor {
        if let Some(out) = jvp_unary(self, Operation::Log) {
            return out;
        }
        Tensor::from_concrete(unary_map(self.expect_concrete("log"), |x| x.ln()))
    }

    pub fn sum_all(&self) -> Tensor {
        if let Some(out) = jvp_unary(self, Operation::SumAll) {
            return out;
        }
        let tensor = self.expect_concrete("sum_all");
        let sum = tensor.data.iter().copied().sum();
        Tensor::from_vec(vec![sum], vec![1])
    }

    pub fn mean_all(&self) -> Tensor {
        if let Some(out) = jvp_unary(self, Operation::MeanAll) {
            return out;
        }
        let tensor = self.expect_concrete("mean_all");
        let sum: f32 = tensor.data.iter().copied().sum();
        Tensor::from_vec(vec![sum / tensor.data.len() as f32], vec![1])
    }

    pub(crate) fn from_concrete(tensor: DenseTensor) -> Self {
        Self {
            inner: TensorInner::Concrete(tensor),
        }
    }

    pub(crate) fn from_jvp(primal: DenseTensor, tangent: TracedTensor) -> Self {
        Self {
            inner: TensorInner::Jvp(JvpTensor { primal, tangent }),
        }
    }

    pub(crate) fn expect_concrete(&self, op_name: &str) -> &DenseTensor {
        match &self.inner {
            TensorInner::Concrete(tensor) => tensor,
            TensorInner::Jvp(_) => {
                panic!(
                    "tangent::tensor::Tensor::{op_name} cannot execute eagerly on a traced tensor"
                )
            }
        }
    }

    pub(crate) fn as_jvp(&self) -> Option<&JvpTensor> {
        match &self.inner {
            TensorInner::Concrete(_) => None,
            TensorInner::Jvp(tensor) => Some(tensor),
        }
    }
}
