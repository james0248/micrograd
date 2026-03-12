mod dense;
mod traced;

use crate::autodiff::{Operation, jvp_binary, jvp_unary};

pub(crate) use dense::{
    DenseTensor, elementwise_binary, matmul, max_axis, mean_all, relu, sum_all, sum_axis, unary_map,
};
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

    pub fn to_vec(&self) -> Vec<f32> {
        match &self.inner {
            TensorInner::Concrete(tensor) => tensor.to_vec(),
            TensorInner::Jvp(_) => {
                panic!("tangent::tensor::Tensor::to_vec() is unavailable while tracing")
            }
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.assert_broadcast_autodiff_supported(other, "add");
        if let Some(out) = jvp_binary(self, other, Operation::Add) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("add"),
            other.expect_concrete("add"),
            |x, y| x + y,
        ))
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.assert_broadcast_autodiff_supported(other, "sub");
        if let Some(out) = jvp_binary(self, other, Operation::Sub) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("sub"),
            other.expect_concrete("sub"),
            |x, y| x - y,
        ))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.assert_broadcast_autodiff_supported(other, "mul");
        if let Some(out) = jvp_binary(self, other, Operation::Mul) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("mul"),
            other.expect_concrete("mul"),
            |x, y| x * y,
        ))
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        self.assert_broadcast_autodiff_supported(other, "div");
        if let Some(out) = jvp_binary(self, other, Operation::Div) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("div"),
            other.expect_concrete("div"),
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

    pub fn relu(&self) -> Tensor {
        self.assert_autodiff_op_unavailable("relu");
        Tensor::from_concrete(relu(self.expect_concrete("relu")))
    }

    pub fn sum(&self, axis: usize, keepdim: bool) -> Tensor {
        self.assert_autodiff_op_unavailable("sum");
        Tensor::from_concrete(sum_axis(self.expect_concrete("sum"), axis, keepdim))
    }

    pub fn max(&self, axis: usize, keepdim: bool) -> Tensor {
        self.assert_autodiff_op_unavailable("max");
        Tensor::from_concrete(max_axis(self.expect_concrete("max"), axis, keepdim))
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        self.assert_binary_autodiff_op_unavailable(other, "matmul");
        Tensor::from_concrete(matmul(
            self.expect_concrete("matmul"),
            other.expect_concrete("matmul"),
        ))
    }

    pub fn sum_all(&self) -> Tensor {
        if let Some(out) = jvp_unary(self, Operation::SumAll) {
            return out;
        }
        Tensor::from_concrete(sum_all(self.expect_concrete("sum_all")))
    }

    pub fn mean_all(&self) -> Tensor {
        if let Some(out) = jvp_unary(self, Operation::MeanAll) {
            return out;
        }
        Tensor::from_concrete(mean_all(self.expect_concrete("mean_all")))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        if let Some(out) = jvp_unary(self, Operation::Transpose { dim0, dim1 }) {
            return out;
        }
        Tensor::from_concrete(self.expect_concrete("transpose").transpose(dim0, dim1))
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

    fn assert_broadcast_autodiff_supported(&self, other: &Tensor, op_name: &str) {
        if (self.as_jvp().is_some() || other.as_jvp().is_some()) && self.shape() != other.shape() {
            panic!("autodiff does not support broadcasted {op_name} yet");
        }
    }

    fn assert_autodiff_op_unavailable(&self, op_name: &str) {
        if self.as_jvp().is_some() {
            panic!("autodiff does not support {op_name} yet");
        }
    }

    fn assert_binary_autodiff_op_unavailable(&self, other: &Tensor, op_name: &str) {
        if self.as_jvp().is_some() || other.as_jvp().is_some() {
            panic!("autodiff does not support {op_name} yet");
        }
    }
}

#[cfg(test)]
mod tests;
