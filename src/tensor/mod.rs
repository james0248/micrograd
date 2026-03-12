mod dense;
mod traced;

use crate::autodiff::{Operation, linearize_binary, linearize_unary};

pub(crate) use dense::{
    DenseTensor, elementwise_binary, expand_to_shape, matmul, matmul_shape, max_axis,
    max_axis_weights, mean, normalize_reduction_axes, reduction_size, relu, sum, sum_to_shape,
    unary_map,
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

    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        match &self.inner {
            TensorInner::Concrete(tensor) => tensor.to_vec(),
            TensorInner::Jvp(jvp) => jvp.primal.to_vec(),
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        if let Some(out) = linearize_binary(self, other, Operation::Add) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("add"),
            other.expect_concrete("add"),
            |x, y| x + y,
        ))
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        if let Some(out) = linearize_binary(self, other, Operation::Sub) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("sub"),
            other.expect_concrete("sub"),
            |x, y| x - y,
        ))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        if let Some(out) = linearize_binary(self, other, Operation::Mul) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("mul"),
            other.expect_concrete("mul"),
            |x, y| x * y,
        ))
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        if let Some(out) = linearize_binary(self, other, Operation::Div) {
            return out;
        }
        Tensor::from_concrete(elementwise_binary(
            self.expect_concrete("div"),
            other.expect_concrete("div"),
            |x, y| x / y,
        ))
    }

    pub fn exp(&self) -> Tensor {
        if let Some(out) = linearize_unary(self, Operation::Exp) {
            return out;
        }
        Tensor::from_concrete(unary_map(self.expect_concrete("exp"), |x| x.exp()))
    }

    pub fn log(&self) -> Tensor {
        if let Some(out) = linearize_unary(self, Operation::Log) {
            return out;
        }
        Tensor::from_concrete(unary_map(self.expect_concrete("log"), |x| x.ln()))
    }

    pub fn relu(&self) -> Tensor {
        if let Some(out) = linearize_unary(self, Operation::Relu) {
            return out;
        }
        Tensor::from_concrete(relu(self.expect_concrete("relu")))
    }

    pub fn sum(&self, axes: &[usize], keepdim: bool) -> Tensor {
        let axes = normalize_reduction_axes(self.shape(), axes);
        if let Some(out) = linearize_unary(
            self,
            Operation::Sum {
                axes: axes.clone(),
                keepdim,
            },
        ) {
            return out;
        }
        Tensor::from_concrete(sum(self.expect_concrete("sum"), &axes, keepdim))
    }

    pub fn mean(&self, axes: &[usize], keepdim: bool) -> Tensor {
        let axes = normalize_reduction_axes(self.shape(), axes);
        if let Some(out) = linearize_unary(
            self,
            Operation::Mean {
                axes: axes.clone(),
                keepdim,
            },
        ) {
            return out;
        }
        Tensor::from_concrete(mean(self.expect_concrete("mean"), &axes, keepdim))
    }

    pub fn max(&self, axis: usize, keepdim: bool) -> Tensor {
        if let Some(out) = linearize_unary(self, Operation::Max { axis, keepdim }) {
            return out;
        }
        Tensor::from_concrete(max_axis(self.expect_concrete("max"), axis, keepdim))
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        if let Some(out) = linearize_binary(self, other, Operation::MatMul) {
            return out;
        }
        Tensor::from_concrete(matmul(
            self.expect_concrete("matmul"),
            other.expect_concrete("matmul"),
        ))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        if let Some(out) = linearize_unary(self, Operation::Transpose { dim0, dim1 }) {
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
}

#[cfg(test)]
mod tests;
