use super::dense::{DenseTensor, numel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DType {
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorSpec {
    pub(crate) shape: Vec<usize>,
    dtype: DType,
}

impl TensorSpec {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            dtype: DType::F32,
        }
    }

    pub(crate) fn numel(&self) -> usize {
        numel(&self.shape)
    }
}

pub(crate) type ValueId = usize;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TracedTensor {
    pub(crate) var: ValueId,
    pub(crate) spec: TensorSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct JvpTensor {
    pub(crate) primal: DenseTensor,
    pub(crate) tangent: TracedTensor,
}
