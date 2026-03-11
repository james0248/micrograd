use super::traced::TensorSpec;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DenseTensor {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<usize>,
}

impl DenseTensor {
    pub(crate) fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            numel(&shape),
            "data length ({}) must match shape numel ({})",
            data.len(),
            numel(&shape)
        );
        Self { data, shape }
    }

    pub(crate) fn zeros(shape: Vec<usize>) -> Self {
        let data = vec![0.0; numel(&shape)];
        Self { data, shape }
    }

    pub(crate) fn filled(shape: Vec<usize>, value: f32) -> Self {
        let data = vec![value; numel(&shape)];
        Self { data, shape }
    }

    pub(crate) fn spec(&self) -> TensorSpec {
        TensorSpec::new(self.shape.clone())
    }
}

pub(crate) fn elementwise_binary(
    lhs: &DenseTensor,
    rhs: &DenseTensor,
    op_name: &str,
    f: impl Fn(f32, f32) -> f32,
) -> DenseTensor {
    assert_eq!(
        lhs.shape, rhs.shape,
        "same-shape elementwise op {op_name} requires equal shapes, got {:?} and {:?}",
        lhs.shape, rhs.shape
    );
    let data = lhs
        .data
        .iter()
        .zip(rhs.data.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    DenseTensor::from_vec(data, lhs.shape.clone())
}

pub(crate) fn unary_map(input: &DenseTensor, f: impl Fn(f32) -> f32) -> DenseTensor {
    let data = input.data.iter().copied().map(f).collect();
    DenseTensor::from_vec(data, input.shape.clone())
}

pub(crate) fn numel(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>()
}
