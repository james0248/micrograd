use super::Op;
use super::engine::{Handle, Parents};
use super::kernels::matmul_forward;
use super::with_engine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub(super) handle: Handle,
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn shape2(shape: &[usize]) -> (usize, usize) {
    assert_eq!(shape.len(), 2, "expected rank-2 tensor, got {:?}", shape);
    (shape[0], shape[1])
}

fn idx2(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            numel(&shape),
            "tensor data length ({}) must match shape numel ({})",
            data.len(),
            numel(&shape)
        );
        with_engine(|engine| engine.create_temp_leaf(data, shape))
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let data = vec![0.0; numel(&shape)];
        Self::from_vec(data, shape)
    }

    pub fn parameter(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            numel(&shape),
            "tensor data length ({}) must match shape numel ({})",
            data.len(),
            numel(&shape)
        );
        with_engine(|engine| engine.create_parameter(data, shape))
    }

    pub fn id(&self) -> usize {
        with_engine(|engine| engine.id_of(*self))
    }

    pub fn shape(&self) -> Vec<usize> {
        with_engine(|engine| engine.shape_of(*self))
    }

    pub fn numel(&self) -> usize {
        with_engine(|engine| engine.numel_of(*self))
    }

    pub fn data(&self) -> Vec<f32> {
        with_engine(|engine| engine.data_of(*self))
    }

    pub fn grad(&self) -> Vec<f32> {
        with_engine(|engine| engine.grad_of(*self))
    }

    pub fn set_data(&self, data: &[f32]) {
        with_engine(|engine| engine.set_data(*self, data));
    }

    pub fn set_grad(&self, grad: &[f32]) {
        with_engine(|engine| engine.set_grad(*self, grad));
    }

    pub fn add_grad(&self, delta: &[f32]) {
        with_engine(|engine| engine.add_grad(*self, delta));
    }

    pub fn zero_grad(&self) {
        with_engine(|engine| engine.zero_grad(*self));
    }

    pub fn is_leaf(&self) -> bool {
        with_engine(|engine| matches!(engine.parents_of(*self), Parents::None))
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*other);
            let (m, k) = shape2(&a_shape);
            let (k_b, n) = shape2(&b_shape);
            assert_eq!(
                k, k_b,
                "matmul shape mismatch: left={:?}, right={:?}",
                a_shape, b_shape
            );

            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let out = matmul_forward(&a, &b, m, k, n);

            engine.create_from_op(
                out,
                vec![m, n],
                Op::MatMul2D,
                Parents::Binary(*self, *other),
            )
        })
    }

    pub fn add_row_bias(&self, bias: &Tensor) -> Tensor {
        with_engine(|engine| {
            let x_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*bias);
            let (rows, cols) = shape2(&x_shape);
            assert_eq!(
                b_shape,
                vec![cols],
                "add_row_bias expects bias shape [{cols}], got {:?}",
                b_shape
            );

            let x = engine.data_of(*self);
            let b = engine.data_of(*bias);
            let mut out = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[idx2(i, j, cols)] = x[idx2(i, j, cols)] + b[j];
                }
            }

            engine.create_from_op(
                out,
                vec![rows, cols],
                Op::AddRowBias,
                Parents::Binary(*self, *bias),
            )
        })
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*other);
            assert_eq!(
                a_shape, b_shape,
                "add shape mismatch: left={:?}, right={:?}",
                a_shape, b_shape
            );

            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let out: Vec<f32> = a.into_iter().zip(b).map(|(x, y)| x + y).collect();
            engine.create_from_op(out, a_shape, Op::Add, Parents::Binary(*self, *other))
        })
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*other);
            assert_eq!(
                a_shape, b_shape,
                "sub shape mismatch: left={:?}, right={:?}",
                a_shape, b_shape
            );

            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let out: Vec<f32> = a.into_iter().zip(b).map(|(x, y)| x - y).collect();
            engine.create_from_op(out, a_shape, Op::Sub, Parents::Binary(*self, *other))
        })
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*other);
            assert_eq!(
                a_shape, b_shape,
                "mul shape mismatch: left={:?}, right={:?}",
                a_shape, b_shape
            );

            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let out: Vec<f32> = a.into_iter().zip(b).map(|(x, y)| x * y).collect();
            engine.create_from_op(out, a_shape, Op::Mul, Parents::Binary(*self, *other))
        })
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*other);
            assert_eq!(
                a_shape, b_shape,
                "div shape mismatch: left={:?}, right={:?}",
                a_shape, b_shape
            );

            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let out: Vec<f32> = a.into_iter().zip(b).map(|(x, y)| x / y).collect();
            engine.create_from_op(out, a_shape, Op::Div, Parents::Binary(*self, *other))
        })
    }

    pub fn exp(&self) -> Tensor {
        with_engine(|engine| {
            let shape = engine.shape_of(*self);
            let a = engine.data_of(*self);
            let out: Vec<f32> = a.into_iter().map(f32::exp).collect();
            engine.create_from_op(out, shape, Op::Exp, Parents::Unary(*self))
        })
    }

    pub fn log(&self) -> Tensor {
        with_engine(|engine| {
            let shape = engine.shape_of(*self);
            let a = engine.data_of(*self);
            let out: Vec<f32> = a.into_iter().map(f32::ln).collect();
            engine.create_from_op(out, shape, Op::Log, Parents::Unary(*self))
        })
    }

    pub fn sum_rows_keepdim(&self) -> Tensor {
        with_engine(|engine| {
            let shape = engine.shape_of(*self);
            let (rows, cols) = shape2(&shape);
            let a = engine.data_of(*self);
            let mut out = vec![0.0; rows];
            for i in 0..rows {
                let mut acc = 0.0;
                for j in 0..cols {
                    acc += a[idx2(i, j, cols)];
                }
                out[i] = acc;
            }
            engine.create_from_op(
                out,
                vec![rows, 1],
                Op::SumRowsKeepDim,
                Parents::Unary(*self),
            )
        })
    }

    pub fn sub_rowwise(&self, row_vec: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*row_vec);
            let (rows, cols) = shape2(&a_shape);
            assert_eq!(
                b_shape,
                vec![rows, 1],
                "sub_rowwise expects right shape [{rows}, 1], got {:?}",
                b_shape
            );

            let a = engine.data_of(*self);
            let b = engine.data_of(*row_vec);
            let mut out = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[idx2(i, j, cols)] = a[idx2(i, j, cols)] - b[i];
                }
            }
            engine.create_from_op(
                out,
                vec![rows, cols],
                Op::SubRowwise,
                Parents::Binary(*self, *row_vec),
            )
        })
    }

    pub fn div_rowwise(&self, row_vec: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*row_vec);
            let (rows, cols) = shape2(&a_shape);
            assert_eq!(
                b_shape,
                vec![rows, 1],
                "div_rowwise expects right shape [{rows}, 1], got {:?}",
                b_shape
            );

            let a = engine.data_of(*self);
            let b = engine.data_of(*row_vec);
            let mut out = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[idx2(i, j, cols)] = a[idx2(i, j, cols)] / b[i];
                }
            }
            engine.create_from_op(
                out,
                vec![rows, cols],
                Op::DivRowwise,
                Parents::Binary(*self, *row_vec),
            )
        })
    }

    pub fn relu(&self) -> Tensor {
        with_engine(|engine| {
            let shape = engine.shape_of(*self);
            let x = engine.data_of(*self);
            let out: Vec<f32> = x
                .into_iter()
                .map(|v| if v > 0.0 { v } else { 0.0 })
                .collect();
            engine.create_from_op(out, shape, Op::Relu, Parents::Unary(*self))
        })
    }

    pub fn mean(&self) -> Tensor {
        with_engine(|engine| {
            let x = engine.data_of(*self);
            let n = x.len();
            assert!(n > 0, "mean requires non-empty tensor");
            let sum: f32 = x.iter().sum();
            let out = vec![sum / n as f32];
            engine.create_from_op(out, vec![1], Op::Mean, Parents::Unary(*self))
        })
    }

    pub fn backward(&self) {
        with_engine(|engine| {
            engine.assert_backward_allowed(*self);
            let order = engine.collect_reachable_heap_order(*self);
            engine.set_grad(*self, &[1.0]);
            for node in order {
                engine.backward_step(node);
            }
        });
    }
}
