use super::Op;
use super::engine::{Handle, Parents};
use super::kernels::matmul_forward;
use super::shape::{
    broadcast_shape, broadcast_strides_for, contiguous_strides, for_each_index, numel,
    offset_from_coords, reduced_offset_from_input_coords, reduced_shape,
};
use super::with_engine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub(super) handle: Handle,
}

fn binary_broadcast_forward(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    f: impl Fn(f32, f32) -> f32,
) -> (Vec<f32>, Vec<usize>) {
    let out_shape = broadcast_shape(a_shape, b_shape);
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];

    let a_strides = contiguous_strides(a_shape);
    let b_strides = contiguous_strides(b_shape);
    let a_bstrides = broadcast_strides_for(a_shape, &a_strides, &out_shape);
    let b_bstrides = broadcast_strides_for(b_shape, &b_strides, &out_shape);

    let mut out_i = 0usize;
    for_each_index(&out_shape, |coords| {
        let a_idx = offset_from_coords(coords, &a_bstrides);
        let b_idx = offset_from_coords(coords, &b_bstrides);
        out[out_i] = f(a_data[a_idx], b_data[b_idx]);
        out_i += 1;
    });

    (out, out_shape)
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            numel(&shape),
            "data length ({}) must match shape numel ({})",
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
            "data length ({}) must match shape numel ({})",
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
            assert!(
                a_shape.len() >= 2 && b_shape.len() >= 2,
                "matmul expects rank >= 2: left={:?}, right={:?}",
                a_shape,
                b_shape
            );

            let m = a_shape[a_shape.len() - 2];
            let k = a_shape[a_shape.len() - 1];
            let k_b = b_shape[b_shape.len() - 2];
            let n = b_shape[b_shape.len() - 1];
            assert_eq!(
                k, k_b,
                "matmul shape mismatch: left={:?}, right={:?}",
                a_shape, b_shape
            );

            let a_batch = &a_shape[..a_shape.len() - 2];
            let b_batch = &b_shape[..b_shape.len() - 2];
            let batch_shape = broadcast_shape(a_batch, b_batch);

            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let a_strides = contiguous_strides(&a_shape);
            let b_strides = contiguous_strides(&b_shape);
            let a_batch_strides =
                broadcast_strides_for(a_batch, &a_strides[..a_batch.len()], &batch_shape);
            let b_batch_strides =
                broadcast_strides_for(b_batch, &b_strides[..b_batch.len()], &batch_shape);
            let batch_strides = contiguous_strides(&batch_shape);

            let a_block = m * k;
            let b_block = k * n;
            let out_block = m * n;
            let mut out_shape = batch_shape.clone();
            out_shape.push(m);
            out_shape.push(n);
            let mut out = vec![0.0; numel(&out_shape)];

            for_each_index(&batch_shape, |batch_coords| {
                let a_off = offset_from_coords(batch_coords, &a_batch_strides);
                let b_off = offset_from_coords(batch_coords, &b_batch_strides);
                let batch_off = offset_from_coords(batch_coords, &batch_strides);
                let out_off = batch_off * out_block;

                let a_slice = &a[a_off..a_off + a_block];
                let b_slice = &b[b_off..b_off + b_block];
                let block = matmul_forward(a_slice, b_slice, m, k, n);
                out[out_off..out_off + out_block].copy_from_slice(&block);
            });

            engine.create_from_op(out, out_shape, Op::MatMul, Parents::Binary(*self, *other))
        })
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*other);
            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let (out, out_shape) =
                binary_broadcast_forward(&a, &a_shape, &b, &b_shape, |x, y| x + y);
            engine.create_from_op(out, out_shape, Op::Add, Parents::Binary(*self, *other))
        })
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*other);
            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let (out, out_shape) =
                binary_broadcast_forward(&a, &a_shape, &b, &b_shape, |x, y| x - y);
            engine.create_from_op(out, out_shape, Op::Sub, Parents::Binary(*self, *other))
        })
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*other);
            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let (out, out_shape) =
                binary_broadcast_forward(&a, &a_shape, &b, &b_shape, |x, y| x * y);
            engine.create_from_op(out, out_shape, Op::Mul, Parents::Binary(*self, *other))
        })
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_shape = engine.shape_of(*self);
            let b_shape = engine.shape_of(*other);
            let a = engine.data_of(*self);
            let b = engine.data_of(*other);
            let (out, out_shape) =
                binary_broadcast_forward(&a, &a_shape, &b, &b_shape, |x, y| x / y);
            engine.create_from_op(out, out_shape, Op::Div, Parents::Binary(*self, *other))
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

    pub fn sum(&self, axis: usize, keepdim: bool) -> Tensor {
        with_engine(|engine| {
            let shape = engine.shape_of(*self);
            let out_shape = reduced_shape(&shape, axis, keepdim);
            let a = engine.data_of(*self);
            let out_strides = contiguous_strides(&out_shape);
            let mut out = vec![0.0; numel(&out_shape)];
            let mut in_i = 0usize;

            for_each_index(&shape, |coords| {
                let out_i = reduced_offset_from_input_coords(
                    coords,
                    &out_shape,
                    &out_strides,
                    axis,
                    keepdim,
                );
                out[out_i] += a[in_i];
                in_i += 1;
            });

            engine.create_from_op(
                out,
                out_shape,
                Op::Sum { axis, keepdim },
                Parents::Unary(*self),
            )
        })
    }

    pub fn max(&self, axis: usize, keepdim: bool) -> Tensor {
        with_engine(|engine| {
            let shape = engine.shape_of(*self);
            let out_shape = reduced_shape(&shape, axis, keepdim);
            let out_strides = contiguous_strides(&out_shape);
            let x = engine.data_of(*self);

            let mut out = vec![f32::NEG_INFINITY; numel(&out_shape)];
            let mut in_i = 0usize;

            for_each_index(&shape, |coords| {
                let out_i = reduced_offset_from_input_coords(
                    coords,
                    &out_shape,
                    &out_strides,
                    axis,
                    keepdim,
                );
                out[out_i] = out[out_i].max(x[in_i]);
                in_i += 1;
            });

            engine.create_from_op(
                out,
                out_shape,
                Op::Max { axis, keepdim },
                Parents::Unary(*self),
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
