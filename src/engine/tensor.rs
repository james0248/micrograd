use super::Op;
use super::engine::{Handle, Parents};
use super::kernels::{MatRef, matmul};
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
    a_strides: &[usize],
    a_offset: usize,
    b_data: &[f32],
    b_shape: &[usize],
    b_strides: &[usize],
    b_offset: usize,
    f: impl Fn(f32, f32) -> f32,
) -> (Vec<f32>, Vec<usize>) {
    let out_shape = broadcast_shape(a_shape, b_shape);
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];

    let a_bstrides = broadcast_strides_for(a_shape, a_strides, &out_shape);
    let b_bstrides = broadcast_strides_for(b_shape, b_strides, &out_shape);

    let mut out_i = 0usize;
    for_each_index(&out_shape, |coords| {
        let a_idx = a_offset + offset_from_coords(coords, &a_bstrides);
        let b_idx = b_offset + offset_from_coords(coords, &b_bstrides);
        out[out_i] = f(a_data[a_idx], b_data[b_idx]);
        out_i += 1;
    });

    (out, out_shape)
}

fn binary_elementwise(a: &Tensor, b: &Tensor, op: Op, f: impl Fn(f32, f32) -> f32) -> Tensor {
    with_engine(|engine| {
        let a_layout = engine.layout_of(*a);
        let b_layout = engine.layout_of(*b);
        let a_data = engine.buffer_of(a_layout.buffer_id);
        let b_data = engine.buffer_of(b_layout.buffer_id);
        let (out, out_shape) = binary_broadcast_forward(
            a_data,
            &a_layout.shape,
            &a_layout.strides,
            a_layout.offset,
            b_data,
            &b_layout.shape,
            &b_layout.strides,
            b_layout.offset,
            f,
        );
        engine.create_from_op(out, out_shape, op, Parents::Binary(*a, *b))
    })
}

fn unary_map(a: &Tensor, op: Op, f: impl Fn(f32) -> f32) -> Tensor {
    with_engine(|engine| {
        let layout = engine.layout_of(*a);
        let data = engine.buffer_of(layout.buffer_id);
        let mut out = vec![0.0; numel(&layout.shape)];
        let mut out_i = 0usize;
        for_each_index(&layout.shape, |coords| {
            let i = layout.offset + offset_from_coords(coords, &layout.strides);
            out[out_i] = f(data[i]);
            out_i += 1;
        });
        engine.create_from_op(out, layout.shape, op, Parents::Unary(*a))
    })
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

    pub fn sgd_step(&self, lr: f32) {
        with_engine(|engine| engine.sgd_step(*self, lr));
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

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        with_engine(|engine| {
            let a_layout = engine.layout_of(*self);
            let b_layout = engine.layout_of(*other);
            let a_shape = a_layout.shape.clone();
            let b_shape = b_layout.shape.clone();
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

            let a = engine.buffer_of(a_layout.buffer_id);
            let b = engine.buffer_of(b_layout.buffer_id);
            let a_batch_strides =
                broadcast_strides_for(a_batch, &a_layout.strides[..a_batch.len()], &batch_shape);
            let b_batch_strides =
                broadcast_strides_for(b_batch, &b_layout.strides[..b_batch.len()], &batch_shape);
            let batch_strides = contiguous_strides(&batch_shape);

            let out_block = m * n;
            let mut out_shape = batch_shape.clone();
            out_shape.push(m);
            out_shape.push(n);
            let mut out = vec![0.0; numel(&out_shape)];

            for_each_index(&batch_shape, |batch_coords| {
                let a_off = a_layout.offset + offset_from_coords(batch_coords, &a_batch_strides);
                let b_off = b_layout.offset + offset_from_coords(batch_coords, &b_batch_strides);
                let batch_off = offset_from_coords(batch_coords, &batch_strides);
                let out_off = batch_off * out_block;

                let block = matmul(
                    a,
                    MatRef {
                        rows: m,
                        cols: k,
                        row_stride: a_layout.strides[a_layout.strides.len() - 2],
                        col_stride: a_layout.strides[a_layout.strides.len() - 1],
                        offset: a_off,
                    },
                    b,
                    MatRef {
                        rows: k,
                        cols: n,
                        row_stride: b_layout.strides[b_layout.strides.len() - 2],
                        col_stride: b_layout.strides[b_layout.strides.len() - 1],
                        offset: b_off,
                    },
                );
                out[out_off..out_off + out_block].copy_from_slice(&block);
            });

            engine.create_from_op(out, out_shape, Op::MatMul, Parents::Binary(*self, *other))
        })
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        binary_elementwise(self, other, Op::Add, |x, y| x + y)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        binary_elementwise(self, other, Op::Sub, |x, y| x - y)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        binary_elementwise(self, other, Op::Mul, |x, y| x * y)
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        binary_elementwise(self, other, Op::Div, |x, y| x / y)
    }

    pub fn exp(&self) -> Tensor {
        unary_map(self, Op::Exp, f32::exp)
    }

    pub fn log(&self) -> Tensor {
        unary_map(self, Op::Log, f32::ln)
    }

    pub fn sum(&self, axis: usize, keepdim: bool) -> Tensor {
        with_engine(|engine| {
            let layout = engine.layout_of(*self);
            let shape = layout.shape.clone();
            let out_shape = reduced_shape(&shape, axis, keepdim);
            let a = engine.buffer_of(layout.buffer_id);
            let out_strides = contiguous_strides(&out_shape);
            let mut out = vec![0.0; numel(&out_shape)];
            for_each_index(&shape, |coords| {
                let in_i = layout.offset + offset_from_coords(coords, &layout.strides);
                let out_i = reduced_offset_from_input_coords(
                    coords,
                    &out_shape,
                    &out_strides,
                    axis,
                    keepdim,
                );
                out[out_i] += a[in_i];
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
            let layout = engine.layout_of(*self);
            let shape = layout.shape.clone();
            let out_shape = reduced_shape(&shape, axis, keepdim);
            let out_strides = contiguous_strides(&out_shape);
            let x = engine.buffer_of(layout.buffer_id);

            let mut out = vec![f32::NEG_INFINITY; numel(&out_shape)];

            for_each_index(&shape, |coords| {
                let in_i = layout.offset + offset_from_coords(coords, &layout.strides);
                let out_i = reduced_offset_from_input_coords(
                    coords,
                    &out_shape,
                    &out_strides,
                    axis,
                    keepdim,
                );
                out[out_i] = out[out_i].max(x[in_i]);
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
        unary_map(self, Op::Relu, |v| if v > 0.0 { v } else { 0.0 })
    }

    pub fn mean(&self) -> Tensor {
        with_engine(|engine| {
            let layout = engine.layout_of(*self);
            let x = engine.buffer_of(layout.buffer_id);
            let n = numel(&layout.shape);
            assert!(n > 0, "mean requires non-empty tensor");
            let mut sum = 0.0f32;
            for_each_index(&layout.shape, |coords| {
                let i = layout.offset + offset_from_coords(coords, &layout.strides);
                sum += x[i];
            });
            let out = vec![sum / n as f32];
            engine.create_from_op(out, vec![1], Op::Mean, Parents::Unary(*self))
        })
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        with_engine(|engine| engine.create_transpose_view(*self, dim0, dim1))
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
