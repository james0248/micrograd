use std::sync::Arc;
use std::thread;

use super::traced::TensorSpec;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DenseTensor {
    pub(crate) storage: Arc<[f32]>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
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
        Self {
            storage: Arc::<[f32]>::from(data.into_boxed_slice()),
            strides: contiguous_strides(&shape),
            shape,
            offset: 0,
        }
    }

    pub(crate) fn zeros(shape: Vec<usize>) -> Self {
        Self::filled(shape, 0.0)
    }

    pub(crate) fn filled(shape: Vec<usize>, value: f32) -> Self {
        let data = vec![value; numel(&shape)];
        Self::from_vec(data, shape)
    }

    pub(crate) fn spec(&self) -> TensorSpec {
        TensorSpec::new(self.shape.clone())
    }

    pub(crate) fn to_vec(&self) -> Vec<f32> {
        if self.is_contiguous() {
            let len = numel(&self.shape);
            return self.storage[self.offset..self.offset + len].to_vec();
        }

        let mut out = Vec::with_capacity(numel(&self.shape));
        for_each_index(&self.shape, |coords| {
            out.push(self.value_at(coords));
        });
        out
    }

    pub(crate) fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        assert!(
            dim0 < self.shape.len() && dim1 < self.shape.len(),
            "transpose dims out of bounds: dim0={}, dim1={}, rank={}",
            dim0,
            dim1,
            self.shape.len()
        );

        if dim0 == dim1 {
            return self.clone();
        }

        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);

        Self {
            storage: self.storage.clone(),
            shape,
            strides,
            offset: self.offset,
        }
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        self.offset + numel(&self.shape) <= self.storage.len()
            && self.strides == contiguous_strides(&self.shape)
    }

    pub(crate) fn value_at(&self, coords: &[usize]) -> f32 {
        let index = self.offset + offset_from_coords(coords, &self.strides);
        self.storage[index]
    }
}

pub(crate) fn elementwise_binary(
    lhs: &DenseTensor,
    rhs: &DenseTensor,
    f: impl Fn(f32, f32) -> f32,
) -> DenseTensor {
    let out_shape = broadcast_shape(&lhs.shape, &rhs.shape);
    let lhs_bstrides = broadcast_strides_for(&lhs.shape, &lhs.strides, &out_shape);
    let rhs_bstrides = broadcast_strides_for(&rhs.shape, &rhs.strides, &out_shape);
    let mut data = Vec::with_capacity(numel(&out_shape));

    for_each_index(&out_shape, |coords| {
        let lhs_i = lhs.offset + offset_from_coords(coords, &lhs_bstrides);
        let rhs_i = rhs.offset + offset_from_coords(coords, &rhs_bstrides);
        data.push(f(lhs.storage[lhs_i], rhs.storage[rhs_i]));
    });

    DenseTensor::from_vec(data, out_shape)
}

pub(crate) fn unary_map(input: &DenseTensor, f: impl Fn(f32) -> f32) -> DenseTensor {
    let mut data = Vec::with_capacity(numel(&input.shape));
    for_each_index(&input.shape, |coords| {
        data.push(f(input.value_at(coords)));
    });
    DenseTensor::from_vec(data, input.shape.clone())
}

pub(crate) fn relu(input: &DenseTensor) -> DenseTensor {
    unary_map(input, |x| if x > 0.0 { x } else { 0.0 })
}

pub(crate) fn sum_all(input: &DenseTensor) -> DenseTensor {
    let mut sum = 0.0;
    for_each_index(&input.shape, |coords| {
        sum += input.value_at(coords);
    });
    DenseTensor::from_vec(vec![sum], vec![1])
}

pub(crate) fn mean_all(input: &DenseTensor) -> DenseTensor {
    let total = numel(&input.shape);
    assert!(
        total > 0,
        "mean_all requires a tensor with at least one element"
    );
    let mut sum = 0.0;
    for_each_index(&input.shape, |coords| {
        sum += input.value_at(coords);
    });
    DenseTensor::from_vec(vec![sum / total as f32], vec![1])
}

fn reduce_axis(
    input: &DenseTensor,
    axis: usize,
    keepdim: bool,
    init: f32,
    combine: impl Fn(f32, f32) -> f32,
) -> DenseTensor {
    let out_shape = reduced_shape(&input.shape, axis, keepdim);
    let out_strides = contiguous_strides(&out_shape);
    let mut out = vec![init; numel(&out_shape)];

    for_each_index(&input.shape, |coords| {
        let in_i = input.offset + offset_from_coords(coords, &input.strides);
        let out_i =
            reduced_offset_from_input_coords(coords, &out_shape, &out_strides, axis, keepdim);
        out[out_i] = combine(out[out_i], input.storage[in_i]);
    });

    DenseTensor::from_vec(out, out_shape)
}

pub(crate) fn sum_axis(input: &DenseTensor, axis: usize, keepdim: bool) -> DenseTensor {
    reduce_axis(input, axis, keepdim, 0.0, |acc, x| acc + x)
}

pub(crate) fn max_axis(input: &DenseTensor, axis: usize, keepdim: bool) -> DenseTensor {
    reduce_axis(input, axis, keepdim, f32::NEG_INFINITY, f32::max)
}

pub(crate) fn matmul(lhs: &DenseTensor, rhs: &DenseTensor) -> DenseTensor {
    let lhs_shape = &lhs.shape;
    let rhs_shape = &rhs.shape;
    assert!(
        lhs_shape.len() >= 2 && rhs_shape.len() >= 2,
        "matmul expects rank >= 2: left={:?}, right={:?}",
        lhs_shape,
        rhs_shape
    );

    let m = lhs_shape[lhs_shape.len() - 2];
    let k = lhs_shape[lhs_shape.len() - 1];
    let rhs_k = rhs_shape[rhs_shape.len() - 2];
    let n = rhs_shape[rhs_shape.len() - 1];
    assert_eq!(
        k, rhs_k,
        "matmul shape mismatch: left={:?}, right={:?}",
        lhs_shape, rhs_shape
    );

    let lhs_batch = &lhs_shape[..lhs_shape.len() - 2];
    let rhs_batch = &rhs_shape[..rhs_shape.len() - 2];
    let batch_shape = broadcast_shape(lhs_batch, rhs_batch);
    let lhs_batch_strides =
        broadcast_strides_for(lhs_batch, &lhs.strides[..lhs_batch.len()], &batch_shape);
    let rhs_batch_strides =
        broadcast_strides_for(rhs_batch, &rhs.strides[..rhs_batch.len()], &batch_shape);
    let batch_strides = contiguous_strides(&batch_shape);

    let out_block = m * n;
    let mut out_shape = batch_shape.clone();
    out_shape.push(m);
    out_shape.push(n);
    let mut out = vec![0.0; numel(&out_shape)];

    for_each_index(&batch_shape, |batch_coords| {
        let lhs_offset = lhs.offset + offset_from_coords(batch_coords, &lhs_batch_strides);
        let rhs_offset = rhs.offset + offset_from_coords(batch_coords, &rhs_batch_strides);
        let batch_offset = offset_from_coords(batch_coords, &batch_strides);
        let out_offset = batch_offset * out_block;

        let block = matmul_kernel(
            &lhs.storage,
            MatRef {
                rows: m,
                cols: k,
                row_stride: lhs.strides[lhs.strides.len() - 2],
                col_stride: lhs.strides[lhs.strides.len() - 1],
                offset: lhs_offset,
            },
            &rhs.storage,
            MatRef {
                rows: k,
                cols: n,
                row_stride: rhs.strides[rhs.strides.len() - 2],
                col_stride: rhs.strides[rhs.strides.len() - 1],
                offset: rhs_offset,
            },
        );
        out[out_offset..out_offset + out_block].copy_from_slice(&block);
    });

    DenseTensor::from_vec(out, out_shape)
}

pub(crate) fn numel(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>()
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![0; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc = acc
            .checked_mul(shape[i])
            .expect("shape product overflow while computing strides");
    }
    strides
}

pub(crate) fn offset_from_coords(coords: &[usize], strides: &[usize]) -> usize {
    debug_assert_eq!(coords.len(), strides.len());
    coords
        .iter()
        .zip(strides.iter())
        .map(|(coord, stride)| coord * stride)
        .sum::<usize>()
}

pub(crate) fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    let rank = shape.len();
    let total = numel(shape);
    if total == 0 {
        return;
    }
    if rank == 0 {
        f(&[]);
        return;
    }

    let mut coords = vec![0usize; rank];
    for _ in 0..total {
        f(&coords);
        for axis in (0..rank).rev() {
            coords[axis] += 1;
            if coords[axis] < shape[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}

pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];

    for i in 0..rank {
        let a_dim = if i >= rank - a.len() {
            a[i - (rank - a.len())]
        } else {
            1
        };
        let b_dim = if i >= rank - b.len() {
            b[i - (rank - b.len())]
        } else {
            1
        };

        assert!(
            a_dim == b_dim || a_dim == 1 || b_dim == 1,
            "broadcast shape mismatch: left={:?}, right={:?}",
            a,
            b
        );
        out[i] = a_dim.max(b_dim);
    }

    out
}

pub(crate) fn broadcast_strides_for(
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> Vec<usize> {
    assert_eq!(
        input_shape.len(),
        input_strides.len(),
        "input shape/stride rank mismatch"
    );
    assert!(
        input_shape.len() <= output_shape.len(),
        "cannot broadcast higher-rank input: input={:?}, output={:?}",
        input_shape,
        output_shape
    );

    let out_rank = output_shape.len();
    let mut out = vec![0usize; out_rank];
    let offset = out_rank - input_shape.len();

    for out_axis in 0..out_rank {
        if out_axis < offset {
            out[out_axis] = 0;
            continue;
        }

        let in_axis = out_axis - offset;
        let in_dim = input_shape[in_axis];
        let out_dim = output_shape[out_axis];
        assert!(
            in_dim == out_dim || in_dim == 1,
            "broadcast shape mismatch at axis {}: input={:?}, output={:?}",
            out_axis,
            input_shape,
            output_shape
        );
        out[out_axis] = if in_dim == 1 {
            0
        } else {
            input_strides[in_axis]
        };
    }

    out
}

pub(crate) fn reduced_shape(shape: &[usize], axis: usize, keepdim: bool) -> Vec<usize> {
    assert!(
        axis < shape.len(),
        "axis out of bounds: axis={} for shape {:?}",
        axis,
        shape
    );

    if keepdim {
        let mut out = shape.to_vec();
        out[axis] = 1;
        out
    } else if shape.len() == 1 {
        vec![1]
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, &dim)| if i == axis { None } else { Some(dim) })
            .collect()
    }
}

pub(crate) fn reduced_offset_from_input_coords(
    input_coords: &[usize],
    output_shape: &[usize],
    output_strides: &[usize],
    axis: usize,
    keepdim: bool,
) -> usize {
    if !keepdim && input_coords.len() == 1 {
        return 0;
    }

    if keepdim {
        let mut off = 0usize;
        for i in 0..input_coords.len() {
            let coord = if i == axis { 0 } else { input_coords[i] };
            off += coord * output_strides[i];
        }
        off
    } else {
        let mut off = 0usize;
        let mut out_i = 0usize;
        for (in_i, &coord) in input_coords.iter().enumerate() {
            if in_i == axis {
                continue;
            }
            off += coord * output_strides[out_i];
            out_i += 1;
        }
        debug_assert_eq!(out_i, output_shape.len());
        off
    }
}

#[derive(Debug, Clone, Copy)]
struct MatRef {
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
    offset: usize,
}

fn validate_matref_bounds(data_len: usize, mat: MatRef, side: &str) {
    if mat.rows == 0 || mat.cols == 0 {
        assert!(
            mat.offset <= data_len,
            "{side} matrix offset out of bounds: offset={}, len={}",
            mat.offset,
            data_len
        );
        return;
    }

    let row_term = (mat.rows - 1)
        .checked_mul(mat.row_stride)
        .expect("row stride overflow");
    let col_term = (mat.cols - 1)
        .checked_mul(mat.col_stride)
        .expect("column stride overflow");
    let last = mat
        .offset
        .checked_add(row_term)
        .and_then(|value| value.checked_add(col_term))
        .expect("matrix index overflow");
    assert!(
        last < data_len,
        "{side} matrix out of bounds: last_index={}, len={}",
        last,
        data_len
    );
}

fn available_threads() -> usize {
    use std::sync::OnceLock;

    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
    })
}

fn worker_count(rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    available_threads().min(rows).max(1)
}

fn matmul_rows(
    out: &mut [f32],
    lhs: &[f32],
    lhs_mat: MatRef,
    rhs: &[f32],
    rhs_mat: MatRef,
    row_start: usize,
    row_count: usize,
) {
    let k = lhs_mat.cols;
    let cols = rhs_mat.cols;

    for local_row in 0..row_count {
        let row = row_start + local_row;
        let out_row = &mut out[local_row * cols..(local_row + 1) * cols];
        let lhs_base = lhs_mat.offset + row * lhs_mat.row_stride;
        for col in 0..cols {
            let rhs_base = rhs_mat.offset + col * rhs_mat.col_stride;
            let mut acc = 0.0f32;
            for kk in 0..k {
                let lhs_i = lhs_base + kk * lhs_mat.col_stride;
                let rhs_i = rhs_base + kk * rhs_mat.row_stride;
                acc += lhs[lhs_i] * rhs[rhs_i];
            }
            out_row[col] = acc;
        }
    }
}

fn matmul_kernel(lhs: &[f32], lhs_mat: MatRef, rhs: &[f32], rhs_mat: MatRef) -> Vec<f32> {
    assert_eq!(
        lhs_mat.cols,
        rhs_mat.rows,
        "matmul inner-dimension mismatch: left={:?}, right={:?}",
        (lhs_mat.rows, lhs_mat.cols),
        (rhs_mat.rows, rhs_mat.cols)
    );
    validate_matref_bounds(lhs.len(), lhs_mat, "left");
    validate_matref_bounds(rhs.len(), rhs_mat, "right");

    let rows = lhs_mat.rows;
    let k = lhs_mat.cols;
    let cols = rhs_mat.cols;
    let mut out = vec![0.0; rows * cols];
    if rows == 0 || cols == 0 || k == 0 {
        return out;
    }

    let workers = worker_count(rows);
    let ops = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(k))
        .unwrap_or(usize::MAX);
    if workers == 1 || ops < 256_000 {
        matmul_rows(&mut out, lhs, lhs_mat, rhs, rhs_mat, 0, rows);
        return out;
    }

    let rows_per_chunk = rows.div_ceil(workers);
    thread::scope(|scope| {
        for (chunk_index, out_chunk) in out.chunks_mut(rows_per_chunk * cols).enumerate() {
            let row_start = chunk_index * rows_per_chunk;
            let chunk_rows = out_chunk.len() / cols;
            scope.spawn(move || {
                matmul_rows(out_chunk, lhs, lhs_mat, rhs, rhs_mat, row_start, chunk_rows);
            });
        }
    });

    out
}
