use std::thread;

fn worker_count(rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    let available = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    available.min(rows).max(1)
}

#[derive(Debug, Clone, Copy)]
struct View2D<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
    offset: usize,
}

impl<'a> View2D<'a> {
    fn contiguous(data: &'a [f32], rows: usize, cols: usize, side: &str) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "{side} matrix size mismatch: got {}, expected {}",
            data.len(),
            rows * cols
        );
        Self {
            data,
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
            offset: 0,
        }
    }

    fn transposed(self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            offset: self.offset,
        }
    }

    fn validate_bounds(self, side: &str) {
        if self.rows == 0 || self.cols == 0 {
            assert!(
                self.offset <= self.data.len(),
                "{side} view offset out of bounds: offset={}, len={}",
                self.offset,
                self.data.len()
            );
            return;
        }

        let row_term = (self.rows - 1)
            .checked_mul(self.row_stride)
            .expect("row stride overflow");
        let col_term = (self.cols - 1)
            .checked_mul(self.col_stride)
            .expect("column stride overflow");
        let last = self
            .offset
            .checked_add(row_term)
            .and_then(|v| v.checked_add(col_term))
            .expect("view index overflow");
        assert!(
            last < self.data.len(),
            "{side} view out of bounds: last_index={}, len={}",
            last,
            self.data.len()
        );
    }

    #[inline]
    fn get(self, row: usize, col: usize) -> f32 {
        debug_assert!(row < self.rows, "row out of bounds");
        debug_assert!(col < self.cols, "column out of bounds");
        let idx = self.offset + row * self.row_stride + col * self.col_stride;
        self.data[idx]
    }
}

fn matmul_views(a: View2D<'_>, b: View2D<'_>) -> Vec<f32> {
    assert_eq!(
        a.cols,
        b.rows,
        "matmul inner-dimension mismatch: left={:?}, right={:?}",
        (a.rows, a.cols),
        (b.rows, b.cols)
    );
    a.validate_bounds("left");
    b.validate_bounds("right");

    let rows = a.rows;
    let k = a.cols;
    let cols = b.cols;
    let mut out = vec![0.0; rows * cols];
    if rows == 0 || cols == 0 || k == 0 {
        return out;
    }

    let workers = worker_count(rows);
    let rows_per_chunk = rows.div_ceil(workers);

    thread::scope(|scope| {
        for (chunk_idx, out_chunk) in out.chunks_mut(rows_per_chunk * cols).enumerate() {
            let row_start = chunk_idx * rows_per_chunk;
            let chunk_rows = out_chunk.len() / cols;

            scope.spawn(move || {
                for local_i in 0..chunk_rows {
                    let i = row_start + local_i;
                    let out_row = &mut out_chunk[local_i * cols..(local_i + 1) * cols];

                    for j in 0..cols {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            acc += a.get(i, kk) * b.get(kk, j);
                        }
                        out_row[j] = acc;
                    }
                }
            });
        }
    });

    out
}

pub(super) fn matmul_forward(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let a_view = View2D::contiguous(a, m, k, "left");
    let b_view = View2D::contiguous(b, k, n, "right");
    matmul_views(a_view, b_view)
}

pub(super) fn matmul_backward_da(
    out_grad: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let d_out = View2D::contiguous(out_grad, m, n, "out_grad");
    let right = View2D::contiguous(b, k, n, "right").transposed();
    // dA = dOut * B^T
    matmul_views(d_out, right)
}

pub(super) fn matmul_backward_db(
    a: &[f32],
    out_grad: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let left = View2D::contiguous(a, m, k, "left").transposed();
    let d_out = View2D::contiguous(out_grad, m, n, "out_grad");
    // dB = A^T * dOut
    matmul_views(left, d_out)
}

#[cfg(test)]
mod tests {
    use super::{View2D, matmul_backward_da, matmul_backward_db, matmul_forward};

    fn close_vec(lhs: &[f32], rhs: &[f32], eps: f32) {
        assert_eq!(lhs.len(), rhs.len(), "length mismatch");
        for (i, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (a - b).abs() <= eps,
                "index {i}: expected {b}, got {a}, eps={eps}"
            );
        }
    }

    #[test]
    fn matmul_forward_matches_known_result() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let out = matmul_forward(&a, &b, 2, 2, 2);
        close_vec(&out, &[19.0, 22.0, 43.0, 50.0], 1e-6);
    }

    #[test]
    fn matmul_backward_terms_match_hand_computation() {
        // A: [2,2], B: [2,2], dOut: [2,2]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let d_out = vec![1.0, 2.0, 3.0, 4.0];

        // dA = dOut * B^T
        let d_a = matmul_backward_da(&d_out, &b, 2, 2, 2);
        close_vec(&d_a, &[17.0, 23.0, 39.0, 53.0], 1e-6);

        // dB = A^T * dOut
        let d_b = matmul_backward_db(&a, &d_out, 2, 2, 2);
        close_vec(&d_b, &[10.0, 14.0, 14.0, 20.0], 1e-6);
    }

    #[test]
    fn transposed_view_reads_without_copy() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = View2D::contiguous(&src, 2, 3, "test");
        let t = v.transposed();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        close_vec(
            &[
                t.get(0, 0),
                t.get(0, 1),
                t.get(1, 0),
                t.get(1, 1),
                t.get(2, 0),
                t.get(2, 1),
            ],
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            1e-6,
        );
    }

    #[test]
    fn matmul_forward_rectangular_matches_known_result() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let b = vec![1.0, 2.0, 0.0, 1.0, 2.0, 3.0]; // [3,2]
        let out = matmul_forward(&a, &b, 2, 3, 2);
        close_vec(&out, &[7.0, 13.0, 16.0, 31.0], 1e-6);
    }
}
