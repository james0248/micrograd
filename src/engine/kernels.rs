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
struct StridedMatrix {
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
    offset: usize,
}

fn validate_bounds(data_len: usize, m: StridedMatrix, side: &str) {
    if m.rows == 0 || m.cols == 0 {
        assert!(
            m.offset <= data_len,
            "{side} matrix offset out of bounds: offset={}, len={}",
            m.offset,
            data_len
        );
        return;
    }

    let row_term = (m.rows - 1)
        .checked_mul(m.row_stride)
        .expect("row stride overflow");
    let col_term = (m.cols - 1)
        .checked_mul(m.col_stride)
        .expect("column stride overflow");
    let last = m
        .offset
        .checked_add(row_term)
        .and_then(|v| v.checked_add(col_term))
        .expect("matrix index overflow");
    assert!(
        last < data_len,
        "{side} matrix out of bounds: last_index={}, len={}",
        last,
        data_len
    );
}

fn matmul_strided(a: &[f32], a_mat: StridedMatrix, b: &[f32], b_mat: StridedMatrix) -> Vec<f32> {
    assert_eq!(
        a_mat.cols,
        b_mat.rows,
        "matmul inner-dimension mismatch: left={:?}, right={:?}",
        (a_mat.rows, a_mat.cols),
        (b_mat.rows, b_mat.cols)
    );
    validate_bounds(a.len(), a_mat, "left");
    validate_bounds(b.len(), b_mat, "right");

    let rows = a_mat.rows;
    let k = a_mat.cols;
    let cols = b_mat.cols;
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
                    let a_base = a_mat.offset + i * a_mat.row_stride;

                    for j in 0..cols {
                        let b_base = b_mat.offset + j * b_mat.col_stride;
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            let a_idx = a_base + kk * a_mat.col_stride;
                            let b_idx = b_base + kk * b_mat.row_stride;
                            acc += a[a_idx] * b[b_idx];
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
    let a_mat = StridedMatrix {
        rows: m,
        cols: k,
        row_stride: k,
        col_stride: 1,
        offset: 0,
    };
    let b_mat = StridedMatrix {
        rows: k,
        cols: n,
        row_stride: n,
        col_stride: 1,
        offset: 0,
    };
    matmul_strided(a, a_mat, b, b_mat)
}

pub(super) fn matmul_backward_da(
    out_grad: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let d_out = StridedMatrix {
        rows: m,
        cols: n,
        row_stride: n,
        col_stride: 1,
        offset: 0,
    };
    // Logical right matrix is B^T with shape [n, k], mapped into B[k, n].
    let right = StridedMatrix {
        rows: n,
        cols: k,
        row_stride: 1,
        col_stride: n,
        offset: 0,
    };
    // dA = dOut * B^T
    matmul_strided(out_grad, d_out, b, right)
}

pub(super) fn matmul_backward_db(
    a: &[f32],
    out_grad: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    // Logical left matrix is A^T with shape [k, m], mapped into A[m, k].
    let left = StridedMatrix {
        rows: k,
        cols: m,
        row_stride: 1,
        col_stride: k,
        offset: 0,
    };
    let d_out = StridedMatrix {
        rows: m,
        cols: n,
        row_stride: n,
        col_stride: 1,
        offset: 0,
    };
    // dB = A^T * dOut
    matmul_strided(a, left, out_grad, d_out)
}

#[cfg(test)]
mod tests {
    use super::{matmul_backward_da, matmul_backward_db, matmul_forward};

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
    fn matmul_forward_rectangular_matches_known_result() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let b = vec![1.0, 2.0, 0.0, 1.0, 2.0, 3.0]; // [3,2]
        let out = matmul_forward(&a, &b, 2, 3, 2);
        close_vec(&out, &[7.0, 13.0, 16.0, 31.0], 1e-6);
    }
}
