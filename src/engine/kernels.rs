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
pub(super) struct MatRef {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) row_stride: usize,
    pub(super) col_stride: usize,
    pub(super) offset: usize,
}

fn validate_bounds(data_len: usize, m: MatRef, side: &str) {
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

pub(super) fn matmul(a: &[f32], a_mat: MatRef, b: &[f32], b_mat: MatRef) -> Vec<f32> {
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
    let ops = rows
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(k))
        .unwrap_or(usize::MAX);
    if workers == 1 || ops < 256_000 {
        for i in 0..rows {
            let out_row = &mut out[i * cols..(i + 1) * cols];
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
        return out;
    }

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

#[cfg(test)]
mod tests {
    use super::{MatRef, matmul};

    fn contiguous(rows: usize, cols: usize) -> MatRef {
        MatRef {
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
            offset: 0,
        }
    }

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
        let out = matmul(&a, contiguous(2, 2), &b, contiguous(2, 2));
        close_vec(&out, &[19.0, 22.0, 43.0, 50.0], 1e-6);
    }

    #[test]
    fn matmul_backward_terms_match_hand_computation() {
        // A: [2,2], B: [2,2], dOut: [2,2]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let d_out = vec![1.0, 2.0, 3.0, 4.0];

        // dA = dOut * B^T
        let d_a = matmul(
            &d_out,
            contiguous(2, 2),
            &b,
            MatRef {
                rows: 2,
                cols: 2,
                row_stride: 1,
                col_stride: 2,
                offset: 0,
            },
        );
        close_vec(&d_a, &[17.0, 23.0, 39.0, 53.0], 1e-6);

        // dB = A^T * dOut
        let d_b = matmul(
            &a,
            MatRef {
                rows: 2,
                cols: 2,
                row_stride: 1,
                col_stride: 2,
                offset: 0,
            },
            &d_out,
            contiguous(2, 2),
        );
        close_vec(&d_b, &[10.0, 14.0, 14.0, 20.0], 1e-6);
    }

    #[test]
    fn matmul_forward_rectangular_matches_known_result() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let b = vec![1.0, 2.0, 0.0, 1.0, 2.0, 3.0]; // [3,2]
        let out = matmul(&a, contiguous(2, 3), &b, contiguous(3, 2));
        close_vec(&out, &[7.0, 13.0, 16.0, 31.0], 1e-6);
    }
}
