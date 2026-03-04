use std::thread;

fn idx2(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

fn worker_count(rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    let available = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    available.min(rows).max(1)
}

pub(super) fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        src.len(),
        rows * cols,
        "transpose input size mismatch: got {}, expected {}",
        src.len(),
        rows * cols
    );
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[idx2(j, i, rows)] = src[idx2(i, j, cols)];
        }
    }
    out
}

pub(super) fn matmul_with_b_transposed(
    a: &[f32],
    b_t: &[f32],
    rows: usize,
    k: usize,
    cols: usize,
) -> Vec<f32> {
    assert_eq!(
        a.len(),
        rows * k,
        "left matrix size mismatch: got {}, expected {}",
        a.len(),
        rows * k
    );
    assert_eq!(
        b_t.len(),
        cols * k,
        "transposed right matrix size mismatch: got {}, expected {}",
        b_t.len(),
        cols * k
    );

    let mut out = vec![0.0; rows * cols];
    let workers = worker_count(rows);
    let rows_per_chunk = rows.div_ceil(workers);

    thread::scope(|scope| {
        for (chunk_idx, out_chunk) in out.chunks_mut(rows_per_chunk * cols).enumerate() {
            let row_start = chunk_idx * rows_per_chunk;
            let chunk_rows = out_chunk.len() / cols;

            scope.spawn(move || {
                for local_i in 0..chunk_rows {
                    let i = row_start + local_i;
                    let a_row = &a[i * k..(i + 1) * k];
                    let out_row = &mut out_chunk[local_i * cols..(local_i + 1) * cols];

                    for j in 0..cols {
                        let b_row = &b_t[j * k..(j + 1) * k];
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            acc += a_row[kk] * b_row[kk];
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
    assert_eq!(
        b.len(),
        k * n,
        "right matrix size mismatch: got {}, expected {}",
        b.len(),
        k * n
    );
    let b_t = transpose_2d(b, k, n);
    matmul_with_b_transposed(a, &b_t, m, k, n)
}

pub(super) fn matmul_backward_da(
    out_grad: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(
        out_grad.len(),
        m * n,
        "out_grad size mismatch: got {}, expected {}",
        out_grad.len(),
        m * n
    );
    assert_eq!(
        b.len(),
        k * n,
        "right matrix size mismatch: got {}, expected {}",
        b.len(),
        k * n
    );

    // dA = dOut * B^T. Our kernel expects a transposed right matrix;
    // transpose(B^T) is B itself.
    matmul_with_b_transposed(out_grad, b, m, n, k)
}

pub(super) fn matmul_backward_db(
    a: &[f32],
    out_grad: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(
        a.len(),
        m * k,
        "left matrix size mismatch: got {}, expected {}",
        a.len(),
        m * k
    );
    assert_eq!(
        out_grad.len(),
        m * n,
        "out_grad size mismatch: got {}, expected {}",
        out_grad.len(),
        m * n
    );

    // dB = A^T * dOut
    let a_t = transpose_2d(a, m, k); // [k, m]
    let d_out_t = transpose_2d(out_grad, m, n); // [n, m]
    matmul_with_b_transposed(&a_t, &d_out_t, k, m, n)
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
}
