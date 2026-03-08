use micrograd::engine::{Tensor, clear_graph, no_grad, reset_state, with_grad};
use micrograd::losses::cross_entropy_with_logits;

fn assert_close(actual: f32, expected: f32, eps: f32) {
    assert!(
        (actual - expected).abs() <= eps,
        "expected {expected:.8}, got {actual:.8} (eps={eps})"
    );
}

#[test]
fn tensor_from_vec_shape_and_numel() {
    reset_state();
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(t.shape(), vec![2, 2]);
    assert_eq!(t.numel(), 4);
    assert_eq!(t.data(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matmul_and_add_bias_forward() {
    reset_state();
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let w = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![1.0, -1.0], vec![2]);
    let y = x.matmul(&w).add(&b);
    assert_eq!(y.shape(), vec![2, 2]);
    assert_eq!(y.data(), vec![20.0, 21.0, 44.0, 49.0]);
}

#[test]
fn matmul_rectangular_forward_is_correct() {
    reset_state();
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let w = Tensor::from_vec(vec![1.0, 2.0, 0.0, 1.0, 2.0, 3.0], vec![3, 2]);
    let y = x.matmul(&w);
    assert_eq!(y.shape(), vec![2, 2]);
    assert_eq!(y.data(), vec![7.0, 13.0, 16.0, 31.0]);
}

#[test]
fn transpose_view_is_metadata_only_and_set_data_writes_through() {
    reset_state();
    let x = Tensor::parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let t = x.transpose(0, 1);

    assert_eq!(t.shape(), vec![3, 2]);
    assert_eq!(t.data(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    t.set_data(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    assert_eq!(t.data(), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    assert_eq!(x.data(), vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0]);
}

#[test]
fn transpose_backward_maps_gradients_to_source_layout() {
    reset_state();
    let x = Tensor::parameter(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let w = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let loss = x.transpose(0, 1).mul(&w).mean();
    loss.backward();
    let g = x.grad();
    assert_eq!(g.len(), 4);
    assert_close(g[0], 0.25, 1e-6);
    assert_close(g[1], 0.75, 1e-6);
    assert_close(g[2], 0.50, 1e-6);
    assert_close(g[3], 1.00, 1e-6);
}

#[test]
fn matmul_accepts_transposed_view_input() {
    reset_state();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = a.transpose(0, 1).matmul(&b);
    assert_eq!(out.shape(), vec![3, 2]);
    assert_eq!(out.data(), vec![13.0, 18.0, 17.0, 24.0, 21.0, 30.0]);
}

#[test]
fn matmul_backward_mean_matches_expected_gradients() {
    reset_state();
    let a = Tensor::parameter(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::parameter(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let loss = a.matmul(&b).mean();
    loss.backward();

    let a_grad = a.grad();
    let b_grad = b.grad();

    let expected_a = [2.75, 3.75, 2.75, 3.75];
    let expected_b = [1.0, 1.0, 1.5, 1.5];

    for (actual, expected) in a_grad.iter().zip(expected_a.iter()) {
        assert_close(*actual, *expected, 1e-5);
    }
    for (actual, expected) in b_grad.iter().zip(expected_b.iter()) {
        assert_close(*actual, *expected, 1e-5);
    }
}

#[test]
fn relu_mean_backward_is_correct() {
    reset_state();
    let x = Tensor::parameter(vec![-1.0, 2.0, 3.0], vec![1, 3]);
    let loss = x.relu().mean();
    loss.backward();
    let g = x.grad();
    assert_eq!(g.len(), 3);
    assert_close(g[0], 0.0, 1e-6);
    assert_close(g[1], 1.0 / 3.0, 1e-6);
    assert_close(g[2], 1.0 / 3.0, 1e-6);
}

#[test]
fn cross_entropy_backward_row_sums_are_zero() {
    reset_state();
    let logits = Tensor::parameter(vec![1.0, 2.0, 3.0, 0.5, 1.0, -1.0], vec![2, 3]);
    let loss = cross_entropy_with_logits(&logits, &[2, 0]);
    loss.backward();
    let g = logits.grad();
    assert_eq!(g.len(), 6);
    let row0 = g[0] + g[1] + g[2];
    let row1 = g[3] + g[4] + g[5];
    assert_close(row0, 0.0, 1e-5);
    assert_close(row1, 0.0, 1e-5);
}

#[test]
fn no_grad_backward_panics() {
    reset_state();
    let result = std::panic::catch_unwind(|| {
        no_grad(|| {
            let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
            let loss = cross_entropy_with_logits(&logits, &[2]);
            loss.backward();
        });
    });
    assert!(result.is_err());
}

#[test]
fn stale_inner_tensor_panics_after_scope_exit() {
    reset_state();
    with_grad(|| {
        let inner = with_grad(|| Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]));
        let stale = std::panic::catch_unwind(|| inner.data());
        assert!(stale.is_err());
    });
}

#[test]
fn tensor_parameters_survive_clear_graph() {
    reset_state();
    let p = Tensor::parameter(vec![2.0], vec![1]);
    with_grad(|| {
        let x = Tensor::from_vec(vec![3.0], vec![1, 1]);
        let y = x.matmul(&Tensor::from_vec(vec![1.0], vec![1, 1])).mean();
        let _ = y.data();
    });
    clear_graph();
    assert_eq!(p.data(), vec![2.0]);
}

#[test]
fn broadcast_add_forward_matches_expected() {
    reset_state();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
    let out = a.add(&b);
    assert_eq!(out.shape(), vec![2, 3]);
    assert_eq!(out.data(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn broadcast_mul_backward_reduces_to_input_shapes() {
    reset_state();
    let a = Tensor::parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::parameter(vec![10.0, 20.0, 30.0], vec![3]);
    let loss = a.mul(&b).mean();
    loss.backward();

    let ga = a.grad();
    let gb = b.grad();
    let expected_a = [
        10.0 / 6.0,
        20.0 / 6.0,
        30.0 / 6.0,
        10.0 / 6.0,
        20.0 / 6.0,
        30.0 / 6.0,
    ];
    let expected_b = [5.0 / 6.0, 7.0 / 6.0, 9.0 / 6.0];

    for (actual, expected) in ga.iter().zip(expected_a.iter()) {
        assert_close(*actual, *expected, 1e-6);
    }
    for (actual, expected) in gb.iter().zip(expected_b.iter()) {
        assert_close(*actual, *expected, 1e-6);
    }
}

#[test]
fn sum_axis_shapes_and_values_are_correct() {
    reset_state();
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let s_row_keep = x.sum(1, true);
    assert_eq!(s_row_keep.shape(), vec![2, 1]);
    assert_eq!(s_row_keep.data(), vec![6.0, 15.0]);

    let s_row = x.sum(1, false);
    assert_eq!(s_row.shape(), vec![2]);
    assert_eq!(s_row.data(), vec![6.0, 15.0]);

    let s_col = x.sum(0, false);
    assert_eq!(s_col.shape(), vec![3]);
    assert_eq!(s_col.data(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn max_backward_splits_tie_gradients_evenly() {
    reset_state();
    let x = Tensor::parameter(vec![2.0, 2.0, 1.0], vec![1, 3]);
    let y = x.max(1, false);
    y.backward();
    let g = x.grad();
    assert_eq!(g.len(), 3);
    assert_close(g[0], 0.5, 1e-6);
    assert_close(g[1], 0.5, 1e-6);
    assert_close(g[2], 0.0, 1e-6);
}

#[test]
fn batched_matmul_forward_supports_batch_broadcast() {
    reset_state();
    let a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
    );
    let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]);
    let out = a.matmul(&b);
    assert_eq!(out.shape(), vec![2, 2, 2]);
    assert_eq!(
        out.data(),
        vec![4.0, 5.0, 10.0, 11.0, 16.0, 17.0, 22.0, 23.0]
    );
}

#[test]
fn batched_matmul_backward_accumulates_broadcasted_batch_grad() {
    reset_state();
    let a = Tensor::parameter(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
    let b = Tensor::parameter(vec![5.0, 6.0], vec![2, 1]);
    let loss = a.matmul(&b).mean();
    loss.backward();

    let ga = a.grad();
    let gb = b.grad();
    let expected_a = [2.5, 3.0, 2.5, 3.0];
    let expected_b = [2.0, 3.0];

    for (actual, expected) in ga.iter().zip(expected_a.iter()) {
        assert_close(*actual, *expected, 1e-6);
    }
    for (actual, expected) in gb.iter().zip(expected_b.iter()) {
        assert_close(*actual, *expected, 1e-6);
    }
}
