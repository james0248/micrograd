use tangent::engine::{Tensor, reset_state};
use tangent::losses::cross_entropy_with_logits;

fn assert_close(actual: f32, expected: f32, eps: f32) {
    assert!(
        (actual - expected).abs() <= eps,
        "expected {expected:.8}, got {actual:.8} (eps={eps})"
    );
}

#[test]
fn cross_entropy_value_matches_reference() {
    reset_state();
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let loss = cross_entropy_with_logits(&logits, &[2]);
    let value = loss.data()[0];

    let max = 3.0f32;
    let sum_exp = (1.0f32 - max).exp() + (2.0f32 - max).exp() + (3.0f32 - max).exp();
    let expected = max + sum_exp.ln() - 3.0f32;
    assert_close(value, expected, 1e-5);
}

#[test]
fn cross_entropy_backward_row_sums_are_zero() {
    reset_state();
    let logits = Tensor::parameter(vec![1.0, 2.0, 3.0, 0.5, 1.0, -1.0], vec![2, 3]);
    let loss = cross_entropy_with_logits(&logits, &[2, 0]);
    loss.backward();
    let g = logits.grad();

    let row0 = g[0] + g[1] + g[2];
    let row1 = g[3] + g[4] + g[5];
    assert_close(row0, 0.0, 1e-5);
    assert_close(row1, 0.0, 1e-5);
}

#[test]
#[should_panic(expected = "target length mismatch")]
fn cross_entropy_panics_on_length_mismatch() {
    reset_state();
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 0.5, 1.0, -1.0], vec![2, 3]);
    let _ = cross_entropy_with_logits(&logits, &[2]);
}

#[test]
#[should_panic(expected = "target out of range")]
fn cross_entropy_panics_on_invalid_target() {
    reset_state();
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let _ = cross_entropy_with_logits(&logits, &[9]);
}
