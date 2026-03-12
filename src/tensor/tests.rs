use super::Tensor;

#[test]
fn transpose_remains_a_noncontiguous_view() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let xt = x.transpose(0, 1);

    assert!(!x.expect_concrete("test").transpose(0, 1).is_contiguous());
    assert!(!xt.expect_concrete("test").is_contiguous());
}

#[test]
fn broadcasted_elementwise_ops_return_contiguous_outputs() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = Tensor::from_vec(vec![10.0, 20.0], vec![2]);

    assert!(x.add(&y).expect_concrete("test").is_contiguous());
    assert!(x.sub(&y).expect_concrete("test").is_contiguous());
    assert!(x.mul(&y).expect_concrete("test").is_contiguous());
    assert!(x.div(&y).expect_concrete("test").is_contiguous());
}

#[test]
fn reduction_relu_and_matmul_outputs_are_contiguous() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).transpose(0, 1);
    let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    assert!(x.sum(&[1], true).expect_concrete("test").is_contiguous());
    assert!(x.max(1, false).expect_concrete("test").is_contiguous());
    assert!(x.relu().expect_concrete("test").is_contiguous());
    assert!(x.matmul(&w).expect_concrete("test").is_contiguous());
}
