use crate::tensor::Tensor;

pub fn cross_entropy_with_logits(logits: &Tensor, targets: &[u8]) -> Tensor {
    let shape = logits.shape();
    assert_eq!(shape.len(), 2, "logits must be rank-2 [batch, classes]");
    let batch = shape[0];
    let classes = shape[1];
    assert!(
        batch > 0 && classes > 0,
        "logits shape must have positive dimensions, got {:?}",
        shape
    );
    assert_eq!(
        targets.len(),
        batch,
        "target length mismatch: expected {}, got {}",
        batch,
        targets.len()
    );

    let logits_data = logits.data();
    let mut row_max = vec![0.0; batch];
    for i in 0..batch {
        let row_start = i * classes;
        let row = &logits_data[row_start..row_start + classes];
        row_max[i] = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }

    let mut one_hot = vec![0.0; batch * classes];
    for (i, &target) in targets.iter().enumerate() {
        let t = target as usize;
        assert!(
            t < classes,
            "target out of range at row {}: {} >= {}",
            i,
            t,
            classes
        );
        one_hot[i * classes + t] = 1.0;
    }

    let row_max_t = Tensor::from_vec(row_max, vec![batch, 1]);
    let shifted = logits.sub_rowwise(&row_max_t);
    let lse = shifted.exp().sum_rows_keepdim().log();
    let one_hot_t = Tensor::from_vec(one_hot, vec![batch, classes]);
    let target_logits = shifted.mul(&one_hot_t).sum_rows_keepdim();
    lse.sub(&target_logits).mean()
}
