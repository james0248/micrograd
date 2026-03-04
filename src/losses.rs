use crate::engine::Tensor;

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

    let row_max = logits.max(1, true);
    let shifted = logits.sub(&row_max);
    let lse = shifted.exp().sum(1, true).log();
    let one_hot_t = Tensor::from_vec(one_hot, vec![batch, classes]);
    let target_logits = shifted.mul(&one_hot_t).sum(1, true);
    lse.sub(&target_logits).mean()
}
