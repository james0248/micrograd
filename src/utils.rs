pub fn split_train_and_eval<T>(mut items: Vec<T>, eval_ratio: f64) -> (Vec<T>, Vec<T>) {
    assert!(
        (0.0..=1.0).contains(&eval_ratio),
        "eval_ratio must be in [0.0, 1.0]"
    );

    let eval_len = ((items.len() as f64) * eval_ratio).floor() as usize;
    let split_at = items.len().saturating_sub(eval_len);
    let eval = items.split_off(split_at);
    (items, eval)
}
