use micrograd::utils::split_train_and_eval;

#[test]
fn split_train_and_eval_keeps_prefix_for_train_and_suffix_for_eval() {
    let data = vec![1, 2, 3, 4, 5];
    let (train, eval) = split_train_and_eval(data, 0.4);
    assert_eq!(train, vec![1, 2, 3]);
    assert_eq!(eval, vec![4, 5]);
}

#[test]
fn split_train_and_eval_handles_ratio_edges() {
    let data = vec![10, 20, 30];
    let (train_all, eval_none) = split_train_and_eval(data.clone(), 0.0);
    assert_eq!(train_all, data);
    assert!(eval_none.is_empty());

    let (train_none, eval_all) = split_train_and_eval(vec![10, 20, 30], 1.0);
    assert!(train_none.is_empty());
    assert_eq!(eval_all, vec![10, 20, 30]);
}

#[test]
fn split_train_and_eval_handles_empty_input() {
    let (train, eval): (Vec<i32>, Vec<i32>) = split_train_and_eval(Vec::new(), 0.2);
    assert!(train.is_empty());
    assert!(eval.is_empty());
}
