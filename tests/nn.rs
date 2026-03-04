use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Serialize;

use micrograd::engine::{Tensor, no_grad, reset_state, with_grad};
use micrograd::losses::cross_entropy_with_logits;
use micrograd::nn::Mlp;
use micrograd::optim::{Optimizer, Sgd};

fn tiny_dataset() -> Vec<([f32; 2], u8)> {
    vec![
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
        ([1.0, 1.0], 1),
    ]
}

fn build_batch(rows: &[([f32; 2], u8)]) -> (Vec<f32>, Vec<u8>) {
    let mut x = Vec::with_capacity(rows.len() * 2);
    let mut y = Vec::with_capacity(rows.len());
    for (xy, label) in rows {
        x.push(xy[0]);
        x.push(xy[1]);
        y.push(*label);
    }
    (x, y)
}

fn mean_loss(model: &Mlp, rows: &[([f32; 2], u8)]) -> f32 {
    no_grad(|| {
        let (x, y) = build_batch(rows);
        let xb = Tensor::from_vec(x, vec![rows.len(), 2]);
        let logits = model.forward(&xb);
        let loss = cross_entropy_with_logits(&logits, &y);
        loss.data()[0]
    })
}

fn temp_checkpoint_path(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be monotonic here")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "micrograd_{tag}_{}_{}.ckpt",
        std::process::id(),
        nanos
    ))
}

fn snapshot_params(model: &Mlp) -> Vec<Vec<f32>> {
    model.parameters().iter().map(Tensor::data).collect()
}

#[test]
fn mlp_tensor_forward_shape_is_batch_by_output() {
    reset_state();
    let model = Mlp::new(&[784, 128, 10], 7);
    with_grad(|| {
        let x = Tensor::from_vec(vec![0.0; 8 * 784], vec![8, 784]);
        let out = model.forward(&x);
        assert_eq!(out.shape(), vec![8, 10]);
    });
}

#[test]
fn mlp_tensor_parameter_count_matches_dims() {
    reset_state();
    let model = Mlp::new(&[784, 128, 10], 7);
    let params = model.parameters();
    assert_eq!(params.len(), 4);
    let total_numel: usize = params.iter().map(Tensor::numel).sum();
    assert_eq!(total_numel, 784 * 128 + 128 + 128 * 10 + 10);
}

#[test]
fn tensor_training_smoke_loss_decreases() {
    reset_state();
    let model = Mlp::new(&[2, 8, 2], 42);
    let mut optimizer = Sgd::new(model.parameters(), 0.1);
    let mut rng = StdRng::seed_from_u64(11);
    let mut rows = tiny_dataset();
    let initial = mean_loss(&model, &rows);

    for _ in 0..200 {
        rows.shuffle(&mut rng);
        optimizer.zero_grad();

        let (x, y) = build_batch(&rows);
        with_grad(|| {
            let xb = Tensor::from_vec(x, vec![rows.len(), 2]);
            let logits = model.forward(&xb);
            let loss = cross_entropy_with_logits(&logits, &y);
            loss.backward();
        });

        optimizer.step();
    }

    let final_loss = mean_loss(&model, &rows);
    assert!(
        final_loss < initial,
        "expected final loss < initial loss, got initial={initial}, final={final_loss}"
    );
}

#[test]
fn mlp_weights_roundtrip_save_load() {
    reset_state();
    let src = Mlp::new(&[2, 8, 2], 42);
    let dst = Mlp::new(&[2, 8, 2], 7);
    let path = temp_checkpoint_path("nn_roundtrip");

    let src_params = snapshot_params(&src);
    let dst_before = snapshot_params(&dst);
    assert_ne!(src_params, dst_before);

    src.save_weights(&path).expect("save must succeed");
    dst.load_weights(&path).expect("load must succeed");
    fs::remove_file(path).ok();

    let dst_after = snapshot_params(&dst);
    assert_eq!(src_params, dst_after);
}

#[test]
fn mlp_weights_load_rejects_dims_mismatch() {
    reset_state();
    let src = Mlp::new(&[2, 8, 2], 42);
    let dst = Mlp::new(&[2, 4, 2], 7);
    let path = temp_checkpoint_path("nn_dims_mismatch");

    src.save_weights(&path).expect("save must succeed");
    let dst_before = snapshot_params(&dst);
    let err = dst.load_weights(&path).expect_err("load must fail");
    fs::remove_file(path).ok();

    let dst_after = snapshot_params(&dst);
    assert_eq!(dst_before, dst_after, "model weights must be unchanged");
    assert!(err.contains("dims mismatch"), "{err}");
}

#[test]
fn mlp_weights_load_rejects_corrupted_file() {
    reset_state();
    let model = Mlp::new(&[2, 8, 2], 7);
    let path = temp_checkpoint_path("nn_corrupted");

    fs::write(&path, b"not a checkpoint").expect("write must succeed");
    let err = model.load_weights(&path).expect_err("load must fail");
    fs::remove_file(path).ok();

    assert!(err.contains("failed to deserialize checkpoint"), "{err}");
}

#[derive(Serialize)]
struct RawCheckpointV1 {
    version: u32,
    dims: Vec<usize>,
    params: Vec<Vec<f32>>,
}

#[test]
fn mlp_weights_load_rejects_param_count_mismatch_without_mutation() {
    reset_state();
    let model = Mlp::new(&[2, 8, 2], 7);
    let path = temp_checkpoint_path("nn_param_count");
    let payload = RawCheckpointV1 {
        version: 1,
        dims: model.dims(),
        params: vec![vec![0.0; 16]],
    };

    let file = fs::File::create(&path).expect("create must succeed");
    let mut writer = BufWriter::new(file);
    bincode::serialize_into(&mut writer, &payload).expect("serialize must succeed");
    writer.flush().expect("flush must succeed");

    let before = snapshot_params(&model);
    let err = model.load_weights(&path).expect_err("load must fail");
    fs::remove_file(path).ok();

    let after = snapshot_params(&model);
    assert_eq!(before, after, "model weights must be unchanged");
    assert!(err.contains("parameter count mismatch"), "{err}");
}
