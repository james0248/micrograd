use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::autodiff;
use crate::data::{MnistSample, load_and_split_mnist};
use crate::losses::cross_entropy_with_logits;
use crate::nn::Mlp;
use crate::optim::{Optimizer, Sgd};
use crate::tensor::Tensor;

const DATA_PATH: &str = "data/train.csv";
const EVAL_RATIO: f64 = 0.10;
const SPLIT_SEED: u64 = 7;
const SHUFFLE_SEED: u64 = 19;
const MODEL_SEED: u64 = 42;
const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 10;
const CHECKPOINT_PATH: &str = "checkpoints/mnist_mlp.ckpt";

#[derive(Debug, Clone)]
struct FlatMnist {
    images: Vec<f32>,
    labels: Vec<u8>,
    rows: usize,
    cols: usize,
}

fn flatten_samples(samples: Vec<MnistSample>) -> FlatMnist {
    let rows = samples.len();
    let cols = 784;
    let mut images = Vec::with_capacity(rows * cols);
    let mut labels = Vec::with_capacity(rows);

    for sample in samples {
        labels.push(sample.label);
        images.extend_from_slice(&sample.pixels);
    }

    FlatMnist {
        images,
        labels,
        rows,
        cols,
    }
}

fn build_batch(dataset: &FlatMnist, indices: &[usize]) -> (Vec<f32>, Vec<u8>) {
    let mut xb = Vec::with_capacity(indices.len() * dataset.cols);
    let mut yb = Vec::with_capacity(indices.len());

    for &idx in indices {
        let start = idx * dataset.cols;
        let end = start + dataset.cols;
        xb.extend_from_slice(&dataset.images[start..end]);
        yb.push(dataset.labels[idx]);
    }

    (xb, yb)
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .expect("row must not be empty")
}

fn count_correct(logits_data: &[f32], classes: usize, targets: &[u8]) -> usize {
    targets
        .iter()
        .enumerate()
        .filter(|&(i, target)| {
            let row_start = i * classes;
            argmax(&logits_data[row_start..row_start + classes]) as u8 == *target
        })
        .count()
}

fn evaluate(model: &Mlp, dataset: &FlatMnist) -> (f32, f32) {
    if dataset.rows == 0 {
        return (0.0, 0.0);
    }

    let mut total_loss = 0.0f32;
    let mut total_correct = 0usize;
    let mut seen = 0usize;

    let indices: Vec<usize> = (0..dataset.rows).collect();
    for chunk in indices.chunks(BATCH_SIZE) {
        let (xb, yb) = build_batch(dataset, chunk);
        let batch = yb.len();
        let x = Tensor::from_vec(xb, vec![batch, dataset.cols]);
        let logits = model.forward(&x);
        let loss = cross_entropy_with_logits(&logits, &yb);
        let loss_value = loss.to_vec()[0];
        total_loss += loss_value * batch as f32;

        let logits_data = logits.to_vec();
        let classes = logits.shape()[1];
        total_correct += count_correct(&logits_data, classes, &yb);
        seen += batch;
    }

    (total_loss / seen as f32, total_correct as f32 / seen as f32)
}

fn learning_rate_for_epoch(epoch: usize) -> f32 {
    if epoch < 6 { 0.1 } else { 0.01 }
}

fn save_model_checkpoint(model: &Mlp, path: &Path) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Err(format!(
            "checkpoint path has no parent directory: {}",
            path.display()
        ));
    };

    fs::create_dir_all(parent).map_err(|err| {
        format!(
            "failed to create checkpoint directory {}: {err}",
            parent.display()
        )
    })?;

    model
        .save_weights(path)
        .map_err(|err| format!("failed to save checkpoint {}: {err}", path.display()))
}

pub fn run() -> Result<(), String> {
    let (train_samples, eval_samples) = load_and_split_mnist(DATA_PATH, EVAL_RATIO, SPLIT_SEED)?;
    let train = flatten_samples(train_samples);
    let eval = flatten_samples(eval_samples);

    println!(
        "loaded mnist: train={} eval={} batch_size={}",
        train.rows, eval.rows, BATCH_SIZE
    );

    let mut model = Mlp::new(&[train.cols, 32, 10], MODEL_SEED);
    let mut optimizer = Sgd::new(learning_rate_for_epoch(0));

    let mut rng = StdRng::seed_from_u64(SHUFFLE_SEED);
    let mut train_indices: Vec<usize> = (0..train.rows).collect();

    for epoch in 0..EPOCHS {
        let epoch_start = Instant::now();
        train_indices.shuffle(&mut rng);
        let lr = learning_rate_for_epoch(epoch);
        optimizer.set_lr(lr);
        let mut total_loss = 0.0f32;
        let mut total_correct = 0usize;
        let mut seen = 0usize;

        for chunk in train_indices.chunks(BATCH_SIZE) {
            let (xb, yb) = build_batch(&train, chunk);
            let batch = yb.len();
            let x = Tensor::from_vec(xb, vec![batch, train.cols]);

            let logits_cell: RefCell<Option<(Vec<f32>, usize)>> = RefCell::new(None);

            let params = model.parameters();
            let (loss, grads) = autodiff::value_and_grad(
                |ps| {
                    let logits = model.forward_with_params(ps, &x);
                    logits_cell.borrow_mut().replace((logits.to_vec(), logits.shape()[1]));
                    cross_entropy_with_logits(&logits, &yb)
                },
                &params,
            );
            let loss_value = loss.to_vec()[0];
            let (logits_data, classes) = logits_cell.into_inner().expect("closure must run");
            let correct = count_correct(&logits_data, classes, &yb);

            optimizer.step(&mut model, &grads);

            total_loss += loss_value * batch as f32;
            total_correct += correct;
            seen += batch;
        }

        let train_loss = total_loss / seen as f32;
        let train_acc = total_correct as f32 / seen as f32;
        let (eval_loss, eval_acc) = evaluate(&model, &eval);
        let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "epoch {:>2}: lr={:.4} train_loss={:.5} train_acc={:.4} eval_loss={:.5} eval_acc={:.4} epoch_time_ms={:.2}",
            epoch + 1,
            lr,
            train_loss,
            train_acc,
            eval_loss,
            eval_acc,
            epoch_ms
        );
    }

    let checkpoint_path = Path::new(CHECKPOINT_PATH);
    save_model_checkpoint(&model, checkpoint_path)?;
    println!("saved checkpoint: {}", checkpoint_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(tag: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic here");
        std::env::temp_dir().join(format!(
            "tangent_mnist_{tag}_{}_{}",
            std::process::id(),
            now.as_nanos()
        ))
    }

    #[test]
    fn save_model_checkpoint_creates_file() {
        let dir = temp_path("save_ok");
        let model = Mlp::new(&[2, 4, 2], 7);
        let path = dir.join("weights.ckpt");

        save_model_checkpoint(&model, &path).expect("save must succeed");
        let metadata = fs::metadata(&path).expect("checkpoint file must exist");
        assert!(metadata.is_file());
        assert!(metadata.len() > 0, "checkpoint file must be non-empty");

        fs::remove_file(path).ok();
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn save_model_checkpoint_returns_err_for_file_as_directory() {
        let base = temp_path("save_err");
        let file_path = base.with_extension("tmp");
        std::fs::File::create(&file_path).expect("create file must succeed");
        let model = Mlp::new(&[2, 4, 2], 7);
        let checkpoint_path = file_path.join("weights.ckpt");

        let err = save_model_checkpoint(&model, &checkpoint_path)
            .expect_err("save must fail when directory path is a file");
        assert!(
            err.contains("failed to create checkpoint directory"),
            "{err}"
        );

        fs::remove_file(file_path).ok();
    }
}
