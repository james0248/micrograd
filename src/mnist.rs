use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::data::{MnistSample, load_and_split_mnist};
use crate::engine::{Tensor, no_grad, with_grad};
use crate::losses::cross_entropy_with_logits;
use crate::nn::Mlp;
use crate::optim::{Optimizer, Sgd};

const DATA_PATH: &str = "data/train.csv";
const EVAL_RATIO: f64 = 0.10;
const SPLIT_SEED: u64 = 7;
const SHUFFLE_SEED: u64 = 19;
const MODEL_SEED: u64 = 42;
const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 10;
const CHECKPOINT_DIR: &str = "artifacts";
const CHECKPOINT_PREFIX: &str = "mnist";

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
        images.extend(sample.pixels.into_iter().map(|v| v as f32));
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

fn evaluate(model: &Mlp, dataset: &FlatMnist) -> (f32, f32) {
    if dataset.rows == 0 {
        return (0.0, 0.0);
    }

    no_grad(|| {
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
            let loss_value = loss.data()[0];
            total_loss += loss_value * batch as f32;

            let logits_data = logits.data();
            let classes = logits.shape()[1];
            for (i, target) in yb.iter().enumerate() {
                let row_start = i * classes;
                let row_end = row_start + classes;
                let pred = argmax(&logits_data[row_start..row_end]) as u8;
                if pred == *target {
                    total_correct += 1;
                }
            }
            seen += batch;
        }

        (total_loss / seen as f32, total_correct as f32 / seen as f32)
    })
}

fn learning_rate_for_epoch(epoch: usize) -> f32 {
    if epoch < 6 { 0.1 } else { 0.01 }
}

fn checkpoint_path_in_dir(base_dir: &Path, prefix: &str) -> Result<PathBuf, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("failed to get current timestamp: {err}"))?;
    let filename = format!("{prefix}_{}_{}.ckpt", now.as_secs(), now.subsec_nanos());
    Ok(base_dir.join(filename))
}

fn save_model_checkpoint(model: &Mlp, base_dir: &Path, prefix: &str) -> Result<PathBuf, String> {
    fs::create_dir_all(base_dir).map_err(|err| {
        format!(
            "failed to create checkpoint directory {}: {err}",
            base_dir.display()
        )
    })?;
    let path = checkpoint_path_in_dir(base_dir, prefix)?;
    model
        .save_weights(&path)
        .map_err(|err| format!("failed to save checkpoint {}: {err}", path.display()))?;
    Ok(path)
}

pub fn run() -> Result<(), String> {
    let (train_samples, eval_samples) = load_and_split_mnist(DATA_PATH, EVAL_RATIO, SPLIT_SEED)?;
    let train = flatten_samples(train_samples);
    let eval = flatten_samples(eval_samples);

    println!(
        "loaded mnist: train={} eval={} batch_size={}",
        train.rows, eval.rows, BATCH_SIZE
    );

    let model = Mlp::new(&[train.cols, 128, 10], MODEL_SEED);
    let mut optimizer = Sgd::new(model.parameters(), learning_rate_for_epoch(0));

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
            optimizer.zero_grad();

            let (xb, yb) = build_batch(&train, chunk);
            let batch = yb.len();

            let (loss_value, correct) = with_grad(|| {
                let x = Tensor::from_vec(xb, vec![batch, train.cols]);
                let logits = model.forward(&x);
                let classes = logits.shape()[1];
                let logits_data = logits.data();

                let mut batch_correct = 0usize;
                for (i, target) in yb.iter().enumerate() {
                    let row_start = i * classes;
                    let row_end = row_start + classes;
                    let pred = argmax(&logits_data[row_start..row_end]) as u8;
                    if pred == *target {
                        batch_correct += 1;
                    }
                }

                let loss = cross_entropy_with_logits(&logits, &yb);
                let loss_value = loss.data()[0];
                loss.backward();
                (loss_value, batch_correct)
            });

            optimizer.step();

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

    let checkpoint_path =
        save_model_checkpoint(&model, Path::new(CHECKPOINT_DIR), CHECKPOINT_PREFIX)?;
    println!("saved checkpoint: {}", checkpoint_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::reset_state;

    fn temp_path(tag: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic here");
        std::env::temp_dir().join(format!(
            "micrograd_mnist_{tag}_{}_{}",
            std::process::id(),
            now.as_nanos()
        ))
    }

    #[test]
    fn checkpoint_filename_has_expected_shape() {
        let path = checkpoint_path_in_dir(Path::new(CHECKPOINT_DIR), CHECKPOINT_PREFIX)
            .expect("path generation must succeed");
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .expect("filename must be valid utf-8");

        assert!(file_name.starts_with("mnist_"), "{file_name}");
        assert!(file_name.ends_with(".ckpt"), "{file_name}");
        let stem = file_name.strip_suffix(".ckpt").expect("must end with ckpt");
        let parts: Vec<&str> = stem.split('_').collect();
        assert_eq!(parts.len(), 3, "{file_name}");
        assert_eq!(parts[0], "mnist");
        assert!(parts[1].parse::<u64>().is_ok(), "{file_name}");
        assert!(parts[2].parse::<u32>().is_ok(), "{file_name}");
    }

    #[test]
    fn save_model_checkpoint_creates_file() {
        reset_state();
        let dir = temp_path("save_ok");
        let model = Mlp::new(&[2, 4, 2], 7);

        let path = save_model_checkpoint(&model, &dir, "mnist_test").expect("save must succeed");
        let metadata = fs::metadata(&path).expect("checkpoint file must exist");
        assert!(metadata.is_file());
        assert!(metadata.len() > 0, "checkpoint file must be non-empty");

        fs::remove_file(path).ok();
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn save_model_checkpoint_returns_err_for_file_as_directory() {
        reset_state();
        let base = temp_path("save_err");
        let file_path = base.with_extension("tmp");
        std::fs::File::create(&file_path).expect("create file must succeed");
        let model = Mlp::new(&[2, 4, 2], 7);

        let err = save_model_checkpoint(&model, &file_path, "mnist_test")
            .expect_err("save must fail when directory path is a file");
        assert!(
            err.contains("failed to create checkpoint directory"),
            "{err}"
        );

        fs::remove_file(file_path).ok();
    }
}
