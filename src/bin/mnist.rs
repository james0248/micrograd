use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::cell::RefCell;
use std::fs;
use std::time::Instant;

use tangent::autodiff;
use tangent::losses::cross_entropy_with_logits;
use tangent::nn::dense::Dense;
use tangent::nn::module::Module;
use tangent::nn::tree::TensorTree;
use tangent::tensor::Tensor;
use tangent::utils::split_train_and_eval;

const DATA_PATH: &str = "data/train.csv";
const EVAL_RATIO: f64 = 0.10;
const SPLIT_SEED: u64 = 7;
const SHUFFLE_SEED: u64 = 19;
const MODEL_SEED: u64 = 42;
const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 10;
const LR: f32 = 0.1;

// ---------------------------------------------------------------------------
// Data loading (self-contained, mirrors data.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct MnistSample {
    label: u8,
    pixels: [f32; 784],
}

fn load_mnist_csv(path: &str) -> Result<Vec<MnistSample>, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("failed to read '{path}': {e}"))?;
    let mut lines = raw.lines();

    let header = lines
        .next()
        .ok_or_else(|| "mnist csv is empty".to_string())?;
    let header_cols = header.split(',').count();
    if header_cols != 785 {
        return Err(format!(
            "invalid header: expected 785 columns (label + 784 pixels), got {header_cols}"
        ));
    }

    let mut out = Vec::new();

    for (idx, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row_num = idx + 2;
        let mut cols = line.split(',');

        let label_tok = cols
            .next()
            .ok_or_else(|| format!("row {row_num}: missing label"))?;
        let label = label_tok
            .parse::<u8>()
            .map_err(|e| format!("row {row_num}: invalid label '{label_tok}': {e}"))?;
        if label > 9 {
            return Err(format!(
                "row {row_num}: label out of range [0, 9], got {label}"
            ));
        }

        let mut pixels = [0.0f32; 784];
        for (pix_idx, slot) in pixels.iter_mut().enumerate() {
            let tok = cols.next().ok_or_else(|| {
                format!(
                    "row {row_num}: expected 784 pixels, missing pixel{}",
                    pix_idx
                )
            })?;
            let raw_px = tok.parse::<u16>().map_err(|e| {
                format!("row {row_num}: invalid pixel{} value '{tok}': {e}", pix_idx)
            })?;
            if raw_px > 255 {
                return Err(format!(
                    "row {row_num}: pixel{} out of range [0, 255], got {}",
                    pix_idx, raw_px
                ));
            }
            *slot = raw_px as f32 / 255.0;
        }

        if cols.next().is_some() {
            return Err(format!(
                "row {row_num}: expected exactly 785 columns (label + 784 pixels)"
            ));
        }

        out.push(MnistSample { label, pixels });
    }

    Ok(out)
}

fn load_and_split_mnist(
    path: &str,
    eval_ratio: f64,
    seed: u64,
) -> Result<(Vec<MnistSample>, Vec<MnistSample>), String> {
    let mut samples = load_mnist_csv(path)?;
    let mut rng = StdRng::seed_from_u64(seed);
    samples.shuffle(&mut rng);
    Ok(split_train_and_eval(samples, eval_ratio))
}

// ---------------------------------------------------------------------------
// Flat batch helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Accuracy helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Model definition — local MLP using Dense
// ---------------------------------------------------------------------------

struct Mlp {
    dims: Vec<usize>,
}

impl Module for Mlp {
    fn forward(&self, input: &Tensor) -> Tensor {
        let last = self.dims.len() - 1;
        let mut x = input.clone();
        for (i, &dim) in self.dims.iter().enumerate() {
            x = Dense::new(dim, true).call(&x);
            if i < last {
                x = x.relu();
            }
        }
        x
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

fn evaluate(model: &Mlp, params: &TensorTree, dataset: &FlatMnist) -> (f32, f32) {
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
        let logits = model.apply(params.clone(), &x);
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

// ---------------------------------------------------------------------------
// Training entry point
// ---------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let (train_samples, eval_samples) = load_and_split_mnist(DATA_PATH, EVAL_RATIO, SPLIT_SEED)?;
    let train = flatten_samples(train_samples);
    let eval = flatten_samples(eval_samples);

    println!(
        "loaded mnist: train={} eval={} batch_size={}",
        train.rows, eval.rows, BATCH_SIZE
    );

    // JAX-style init: build model, run init with dummy input to collect params
    let model = Mlp { dims: vec![32, 10] };
    let dummy = Tensor::zeros(vec![1, 784]);
    let mut rng = StdRng::seed_from_u64(MODEL_SEED);
    let mut params = model.init(&mut rng, &dummy);

    let mut shuffle_rng = StdRng::seed_from_u64(SHUFFLE_SEED);
    let mut train_indices: Vec<usize> = (0..train.rows).collect();

    for epoch in 0..EPOCHS {
        let epoch_start = Instant::now();
        train_indices.shuffle(&mut shuffle_rng);
        let mut total_loss = 0.0f32;
        let mut total_correct = 0usize;
        let mut seen = 0usize;

        for chunk in train_indices.chunks(BATCH_SIZE) {
            let (xb, yb) = build_batch(&train, chunk);
            let batch = yb.len();
            let x = Tensor::from_vec(xb, vec![batch, train.cols]);

            let logits_cell: RefCell<Option<(Vec<f32>, usize)>> = RefCell::new(None);

            let leaves = params.leaves();
            let (loss, grads) = autodiff::value_and_grad(
                |ps| {
                    let traced_params = params.replace_leaves(ps);
                    let logits = model.apply(traced_params, &x);
                    logits_cell
                        .borrow_mut()
                        .replace((logits.to_vec(), logits.shape()[1]));
                    cross_entropy_with_logits(&logits, &yb)
                },
                &leaves,
            );
            let loss_value = loss.to_vec()[0];
            let (logits_data, classes) = logits_cell.into_inner().expect("closure must run");
            let correct = count_correct(&logits_data, classes, &yb);

            // Functional SGD update
            let lr_t = Tensor::from_vec(vec![LR], vec![1]);
            let new_leaves: Vec<Tensor> = leaves
                .iter()
                .zip(grads.iter())
                .map(|(p, g)| p.sub(&g.mul(&lr_t)))
                .collect();
            params = params.replace_leaves(&new_leaves);

            total_loss += loss_value * batch as f32;
            total_correct += correct;
            seen += batch;
        }

        let train_loss = total_loss / seen as f32;
        let train_acc = total_correct as f32 / seen as f32;
        let (eval_loss, eval_acc) = evaluate(&model, &params, &eval);
        let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "epoch {:>2}: lr={:.4} train_loss={:.5} train_acc={:.4} eval_loss={:.5} eval_acc={:.4} epoch_time_ms={:.2}",
            epoch + 1,
            LR,
            train_loss,
            train_acc,
            eval_loss,
            eval_acc,
            epoch_ms
        );
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        panic!("tangent failed: {err}");
    }
}
