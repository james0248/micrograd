use micrograd::data::{MnistSample, load_and_split_mnist};
use micrograd::engine::{Value, no_grad, with_grad};
use micrograd::nn::Mlp;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

const DATA_PATH: &str = "data/train.csv";
const EVAL_RATIO: f64 = 0.10;
const SPLIT_SEED: u64 = 7;
const SHUFFLE_SEED: u64 = 19;
const MODEL_SEED: u64 = 42;
const LEARNING_RATE: f64 = 0.03;
const EPOCHS: usize = 1;
const HIDDEN_SIZE: usize = 16;

fn sample_to_values(sample: &MnistSample) -> Vec<Value> {
    sample.pixels.iter().map(|&px| Value::new(px)).collect()
}

fn argmax_values(values: &[Value]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.data().total_cmp(&b.data()))
        .map(|(idx, _)| idx)
        .expect("logits cannot be empty")
}

fn argmax_f64(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .expect("logits cannot be empty")
}

fn softmax_cross_entropy(logits: &[Value], target: u8) -> Value {
    assert_eq!(logits.len(), 10, "mnist head must output 10 logits");
    let target_idx = target as usize;
    assert!(target_idx < logits.len(), "target label out of range");

    // Shift by scalar max(logits) for numerical stability.
    let max_logit = logits
        .iter()
        .map(Value::data)
        .fold(f64::NEG_INFINITY, f64::max);
    let shift = Value::new(max_logit);

    let mut sum_exp = Value::new(0.0);
    for logit in logits {
        sum_exp = &sum_exp + &logit.sub(&shift).exp();
    }

    shift.add(&sum_exp.log()).sub(&logits[target_idx])
}

fn softmax_cross_entropy_scalar(logits: &[f64], target: u8) -> f64 {
    let target_idx = target as usize;
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = logits.iter().map(|&z| (z - max_logit).exp()).sum();
    let logsumexp = max_logit + sum_exp.ln();
    logsumexp - logits[target_idx]
}

fn evaluate(mlp: &Mlp, dataset: &[MnistSample]) -> (f64, f64) {
    if dataset.is_empty() {
        return (0.0, 0.0);
    }

    no_grad(|| {
        let mut total_loss = 0.0;
        let mut correct = 0usize;

        for sample in dataset {
            let input = sample_to_values(sample);
            let logits = mlp.forward(&input);
            let logits_data: Vec<f64> = logits.iter().map(Value::data).collect();
            total_loss += softmax_cross_entropy_scalar(&logits_data, sample.label);

            let pred = argmax_f64(&logits_data) as u8;
            if pred == sample.label {
                correct += 1;
            }
        }

        (
            total_loss / dataset.len() as f64,
            correct as f64 / dataset.len() as f64,
        )
    })
}

fn main() {
    let (train_dataset, eval_dataset) = load_and_split_mnist(DATA_PATH, EVAL_RATIO, SPLIT_SEED)
        .unwrap_or_else(|e| panic!("failed to load mnist csv at {DATA_PATH}: {e}"));

    println!(
        "loaded mnist: train={} eval={} (eval_ratio={:.2})",
        train_dataset.len(),
        eval_dataset.len(),
        EVAL_RATIO
    );

    let mlp = Mlp::new(&[784, HIDDEN_SIZE, 10], MODEL_SEED);
    let (weights, biases) = mlp.parameters();
    let mut params = weights;
    params.extend(biases);

    let mut shuffle_rng = StdRng::seed_from_u64(SHUFFLE_SEED);
    let mut order: Vec<usize> = (0..train_dataset.len()).collect();

    for epoch in 0..EPOCHS {
        order.shuffle(&mut shuffle_rng);
        let mut train_loss = 0.0;
        let mut train_correct = 0usize;

        for &row_idx in &order {
            for p in &params {
                p.zero_grad();
            }

            let sample = &train_dataset[row_idx];
            let (sample_loss, pred) = with_grad(|| {
                let input = sample_to_values(sample);
                let logits = mlp.forward(&input);
                let pred = argmax_values(&logits) as u8;
                let loss = softmax_cross_entropy(&logits, sample.label);
                let loss_data = loss.data();
                loss.backward();
                (loss_data, pred)
            });

            for p in &params {
                p.set_data(p.data() - LEARNING_RATE * p.grad());
            }

            train_loss += sample_loss;
            if pred == sample.label {
                train_correct += 1;
            }
        }

        let train_loss = train_loss / train_dataset.len() as f64;
        let train_acc = train_correct as f64 / train_dataset.len() as f64;
        let (eval_loss, eval_acc) = evaluate(&mlp, &eval_dataset);

        println!(
            "epoch {:>3}: train_loss = {:.6}, train_acc = {:.4}, eval_loss = {:.6}, eval_acc = {:.4}",
            epoch + 1,
            train_loss,
            train_acc,
            eval_loss,
            eval_acc
        );
    }
}
