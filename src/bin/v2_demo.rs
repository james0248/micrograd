use micrograd::engine_v2::{Value, no_grad, with_grad};
use micrograd::nn_v2::Mlp;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn generate_xor_dataset(samples_per_corner: usize, noise: f64, seed: u64) -> Vec<([f64; 2], f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let corners = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    let mut dataset = Vec::with_capacity(samples_per_corner * corners.len());
    for _ in 0..samples_per_corner {
        for (xy, target) in corners {
            let nx = rng.gen_range(-noise..noise);
            let ny = rng.gen_range(-noise..noise);
            dataset.push(([xy[0] + nx, xy[1] + ny], target));
        }
    }

    dataset
}

fn sigmoid(logit: &Value) -> Value {
    let one = Value::new(1.0);
    let denom = one + (-*logit).exp();
    one.div(&denom)
}

fn bce_from_prob(prob: &Value, y: f64) -> Value {
    let one = Value::new(1.0);
    let target = Value::new(y);
    let eps = Value::new(1e-8);

    let p = prob.add(&eps);
    let one_minus_p = one.sub(prob).add(&eps);
    let term_pos = target.mul(&p.log());
    let term_neg = one.sub(&target).mul(&one_minus_p.log());
    -(term_pos.add(&term_neg))
}

fn bce_scalar(prob: f64, y: f64) -> f64 {
    let eps = 1e-8;
    let p = (prob + eps).clamp(eps, 1.0 - eps);
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

fn evaluate(mlp: &Mlp, dataset: &[([f64; 2], f64)]) -> (f64, f64) {
    no_grad(|| {
        let mut total_bce = 0.0;
        let mut correct = 0usize;

        for (x_raw, y_raw) in dataset.iter().copied() {
            let x = vec![Value::new(x_raw[0]), Value::new(x_raw[1])];
            let logit = mlp.forward(&x).into_iter().next().expect("single output");
            let prob = 1.0 / (1.0 + (-logit.data()).exp());
            total_bce += bce_scalar(prob, y_raw);

            let pred_label = if prob >= 0.5 { 1.0 } else { 0.0 };
            if (pred_label - y_raw).abs() < f64::EPSILON {
                correct += 1;
            }
        }

        (
            total_bce / dataset.len() as f64,
            correct as f64 / dataset.len() as f64,
        )
    })
}

fn main() {
    let train_dataset = generate_xor_dataset(16, 0.08, 7);
    let eval_dataset = generate_xor_dataset(16, 0.08, 77);
    let mlp = Mlp::new(&[2, 4, 1], 42);
    let learning_rate = 5e-2;
    let epochs = 500;

    for epoch in 0..epochs {
        let (weights, biases) = mlp.parameters();
        for p in weights.iter().chain(biases.iter()) {
            p.zero_grad();
        }

        let (train_bce, train_acc) = with_grad(|| {
            let mut total_loss = Value::new(0.0);
            let mut correct = 0usize;
            for (x_raw, y_raw) in train_dataset.iter().copied() {
                let x = vec![Value::new(x_raw[0]), Value::new(x_raw[1])];
                let logit = mlp.forward(&x).into_iter().next().expect("single output");
                let prob = sigmoid(&logit);
                let pred_label = if prob.data() >= 0.5 { 1.0 } else { 0.0 };
                if (pred_label - y_raw).abs() < f64::EPSILON {
                    correct += 1;
                }
                let sample_loss = bce_from_prob(&prob, y_raw);
                total_loss = total_loss + sample_loss;
            }

            let mean_loss = total_loss.div(&Value::new(train_dataset.len() as f64));
            mean_loss.backward();

            (
                mean_loss.data(),
                correct as f64 / train_dataset.len() as f64,
            )
        });

        for p in weights.iter().chain(biases.iter()) {
            let next = p.data() - learning_rate * p.grad();
            p.set_data(next);
        }

        if epoch % 25 == 0 || epoch == epochs - 1 {
            let (eval_bce, eval_acc) = evaluate(&mlp, &eval_dataset);
            println!(
                "epoch {:>3}: train_bce = {:.6}, train_acc = {:.3}, eval_bce = {:.6}, eval_acc = {:.3}",
                epoch, train_bce, train_acc, eval_bce, eval_acc
            );
        }
    }
}
