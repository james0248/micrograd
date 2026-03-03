use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use micrograd::engine_v2::{Value, no_grad, reset_state, with_grad};
use micrograd::nn_v2::{Mlp, Neuron};

fn xor_dataset(samples_per_corner: usize, noise: f64, seed: u64) -> Vec<([f64; 2], f64)> {
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

fn mean_loss_value(mlp: &Mlp, dataset: &[([f64; 2], f64)]) -> Value {
    let mut total_loss = Value::new(0.0);

    for (x_raw, y_raw) in dataset.iter().copied() {
        let x = vec![Value::new(x_raw[0]), Value::new(x_raw[1])];
        let logit = mlp.forward(&x).into_iter().next().expect("single output");
        let prob = sigmoid(&logit);
        total_loss = total_loss + bce_from_prob(&prob, y_raw);
    }

    total_loss.div(&Value::new(dataset.len() as f64))
}

fn mean_loss_scalar(mlp: &Mlp, dataset: &[([f64; 2], f64)]) -> f64 {
    no_grad(|| mean_loss_value(mlp, dataset).data())
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

#[test]
fn mlp_parameter_count_2_4_1_is_17() {
    reset_state();
    let mlp = Mlp::new(&[2, 4, 1], 7);
    let (weights, biases) = mlp.parameters();
    assert_eq!(weights.len(), 12);
    assert_eq!(biases.len(), 5);
    assert_eq!(weights.len() + biases.len(), 17);
}

#[test]
fn mlp_forward_output_shape_is_one() {
    reset_state();
    let mlp = Mlp::new(&[2, 4, 1], 7);
    with_grad(|| {
        let x = vec![Value::new(0.25), Value::new(-0.75)];
        let out = mlp.forward(&x);
        assert_eq!(out.len(), 1);
    });
}

#[test]
fn mlp_init_is_deterministic_for_same_seed() {
    reset_state();
    let a = Mlp::new(&[2, 4, 1], 1234);
    let b = Mlp::new(&[2, 4, 1], 1234);

    let (wa, ba) = a.parameters();
    let (wb, bb) = b.parameters();
    assert_eq!(wa.len(), wb.len());
    assert_eq!(ba.len(), bb.len());

    for (va, vb) in wa.iter().zip(wb.iter()) {
        assert_eq!(va.data(), vb.data());
    }
    for (va, vb) in ba.iter().zip(bb.iter()) {
        assert_eq!(va.data(), vb.data());
    }
}

#[test]
#[should_panic(expected = "input length must match neuron weight count")]
fn neuron_forward_panics_on_input_len_mismatch() {
    reset_state();
    let mut rng = StdRng::seed_from_u64(99);
    let neuron = Neuron::new(2, &mut rng);
    with_grad(|| {
        let x = vec![Value::new(1.0)];
        let _ = neuron.forward(&x);
    });
}

#[test]
fn training_smoke_loss_decreases() {
    reset_state();
    let dataset = xor_dataset(16, 0.08, 7);
    let mlp = Mlp::new(&[2, 4, 1], 42);
    let initial_loss = mean_loss_scalar(&mlp, &dataset);
    let learning_rate = 1e-2;

    for _ in 0..500 {
        let (weights, biases) = mlp.parameters();
        for p in weights.iter().chain(biases.iter()) {
            p.zero_grad();
        }

        with_grad(|| {
            let mean_loss = mean_loss_value(&mlp, &dataset);
            mean_loss.backward();
        });

        for p in weights.iter().chain(biases.iter()) {
            p.set_data(p.data() - learning_rate * p.grad());
        }
    }

    let final_loss = mean_loss_scalar(&mlp, &dataset);
    assert!(
        final_loss < initial_loss,
        "classification BCE did not decrease: initial={initial_loss}, final={final_loss}"
    );
}
