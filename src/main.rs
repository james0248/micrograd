use micrograd::engine::Value;
use micrograd::nn::{Layer, Mlp, Neuron};

fn main() {
    let probe = Value::new(0.0);
    let _skeleton = Mlp {
        layers: vec![Layer {
            neurons: vec![Neuron {
                weights: vec![Value::new(0.5), Value::new(-0.5)],
                bias: Value::new(0.0),
            }],
        }],
    };

    println!("stage 1 skeleton ready: probe={}", probe.data());
}
