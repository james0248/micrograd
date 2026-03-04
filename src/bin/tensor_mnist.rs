fn main() {
    if let Err(err) = micrograd::tensor_mnist::run() {
        panic!("tensor_mnist failed: {err}");
    }
}
