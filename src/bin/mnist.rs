fn main() {
    if let Err(err) = micrograd::mnist::run() {
        panic!("mnist failed: {err}");
    }
}
