fn main() {
    if let Err(err) = tangent::mnist::run() {
        panic!("tangent failed: {err}");
    }
}
