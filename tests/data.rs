use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use micrograd::data::{load_and_split_mnist, load_mnist_csv};

fn header_line() -> String {
    let mut header = String::from("label");
    for i in 0..784 {
        header.push(',');
        header.push_str(&format!("pixel{i}"));
    }
    header
}

fn row_line(label: &str, pixel: u8) -> String {
    let mut row = String::from(label);
    let px = pixel.to_string();
    for _ in 0..784 {
        row.push(',');
        row.push_str(&px);
    }
    row
}

fn write_temp_csv(body: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be monotonic here")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "micrograd_mnist_test_{}_{}.csv",
        std::process::id(),
        nanos
    ));
    fs::write(&path, body).expect("temp csv write must succeed");
    path
}

#[test]
fn load_mnist_csv_parses_and_normalizes() {
    let csv = format!(
        "{}\n{}\n{}\n",
        header_line(),
        row_line("3", 255),
        row_line("7", 0)
    );
    let path = write_temp_csv(&csv);
    let samples = load_mnist_csv(path.to_str().expect("valid path")).expect("csv should parse");
    fs::remove_file(path).ok();

    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].label, 3);
    assert_eq!(samples[1].label, 7);
    assert!((samples[0].pixels[0] - 1.0).abs() < 1e-12);
    assert!((samples[1].pixels[0] - 0.0).abs() < 1e-12);
}

#[test]
fn load_mnist_csv_rejects_bad_header_column_count() {
    let csv = "label,pixel0\n1,10\n";
    let path = write_temp_csv(csv);
    let err = load_mnist_csv(path.to_str().expect("valid path")).expect_err("must fail");
    fs::remove_file(path).ok();
    assert!(err.contains("expected 785 columns"), "{err}");
}

#[test]
fn load_mnist_csv_rejects_invalid_label() {
    let csv = format!("{}\n{}\n", header_line(), row_line("x", 10));
    let path = write_temp_csv(&csv);
    let err = load_mnist_csv(path.to_str().expect("valid path")).expect_err("must fail");
    fs::remove_file(path).ok();
    assert!(err.contains("invalid label"), "{err}");
}

#[test]
fn load_and_split_mnist_is_deterministic_for_same_seed() {
    let mut lines = vec![header_line()];
    for i in 0..20 {
        lines.push(row_line(&(i % 10).to_string(), i as u8));
    }
    let csv = lines.join("\n") + "\n";
    let path = write_temp_csv(&csv);
    let path_str = path.to_str().expect("valid path");

    let (train_a, eval_a) = load_and_split_mnist(path_str, 0.2, 123).expect("first load ok");
    let (train_b, eval_b) = load_and_split_mnist(path_str, 0.2, 123).expect("second load ok");
    fs::remove_file(path).ok();

    assert_eq!(train_a.len(), 16);
    assert_eq!(eval_a.len(), 4);
    assert_eq!(train_a, train_b);
    assert_eq!(eval_a, eval_b);
}
