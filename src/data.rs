use std::fs;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::utils::split_train_and_eval;

#[derive(Debug, Clone, PartialEq)]
pub struct MnistSample {
    pub label: u8,
    pub pixels: [f64; 784],
}

pub fn load_mnist_csv(path: &str) -> Result<Vec<MnistSample>, String> {
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

        let mut pixels = [0.0; 784];
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
            *slot = raw_px as f64 / 255.0;
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

pub fn load_and_split_mnist(
    path: &str,
    eval_ratio: f64,
    seed: u64,
) -> Result<(Vec<MnistSample>, Vec<MnistSample>), String> {
    let mut samples = load_mnist_csv(path)?;
    let mut rng = StdRng::seed_from_u64(seed);
    samples.shuffle(&mut rng);
    Ok(split_train_and_eval(samples, eval_ratio))
}
