use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

pub(crate) const MLP_CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CheckpointV1 {
    pub(crate) version: u32,
    pub(crate) dims: Vec<usize>,
    pub(crate) params: Vec<Vec<f32>>,
}

pub(crate) fn save_mlp_checkpoint<P: AsRef<Path>>(
    path: P,
    dims: &[usize],
    params: &[Vec<f32>],
) -> Result<(), String> {
    let path_ref = path.as_ref();
    let file = File::create(path_ref)
        .map_err(|err| format!("failed to create checkpoint {}: {err}", path_ref.display()))?;
    let mut writer = BufWriter::new(file);
    let payload = CheckpointV1 {
        version: MLP_CHECKPOINT_VERSION,
        dims: dims.to_vec(),
        params: params.to_vec(),
    };

    bincode::serialize_into(&mut writer, &payload).map_err(|err| {
        format!(
            "failed to serialize checkpoint {}: {err}",
            path_ref.display()
        )
    })?;

    writer
        .flush()
        .map_err(|err| format!("failed to flush checkpoint {}: {err}", path_ref.display()))?;

    Ok(())
}

pub(crate) fn load_mlp_checkpoint<P: AsRef<Path>>(path: P) -> Result<CheckpointV1, String> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref)
        .map_err(|err| format!("failed to open checkpoint {}: {err}", path_ref.display()))?;
    let mut reader = BufReader::new(file);
    bincode::deserialize_from(&mut reader).map_err(|err| {
        format!(
            "failed to deserialize checkpoint {}: {err}",
            path_ref.display()
        )
    })
}
