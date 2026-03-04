pub(super) fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub(super) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![0; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc = acc
            .checked_mul(shape[i])
            .expect("shape product overflow while computing strides");
    }
    strides
}

pub(super) fn offset_from_coords(coords: &[usize], strides: &[usize]) -> usize {
    debug_assert_eq!(coords.len(), strides.len());
    coords
        .iter()
        .zip(strides.iter())
        .map(|(c, s)| c * s)
        .sum::<usize>()
}

pub(super) fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];

    for i in 0..rank {
        let a_dim = if i >= rank - a.len() {
            a[i - (rank - a.len())]
        } else {
            1
        };
        let b_dim = if i >= rank - b.len() {
            b[i - (rank - b.len())]
        } else {
            1
        };

        assert!(
            a_dim == b_dim || a_dim == 1 || b_dim == 1,
            "broadcast shape mismatch: left={:?}, right={:?}",
            a,
            b
        );
        out[i] = a_dim.max(b_dim);
    }

    out
}

pub(super) fn broadcast_strides_for(
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> Vec<usize> {
    assert_eq!(
        input_shape.len(),
        input_strides.len(),
        "input shape/stride rank mismatch"
    );
    assert!(
        input_shape.len() <= output_shape.len(),
        "cannot broadcast higher-rank input: input={:?}, output={:?}",
        input_shape,
        output_shape
    );

    let out_rank = output_shape.len();
    let mut out = vec![0usize; out_rank];
    let offset = out_rank - input_shape.len();

    for out_axis in 0..out_rank {
        if out_axis < offset {
            out[out_axis] = 0;
            continue;
        }

        let in_axis = out_axis - offset;
        let in_dim = input_shape[in_axis];
        let out_dim = output_shape[out_axis];
        assert!(
            in_dim == out_dim || in_dim == 1,
            "broadcast shape mismatch at axis {}: input={:?}, output={:?}",
            out_axis,
            input_shape,
            output_shape
        );
        out[out_axis] = if in_dim == 1 {
            0
        } else {
            input_strides[in_axis]
        };
    }

    out
}

pub(super) fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    let rank = shape.len();
    let total = numel(shape);
    if total == 0 {
        return;
    }
    if rank == 0 {
        f(&[]);
        return;
    }

    let mut coords = vec![0usize; rank];
    for _ in 0..total {
        f(&coords);
        for axis in (0..rank).rev() {
            coords[axis] += 1;
            if coords[axis] < shape[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}

pub(super) fn reduced_shape(shape: &[usize], axis: usize, keepdim: bool) -> Vec<usize> {
    assert!(
        axis < shape.len(),
        "axis out of bounds: axis={} for shape {:?}",
        axis,
        shape
    );

    if keepdim {
        let mut out = shape.to_vec();
        out[axis] = 1;
        out
    } else if shape.len() == 1 {
        vec![1]
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == axis { None } else { Some(d) })
            .collect()
    }
}

pub(super) fn reduced_offset_from_input_coords(
    input_coords: &[usize],
    output_shape: &[usize],
    output_strides: &[usize],
    axis: usize,
    keepdim: bool,
) -> usize {
    if !keepdim && input_coords.len() == 1 {
        return 0;
    }

    if keepdim {
        let mut off = 0usize;
        for i in 0..input_coords.len() {
            let c = if i == axis { 0 } else { input_coords[i] };
            off += c * output_strides[i];
        }
        off
    } else {
        let mut off = 0usize;
        let mut out_i = 0usize;
        for (in_i, &coord) in input_coords.iter().enumerate() {
            if in_i == axis {
                continue;
            }
            off += coord * output_strides[out_i];
            out_i += 1;
        }
        debug_assert_eq!(out_i, output_shape.len());
        off
    }
}
