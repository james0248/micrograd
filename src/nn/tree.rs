use std::collections::HashMap;

use crate::tensor::Tensor;

pub enum TensorTree {
    Leaf(Tensor),
    List(Vec<TensorTree>),
    Dict(HashMap<String, TensorTree>),
}
