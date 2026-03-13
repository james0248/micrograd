use std::collections::HashMap;

use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub enum TensorTree {
    Leaf(Tensor),
    List(Vec<TensorTree>),
    Dict(HashMap<String, TensorTree>),
}

impl TensorTree {
    pub fn leaves(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves(&self, out: &mut Vec<Tensor>) {
        match self {
            TensorTree::Leaf(t) => out.push(t.clone()),
            TensorTree::List(items) => {
                for item in items {
                    item.collect_leaves(out);
                }
            }
            TensorTree::Dict(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                for key in keys {
                    map[key].collect_leaves(out);
                }
            }
        }
    }

    pub fn replace_leaves(&self, new_leaves: &[Tensor]) -> TensorTree {
        let mut idx = 0;
        let result = self.replace_leaves_inner(new_leaves, &mut idx);
        assert_eq!(
            idx,
            new_leaves.len(),
            "replace_leaves: expected {} leaves, got {}",
            idx,
            new_leaves.len()
        );
        result
    }

    fn replace_leaves_inner(&self, new_leaves: &[Tensor], idx: &mut usize) -> TensorTree {
        match self {
            TensorTree::Leaf(_) => {
                assert!(
                    *idx < new_leaves.len(),
                    "replace_leaves: not enough tensors in slice"
                );
                let t = new_leaves[*idx].clone();
                *idx += 1;
                TensorTree::Leaf(t)
            }
            TensorTree::List(items) => {
                let new_items = items
                    .iter()
                    .map(|item| item.replace_leaves_inner(new_leaves, idx))
                    .collect();
                TensorTree::List(new_items)
            }
            TensorTree::Dict(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut new_map = HashMap::new();
                for key in keys {
                    new_map.insert(key.clone(), map[key].replace_leaves_inner(new_leaves, idx));
                }
                TensorTree::Dict(new_map)
            }
        }
    }
}
