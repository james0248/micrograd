use std::cell::RefCell;
use std::collections::HashMap;

use crate::{nn::tree::TensorTree, tensor::Tensor};

use rand::rngs::StdRng;

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = RefCell::new(None);
}

enum RunMode {
    Init,
    Apply,
}

struct Scope {
    run_mode: RunMode,
    pub params: TensorTree,
    prng_key: Option<StdRng>,

    path_stack: Vec<String>,
    counters: HashMap<String, HashMap<String, usize>>,
}

impl Scope {
    fn push_path(&mut self, type_name: &str) {
        let current_path = self.path_stack.join("/");

        let counter = self
            .counters
            .entry(current_path)
            .or_insert_with(HashMap::new);
        let count = counter.entry(type_name.to_string()).or_insert(0);

        let name = format!("{}_{}", type_name, count);
        *count += 1;

        self.path_stack.push(name);
    }

    fn pop_path(&mut self) {
        self.path_stack.pop();
    }

    fn current_dict_mut(&mut self) -> &mut HashMap<String, TensorTree> {
        let mut current_node = &mut self.params;

        // Traverse the path stack to find the current dictionary
        for path in &self.path_stack {
            if let TensorTree::Dict(map) = current_node {
                current_node = map
                    .entry(path.to_string())
                    .or_insert(TensorTree::Dict(HashMap::new()));
            } else {
                panic!("Corrupted track: Expected dictionary at {}", path);
            }
        }

        if let TensorTree::Dict(map) = current_node {
            map
        } else {
            unreachable!()
        }
    }

    fn param<F>(&mut self, name: &str, init_fn: F, shape: Vec<usize>) -> Tensor
    where
        F: Fn(&mut StdRng, Vec<usize>) -> Tensor,
    {
        match self.run_mode {
            RunMode::Init => {
                let prng_key = self
                    .prng_key
                    .as_mut()
                    .expect("PRNG key not found in Init mode");
                let param = init_fn(prng_key, shape);

                let current_dict = self.current_dict_mut();
                current_dict.insert(name.to_string(), TensorTree::Leaf(param.clone()));

                param
            }

            RunMode::Apply => {
                let current_dict = self.current_dict_mut();
                let param = current_dict
                    .get(name)
                    .unwrap_or_else(|| panic!("Parameter {} not found in scope", name));

                match param {
                    TensorTree::Leaf(tensor) => tensor.clone(),
                    _ => panic!("Parameter {} is not a leaf", name),
                }
            }
        }
    }
}

pub(crate) fn enter_init_scope(prng_key: StdRng) {
    let new_scope = Scope {
        run_mode: RunMode::Init,
        params: TensorTree::Dict(HashMap::new()),
        prng_key: Some(prng_key),
        path_stack: Vec::new(),
        counters: HashMap::new(),
    };

    SCOPE.with(|cell| {
        cell.borrow_mut().replace(new_scope);
    });
}

pub(crate) fn enter_apply_scope(params: TensorTree) {
    let new_scope = Scope {
        run_mode: RunMode::Apply,
        params,
        prng_key: Option::None,
        path_stack: Vec::new(),
        counters: HashMap::new(),
    };
    SCOPE.with(|cell| {
        cell.borrow_mut().replace(new_scope);
    });
}

pub(crate) fn exit_scope() -> TensorTree {
    SCOPE.with(|cell| {
        let scope = cell
            .borrow_mut()
            .take()
            .expect("Tried to exit scope but no scope found");
        scope.params
    })
}
pub(crate) fn push_path(type_name: &str) {
    SCOPE.with(|cell| {
        let mut scope = cell.borrow_mut();
        scope.as_mut().unwrap().push_path(type_name);
    });
}

pub(crate) fn pop_path() {
    SCOPE.with(|cell| {
        let mut scope = cell.borrow_mut();
        scope.as_mut().unwrap().pop_path();
    });
}

pub fn param<F>(name: &str, init_fn: F, shape: Vec<usize>) -> Tensor
where
    F: Fn(&mut StdRng, Vec<usize>) -> Tensor,
{
    SCOPE.with(|cell| {
        let mut scope = cell.borrow_mut();
        scope.as_mut().unwrap().param(name, init_fn, shape)
    })
}
