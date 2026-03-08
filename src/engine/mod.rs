mod engine;
mod kernels;
mod shape;
mod tensor;

pub use tensor::Tensor;

use std::cell::RefCell;

use engine::{ContextKind, Engine};

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    MatMul,
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Log,
    Sum { axis: usize, keepdim: bool },
    Max { axis: usize, keepdim: bool },
    Transpose { dim0: usize, dim1: usize },
    Relu,
    Mean,
}

thread_local! {
    static ENGINE: RefCell<Engine> = RefCell::new(Engine::new());
}

fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> R {
    ENGINE.with(|engine| f(&mut engine.borrow_mut()))
}

struct ContextGuard {
    kind: ContextKind,
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        with_engine(|engine| engine.exit_context(self.kind));
    }
}

fn with_context<F, T>(kind: ContextKind, f: F) -> T
where
    F: FnOnce() -> T,
{
    with_engine(|engine| engine.enter_context(kind));
    let guard = ContextGuard { kind };
    let out = f();
    drop(guard);
    out
}

pub fn with_grad<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    with_context(ContextKind::WithGrad, f)
}

pub fn no_grad<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    with_context(ContextKind::NoGrad, f)
}

pub fn clear_graph() {
    with_engine(|engine| engine.clear_graph());
}

pub fn reset_state() {
    with_engine(|engine| {
        *engine = Engine::new();
    });
}
