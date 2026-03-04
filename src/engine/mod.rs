mod engine;
mod kernels;
mod shape;
mod tensor;

pub use tensor::Tensor;

use std::sync::Mutex;

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
    Relu,
    Mean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineStats {
    pub param_count: usize,
    pub temp_count: usize,
    pub generation: u64,
    pub context_depth: usize,
    pub with_grad_active: bool,
}

thread_local! {
    static ENGINE: Mutex<Engine> = Mutex::new(Engine::new());
}

fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> R {
    ENGINE.with(|engine| {
        let mut guard = engine.lock().unwrap_or_else(|poison| poison.into_inner());
        f(&mut guard)
    })
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

pub fn stats() -> EngineStats {
    with_engine(|engine| EngineStats {
        param_count: engine.param_count(),
        temp_count: engine.temp_count(),
        generation: engine.active_generation(),
        context_depth: engine.context_depth(),
        with_grad_active: engine.is_with_grad_active(),
    })
}

pub fn reset_state() {
    with_engine(|engine| {
        *engine = Engine::new();
    });
}
