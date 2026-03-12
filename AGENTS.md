# AGENTS.md

## Project Objective

- Maintain and improve the current `tangent` tensor/autodiff engine.
- Prioritize correctness, explicit data flow, and small understandable APIs.
- Treat the legacy tape-engine migration as complete; current work is feature, cleanup, and performance work on the new engine.

## Current Architecture

- The runtime engine is `tensor` + `autodiff`.
- Autodiff is based on internal tracing, linearization, and transpose-based pullback construction.
- Public differentiation APIs are:
  - `grad(f, inputs)`
  - `value_and_grad(f, inputs)`
- Model training uses explicit parameter tensors and explicit gradient application.
- Tensor storage is dense with generic strided layouts for view semantics.
- Public tensor materialization is explicit via `to_vec()`.

## Rust Conventions

- Prefer explicit ownership and borrowing over hidden behavior.
- Keep APIs small, readable, and easy to debug.
- Use `f32` for the current tensor/autodiff path.
- Use `Result` for fallible I/O and persistence paths where appropriate.
- Avoid unnecessary `unwrap`; keep panic paths explicit and descriptive when invariants are intentionally enforced.

## Dependency Policy

- Keep dependencies minimal.
- Current repo-approved dependencies already include `rand`, `serde`, and `bincode`.
- Do not add new dependencies without a clear justification.

## Testing Requirements

- Add explicit forward and backward tests for new tensor/autodiff behavior.
- Keep finite-difference checks in the test suite for gradient-sensitive changes.
- Treat `cargo test` as a mandatory quality gate.
- Prefer deterministic tests and seeded randomness.

## Documentation

- Keep `MIGRATION.md` as a concise migration summary, not an active work log.
- Track remaining work in:
  - `docs/PERF.md`
  - `docs/IMPROVEMENTS.md`
  - `docs/BACKLOG.md`
