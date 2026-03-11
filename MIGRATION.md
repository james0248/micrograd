# Autodiff Migration

## Purpose

This document tracks the migration from the legacy dynamic, tape-style autograd engine toward a more JAX-like functional tracing and transformation system.

The legacy engine remains in the repository until the new autodiff path is validated and trusted.

## Current State

- The package and crate name is currently `tangent`.
- The legacy engine still exists and remains the correctness baseline.
- The legacy tape-based engine is also used as an external parity reference for overlapping public operations.
- A new autodiff path already exists beside the legacy engine.
- The new autodiff path currently uses:
  - a flat SSA-like `Trace`
  - internal tracing via thread-local recorder state
  - interpreter-based execution
  - transpose-based reverse execution for the public `value_and_grad` path
- Internal linearization and transpose-based pullback now define the full autodiff path.
- Regular primal trace mode and the old direct-VJP baseline have been removed.
- `Recorder` is currently the generic low-level `Trace` builder, while `JvpRecorder` is a linearization-specific wrapper that manages tangent inputs and residual capture on top of `Recorder`.
- The public autodiff API is still:
  - `grad(f, inputs)`
  - `value_and_grad(f, inputs)`
- The scalar-output restriction currently belongs to the public `grad` / `value_and_grad` entrypoints, not tracing or linearization.
- The current MVP op set is:
  - `add`
  - `sub`
  - `mul`
  - `div`
  - `exp`
  - `log`
  - `sum_all`
  - `mean_all`

## Desired State

- Trace pure tensor functions internally without exposing tracer types in the public API.
- Add internal linearization support before changing public autodiff APIs.
- Treat regular primal trace mode as temporary scaffolding, not as part of the target architecture.
- Eventually support a more JAX-like transform stack centered around:
  - linearization
  - transpose and pullback construction
  - `vjp`
  - later, possibly `grad(f)(inputs)`-style public APIs
- Keep the migration validation-first: the old engine stays in place until the new path is reliable.

## Key Decisions

- `linearize` should land internal-first, not public-first.
- Existing `grad` and `value_and_grad` stay unchanged during the first linearization step.
- `MIGRATION.md` tracks autodiff migration only, not unrelated refactors.
- Progress is tracked by stage sections instead of a chronological work log.
- Linearize plus transpose is the current architectural direction.
- Linearized traces should capture forward-dependent coefficients as separate residuals, not as input-dependent trace constants.
- Public `value_and_grad` should run through the transpose-based path.
- `grad` should remain a thin wrapper over `value_and_grad` for now.
- During validation, the new autodiff path should also be compared against the older tape-based engine on overlapping forward and gradient behavior.
- Regular trace mode is not part of the target architecture and has been removed.
- `Recorder` should remain as the generic low-level IR/trace builder. `JvpRecorder` should not replace it one-for-one; after migration it can stay as a linearization-specific wrapper or be renamed if a better name emerges.

## Progress

### Done

- A side-by-side new autodiff path exists beside the legacy engine.
- The new code is split into `tensor` and `autodiff` modules.
- The test-only debug API has been removed.
- Scalar-output validation has moved from tracing to the public autodiff entrypoints.
- Root integration tests and internal white-box tests are split cleanly.
- Internal `linearize` exists.
- Separate-residual linearization exists.
- Internal transpose-based pullback exists.
- Public `value_and_grad` is routed through the transpose-based path.
- Direct VJP and regular primal tracing have been removed.
- Integration tests compare the new autodiff API against the legacy tape-based engine on overlapping operations.
- Integration tests include randomized stress coverage against finite differences and the legacy tape-based engine.

### Current

- Improve the new path itself rather than carrying migration scaffolding.
- The immediate engine task is residual deduplication inside linearization and pullback construction.

### Next

- Add residual deduplication once the new engine behavior is considered correct.
- Revisit naming cleanup around `JvpRecorder` and other migration-era terms if they still feel temporary.

### Later

- Only after that, consider internal or public `vjp` and possible `grad(f)(inputs)`-style APIs.

## Immediate Plan

- Reuse the existing `Trace`, `Recorder`, and interpreter infrastructure.
- Keep the public autodiff API surface unchanged while improving the new implementation.
- Validate transpose-based `value_and_grad` against the legacy tape-based engine and finite differences on the current MVP op set.
- Keep `Recorder` as the shared low-level builder while the migration converges. Treat `JvpRecorder` as a higher-level linearization helper rather than the permanent replacement for `Recorder`.
- Implement residual deduplication without changing the user-facing API.

## Open Questions

- When, if ever, should `linearize` become a public API?
- Should `TensorSpec` remain as a shared metadata type, or be simplified later?
- Should `JvpRecorder` keep its current name, or be renamed to something more architecture-neutral once the migration settles?
- What should the post-stabilization public transform surface look like: keep only `grad/value_and_grad`, or add `vjp` before changing higher-level APIs?
