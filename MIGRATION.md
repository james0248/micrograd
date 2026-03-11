# Autodiff Migration

## Purpose

This document tracks the migration from the legacy dynamic, tape-style autograd engine toward a more JAX-like functional tracing and transformation system.

The legacy engine remains in the repository until the new autodiff path is validated and trusted.

## Current State

- The package and crate name is currently `tangent`.
- The legacy engine still exists and remains the correctness baseline.
- A new autodiff path already exists beside the legacy engine.
- The new autodiff path currently uses:
  - a flat SSA-like `Trace`
  - internal tracing via thread-local recorder state
  - interpreter-based execution
  - direct VJP generation from the forward trace
- Regular primal trace mode still exists, but only as migration scaffolding for the current direct-VJP implementation.
- The public autodiff API is still:
  - `grad(f, inputs)`
  - `value_and_grad(f, inputs)`
- The scalar-output restriction currently belongs to VJP, not tracing.
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
- Direct VJP is the current implementation baseline.
- Linearize plus transpose is the current architectural direction.
- Regular trace mode is temporary and will be removed or isolated after transpose-based VJP is validated.

## Progress

### Done

- A side-by-side new autodiff path exists beside the legacy engine.
- The new code is split into `tensor` and `autodiff` modules.
- The test-only debug API has been removed.
- Scalar-output validation has moved from tracing to VJP.
- Root integration tests and internal white-box tests are split cleanly.

### Current

- Internal `linearize` foundation.

### Next

- Add an internal JVP and `linearize` path over the current op set.
- Validate linearization with analytic checks and directional finite-difference tests.

### Later

- Add transpose-based pullback and VJP construction on top of linearized traces.
- Compare and validate the transpose-based path against the current direct VJP path.
- Only after that, consider changing public autodiff APIs.

## Immediate Plan

- Land the internal `jvp` and `linearize` module.
- Reuse the existing `Trace` and `Recorder` infrastructure.
- Keep the public autodiff API unchanged.
- Validate JVP rules for the full current MVP op set.

## Open Questions

- When, if ever, should `linearize` become a public API?
- Should direct VJP remain after transpose-based VJP is validated, or only stay as a migration baseline?
- Should `TensorSpec` remain as a shared metadata type, or be simplified later?
