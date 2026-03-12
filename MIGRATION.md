# Autodiff Migration

## Purpose

This document tracks the migration from the legacy dynamic, tape-style autograd engine toward a more JAX-like functional tracing and transformation system.

## Current State

- The package and crate name is currently `tangent`.
- The new autodiff path is now the only engine/runtime path in the crate.
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
- The current autodiff-supported op set is:
  - `add`
  - `sub`
  - `mul`
  - `div`
  - `exp`
  - `log`
  - `sum_all`
  - `mean_all`
  - `transpose(dim0, dim1)`
- The eager tensor path now also supports the batched forward ops needed by training:
  - broadcasted elementwise `add` / `sub` / `mul` / `div`
  - `sum(axis, keepdim)`
  - `max(axis, keepdim)`
  - `relu`
  - `matmul`
- The new tensor path now uses dense backing storage with generic strided layouts for view semantics.
- Public tensor inspection on the new path is explicit materialization via `to_vec()`.
- The batched op set is now representable and executable in the new `Trace` IR and interpreter.
- `linearize` now records the full current batched training subset:
  - broadcasted elementwise `add` / `sub` / `mul` / `div`
  - `sum(axis, keepdim)`
  - `max(axis, keepdim)`
  - `relu`
  - `matmul`
- Public autodiff now supports the full current batched op set needed by the training path:
  - broadcasted elementwise `add` / `sub` / `mul` / `div`
  - `sum(axis, keepdim)`
  - `max(axis, keepdim)`
  - `relu`
  - `matmul`
- `max(axis, keepdim)` now uses evenly split tie gradients.
- The current training stack (`nn`, `losses`, `optim`, `mnist`) now runs on the new `tensor` + `autodiff` path.
- The legacy engine has been removed from the crate and repository.
- General `sum(axes, keepdim)` and `mean(axes, keepdim)` are part of the long-term direction; the current `sum_all` / `mean_all` API is still transitional.

## Desired State

- Trace pure tensor functions internally without exposing tracer types in the public API.
- Add internal linearization support before changing public autodiff APIs.
- Treat regular primal trace mode as temporary scaffolding, not as part of the target architecture.
- Eventually support a more JAX-like transform stack centered around:
  - linearization
  - transpose and pullback construction
  - `vjp`
  - later, possibly `grad(f)(inputs)`-style public APIs
- Support the existing batched training flow on the new engine, including:
  - broadcasted elementwise ops
  - axis reductions
  - `max(axis, keepdim)`
  - `relu`
  - `matmul`
  - generic strided layout/view semantics for broadcasted and transposed tensors
  - no hidden materialization in public tensor transforms
- Defer `reshape` until there is a clear policy for when a non-copy reshape is layout-compatible and when a copy would otherwise be required.
- Keep the migration validation-first while strengthening the new-engine-only test surface.

## Key Decisions

- `linearize` should land internal-first, not public-first.
- Existing `grad` and `value_and_grad` stay unchanged during the first linearization step.
- `MIGRATION.md` tracks autodiff migration only, not unrelated refactors.
- Progress is tracked by stage sections instead of a chronological work log.
- Linearize plus transpose is the current architectural direction.
- Linearized traces should capture forward-dependent coefficients as separate residuals, not as input-dependent trace constants.
- Public `value_and_grad` should run through the transpose-based path.
- `grad` should remain a thin wrapper over `value_and_grad` for now.
- The training migration should preserve the current batched training style rather than rewriting training into per-sample loops.
- Stage 1 should use dense backing storage with generic strided layouts.
- `transpose(dim0, dim1)` is enough as the first public view transform.
- The new tensor path should not perform hidden materialization; unsupported view transforms should fail rather than silently copy.
- `reshape` is explicitly deferred until after the batched migration because no-copy reshape compatibility rules are not yet part of the design.
- Stage 2 compute ops should return fresh contiguous outputs; only view ops remain metadata-only.
- Stage 3 should keep broadcasting implicit in IR:
  - `Add` / `Sub` / `Mul` / `Div` remain the operations
  - shape inference and the interpreter handle broadcast semantics from operand shapes
  - explicit helper ops such as `BroadcastTo` can be added later only if backward needs them
- Stage 3 reductions should use single-axis operations with `axis: usize` and `keepdim: bool`.
- Stage 3 should trace and interpret the expanded batched op set through `linearize` where the local linearization rule is already settled, but public autodiff should still fail fast on those ops until Stage 4 transpose rules exist.
- Stage 3 `max` work is forward-only and stays out of `linearize`; tie-gradient policy remains a Stage 4 concern.
- Stage 3 `matmul` should use full legacy semantics immediately: rank `>= 2`, inner-dimension match, and batch-dimension broadcasting.
- General `sum(axes, keepdim)` and `mean(axes, keepdim)` should replace `sum_all` / `mean_all` later; Stage 3 does not add them yet.
- Stage 4 `max(axis, keepdim)` should split tie gradients evenly, matching value-only `amax`-style semantics.
- Batched training support should be implemented in stages:
  - layout and view foundation
  - eager batched ops
  - IR and interpreter expansion
  - autodiff rule expansion
  - training stack migration
  - legacy engine removal
- Autodiff expansion for the batched op set should land in this order:
  - broadcasted `add` / `sub` plus `sum(axis, keepdim)` and `SumToShape`
  - broadcasted `mul` / `div`
  - `relu`
  - `matmul`
  - `max(axis, keepdim)`
- Regular trace mode is not part of the target architecture and has been removed.
- `Recorder` should remain as the generic low-level IR/trace builder. `JvpRecorder` should not replace it one-for-one; after migration it can stay as a linearization-specific wrapper or be renamed if a better name emerges.

## Progress

### Done

- The new autodiff path is fully in place and owns the runtime/training path.
- The new code is split into `tensor` and `autodiff` modules.
- The test-only debug API has been removed.
- Scalar-output validation has moved from tracing to the public autodiff entrypoints.
- Root integration tests and internal white-box tests are split cleanly.
- Internal `linearize` exists.
- Separate-residual linearization exists.
- Internal transpose-based pullback exists.
- Public `value_and_grad` is routed through the transpose-based path.
- Direct VJP and regular primal tracing have been removed.
- Integration tests include deterministic and randomized finite-difference stress coverage.
- Stage 1 layout work is complete:
  - dense backing storage plus generic strided layouts
  - explicit `to_vec()` materialization on the new tensor type
  - metadata-only `transpose(dim0, dim1)`
  - transpose support through eager execution, linearization, and pullback
- Stage 2 eager batched forward work is complete:
  - broadcasted elementwise `add` / `sub` / `mul` / `div`
  - `sum(axis, keepdim)`
  - `max(axis, keepdim)`
  - `relu`
  - `matmul`
  - compute ops return contiguous outputs while `transpose` remains a view
  - this forward surface is now fully covered by Stage 4 autodiff
- Stage 3 IR/interpreter expansion is complete:
  - `Trace` and the interpreter now support `sum(axis, keepdim)`, `max(axis, keepdim)`, `relu`, and `matmul`
  - broadcasting remains implicit in IR for `add` / `sub` / `mul` / `div`
  - `linearize` now records broadcasted binaries, `sum(axis, keepdim)`, `relu`, and `matmul`
  - this IR/interpreter expansion is now exercised end-to-end by Stage 4 autodiff
- Stage 4 autodiff expansion is complete:
  - broadcasted `add` / `sub` now reduce pullback cotangents with `SumToShape`
  - broadcasted `mul` / `div` now use coefficient residuals plus `SumToShape`
  - `sum(axis, keepdim)` now pulls back through `ExpandToShape`
  - `relu` now differentiates through captured derivative masks
  - `matmul` now differentiates with full legacy batch-broadcast semantics
  - `max(axis, keepdim)` now linearizes through tie-weight residuals and splits ties evenly
  - public `value_and_grad` and `grad` now work across the full current batched op set
- Stage 5 training-stack migration is complete:
  - `nn`, `losses`, `optim`, and `mnist` now use `tensor::Tensor` and `autodiff::value_and_grad`
  - model updates now use explicit gradients instead of engine-owned grad buffers
  - SGD now consumes explicit grads and mutates model state through `Parameterized`
  - checkpoint save/load remains compatible while using `to_vec()` and mutable weight loading
  - the training loop preserves the current batched style on the new engine
- Stage 6 legacy-engine removal is complete:
  - the `engine` module and engine-specific tests have been deleted
  - public tests now rely on finite differences, white-box autodiff coverage, training smoke tests, and checkpoint/MNIST integration

### Current

- Strengthen the post-migration new-engine baseline:
  - finite-difference coverage
  - new-engine white-box tests
  - training smoke tests
  - checkpoint and MNIST integration tests
- Residual deduplication is the next engine-improvement phase.

### Next

- Add residual deduplication once the batched new-engine path is considered stable.
- Add general `sum(axes, keepdim)` and `mean(axes, keepdim)`, then retire `sum_all` / `mean_all`.

### Later

- Revisit `reshape` support once the layout model is stable and there is a clear explicit policy for copy vs non-copy behavior.
- Revisit naming cleanup around `JvpRecorder` and other migration-era terms if they still feel temporary.
- Only after that, consider internal or public `vjp` and possible `grad(f)(inputs)`-style APIs.

## Immediate Plan

- Reuse the existing `Trace`, `Recorder`, and interpreter infrastructure while expanding the op set.
- Keep the public autodiff API surface unchanged while adding the batched semantics needed by training.
- Implement the migration in six stages:
  - Stage 1: generic strided layout foundation for broadcasted and transposed views, with dense backing storage and no hidden materialization (done)
  - Stage 2: eager batched forward ops (`broadcast`, `sum`, `max`, `relu`, and `matmul`) (done)
  - Stage 3: IR and interpreter support for the new op set plus internal helper ops, with implicit broadcast semantics and single-axis reductions (done)
  - Stage 4: autodiff expansion in the fixed order (done):
    - Stage 4.1: broadcasted `add` / `sub`, `sum(axis, keepdim)`, and `SumToShape`
    - Stage 4.2: broadcasted `mul` / `div`
    - Stage 4.3: `relu`
    - Stage 4.4: `matmul`
    - Stage 4.5: `max(axis, keepdim)`
  - Stage 5: training stack migration onto the new engine (done)
  - Stage 6: legacy engine removal and validation cleanup (done)
- Keep `Recorder` as the shared low-level builder while the migration converges. Treat `JvpRecorder` as a higher-level linearization helper rather than the permanent replacement for `Recorder`.
- Validate each stage with new-engine white-box tests, finite differences, training smoke tests, and checkpoint/MNIST integration.

## Open Questions

- When, if ever, should `linearize` become a public API?
- How much layout generality should remain after batched training is working beyond generic strided broadcast/transpose views?
- Should `transpose(dim0, dim1)` be the only public view op for now, or should a more general permutation API be introduced later?
- Should `TensorSpec` remain as a shared metadata type, or be simplified later?
- Should `JvpRecorder` keep its current name, or be renamed to something more architecture-neutral once the migration settles?
- What should the post-stabilization public transform surface look like: keep only `grad/value_and_grad`, or add `vjp` before changing higher-level APIs?
