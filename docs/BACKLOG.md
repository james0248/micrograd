# BACKLOG

## Current State

These items are intentionally deferred because they need product/API decisions or broader follow-up work.

## Open Areas

### Public Transform Surface

- Should the public surface remain only:
  - `grad`
  - `value_and_grad`
- Or should it later add:
  - `vjp`
  - public `linearize`
  - `grad(f)(inputs)`-style wrappers

### Reshape Semantics

- Define explicit copy vs no-copy behavior before adding `reshape`.
- Do not add hidden materialization semantics.

### View API Scope

- Decide whether `transpose(dim0, dim1)` is enough for now.
- Decide whether general permutation/view APIs are needed later.

### Reduction Surface After Generalization

- Decide whether `max` should remain single-axis only.
- Decide whether other reductions should follow the same multi-axis pattern as `sum` and `mean`.

### Semantic Stability Questions

- Keep reviewing whether evenly split `max` ties remain the long-term policy.
- Revisit naming and public API ergonomics after the post-migration cleanup settles.
