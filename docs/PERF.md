# PERF

## Current State

The new engine is functionally complete for the current training flow, but it is still correctness-first rather than performance-first.

## Priority Order

1. Residual deduplication
2. Contiguous fast paths
3. Matmul optimization
4. Interpreter allocation reuse
5. Training-step update efficiency

## Actionable Items

### 1. Residual Deduplication

- Deduplicate residuals within a single linearization pass.
- Use shared-storage identity plus layout metadata as the dedup key.
- Preserve first-seen order of residual capture.
- Limit the first version to residuals, not generic constants.

### 2. Contiguous Fast Paths

- Add fast paths for contiguous-contiguous elementwise ops.
- Add fast paths for common reduction shapes.
- Avoid generic stride-walking when both inputs and outputs are contiguous.

### 3. Matmul Optimization

- Reduce repeated offset computation in matmul kernels.
- Improve loop ordering for cache behavior.
- Add specialized fast paths for contiguous 2D and common batched cases.

### 4. Interpreter Reuse

- Reuse temporary buffers across trace execution where possible.
- Reduce cloning of `DenseTensor` values during interpreter execution.

### 5. Training-Step Overhead

- Reduce parameter snapshot cloning during `value_and_grad` training loops.
- Make optimizer application cheaper than composing generic tensor ops for every parameter update.
