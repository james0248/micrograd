# Rust Micrograd 1-Day Roadmap

## Stage Tracking

- Current Stage: `Complete (Post Stage 13)`
- Last Updated: `2026-03-04`
- Next Gate: `Stage 14 (performance tuning for ND broadcast kernels)`
- Stage 3 Status: `Accepted (forward ops + tests complete on 2026-03-03)`
- Stage 4 Status: `Accepted (backward engine + tests complete on 2026-03-03)`
- Stage 5 Status: `Accepted (operator ergonomics + parity tests complete on 2026-03-03)`
- Stage 6 Status: `Accepted (scalar MLP + noisy XOR sigmoid/BCE training demo with eval split complete on 2026-03-03)`
- Stage 7 Status: `Accepted (full verification + doc sync complete on 2026-03-03)`
- Stage 8A Status: `Accepted (roadmap lock for tape v2 + gated migration on 2026-03-03)`
- Stage 8B Status: `Accepted (isolated engine_v2 + parity/lifecycle tests complete on 2026-03-03)`
- Stage 8C Status: `Accepted (nn_v2 + v2_demo integration rehearsal complete on 2026-03-03)`
- Stage 8D Status: `Accepted (default path migrated to v2 and validated on 2026-03-03)`
- Stage 8E Status: `Accepted (legacy v1 quarantined + docs synced on 2026-03-03)`
- Stage 9A Status: `Accepted (explicit context lifecycle + naming lock on 2026-03-03)`
- Stage 9B Status: `Accepted (engine_v2 context runtime refactor complete on 2026-03-03)`
- Stage 9C Status: `Accepted (demo/test integration migrated to with_grad/no_grad on 2026-03-03)`
- Stage 9D Status: `Accepted (verification and doc sync complete on 2026-03-03)`
- Stage 10A Status: `Accepted (MNIST CSV loader + generic split utility complete on 2026-03-04)`
- Stage 10B Status: `Accepted (default main migrated from XOR demo to MNIST training loop on 2026-03-04)`
- Stage 10C Status: `Accepted (MNIST data/split tests + shape verification complete on 2026-03-04)`
- Stage 11A Status: `Accepted (rank-2 tensor autograd core + fused CE op complete on 2026-03-04)`
- Stage 11B Status: `Accepted (tensor MLP + batched training pipeline complete on 2026-03-04)`
- Stage 11C Status: `Accepted (tensor engine/nn test coverage and verification complete on 2026-03-04)`
- Stage 11D Status: `Accepted (default main cut over to tensor MNIST runner on 2026-03-04)`
- Stage 11E Status: `Accepted (tensor CE moved to losses module and composed from core ops on 2026-03-04)`
- Stage 11F Status: `Accepted (SGD update logic moved from tensor core into optimizer module on 2026-03-04)`
- Stage 11G Status: `Accepted (tensor training/tests rewired to losses+optimizer modules on 2026-03-04)`
- Stage 12A Status: `Accepted (matmul kernels extracted to dedicated tensor kernels module on 2026-03-04)`
- Stage 12B Status: `Accepted (scoped-thread parallel matmul integrated for forward and backward paths on 2026-03-04)`
- Stage 12C Status: `Accepted (epoch timing instrumentation added to tensor MNIST demo on 2026-03-04)`
- Stage 12D Status: `Accepted (verification complete; multicore speed validation delegated to user machine on 2026-03-04)`
- Stage 13 Status: `Accepted (N-D broadcasting, generic reductions, and batched matmul autograd complete on 2026-03-04)`

## Thematic Priorities

- Parity first: reproduce scalar micrograd behavior before optimization-heavy refactors.
- Correctness first: trust tests and gradient checks before adding convenience features.
- Minimal dependencies: use only `rand` plus Rust standard library.
- Fast delivery: complete a functional end-to-end result in one day.
- Staged performance: optimize only after a correct baseline is in place.

## Done Target

- `cargo test` passes for engine and training-related checks.
- `cargo run` executes deterministic MNIST CSV classification training and reports train/eval loss + accuracy.
- Default runnable path uses tensor autograd for batched MNIST training (scalar path retained for reference/tests).
- Dependency rule is respected (`rand` only).

## Milestones

### Stage 0 - Planning/Spec

Scope:

- Finalize and approve `AGENTS.md` and `ROADMAP.md`.
- Lock defaults for architecture, testing, and scope.

Acceptance:

- Documents exist and align with agreed constraints.

### Stage 1 - Crate Skeleton

Scope:

- Set module layout (`engine`, optional `nn`, `main` demo wiring).
- Ensure project compiles with placeholder structures.

Acceptance:

- `cargo check` passes.

### Stage 2 - Value Core

Scope:

- Implement `Value` internals: `data`, `grad`, graph links, and op metadata.
- Add constructors/accessors and core state mutation methods.

Acceptance:

- Unit tests validate construction/state behavior.

### Stage 3 - Forward Ops

Scope:

- Implement scalar forward ops used by micrograd flow.
- Start with methods and explicit composition.

Acceptance:

- Unit tests verify forward numeric correctness.

### Stage 4 - Backward Engine

Scope:

- Implement topological ordering and reverse traversal.
- Add local gradient rules and accumulation logic.

Acceptance:

- Backward tests pass, including shared-subgraph accumulation.

### Stage 5 - API Ergonomics

Scope:

- Add operator trait overloads after method path is stable.
- Keep method API available for clarity and testing.

Acceptance:

- Expression-style tests pass with identical gradients.

### Stage 6 - Tiny NN + Training Demo

Scope:

- Implement minimal `Neuron`, `Layer`, `MLP` on scalar `Value`.
- Add deterministic initialization via `rand`.
- Run a tiny in-code training loop from `main` with sigmoid + BCE.
- Add a held-out eval split and report train/eval metrics.

Acceptance:

- Demo runs and prints improving BCE/accuracy trends across epochs.

### Stage 7 - Final Verification

Scope:

- Run full validation and finalize known limitations.
- Confirm done target and stage closure.

Acceptance:

- `cargo test` and `cargo run` satisfy done target criteria.

### Stage 8A - Roadmap Lock

Scope:

- Lock architecture direction to a PyTorch-like hidden global tape (`no Rc`, `no RefCell`) for v2.
- Define gated rollout: isolated v2 first, migration second.

Acceptance:

- `ROADMAP.md` includes Stage 8A-8E with clear gates and acceptance criteria.

### Stage 8B - Engine V2 (Isolated)

Scope:

- Implement `engine_v2` in parallel with current engine.
- Keep existing `engine` path untouched during this stage.
- Build parity tests for v2 (`tests/engine_v2.rs`) including lifecycle safety checks.

Acceptance:

- `engine_v2` tests pass.
- Existing default engine path remains functional and unchanged.

### Stage 8C - V2 Integration Rehearsal

Scope:

- Implement `nn_v2` and a dedicated v2 demo entrypoint.
- Validate full training loop behavior on noisy XOR with train/eval reporting.

Acceptance:

- `cargo run --bin v2_demo` shows improving BCE/accuracy trend.
- `tests/nn_v2.rs` passes.

### Stage 8D - Migration Cutover

Scope:

- Switch default `engine`/`nn`/`main` to v2 implementation after isolated validation.
- Preserve current public behavior and interfaces where possible.

Acceptance:

- Default `cargo test` and `cargo run` pass on v2-backed path.

### Stage 8E - Post-Migration Cleanup

Scope:

- Remove legacy engine path (or quarantine as reference module) and sync docs.
- Confirm limitations and stage status reflect post-migration state.

Acceptance:

- No ambiguous dual-default engine path.
- Docs and stage tracking are consistent.

Completion note:

- Legacy implementation is quarantined in `engine_v1`/`nn_v1` while default exports use v2.

### Stage 9A - Context Lifecycle Lock

Scope:

- Lock lifecycle model to explicit contexts: `with_grad(...)` for training graph recording and `no_grad(...)` for eval/non-recording execution.
- Drop `*_scope` naming from the public API direction.
- Keep scalar scope and dependency constraints unchanged.

Acceptance:

- Roadmap and implementation both use plain `with_grad` / `no_grad` naming.

### Stage 9B - Engine V2 Context Runtime

Scope:

- Refactor `engine_v2` to require active context for graph-building ops.
- Remove implicit always-on graph recording assumptions.
- Bind temp-node lifetime to context boundaries and keep parameter lifetime persistent.

Acceptance:

- `engine_v2` supports strict-context execution with stale-handle safety and non-nested `with_grad`.

### Stage 9C - Integration Migration

Scope:

- Move demo and tests to context-bounded execution.
- Keep training in `with_grad` and evaluation in `no_grad`.
- Ensure multi-loss behavior works inside the same `with_grad` context.

Acceptance:

- `main`, `v2_demo`, and test suites run on the explicit context model.

### Stage 9D - Verification + Sync

Scope:

- Run final checks and align roadmap status with implemented behavior.
- Confirm performance-lifecycle notes reflect context-bounded graph memory.

Acceptance:

- `cargo test`, `cargo run`, and `cargo run --bin v2_demo` pass on the Stage 9 model.

### Stage 10A - Data Pipeline (MNIST CSV)

Scope:

- Add MNIST CSV parsing from `data/train.csv` with label/pixel validation and pixel normalization.
- Add generic split utility `split_train_and_eval<T>(Vec<T>, eval_ratio)` for train/eval partitioning.
- Keep split deterministic by shuffling first with seeded RNG.

Acceptance:

- MNIST loader returns valid samples with normalized pixels.
- Generic split utility is reusable and tested independently.

### Stage 10B - MNIST Training Integration

Scope:

- Replace XOR demo in `main` with MNIST training/eval loop.
- Use one-hidden-layer scalar MLP (`[784, hidden, 10]`) and softmax cross-entropy.
- Report epoch-level train/eval metrics.

Acceptance:

- `cargo run` starts MNIST training from `data/train.csv`.
- Logs include train/eval loss and accuracy.

### Stage 10C - Verification + Docs Sync

Scope:

- Add integration tests for data parsing and split determinism.
- Add MNIST-output-shape test in NN suite.
- Sync roadmap stage tracking for Stage 10 completion.

Acceptance:

- `cargo test` passes with new data and utility tests.
- Stage tracking reflects Stage 10 accepted state.

### Stage 11A - Tensor Engine Core

Scope:

- Add additive tensor autograd runtime with rank-2 focused semantics.
- Implement tensor ops required for MNIST training: `matmul`, `add_row_bias`, `relu`, `mean`, and fused `cross_entropy_with_logits`.
- Keep context lifecycle model aligned with scalar (`with_grad` / `no_grad`) and preserve stale-handle safety.

Acceptance:

- Tensor forward/backward core ops are implemented and covered by dedicated tests.
- Scalar engine path remains intact and passing.

### Stage 11B - Tensor NN + Batched Pipeline

Scope:

- Add `nn` with one-hidden-layer MLP support over engine parameters.
- Add batched MNIST training runtime (`mnist`) using mini-batch SGD.
- Keep dependency policy unchanged (`rand` + std only).

Acceptance:

- `cargo run --bin mnist` runs end-to-end batched training and prints train/eval metrics.

### Stage 11C - Verification + Accuracy Gate

Scope:

- Add tensor-specific tests for shape checks, backward behavior, and training smoke.
- Validate tensor runtime and scalar runtime together in default test suite.

Acceptance:

- `cargo test` passes including tensor and scalar suites.

### Stage 11D - Default Path Cutover

Scope:

- Switch default `main` to tensor MNIST runner after tensor verification.
- Keep scalar modules and tests for baseline comparison and learning reference.

Acceptance:

- `cargo run` uses tensor MNIST training path.

### Stage 11E - Loss API Extraction

Scope:

- Remove fused CE from tensor core API/internal op dispatch.
- Introduce `losses` module and compose CE from engine primitives inside grad/no-grad contexts.
- Keep tensor core focused on reusable primitive ops plus activation/reduction basics.

Acceptance:

- No `cross_entropy_with_logits` op/method remains in tensor core internals.
- CE loss is available via `losses::cross_entropy_with_logits`.

### Stage 11F - Optimizer Extraction

Scope:

- Remove parameter update step logic from tensor core.
- Add `optim` module with `Optimizer` trait and `Sgd` implementation.
- Drive param updates through optimizer APIs from training loops.

Acceptance:

- No `sgd_step` logic remains in tensor core engine/tensor APIs.
- Training path updates weights through `optim::Sgd`.

### Stage 11G - Integration + Verification

Scope:

- Rewire tensor training/eval and tests to use external losses and optimizer modules.
- Add dedicated unit/integration coverage for losses and optimizer behavior.

Acceptance:

- `cargo test` passes with new losses/optimizer suites.
- Default tensor training path compiles and runs with extracted module boundaries.

### Stage 12A - Tensor Kernel Extraction

Scope:

- Move matmul compute paths into dedicated `tensor/kernels` module.
- Keep tensor API surface stable while isolating low-level compute details.

Acceptance:

- `Tensor::matmul` and `MatMul2D` backward rely on kernel helpers from `tensor/kernels`.

### Stage 12B - Scoped-Thread Matmul

Scope:

- Implement safe-Rust multithreading for matmul via `std::thread::scope`.
- Apply row-chunk parallelism to forward matmul and both matmul backward gradient paths.
- Use transposed-right-matrix kernel layout for cache-friendly contiguous dot products.

Acceptance:

- Threaded kernel path is active for matmul forward/backward without changing public APIs.

### Stage 12C - Timing Instrumentation

Scope:

- Add epoch-level wall-clock timing to tensor MNIST training logs.
- Keep training metrics and hyperparameters unchanged for before/after comparison.

Acceptance:

- `cargo run` prints per-epoch timing (`epoch_time_ms`) alongside loss/accuracy metrics.

### Stage 12D - Verification + Deployment Note

Scope:

- Validate correctness via full test suite after threaded kernel integration.
- Document that speedup validation is done on user multicore hardware when local runtime reports single logical CPU.

Acceptance:

- `cargo test` passes post-threading refactor.

### Stage 13 - N-D Broadcast Tensor Core

Scope:

- Extend tensor elementwise ops (`add/sub/mul/div`) to NumPy/PyTorch-style broadcasting across arbitrary ranks.
- Add generic reduction ops `sum(axis, keepdim)` and `max(axis, keepdim)` with autograd support.
- Replace rowwise-specialized ops (`add_row_bias`, `sub_rowwise`, `div_rowwise`, `sum_rows_keepdim`) with generic broadcast/reduction usage.
- Extend matmul to rank `>= 2` with broadcasted leading batch dimensions in both forward and backward.
- Update CE and NN paths to depend only on generic tensor primitives.

Acceptance:

- Full `cargo test` suite passes with new broadcast/reduction/matmul coverage.
- Batched matmul gradients accumulate correctly when one side uses broadcasted batch dimensions.
- CE stabilization and training path run through `max/sum` and broadcasted binary ops only.

## Performance Plan

### 1) Baseline Measurement (v1)

- Record current v1 timings (`cargo run` training loop) using `std::time::Instant`.
- Keep seed/dataset/epoch settings identical for v1-v2 comparisons.

### 2) V2 Cost Targets

- Remove `Rc` reference counting overhead.
- Remove `RefCell` runtime borrow checks and churn.
- Reduce per-op temporary allocations where possible.

### 3) Isolated V2 Measurement

- Measure v2 in isolation (`engine_v2` + `v2_demo`) with same workload.
- Compare epoch-level timings and memory growth behavior against v1.

### 3.5) Context-Bounded Memory Policy

- Build and backprop inside `with_grad(...)`; clear temp graph at context exit.
- Use `no_grad(...)` for eval/inference to avoid recording op parents.
- Prevent unbounded tape growth across epochs by design.

### 4) Migration Guardrail

- Do not migrate default path until v2 parity tests and demo behavior are green.

### 5) Correctness Guardrails

- No optimization is accepted unless gradient tests stay green.
- Re-run finite-difference checks after performance-related refactors.

### 6) Dependency Guardrail

- Do not add benchmark/profiling crates in day-1 scope.
- Use explicit std-only timing harnesses.

## Public Interfaces To Build

- `Value` wrapper type for scalar data + grad + graph edges.
- `backward()` autograd entrypoint.
- `backward_with_options(retain_graph: bool)` kept for compatibility while lifecycle is context-bounded.
- Trait ops (`Add`, `Mul`, `Neg`, `Sub`, `Div`) after method-based ops.
- Unary value ops include `tanh`, `exp`, `log`, `relu`.
- `Value::parameter(data)` for persistent trainable scalars.
- `with_grad(|| ...)` helper for explicit graph-recording context.
- `no_grad(|| ...)` helper for eval/inference without graph growth.
- `Mlp::new(dims: &[usize], seed: u64)` and tuple parameters API `(weights, biases)`.

## Required Test Scenarios

- Forward arithmetic correctness for core scalar ops.
- Backward chain-rule correctness on known expressions.
- Gradient accumulation correctness for reused nodes.
- Finite-difference sanity checks vs analytical gradients.
- Training smoke test with measurable BCE decrease on noisy XOR data.
- Deterministic initialization checks and parameter-shape checks.

## Final Limitations

- Scalar-only autograd and MLP implementation (no tensors/batching).
- Hidden global runtime is thread-local with explicit `with_grad` / `no_grad` context boundaries.
- Single-process demo training with full-dataset gradient descent (no mini-batch support).

## Milestone Completion Note

- At the end of each stage, collect concepts used and run an interactive English learning section.
- This learning step is required, but implementation progress remains the primary metric.
