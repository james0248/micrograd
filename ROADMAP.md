# Rust Micrograd 1-Day Roadmap

## Stage Tracking

- Current Stage: `Stage 4 - Backward Engine`
- Last Updated: `2026-03-03`
- Next Gate: `Stage 4 acceptance`
- Stage 3 Status: `Accepted (forward ops + tests complete on 2026-03-03)`

## Thematic Priorities

- Parity first: reproduce scalar micrograd behavior before optimization-heavy refactors.
- Correctness first: trust tests and gradient checks before adding convenience features.
- Minimal dependencies: use only `rand` plus Rust standard library.
- Fast delivery: complete a functional end-to-end result in one day.
- Staged performance: optimize only after a correct baseline is in place.

## Done Target

- `cargo test` passes for engine and training-related checks.
- `cargo run` executes a tiny deterministic training demo and shows a downward loss trend.
- Implementation remains scalar-only.
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
- Run a tiny in-code training loop from `main`.

Acceptance:

- Demo runs and prints improving loss across epochs.

### Stage 7 - Final Verification

Scope:

- Run full validation and finalize known limitations.
- Confirm done target and stage closure.

Acceptance:

- `cargo test` and `cargo run` satisfy done target criteria.

## Performance Plan

### 1) Baseline Measurement (after correctness baseline)

- Time forward/backward/epoch loop using `std::time::Instant`.
- Record baseline timings in roadmap notes or run log.

### 2) Hotspot Identification

- Inspect these expensive paths.
- Frequent `clone` calls on graph values.
- Repeated `borrow`/`borrow_mut` churn.
- Repeated topo vector allocations.
- Repeated graph traversal overhead.

### 3) Low-Risk Optimizations

- Pre-allocate traversal vectors where possible.
- Shorten mutable borrow lifetimes and reduce repeated borrows.
- Remove avoidable temporary allocations/clones.
- Reuse buffers across training iterations when safe.

### 4) Structural Optimization Track (Post Day-1)

- Evaluate arena/index-based graph storage as a follow-up branch.
- Keep this out of day-1 critical path unless baseline is too slow.

### 5) Correctness Guardrails

- No optimization is accepted unless gradient tests stay green.
- Re-run finite-difference checks after performance-related refactors.

### 6) Dependency Guardrail

- Do not add benchmark/profiling crates in day-1 scope.
- Use explicit std-only timing harnesses.

## Public Interfaces To Build

- `Value` wrapper type for scalar data + grad + graph edges.
- `backward()` autograd entrypoint.
- Trait ops (`Add`, `Mul`, `Neg`, `Sub`, `Div`) after method-based ops.
- `Neuron::new`, `Layer::new`, `MLP::new`, `forward`, `parameters`.

## Required Test Scenarios

- Forward arithmetic correctness for core scalar ops.
- Backward chain-rule correctness on known expressions.
- Gradient accumulation correctness for reused nodes.
- Finite-difference sanity checks vs analytical gradients.
- Training smoke test with measurable loss improvement.

## Milestone Completion Note

- At the end of each stage, collect concepts used and run an interactive English learning section.
- This learning step is required, but implementation progress remains the primary metric.
