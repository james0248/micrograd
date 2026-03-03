# AGENTS.md

## Project Objective

- Build a Rust version of Karpathy-style micrograd as a 1-day project.
- Keep scope scalar-first and parity-first, then evolve complexity by roadmap milestone.
- Prioritize project progress and correctness; learning is a guided side effect.

## Idiomatic Rust Conventions

- Use explicit ownership and borrowing; avoid hiding lifetime/ownership behavior.
- Prefer small, clear APIs and explicit data flow over clever abstractions.
- Use `f64` for scalar math unless a milestone explicitly changes this.
- Use `Result` for fallible paths where reasonable; avoid unnecessary `unwrap`.
- Keep code readable and modular (`engine`, `nn`, `main` orchestration).
- External dependency policy: only `rand` is allowed.

## Architecture Defaults

- Use a hidden thread-local runtime with explicit `with_grad(...)` / `no_grad(...)` contexts.
- Represent `Value` as lightweight handles into runtime-managed parameter/temp storage.
- Use DAG topological traversal for `backward`.
- Make gradient accumulation explicit and test it on shared subgraphs.

## Testing Requirements

- Add explicit tests for forward correctness and backward gradients.
- Include finite-difference gradient checks implemented with std-only helpers.
- Treat `cargo test` as a mandatory quality gate for milestone completion.
- Keep tests deterministic where possible (seed randomness in demos/tests).

## Milestone Learning Workflow

- After each roadmap milestone, collect the Rust concepts used in that milestone.
- Ask for an interactive Rust learning section tailored to the implemented work.
- Keep interaction fluid (custom prompts), not a fixed template.
- Maintain implementation progress as the primary objective.

## Roadmap Stage Tracking

- `ROADMAP.md` must always include a visible `Current Stage`.
- Update the stage only when the current stage acceptance criteria are fully met.
- Record `Last Updated` date when stage status changes.
