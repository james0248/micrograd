# IMPROVEMENTS

## Current State

The migration is functionally complete. The main remaining work is API cleanup, readability cleanup, and feature completion around reductions.

## Priority Order

1. General reductions
2. Internal naming cleanup
3. Module/API readability cleanup
4. Post-migration architecture cleanup

## Actionable Items

### 1. General Reductions

- Replace transitional all-reduction APIs with:
  - `sum(axes: &[usize], keepdim: bool)`
  - `mean(axes: &[usize], keepdim: bool)`
- Remove `sum_all` / `mean_all`.
- Keep `max` single-axis for now.

### 2. Internal Naming Cleanup

- Rename migration-era internals to match the current architecture.
- Current target renames:
  - `JvpRecorder` -> `LinearizeRecorder`
  - `JVP_STACK` -> `LINEARIZE_STACK`
  - `jvp_binary` / `jvp_unary` -> `linearize_binary` / `linearize_unary`

### 3. Docs and API Cleanup

- Keep `AGENTS.md` aligned with the actual repo architecture.
- Keep `MIGRATION.md` short and descriptive rather than stage-driven.
- Keep public APIs explicit and minimal.

### 4. Architecture Cleanup

- Revisit internal naming and module boundaries after the reduction transition lands.
- Remove obvious migration-era wording where it no longer helps understanding.
