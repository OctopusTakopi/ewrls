# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed

- Covariance asymmetry no longer builds up under forgetting. The two
  triangles of the rank-1 update rounded differently, and every `1/λ` amplified
  the difference until the next re-symmetrization 256 updates later. At a
  constant `λ ≤ 0.85`, well-excited data broke down within about 90–200
  updates; at `λ = 0.9` the asymmetry grew by several orders of magnitude.
  Rank-1 terms are now bit-for-bit symmetric, at the same cost.
- A long feed gap can no longer silently make `P` indefinite, corrupt `θ` and
  lock the model into rejecting every later update. `λ` must be at least
  `MIN_LAMBDA = 1e-6`. The new `decay` treats forgetting below that floor as
  complete forgetting, keeping `θ` and resetting `P` to the prior. The floor
  applies to the product of all decays since the last update, so a gap split
  into many small decays (e.g. a per-second timer) is handled too. Forgetting
  that accumulates across updates carrying little information (zero rows,
  sparse features, a decay followed by a small `λ`) is bounded by the new
  prior floor (see Changed).
- The persistent ridge is now correct for any forgetting schedule. The refresh
  used only the current `λ`, so with time-varying `λ`, or `λ = 1` updates
  between decays, most coordinates starved (down to `1e-8·γ`), and after a gap
  they took about one half-life to recover. Each coordinate's penalty is now
  tracked and topped back up to its target, reaching it within `d` updates.
  This also removes the first-cycle overshoot above `γ`.
- Restoring a checkpoint is bit-exact. Restore used to re-symmetrize `P` and
  rebuild the covariance bound, so the restored model diverged from the live
  one unless the update count was a multiple of 256. Re-symmetrization now
  averages only entry pairs that differ, a no-op for the states this version
  writes, and a restored asymmetric state is symmetrized before forgetting can
  amplify it.
- `reset_covariance`, `reset` and `fit` no longer allocate.

### Added

- `decay(factor)` for time-based forgetting without an observation, returning
  a `DecayOutcome` (`Applied` or `Reset`). It scales `P` in one vectorized
  pass and carries the covariance bound along, rescanning for the exact bound
  only near overflow, so repeated decays never reset spuriously.
- `Ridge::Relative(κ)`: a ridge proportional to the discounted sample mass
  `W_t`, so the shrinkage per unit of data stays constant as activity changes.
  Also `Builder::ridge`, `EwRls::set_ridge`, `EwRls::ridge` and
  `EwRls::weight_mass`. The mass saturates at `f64::MAX` instead of rejecting
  updates.
- `Builder::ridge_weights` for per-coordinate ridge weights; `0` leaves a
  coordinate, such as an intercept, unpenalized. The prior becomes
  `diag((δ⁻¹ + γ·s_j)⁻¹)`. The builder, `set_ridge` and restore reject an
  overflowing `γ·s_j`, and a ridge whose reset prior would have an infinite
  trace.
- `update_if` / `update_weighted_if`: veto an observation (e.g. a bad tick)
  after seeing its a-priori report, before anything is committed.
- `UpdateReport::innovation_scale = 1/w + xᵀPx/λ` for residual z-scoring.
- The README examples are compiled and run as doctests.

### Changed

- Behaviour: forgetting saturates at the prior (stabilized forgetting). Once
  a coordinate's variance exceeds `2δ`, information `1/δ − 1/P_jj` is added
  back on it as a pseudo-observation at the current `θ_j`, which returns its
  variance to `δ` and leaves `θ` unchanged. Unexcited directions no longer
  wind up without bound when there is no ridge, and after any forgetting
  history relearning is no worse conditioned than learning from the prior.
  With `λ = 1` and no `decay` it never fires, so such models are
  bit-identical.
- Breaking: `λ` below `MIN_LAMBDA = 1e-6` is rejected with `InvalidLambda` by
  the builder, `set_lambda` and checkpoint restore.
- Breaking: `Error`, `BatchError`, `UpdateReport` and the new enums are
  `#[non_exhaustive]`. `Error` has a new `InvalidDecay` variant, and the
  `NumericalBreakdown` message no longer claims that definiteness was lost.
- Checkpoints use format version 2, which adds the ridge state, the sample
  mass, the pending decay and the covariance bound. Version 1 checkpoints still
  load. Their ridge state is reconstructed exactly from 0.2's schedule; their
  sample mass assumes unit weights and a constant `λ`. Both versions load from
  self-describing formats (JSON) and positional binary ones (bincode,
  postcard). Corrupt dimensions now return an error instead of panicking or
  aborting.
- `regularization()` returns the ridge strength (`γ` or `κ`).
- Docs:
  - the objective with arbitrary forgetting;
  - the reset prior `diag((δ⁻¹ + g_j)⁻¹)`, centred on the kept `θ`;
  - the fixed-`θ` versus random-walk variance of residuals and predictions;
  - the mean ridge `ργ`;
  - `serde_json`'s `float_roundtrip` for exact checkpoints;
  - that `MIN_LAMBDA` bounds a single step only (accumulated forgetting is
    bounded by the prior floor);
  - the cost of `decay` versus `set_lambda`.
- Removed the redundant `homepage` manifest field.

## [0.2.0] - 2026-07-17

### Added

- Batch `fit`, `partial_fit`, weighted-fit, and batch prediction APIs.
- Allocation-free `predict_batch_into` output path.
- Indexed errors for invalid or numerically unsafe batch observations.
- Persistent ridge regularization, covariance trace capping, update
  diagnostics, and validated serde checkpoints.

### Changed

- Covariance capping is opt-in so the default follows textbook EW-RLS.
- Normal covariance updates use a guarded fused rank-one path, with an exact
  scalar fallback for extreme floating-point inputs.
- The minimum supported Rust version is 1.97.

### Fixed

- Numerical failures now leave logical model state unchanged.
- Construction, reset, and checkpoint restoration enforce covariance
  finiteness, symmetry, trace, and positive-semidefinite invariants.
- Extreme-scale parameter, covariance, and ridge updates no longer store
  NaN or infinity after returning success.

[0.2.0]: https://github.com/OctopusTakopi/ewrls/releases/tag/v0.2.0
