# Changelog

All notable changes to this project are documented here.

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
