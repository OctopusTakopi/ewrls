# ewrls

Exponentially-weighted recursive least squares (EW-RLS) for online linear regression.

After each update the model holds the exact minimizer of

```text
J_t(θ) = Σ_{i=1..t} Λ_{i,t} w_i (y_i − x_iᵀθ)²
         + Λ_{0,t} δ⁻¹ ‖θ‖² + θᵀΓ_tθ
```

where `Λ_{i,t}` is the product of the forgetting factors applied after sample
`i` — the per-update `λ` and any `decay` calls, so `Λ_{i,t} = λ^{t-i}` for a
constant `λ` — `w_i` is an optional per-sample weight, `δ` the initial
covariance (a prior that is forgotten with the data), and `Γ_t` an optional
persistent diagonal ridge. Each update tops one ridge coordinate `j` back up
to its target (`γ·s_j`, or `κ·W_t·s_j` relative to the discounted sample mass
`W_t`), so every coordinate is back at its target within `d` updates for any
forgetting schedule. With an absolute ridge, `λ = 1` and no `decay` calls,
`Γ_t = γ·diag(s)` exactly. Updates are O(d²) Sherman–Morrison and
allocation-free.

## Usage

```rust
use ewrls::EwRls;

// y = m*x + b, features [x, 1]
let mut model = EwRls::builder(2)
    .lambda(0.97)
    .initial_covariance(1e3)
    .build()?;

let report = model.update(&[0.4, 1.0], 1.9)?;   // prediction, residual, xᵀPx
model.update_weighted(&[0.5, 1.0], 2.1, 3.0)?;  // sample weight 3
let y_hat = model.predict(&[0.6, 1.0])?;
# Ok::<(), ewrls::Error>(())
```

Batch fitting follows familiar `fit`/`partial_fit` semantics while accepting
ordinary contiguous Rust rows (`Vec<f64>`, arrays, or slices):

```rust
use ewrls::EwRls;

let x = [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
let y = [1.0, 4.0, 7.0];
let mut model = EwRls::new(2, 0.99)?;

model.fit(&x, &y)?; // resets, then fits the batch
model.partial_fit(&[[3.0, 1.0]], &[10.0])?; // preserves current state

let predictions = model.predict_batch(&x)?;
let mut reused = [0.0; 3];
model.predict_batch_into(&x, &mut reused)?; // allocation-free output path
# Ok::<(), Box<dyn std::error::Error>>(())
```

Effective window is roughly `1/(1−λ)`: 0.99 ≈ 100 samples, 0.95 ≈ 20.
`λ = 1` is plain RLS and matches the closed-form ridge solution
`(XᵀX + δ⁻¹I)⁻¹Xᵀy` exactly (tested).

## Time-based forgetting, gaps and outliers

For irregularly spaced samples, keep `λ = 1` and forget by elapsed time with
`decay(2^(−Δt/τ))` (half-life `τ`). A relative ridge keeps the shrinkage per
unit of data constant when activity changes, and a ridge weight of `0` leaves
an intercept unpenalized. `update_if` sees the a-priori report before anything
is committed, so a bad tick can be rejected:

```rust
use ewrls::{DecayOutcome, EwRls, Ridge};

let mut model = EwRls::builder(3)
    .lambda(1.0)
    .ridge(Ridge::Relative(0.1))
    .ridge_weights(&[0.0, 1.0, 1.0]) // column 0 is the intercept
    .build()?;
let (half_life, noise_var) = (3600.0, 0.04);
let mut last = 0.0;
for (t, x, y) in [
    (0.0, [1.0, 0.2, -0.1], 0.3),
    (0.4, [1.0, 0.1, 0.4], -0.2),
    (90_000.0, [1.0, -0.3, 0.2], 0.1), // after a 25 h gap
] {
    let outcome = model.decay(0.5_f64.powf((t - last) / half_life))?;
    if outcome == DecayOutcome::Reset {
        // > ~20 half-lives: confidence was reset to the prior, θ kept
    }
    last = t;
    let accepted = model.update_if(&x, y, |r| {
        r.residual.abs() < 10.0 * (noise_var * r.innovation_scale).sqrt()
    })?;
    let _ = accepted; // None if the gate rejected the sample
}
# Ok::<(), ewrls::Error>(())
```

`decay(f)` followed by an update with `λ = 1` minimizes the same objective as
an update with `λ = f`. `decay` costs one O(d²) pass. When every decay is
followed by an update and no gap forgets more than `MIN_LAMBDA`,
`set_lambda(f)` is equivalent and about 20% cheaper overall. `decay` also works
for time buckets without samples and handles gaps of any length.

## Notes

The recursion is hardened for long-running streams:

- the covariance is updated with bit-for-bit symmetric rank-1 terms, so no
  asymmetry can build up and be amplified by forgetting (which used to break
  down within a few hundred updates at `λ ≤ 0.85`)
- forgetting between two updates is bounded. `λ` must be at least
  `MIN_LAMBDA = 1e-6`, and `decay` treats more forgetting than that since the
  last update as complete forgetting, however a long feed gap is split into
  calls: it resets `P` to the prior and keeps `θ`. Beyond what `f64` can
  resolve, covariance-form RLS would silently lose positive definiteness
- forgetting saturates at the prior (stabilized forgetting): once a
  coordinate's variance exceeds `2δ`, information `1/δ − 1/P_jj` is added back
  on it at the current θ, so its variance returns to `δ` and θ is unchanged.
  This bounds the inflation that later updates must cancel for any forgetting
  history, including zero rows and sparse features between gaps. Each such
  step adds a prior term centred on the estimate at the time; with `λ = 1` and
  no `decay` it never happens
- the persistent ridge bounds the covariance under weak excitation; a hard
  `trace(P)` cap (`max_covariance_trace`) also exists but is off by default,
  since capping biases the estimator away from the exact objective
- an observation whose update would produce a non-finite state or a negative
  variance returns `Error::NumericalBreakdown` before any state is modified.
  Definiteness is only checked along the observed direction, so a streak of
  these errors means the covariance is unhealthy: `reset_covariance()`
- NaN/Inf inputs and dimension mismatches are rejected, nothing panics

`reset_covariance()` keeps θ but forgets confidence — useful after a known
regime change. The reset prior is `diag((δ⁻¹ + g_j)⁻¹)` with the ridge at zero
data, and it is centred on the kept θ; `reset()` also zeroes θ.

For residual z-scoring, `UpdateReport::innovation_scale = 1/w + xᵀPx/λ` is the
innovation variance in units of the noise variance `σ²` (random-walk reading
of forgetting); for a fixed θ it is `1/w + c·xᵀPx` with `c = 1` at `λ = 1` and
`c ≈ 1/(1 + λ)` in the steady state of a constant `λ < 1`.

## Features

- `serde` — serialize/deserialize validated, versioned model checkpoints
  (format 2). A restored model continues bit-exactly; with `serde_json`, enable
  its `float_roundtrip` feature or use a binary format (bincode, postcard).
  Format 1 checkpoints from 0.2 still load, in either kind of format.

The minimum supported Rust version is 1.97.

## License

Licensed under the [MIT License](LICENSE).

`cargo test` runs closed-form, exact-objective and equivalence checks (and the
README examples), `cargo bench` the Criterion update/predict benchmarks.
`cargo run --example nist_norris` reproduces the certified values of the NIST
StRD Norris regression benchmark.
