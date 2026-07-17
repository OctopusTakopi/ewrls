# ewrls

Exponentially-weighted recursive least squares (EW-RLS) for online linear regression.

At each step the model holds the exact minimizer of

```text
J_t(θ) = Σ_{i=1..t} λ^{t-i} w_i (y_i − x_iᵀθ)²  +  λ^t δ⁻¹ ‖θ‖²  +  γ ‖θ‖²
```

where `λ ∈ (0, 1]` is the forgetting factor, `w_i` an optional per-sample
weight, `δ` the initial covariance (a decaying ridge prior) and `γ` an
optional persistent L2 penalty (`regularization` in the builder) that holds
under forgetting — exact at `λ = 1`, maintained via cycled pseudo-observations
for `λ < 1`. Updates are O(d²) Sherman–Morrison, allocation-free.

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

## Notes

The recursion is hardened for long-running streams:

- covariance is updated in the symmetric rank-1 form and re-symmetrized
  periodically, so it can't drift off-symmetric over millions of updates
- `regularization` bounds the covariance under weak excitation; a hard
  `trace(P)` cap (`max_covariance_trace`) also exists but is off by default,
  since capping biases the estimator away from the exact objective
- an observation that would lose covariance positive-definiteness or overflow
  returns `Error::NumericalBreakdown` before any state is modified; recover
  with `reset_covariance()`
- NaN/Inf inputs and dimension mismatches are rejected, nothing panics; with
  `serde`, checkpoints are validated on load

`reset_covariance()` keeps θ but forgets confidence — useful after a known
regime change.

## Features

- `serde` — serialize/deserialize model state.

`cargo test` runs closed-form and equivalence checks, `cargo bench` the
Criterion update/predict benchmarks. `cargo run --example nist_norris`
reproduces the certified values of the NIST StRD Norris regression benchmark.
