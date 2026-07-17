//! Exponentially-weighted recursive least squares (EW-RLS) online regression.
//!
//! At every step `t` the model holds the exact minimizer of the exponentially
//! discounted, per-sample weighted least-squares cost
//!
//! ```text
//! J_t(θ) = Σ_{i=1..t} λ^{t-i} w_i (y_i - x_iᵀθ)²  +  λ^t δ⁻¹ ‖θ‖²  +  γ ‖θ‖²
//! ```
//!
//! where `λ ∈ (0, 1]` is the forgetting factor, `w_i > 0` is an optional
//! per-observation weight, `δ` is the initial covariance scale (a ridge
//! prior: large `δ` means a weak prior on `θ = 0`), and `γ ≥ 0` an optional
//! *persistent* ridge penalty that, unlike the `δ` prior, does not decay
//! under forgetting. At `λ = 1` the `γ` term is exact; with `λ < 1` it is
//! maintained by cycled zero-target pseudo-observations (one coordinate per
//! update, scaled so the steady-state penalty is `γ` per direction, rippling
//! down to `γλᵈ` between refreshes).
//!
//! Each update costs `O(d²)` time via the Sherman–Morrison identity and
//! performs no heap allocation. The covariance update uses the symmetric
//! rank-1 form `P ← (P − (Px)(Px)ᵀ/den)/λ`, which preserves symmetry up to
//! rounding; residual drift is removed by periodic re-symmetrization.
//!
//! Safeguards over a naive textbook implementation:
//!
//! - fallible API ([`Error`]) instead of panics, with input validation
//!   (dimension checks, non-finite rejection),
//! - covariance **wind-up protection**: with `λ < 1` and weakly exciting
//!   inputs, `P` grows exponentially in unexcited directions. A persistent
//!   ridge ([`Builder::regularization`]) bounds it consistently with the
//!   objective; a hard trace cap ([`Builder::max_covariance_trace`]) is also
//!   available but off by default, since capping biases the estimator away
//!   from the exact minimizer of `J_t`,
//! - **numerical breakdown detection**: a covariance that has lost positive
//!   definiteness is reported as [`Error::NumericalBreakdown`] rather than
//!   silently corrupting `θ`; recover with [`EwRls::reset_covariance`],
//! - per-update diagnostics ([`UpdateReport`]): a-priori prediction, residual
//!   and predictive variance `xᵀPx`, usable for residual z-scoring.
//!
//! # Example
//!
//! ```
//! use ewrls::EwRls;
//!
//! let mut model = EwRls::new(2, 0.99).unwrap();
//! for t in 0..100 {
//!     let x = [t as f64, 1.0]; // features [x, bias]
//!     let y = 3.0 * x[0] + 1.0;
//!     let report = model.update(&x, y).unwrap();
//!     let _ = report.residual;
//! }
//! assert!((model.params()[0] - 3.0).abs() < 1e-6);
//! assert!((model.params()[1] - 1.0).abs() < 1e-4);
//! ```
//!
//! # Feature flags
//!
//! - `serde` — `Serialize`/`Deserialize` on [`EwRls`] for state persistence.

#![warn(missing_docs)]

use nalgebra::{DMatrix, DVector, DVectorView};

/// Re-symmetrize `P` every this many updates to remove rounding drift.
const SYMMETRIZE_INTERVAL: u64 = 256;

/// Errors returned by [`EwRls`] construction and updates.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum Error {
    /// The model must have at least one feature dimension.
    #[error("dimensions must be greater than 0")]
    ZeroDimensions,
    /// The forgetting factor must satisfy `0 < lambda <= 1`.
    #[error("forgetting factor lambda must be in (0, 1], got {0}")]
    InvalidLambda(f64),
    /// The initial covariance scale must be positive and finite.
    #[error("initial covariance must be positive and finite, got {0}")]
    InvalidInitialCovariance(f64),
    /// The covariance trace cap must be positive.
    #[error("max covariance trace must be positive, got {0}")]
    InvalidMaxTrace(f64),
    /// The ridge penalty must be non-negative and finite.
    #[error("regularization must be non-negative and finite, got {0}")]
    InvalidRegularization(f64),
    /// The input vector length does not match the model dimensions.
    #[error("input has {got} features but model expects {expected}")]
    DimensionMismatch {
        /// Number of features the model was built with.
        expected: usize,
        /// Number of features in the rejected input.
        got: usize,
    },
    /// An input feature, target or weight was NaN or infinite.
    #[error("input contains a non-finite value")]
    NonFiniteInput,
    /// A per-sample weight must be positive and finite.
    #[error("sample weight must be positive and finite, got {0}")]
    InvalidWeight(f64),
    /// Applying the observation would lose covariance positive-definiteness
    /// or overflow; the observation is rejected and the model state is
    /// unchanged. Recover with [`EwRls::reset_covariance`] or
    /// [`EwRls::reset`].
    #[error("numerical breakdown: covariance lost positive-definiteness; reset the covariance")]
    NumericalBreakdown,
}

/// Diagnostics produced by a single update, all computed *before* the
/// parameters were adjusted (a-priori quantities).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdateReport {
    /// A-priori prediction `xᵀθ` using the parameters before this update.
    pub prediction: f64,
    /// A-priori residual `y − xᵀθ` (the innovation).
    pub residual: f64,
    /// Predictive variance scale `xᵀPx` before this update. For residual
    /// z-scoring use `residual / sqrt(predictive_variance + noise_variance)`.
    pub predictive_variance: f64,
}

/// Builder for [`EwRls`] with optional knobs beyond [`EwRls::new`].
#[derive(Debug, Clone)]
pub struct Builder {
    dimensions: usize,
    lambda: f64,
    initial_covariance: f64,
    max_covariance_trace: Option<f64>,
    regularization: f64,
}

impl Builder {
    /// Forgetting factor `λ ∈ (0, 1]`. Default `0.99`.
    ///
    /// The effective sample window is roughly `1 / (1 − λ)`; `λ = 1` weights
    /// all history equally (plain recursive least squares).
    #[must_use]
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Initial covariance scale `δ`: `P₀ = δI`. Default `1e3`.
    ///
    /// Equivalent to a ridge penalty `δ⁻¹‖θ‖²` on the discounted cost; large
    /// values mean low confidence in the zero initialization and fast initial
    /// adaptation.
    #[must_use]
    pub fn initial_covariance(mut self, delta: f64) -> Self {
        self.initial_covariance = delta;
        self
    }

    /// Hard cap on `trace(P)`. Disabled by default (`INFINITY`).
    ///
    /// A last-resort wind-up guard: whenever the trace exceeds the cap —
    /// including at initialization and on reset — `P` is rescaled to meet it.
    /// Rescaling biases the estimator away from the exact minimizer of `J_t`,
    /// so prefer [`Builder::regularization`], which bounds `P` consistently
    /// with the objective.
    #[must_use]
    pub fn max_covariance_trace(mut self, max_trace: f64) -> Self {
        self.max_covariance_trace = Some(max_trace);
        self
    }

    /// Persistent ridge penalty `γ ≥ 0` on `‖θ‖²`. Default `0`.
    ///
    /// Unlike the `δ⁻¹` prior, this penalty does not decay under forgetting:
    /// exact at `λ = 1`, maintained at steady-state strength `≈ γ` per
    /// coordinate for `λ < 1`. Besides shrinking `θ`, it bounds the
    /// covariance in unexcited directions, so it also acts as wind-up
    /// protection.
    #[must_use]
    pub fn regularization(mut self, gamma: f64) -> Self {
        self.regularization = gamma;
        self
    }

    /// Validate the configuration and build the model.
    pub fn build(self) -> Result<EwRls, Error> {
        if self.dimensions == 0 {
            return Err(Error::ZeroDimensions);
        }
        if !(self.lambda > 0.0 && self.lambda <= 1.0) {
            return Err(Error::InvalidLambda(self.lambda));
        }
        if !(self.initial_covariance > 0.0 && self.initial_covariance.is_finite()) {
            return Err(Error::InvalidInitialCovariance(self.initial_covariance));
        }
        let max_trace = self.max_covariance_trace.unwrap_or(f64::INFINITY);
        if max_trace.is_nan() || max_trace <= 0.0 {
            return Err(Error::InvalidMaxTrace(max_trace));
        }
        if !(self.regularization >= 0.0 && self.regularization.is_finite()) {
            return Err(Error::InvalidRegularization(self.regularization));
        }

        // The γ ridge starts in the initial information matrix:
        // P₀ = (δ⁻¹I + γI)⁻¹. Exact and permanent at λ = 1; for λ < 1 the
        // per-update refresh below keeps it from decaying. The trace cap
        // applies from the start.
        let p0 = initial_scale(
            self.initial_covariance,
            self.regularization,
            max_trace,
            self.dimensions,
        );
        // A per-entry scale can be finite while the resulting trace d·p0
        // overflows, leaving a model whose every update breaks down.
        if !(p0 * self.dimensions as f64).is_finite() {
            return Err(Error::InvalidInitialCovariance(self.initial_covariance));
        }

        Ok(EwRls {
            theta: DVector::zeros(self.dimensions),
            p: DMatrix::identity(self.dimensions, self.dimensions) * p0,
            p_x: DVector::zeros(self.dimensions),
            lambda: self.lambda,
            dimensions: self.dimensions,
            initial_covariance: self.initial_covariance,
            max_trace,
            gamma: self.regularization,
            updates: 0,
        })
    }
}

/// Exponentially-weighted recursive least squares regressor.
///
/// See the [crate-level documentation](crate) for the model definition,
/// numerical safeguards and a usage example.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "EwRlsCheckpoint")
)]
pub struct EwRls {
    /// Parameter vector `θ` (`d × 1`).
    theta: DVector<f64>,
    /// Inverse information (covariance) matrix `P` (`d × d`).
    p: DMatrix<f64>,
    /// Workspace for `P·x`, kept to make updates allocation-free.
    p_x: DVector<f64>,
    lambda: f64,
    dimensions: usize,
    initial_covariance: f64,
    /// Serialized as `null` when infinite — JSON has no Inf literal.
    #[cfg_attr(feature = "serde", serde(serialize_with = "serde_inf::serialize"))]
    max_trace: f64,
    gamma: f64,
    updates: u64,
}

/// (De)serialize a possibly-infinite `f64` through `Option`, since formats
/// like JSON cannot represent infinity directly.
#[cfg(feature = "serde")]
mod serde_inf {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_finite() {
            s.serialize_some(v)
        } else {
            s.serialize_none()
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::INFINITY))
    }
}

impl EwRls {
    /// Create a model with `dimensions` features and forgetting factor
    /// `lambda`, using default initial covariance (`1e3`) and wind-up cap.
    ///
    /// Use [`EwRls::builder`] to tune the remaining knobs.
    pub fn new(dimensions: usize, lambda: f64) -> Result<Self, Error> {
        Self::builder(dimensions).lambda(lambda).build()
    }

    /// Start building a model with `dimensions` features.
    #[must_use]
    pub fn builder(dimensions: usize) -> Builder {
        Builder {
            dimensions,
            lambda: 0.99,
            initial_covariance: 1e3,
            max_covariance_trace: None,
            regularization: 0.0,
        }
    }

    /// Predict `xᵀθ` for a feature vector.
    pub fn predict(&self, x: &[f64]) -> Result<f64, Error> {
        self.validate_features(x)?;
        let x = DVectorView::from_slice(x, self.dimensions);
        Ok(x.dot(&self.theta))
    }

    /// Predictive variance scale `xᵀPx` for a feature vector.
    ///
    /// Proportional to the model's parameter uncertainty in the direction of
    /// `x`; combine with an estimate of the observation noise variance for a
    /// full predictive variance. Clamped to be non-negative.
    pub fn prediction_variance(&self, x: &[f64]) -> Result<f64, Error> {
        self.validate_features(x)?;
        let xv = DVectorView::from_slice(x, self.dimensions);
        // Walk columns so accesses stay contiguous in the column-major P.
        let mut acc = 0.0;
        for (j, &xj) in x.iter().enumerate() {
            acc += xj * self.p.column(j).dot(&xv);
        }
        Ok(acc.max(0.0))
    }

    /// Update with an observation `(x, y)` of unit weight.
    pub fn update(&mut self, x: &[f64], y: f64) -> Result<UpdateReport, Error> {
        self.update_weighted(x, y, 1.0)
    }

    /// Update with an observation `(x, y)` carrying weight `w > 0`.
    ///
    /// A weight of `w` makes the sample count `w` times in the least-squares
    /// cost (with `λ = 1`, an integer weight is exactly equivalent to seeing
    /// the observation that many times). Use it to weight samples by trade
    /// size, data quality or inverse noise variance.
    ///
    /// On [`Error::NumericalBreakdown`] the observation is rejected before
    /// any state is modified; recover with [`EwRls::reset_covariance`].
    pub fn update_weighted(&mut self, x: &[f64], y: f64, weight: f64) -> Result<UpdateReport, Error> {
        self.validate_features(x)?;
        if !y.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        if !(weight > 0.0 && weight.is_finite()) {
            return Err(Error::InvalidWeight(weight));
        }

        let x = DVectorView::from_slice(x, self.dimensions);

        // p_x = P·x. P is symmetric, so p_xᵀ = xᵀP as well.
        self.p_x.gemv(1.0, &self.p, &x, 0.0);

        let quad = x.dot(&self.p_x); // xᵀPx ≥ 0 for a healthy P
        let trace = self.p.trace();
        // Rounding bound for the quadratic form: its summands are of order
        // trace(P)·‖x‖², so negatives at eps of that scale are fp noise from
        // an eps-level-indefinite P (unavoidable in covariance-form RLS) and
        // are clamped to zero. Anything more negative is real corruption.
        let tol = 1e-12 * (1.0 + trace * x.norm_squared());
        if !quad.is_finite() || !trace.is_finite() || quad < -tol {
            return Err(Error::NumericalBreakdown);
        }
        let quad = quad.max(0.0);

        // Gain denominator: λ/w + xᵀPx.
        let denominator = self.lambda / weight + quad;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(Error::NumericalBreakdown);
        }

        let prediction = x.dot(&self.theta);
        let residual = y - prediction;
        let gain_scale = residual / denominator;
        if !gain_scale.is_finite() {
            return Err(Error::NumericalBreakdown);
        }

        // Validate the full candidate state before committing anything: every
        // new parameter must stay finite, every new covariance diagonal must
        // stay finite and non-negative, and their trace must not overflow.
        // Otherwise the observation has exhausted f64 precision (or
        // overflowed 1/λ) and is rejected with the state untouched.
        let inv_den = 1.0 / denominator;
        let mut new_trace = 0.0;
        for j in 0..self.dimensions {
            let tj = self.theta[j] + gain_scale * self.p_x[j];
            let dj = (self.p[(j, j)] - self.p_x[j] * self.p_x[j] * inv_den) / self.lambda;
            if !tj.is_finite() || !dj.is_finite() || dj < 0.0 {
                return Err(Error::NumericalBreakdown);
            }
            new_trace += dj;
        }
        if !new_trace.is_finite() {
            return Err(Error::NumericalBreakdown);
        }

        // θ ← θ + k·e with gain k = P·x / den, element-wise with exactly the
        // expression validated above so the committed values cannot differ.
        for j in 0..self.dimensions {
            self.theta[j] += gain_scale * self.p_x[j];
        }

        // P ← (P − (Px)(Px)ᵀ / den) / λ — symmetric rank-1 downdate with the
        // 1/λ forgetting folded into the same pass.
        self.p
            .ger(-inv_den / self.lambda, &self.p_x, &self.p_x, 1.0 / self.lambda);
        // The pre-check and ger may round the last ulp differently; pin any
        // straggler diagonal at zero so no negative variance is ever stored.
        for j in 0..self.dimensions {
            if self.p[(j, j)] < 0.0 {
                self.p[(j, j)] = 0.0;
            }
        }

        // Persistent ridge refresh: the data step decayed all accumulated
        // regularization by λ; re-inject it one coordinate at a time via a
        // zero-target pseudo-observation x = √c·e_j, y = 0 (no forgetting on
        // this step). c = γ(1 − λᵈ) makes the steady-state penalty per
        // direction exactly γ right after its refresh. Shrinks θ_j toward 0
        // and bounds P in direction j.
        if self.gamma > 0.0 && self.lambda < 1.0 {
            let c = self.gamma * (1.0 - self.lambda.powi(self.dimensions as i32));
            let j = (self.updates % self.dimensions as u64) as usize;
            self.p_x.copy_from(&self.p.column(j));
            // den ≥ 1 since c ≥ 0 and P[jj] ≥ 0; skip the refresh on overflow.
            let den = 1.0 + c * self.p_x[j];
            if den.is_finite() {
                let factor = c / den;
                // Bound the largest θ and P corrections before committing,
                // multiplying factor into the (often tiny) column first so a
                // huge γ cannot overflow an intermediate. |P_ik| ≤ trace(P)
                // for a PSD matrix, so these bounds cover every element; skip
                // the refresh rather than store a non-finite value.
                let theta_j = self.theta[j];
                let m = factor * self.p_x.amax();
                let theta_bound = self.theta.amax() + m * theta_j.abs();
                let p_bound = self.p.trace() + m * self.p_x.amax();
                if theta_bound.is_finite() && p_bound.is_finite() {
                    for i in 0..self.dimensions {
                        self.theta[i] -= (factor * self.p_x[i]) * theta_j;
                    }
                    self.p.ger(-factor, &self.p_x, &self.p_x, 1.0);
                }
            }
        }

        // Optional wind-up cap: rescale P if its trace exceeded the cap.
        let trace = self.p.trace();
        if trace > self.max_trace {
            self.p.scale_mut(self.max_trace / trace);
        }

        self.updates += 1;
        if self.updates.is_multiple_of(SYMMETRIZE_INTERVAL) {
            self.symmetrize();
        }

        Ok(UpdateReport {
            prediction,
            residual,
            predictive_variance: quad,
        })
    }

    /// Reset the covariance to `δI`, keeping the learned parameters.
    ///
    /// The standard recovery from [`Error::NumericalBreakdown`], and a common
    /// deliberate move after a known regime change to let the model re-adapt
    /// quickly without discarding `θ`.
    pub fn reset_covariance(&mut self) {
        let p0 = initial_scale(self.initial_covariance, self.gamma, self.max_trace, self.dimensions);
        self.p = DMatrix::identity(self.dimensions, self.dimensions) * p0;
    }

    /// Reset the model to its initial state (`θ = 0`, `P = δI`).
    pub fn reset(&mut self) {
        self.theta.fill(0.0);
        self.reset_covariance();
        self.updates = 0;
    }

    /// Change the forgetting factor at runtime (e.g. per-regime adaptation).
    pub fn set_lambda(&mut self, lambda: f64) -> Result<(), Error> {
        if !(lambda > 0.0 && lambda <= 1.0) {
            return Err(Error::InvalidLambda(lambda));
        }
        self.lambda = lambda;
        Ok(())
    }

    /// The learned parameter vector `θ` as a slice.
    #[must_use]
    pub fn params(&self) -> &[f64] {
        self.theta.as_slice()
    }

    /// The learned parameter vector `θ`.
    #[must_use]
    pub fn theta(&self) -> &DVector<f64> {
        &self.theta
    }

    /// The covariance matrix `P`.
    #[must_use]
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.p
    }

    /// Current `trace(P)` — a cheap health/uncertainty summary.
    #[must_use]
    pub fn covariance_trace(&self) -> f64 {
        self.p.trace()
    }

    /// Number of feature dimensions.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The forgetting factor `λ`.
    #[must_use]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The persistent ridge penalty `γ`.
    #[must_use]
    pub fn regularization(&self) -> f64 {
        self.gamma
    }

    /// Effective sample window `1 / (1 − λ)`; `f64::INFINITY` when `λ = 1`.
    #[must_use]
    pub fn effective_window(&self) -> f64 {
        if self.lambda < 1.0 {
            1.0 / (1.0 - self.lambda)
        } else {
            f64::INFINITY
        }
    }

    /// Total number of successful updates.
    #[must_use]
    pub fn updates(&self) -> u64 {
        self.updates
    }

    fn validate_features(&self, x: &[f64]) -> Result<(), Error> {
        if x.len() != self.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.dimensions,
                got: x.len(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFiniteInput);
        }
        Ok(())
    }

    fn symmetrize(&mut self) {
        for i in 0..self.dimensions {
            for j in (i + 1)..self.dimensions {
                // 0.5a + 0.5b, not (a+b)/2: the sum can overflow for finite
                // same-sign entries near f64::MAX.
                let avg = 0.5 * self.p[(i, j)] + 0.5 * self.p[(j, i)];
                self.p[(i, j)] = avg;
                self.p[(j, i)] = avg;
            }
        }
    }
}

/// Initial covariance scale `P₀ = 1/(δ⁻¹ + γ)`, capped by the trace limit.
/// Computed in a form that survives subnormal `δ` (where `1/δ` overflows)
/// via the algebraically equal `δ/(1 + γδ)`.
fn initial_scale(delta: f64, gamma: f64, max_trace: f64, dimensions: usize) -> f64 {
    let inv = 1.0 / delta + gamma;
    let p0 = if inv.is_finite() {
        1.0 / inv
    } else {
        delta / (1.0 + gamma * delta)
    };
    p0.min(max_trace / dimensions as f64)
}

/// Untrusted mirror of [`EwRls`] used to validate deserialized checkpoints
/// before they become a live model.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct EwRlsCheckpoint {
    theta: DVector<f64>,
    p: DMatrix<f64>,
    #[allow(dead_code)]
    p_x: DVector<f64>,
    lambda: f64,
    dimensions: usize,
    initial_covariance: f64,
    #[serde(deserialize_with = "serde_inf::deserialize")]
    max_trace: f64,
    gamma: f64,
    updates: u64,
}

#[cfg(feature = "serde")]
impl TryFrom<EwRlsCheckpoint> for EwRls {
    type Error = String;

    fn try_from(c: EwRlsCheckpoint) -> Result<Self, String> {
        let d = c.dimensions;
        if d == 0 {
            return Err("dimensions must be greater than 0".into());
        }
        if c.theta.nrows() != d {
            return Err(format!("theta has {} rows, expected {d}", c.theta.nrows()));
        }
        if c.p.nrows() != d || c.p.ncols() != d {
            return Err(format!(
                "covariance is {}x{}, expected {d}x{d}",
                c.p.nrows(),
                c.p.ncols()
            ));
        }
        if !(c.lambda > 0.0 && c.lambda <= 1.0) {
            return Err(format!("lambda {} outside (0, 1]", c.lambda));
        }
        if !(c.initial_covariance > 0.0 && c.initial_covariance.is_finite()) {
            return Err(format!("invalid initial covariance {}", c.initial_covariance));
        }
        if c.max_trace.is_nan() || c.max_trace <= 0.0 {
            return Err(format!("invalid max covariance trace {}", c.max_trace));
        }
        if !(c.gamma >= 0.0 && c.gamma.is_finite()) {
            return Err(format!("invalid regularization {}", c.gamma));
        }
        if c.theta.iter().any(|v| !v.is_finite()) {
            return Err("theta contains a non-finite value".into());
        }
        if c.p.iter().any(|v| !v.is_finite()) {
            return Err("covariance contains a non-finite value".into());
        }
        if (0..d).any(|j| c.p[(j, j)] < 0.0) {
            return Err("covariance has a negative diagonal entry".into());
        }
        // The update algebra relies on P = Pᵀ (it uses P·x for both sides of
        // the rank-1 product). Reject gross asymmetry; the residual rounding
        // drift is removed below.
        for i in 0..d {
            for j in (i + 1)..d {
                let (a, b) = (c.p[(i, j)], c.p[(j, i)]);
                // max-based scale: an additive 1 + |a| + |b| can overflow to
                // infinity near f64::MAX and accept any finite skew.
                let scale = a.abs().max(b.abs()).max(1.0);
                if (a - b).abs() > 1e-8 * scale {
                    return Err("covariance is not symmetric".into());
                }
            }
        }
        let trace: f64 = (0..d).map(|j| c.p[(j, j)]).sum();
        if !trace.is_finite() {
            return Err("covariance trace is not finite".into());
        }
        if trace > c.max_trace {
            return Err(format!(
                "covariance trace {trace} exceeds the stored max trace {}",
                c.max_trace
            ));
        }

        let mut model = EwRls {
            theta: c.theta,
            p: c.p,
            // Workspace is scratch state; always rebuild it.
            p_x: DVector::zeros(d),
            lambda: c.lambda,
            dimensions: d,
            initial_covariance: c.initial_covariance,
            max_trace: c.max_trace,
            gamma: c.gamma,
            updates: c.updates,
        };
        model.symmetrize();
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    const A_TOL: f64 = 1e-9;

    #[test]
    fn constructor_validation() {
        assert_eq!(EwRls::new(0, 0.99).unwrap_err(), Error::ZeroDimensions);
        assert_eq!(EwRls::new(3, 1.1).unwrap_err(), Error::InvalidLambda(1.1));
        assert_eq!(EwRls::new(3, 0.0).unwrap_err(), Error::InvalidLambda(0.0));
        assert_eq!(
            EwRls::builder(3).initial_covariance(-1.0).build().unwrap_err(),
            Error::InvalidInitialCovariance(-1.0)
        );
        assert_eq!(
            EwRls::builder(3).max_covariance_trace(0.0).build().unwrap_err(),
            Error::InvalidMaxTrace(0.0)
        );
        assert!(EwRls::new(3, 1.0).is_ok());
    }

    #[test]
    fn input_validation() {
        let mut model = EwRls::new(2, 0.99).unwrap();
        assert_eq!(
            model.predict(&[1.0]).unwrap_err(),
            Error::DimensionMismatch { expected: 2, got: 1 }
        );
        assert_eq!(
            model.update(&[1.0, f64::NAN], 1.0).unwrap_err(),
            Error::NonFiniteInput
        );
        assert_eq!(
            model.update(&[1.0, 1.0], f64::INFINITY).unwrap_err(),
            Error::NonFiniteInput
        );
        assert_eq!(
            model.update_weighted(&[1.0, 1.0], 1.0, 0.0).unwrap_err(),
            Error::InvalidWeight(0.0)
        );
        assert_eq!(
            model.update_weighted(&[1.0, 1.0], 1.0, -2.0).unwrap_err(),
            Error::InvalidWeight(-2.0)
        );
    }

    #[test]
    fn predict_dot_product() {
        let mut model = EwRls::new(2, 0.98).unwrap();
        // Drive theta to a known value through the public API.
        model.theta = DVector::from_vec(vec![1.5, -2.0]);
        assert!((model.predict(&[10.0, 3.0]).unwrap() - 9.0).abs() < A_TOL);
    }

    #[test]
    fn single_update_manual_values() {
        // λ = 1, δ = 100, x = [2, 3], y = 10 — hand-computed expectation.
        let mut model = EwRls::builder(2)
            .lambda(1.0)
            .initial_covariance(100.0)
            .build()
            .unwrap();

        let report = model.update(&[2.0, 3.0], 10.0).unwrap();
        assert!((report.prediction - 0.0).abs() < A_TOL);
        assert!((report.residual - 10.0).abs() < A_TOL);
        assert!((report.predictive_variance - 1300.0).abs() < A_TOL);

        let expected = [1.53727901614143, 2.3059185242121445];
        assert!((model.params()[0] - expected[0]).abs() < A_TOL);
        assert!((model.params()[1] - expected[1]).abs() < A_TOL);
        // Covariance picked up off-diagonal structure.
        assert!(model.covariance()[(0, 1)].abs() > A_TOL);
    }

    #[test]
    fn matches_closed_form_ridge() {
        // With λ = 1 and P₀ = δI, RLS computes θ = (XᵀX + δ⁻¹I)⁻¹ Xᵀy exactly.
        let mut rng = rand::rng();
        let (n, d) = (60, 4);
        let delta = 1e4;

        let x_mat = DMatrix::from_fn(n, d, |_, _| rng.random_range(-2.0..2.0));
        let y_vec = DVector::from_fn(n, |i, _| {
            x_mat.row(i).iter().sum::<f64>() * 0.7 + rng.random_range(-0.5..0.5)
        });

        let mut model = EwRls::builder(d)
            .lambda(1.0)
            .initial_covariance(delta)
            .max_covariance_trace(f64::INFINITY)
            .build()
            .unwrap();
        for i in 0..n {
            let row: Vec<f64> = x_mat.row(i).iter().copied().collect();
            model.update(&row, y_vec[i]).unwrap();
        }

        let gram = x_mat.transpose() * &x_mat + DMatrix::identity(d, d) / delta;
        let closed_form = gram.try_inverse().unwrap() * x_mat.transpose() * &y_vec;

        let diff = (model.theta() - &closed_form).norm();
        assert!(diff < 1e-6, "RLS deviates from closed-form ridge: {diff}");
    }

    #[test]
    fn integer_weight_equals_repeated_updates() {
        // With λ = 1, weight 3 must equal seeing the same sample 3 times.
        let build = || {
            EwRls::builder(2)
                .lambda(1.0)
                .initial_covariance(50.0)
                .build()
                .unwrap()
        };
        let mut weighted = build();
        let mut repeated = build();

        weighted.update_weighted(&[2.0, -1.0], 4.0, 3.0).unwrap();
        for _ in 0..3 {
            repeated.update(&[2.0, -1.0], 4.0).unwrap();
        }

        let diff = (weighted.theta() - repeated.theta()).norm();
        assert!(diff < 1e-9, "weighted vs repeated theta differ by {diff}");
        let p_diff = (weighted.covariance() - repeated.covariance()).norm();
        assert!(p_diff < 1e-6, "weighted vs repeated P differ by {p_diff}");
    }

    #[test]
    fn regularized_matches_closed_form_ridge_at_lambda_one() {
        // With λ = 1, γ folds into the prior: θ = (XᵀX + (δ⁻¹+γ)I)⁻¹Xᵀy.
        let mut rng = rand::rng();
        let (n, d) = (60, 4);
        let (delta, gamma) = (1e4, 2.5);

        let x_mat = DMatrix::from_fn(n, d, |_, _| rng.random_range(-2.0..2.0));
        let y_vec = DVector::from_fn(n, |i, _| {
            x_mat.row(i).iter().sum::<f64>() * 0.7 + rng.random_range(-0.5..0.5)
        });

        let mut model = EwRls::builder(d)
            .lambda(1.0)
            .initial_covariance(delta)
            .regularization(gamma)
            .max_covariance_trace(f64::INFINITY)
            .build()
            .unwrap();
        for i in 0..n {
            let row: Vec<f64> = x_mat.row(i).iter().copied().collect();
            model.update(&row, y_vec[i]).unwrap();
        }

        let gram = x_mat.transpose() * &x_mat + DMatrix::identity(d, d) * (1.0 / delta + gamma);
        let closed_form = gram.try_inverse().unwrap() * x_mat.transpose() * &y_vec;

        let diff = (model.theta() - &closed_form).norm();
        assert!(diff < 1e-6, "deviates from closed-form ridge: {diff}");
    }

    #[test]
    fn regularization_shrinks_parameters() {
        // Noise-free stream: γ = 0 recovers θ* exactly, a strong γ shrinks it.
        let true_theta = [2.0, -3.0];
        let mut rng = rand::rng();
        let mut run = |gamma: f64| {
            let mut model = EwRls::builder(2)
                .lambda(0.97)
                .regularization(gamma)
                .build()
                .unwrap();
            for _ in 0..2000 {
                let x = [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)];
                let y = x[0] * true_theta[0] + x[1] * true_theta[1];
                model.update(&x, y).unwrap();
            }
            model.theta().norm()
        };

        let true_norm = DVector::from_row_slice(&true_theta).norm();
        assert!((run(0.0) - true_norm).abs() < 1e-6);
        let shrunk = run(10.0);
        assert!(
            shrunk > 0.2 * true_norm && shrunk < 0.8 * true_norm,
            "expected clear shrinkage, got ‖θ‖ = {shrunk} vs ‖θ*‖ = {true_norm}"
        );
    }

    #[test]
    fn regularization_bounds_covariance_without_trace_cap() {
        // Same degenerate excitation as the wind-up test, but with the cap
        // disabled: the persistent ridge alone must keep P bounded.
        let mut model = EwRls::builder(2)
            .lambda(0.9)
            .initial_covariance(100.0)
            .max_covariance_trace(f64::INFINITY)
            .regularization(0.1)
            .build()
            .unwrap();

        for _ in 0..500 {
            model.update(&[1.0, 0.0], 1.0).unwrap();
        }
        // Steady-state information in the unexcited direction is ≥ γλᵈ = 0.081,
        // so P₁₁ ≤ ~12.4; leave generous slack.
        let trace = model.covariance_trace();
        assert!(trace < 100.0, "trace {trace} not bounded by regularization");
    }

    #[test]
    fn regularization_validation() {
        assert_eq!(
            EwRls::builder(2).regularization(-0.1).build().unwrap_err(),
            Error::InvalidRegularization(-0.1)
        );
        assert!(
            EwRls::builder(2)
                .regularization(f64::NAN)
                .build()
                .is_err()
        );
    }

    #[test]
    fn convergence_to_true_parameters() {
        let true_theta = [2.5, -3.0, 1.2];
        let mut model = EwRls::new(3, 0.99).unwrap();
        let mut rng = rand::rng();

        for _ in 0..2000 {
            let x: Vec<f64> = (0..3).map(|_| rng.random_range(-5.0..5.0)).collect();
            let y: f64 = x
                .iter()
                .zip(true_theta)
                .map(|(xi, ti)| xi * ti)
                .sum::<f64>()
                + rng.random_range(-0.1..0.1);
            model.update(&x, y).unwrap();
        }

        for (learned, truth) in model.params().iter().zip(true_theta) {
            assert!((learned - truth).abs() < 0.1, "did not converge");
        }
    }

    #[test]
    fn tracks_drifting_parameters() {
        // The whole point of λ < 1: after a regime switch the model re-adapts.
        let mut model = EwRls::new(2, 0.95).unwrap();
        let mut rng = rand::rng();

        let mut run = |model: &mut EwRls, m: f64, b: f64| {
            for _ in 0..500 {
                let x = [rng.random_range(-1.0..1.0), 1.0];
                model.update(&x, m * x[0] + b).unwrap();
            }
        };
        run(&mut model, 2.0, 1.0);
        run(&mut model, -4.0, 0.5); // regime switch

        assert!((model.params()[0] - -4.0).abs() < 1e-3);
        assert!((model.params()[1] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn covariance_stays_symmetric() {
        let mut model = EwRls::new(8, 0.99).unwrap();
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let x: Vec<f64> = (0..8).map(|_| rng.random_range(-3.0..3.0)).collect();
            model.update(&x, rng.random_range(-1.0..1.0)).unwrap();
        }
        let p = model.covariance();
        let asym = (p - p.transpose()).abs().max();
        assert!(asym < 1e-9, "covariance asymmetry {asym}");
    }

    #[test]
    fn windup_guard_caps_trace() {
        // λ < 1 with only the first direction excited: without the cap,
        // P[1,1] grows by 1/λ per step without bound.
        let mut model = EwRls::builder(2)
            .lambda(0.9)
            .initial_covariance(100.0)
            .max_covariance_trace(1e4)
            .build()
            .unwrap();

        for _ in 0..500 {
            model.update(&[1.0, 0.0], 1.0).unwrap();
        }
        let trace = model.covariance_trace();
        assert!(trace <= 1e4 * (1.0 + 1e-12), "trace {trace} exceeds cap");
        // The guard engaged (unbounded growth would exceed 1e4 in ~44 steps).
        assert!(trace > 1e3, "guard rescaled too aggressively: {trace}");
    }

    #[test]
    fn default_has_no_trace_cap() {
        // Textbook default: the estimator is never silently rescaled, so
        // under degenerate excitation the covariance grows freely.
        let mut model = EwRls::builder(2)
            .lambda(0.9)
            .initial_covariance(100.0)
            .build()
            .unwrap();
        for _ in 0..500 {
            model.update(&[1.0, 0.0], 1.0).unwrap();
        }
        assert!(model.covariance_trace() > 1e6);
    }

    #[test]
    fn cap_applies_at_initialization_and_reset() {
        let mut model = EwRls::builder(1)
            .initial_covariance(1000.0)
            .max_covariance_trace(1.0)
            .build()
            .unwrap();
        assert!((model.covariance_trace() - 1.0).abs() < A_TOL);
        model.update(&[1.0], 1.0).unwrap();
        model.reset_covariance();
        assert!((model.covariance_trace() - 1.0).abs() < A_TOL);
    }

    #[test]
    fn breakdown_leaves_state_unchanged() {
        // λ at the bottom of its valid range: the 1/λ forgetting overflows
        // the covariance. Must be detected before any state is modified.
        let mut model = EwRls::builder(1).lambda(1e-308).build().unwrap();
        let theta_before = model.params().to_vec();
        let trace_before = model.covariance_trace();

        assert_eq!(
            model.update(&[1e-200], 1.0).unwrap_err(),
            Error::NumericalBreakdown
        );
        assert_eq!(model.params(), theta_before.as_slice());
        assert_eq!(model.covariance_trace(), trace_before);
        assert_eq!(model.updates(), 0);
    }

    #[test]
    fn extreme_scale_never_stores_negative_variance() {
        // Catastrophic cancellation: P shrinks by ~17 digits in one step.
        // The observation is either rejected or committed with non-negative
        // diagonals — never a silently negative variance.
        let mut model = EwRls::builder(1)
            .lambda(1.0)
            .initial_covariance(1000.0)
            .build()
            .unwrap();
        let _ = model.update(&[1e7], 1.0);
        assert!(model.covariance()[(0, 0)] >= 0.0);
        assert!(model.prediction_variance(&[1.0]).unwrap() >= 0.0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_and_checkpoint_validation() {
        let mut model = EwRls::new(2, 0.99).unwrap();
        model.update(&[1.0, 2.0], 3.0).unwrap();

        let json = serde_json::to_string(&model).unwrap();
        let restored: EwRls = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.params(), model.params());
        assert_eq!(restored.covariance(), model.covariance());
        assert_eq!(restored.updates(), model.updates());

        // Dimensions inconsistent with the stored matrices must be rejected,
        // as must out-of-range configuration.
        let tampered = json.replace("\"dimensions\":2", "\"dimensions\":3");
        assert!(serde_json::from_str::<EwRls>(&tampered).is_err());
        let tampered = json.replace("\"lambda\":0.99", "\"lambda\":1.5");
        assert!(serde_json::from_str::<EwRls>(&tampered).is_err());
    }

    #[test]
    fn theta_overflow_rejected_before_commit() {
        // The second observation is finite but would push θ past f64::MAX.
        let mut model = EwRls::builder(1).lambda(1.0).build().unwrap();
        model.update(&[1.0], 1.7e308).unwrap();
        let theta_before = model.params().to_vec();

        assert_eq!(
            model.update(&[0.1], 1.7e308).unwrap_err(),
            Error::NumericalBreakdown
        );
        assert_eq!(model.params(), theta_before.as_slice());
        assert!(model.params()[0].is_finite());
    }

    #[test]
    fn eps_level_indefiniteness_is_tolerated() {
        // One huge collinear observation leaves P with an eps-scale negative
        // eigenvalue — inherent to covariance-form RLS. The follow-up update
        // must clamp the eps-level negative quadratic form, not break down.
        let mut model = EwRls::builder(2).lambda(1.0).build().unwrap();
        model.update(&[1e10, 1e10], 1.0).unwrap();
        model.update(&[1e10, 1e10], 1.0).unwrap();
        assert!(model.params().iter().all(|v| v.is_finite()));
        assert!(model.prediction_variance(&[1.0, 1.0]).unwrap() >= 0.0);
    }

    #[test]
    fn trace_overflow_rejected_before_commit() {
        // Each candidate diagonal is finite but their sum overflows; with the
        // default infinite cap this previously slipped past `trace > cap`.
        let mut model = EwRls::builder(2)
            .lambda(0.9)
            .initial_covariance(8.5e307)
            .build()
            .unwrap();
        let trace_before = model.covariance_trace();

        assert_eq!(
            model.update(&[0.0, 0.0], 1.0).unwrap_err(),
            Error::NumericalBreakdown
        );
        assert_eq!(model.covariance_trace(), trace_before);
        assert!(model.covariance_trace().is_finite());
    }

    #[test]
    fn subnormal_initial_covariance_survives() {
        // 1/δ overflows for subnormal δ; P₀ must still be δ, not zero.
        let model = EwRls::builder(1).initial_covariance(1e-320).build().unwrap();
        assert!(model.covariance_trace() > 0.0);
        assert!((model.covariance_trace() - 1e-320).abs() < 1e-321);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_rejects_asymmetric_covariance() {
        let model = EwRls::new(2, 0.99).unwrap();
        let json = serde_json::to_string(&model).unwrap();

        // Fresh P is 1000·I; replace it with a skew matrix.
        let tampered = json.replace("1000.0,0.0,0.0,1000.0", "1.0,0.5,-0.5,1.0");
        assert_ne!(tampered, json, "covariance data pattern not found");
        let err = serde_json::from_str::<EwRls>(&tampered).unwrap_err();
        assert!(err.to_string().contains("symmetric"), "unexpected error: {err}");
    }

    #[test]
    fn ridge_refresh_survives_extreme_gamma() {
        // A huge γ makes factor·θ_j overflow in the naive evaluation order
        // even though the committed corrections are finite.
        let mut model = EwRls::builder(1)
            .lambda(0.99)
            .regularization(1e308)
            .build()
            .unwrap();
        model.update(&[1e150], 1e308).unwrap();
        assert!(model.params()[0].is_finite());
        assert!(model.covariance_trace().is_finite());
    }

    #[test]
    fn overflowing_initial_trace_rejected() {
        // Per-entry δ is finite but d·δ overflows: reject at construction
        // instead of building a model whose every update breaks down.
        assert_eq!(
            EwRls::builder(2).initial_covariance(1e308).build().unwrap_err(),
            Error::InvalidInitialCovariance(1e308)
        );
        // A finite cap constrains the trace, so the same δ is acceptable.
        assert!(
            EwRls::builder(2)
                .initial_covariance(1e308)
                .max_covariance_trace(1e10)
                .build()
                .is_ok()
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_checkpoint_scale_invariants() {
        let model = EwRls::new(2, 0.99).unwrap();
        let json = serde_json::to_string(&model).unwrap();
        let tamper = |data: &str| {
            let t = json.replace("1000.0,0.0,0.0,1000.0", data);
            assert_ne!(t, json, "covariance data pattern not found");
            t
        };

        // 10% skew at 1e308 scale: the additive tolerance scale used to
        // overflow to infinity and accept this.
        let err = serde_json::from_str::<EwRls>(&tamper("1000.0,9e307,1e308,1000.0"))
            .unwrap_err();
        assert!(err.to_string().contains("symmetric"), "unexpected error: {err}");

        // Same-sign off-diagonals near f64::MAX: averaging must not overflow.
        let restored: EwRls =
            serde_json::from_str(&tamper("1000.0,1e308,1e308,1000.0")).unwrap();
        assert!(restored.covariance().iter().all(|v| v.is_finite()));

        // Diagonals whose sum overflows: the trace invariant must reject.
        let err = serde_json::from_str::<EwRls>(&tamper("1e308,0.0,0.0,1e308"))
            .unwrap_err();
        assert!(err.to_string().contains("trace"), "unexpected error: {err}");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_checkpoint_respects_max_trace() {
        let model = EwRls::builder(2)
            .initial_covariance(1000.0)
            .max_covariance_trace(100.0)
            .build()
            .unwrap();
        let json = serde_json::to_string(&model).unwrap();

        // Live P is capped at 50·I; a checkpoint claiming 1000·I under the
        // same stored cap violates the invariant and must be rejected.
        let tampered = json.replace("50.0,0.0,0.0,50.0", "1000.0,0.0,0.0,1000.0");
        assert_ne!(tampered, json, "covariance data pattern not found");
        let err = serde_json::from_str::<EwRls>(&tampered).unwrap_err();
        assert!(err.to_string().contains("max trace"), "unexpected error: {err}");
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut model = EwRls::new(2, 0.99).unwrap();
        model.update(&[1.0, 2.0], 3.0).unwrap();
        assert!(model.updates() == 1);

        model.reset();
        assert_eq!(model.updates(), 0);
        assert!(model.params().iter().all(|&v| v == 0.0));
        assert!((model.covariance_trace() - 2.0 * 1e3).abs() < A_TOL);
    }

    #[test]
    fn effective_window() {
        assert!((EwRls::new(2, 0.99).unwrap().effective_window() - 100.0).abs() < 1e-9);
        assert!(EwRls::new(2, 1.0).unwrap().effective_window().is_infinite());
    }
}
