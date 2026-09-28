//! Exponentially-weighted recursive least squares (EW-RLS) online regression.
//!
//! After every update the model holds the exact minimizer of
//!
//! ```text
//! J_t(θ) = Σ_{i=1..t} Λ_{i,t} w_i (y_i - x_iᵀθ)²
//!          + Λ_{0,t} δ⁻¹ ‖θ‖² + θᵀΓ_tθ
//! ```
//!
//! where `Λ_{i,t}` is the product of every forgetting factor applied after
//! observation `i` — the `λ` of each later update ([`EwRls::set_lambda`]) and
//! every [`EwRls::decay`] call — `w_i > 0` is an optional per-observation
//! weight, `δ` is the initial covariance scale (a prior on `θ = 0` that is
//! forgotten with the data; large `δ` means a weak prior), and `Γ_t` is the
//! optional persistent ridge ([`Ridge`]). With a constant `λ` and no `decay`
//! calls, `Λ_{i,t} = λ^{t-i}` and this is the textbook
//! `Σ λ^{t-i} w_i (y_i - x_iᵀθ)² + λ^t δ⁻¹‖θ‖² + θᵀΓ_tθ`.
//!
//! `Γ_t` is diagonal and is forgotten with the data, but every update tops
//! one coordinate `j` (cycling through all `d`) back up to its target:
//! `γ·s_j` for [`Ridge::Absolute`], or `κ·W_t·s_j` for [`Ridge::Relative`],
//! where `W_t` is the discounted sample mass ([`EwRls::weight_mass`]) and
//! `s_j ≥ 0` a per-coordinate weight ([`Builder::ridge_weights`]; `0` leaves
//! a coordinate such as an intercept unpenalized). Right after its refresh a
//! coordinate holds exactly its target, and in between it has been forgotten
//! by at most `d − 1` updates, for any sequence of forgetting factors. With
//! `λ = 1`, no `decay` calls and an absolute ridge, `Γ_t = γ·diag(s)` exactly.
//!
//! Each update costs `O(d²)` time via the Sherman–Morrison identity and
//! performs no heap allocation. The covariance update
//! `P ← (P − (Px)(Px)ᵀ/den)/λ` is evaluated with bit-for-bit symmetric
//! rank-1 terms, so `P` stays exactly symmetric: rounding asymmetry, which
//! forgetting would amplify by `1/λ` per update, never arises.
//!
//! Safeguards over a naive textbook implementation:
//!
//! - fallible API ([`Error`]) instead of panics, with input validation
//!   (dimension checks, non-finite rejection),
//! - covariance **wind-up protection**: with `λ < 1` and weakly exciting
//!   inputs, `P` grows exponentially in unexcited directions. A persistent
//!   ridge ([`Builder::regularization`], [`Builder::ridge`]) bounds it
//!   consistently with the objective. Whatever the ridge, forgetting
//!   saturates at the prior (stabilized forgetting): once a coordinate's
//!   variance exceeds `2δ`, information `1/δ − 1/P_jj` is added back on it
//!   as a pseudo-observation at the current `θ_j`, so `P_jj` returns to `δ`
//!   and `θ` is unchanged. Each such step adds a prior term centred on the
//!   estimate at the time to `J_t`; with `λ = 1` and no `decay` it never
//!   happens. A hard trace cap ([`Builder::max_covariance_trace`]) is also
//!   available but off by default, since capping biases the estimator away
//!   from the exact minimizer of `J_t`,
//! - **bounded forgetting per step**: one step with factor `λ` inflates `P`
//!   by `1/λ`, and later updates must cancel that inflation, losing about
//!   `log₁₀(1/λ)` digits. Past what `f64` can resolve, the covariance form
//!   silently loses positive definiteness. So `λ` must be at least
//!   [`MIN_LAMBDA`], and [`EwRls::decay`] treats a smaller product of decays
//!   since the last update (a long feed gap) as complete forgetting: it
//!   resets `P` to the prior and keeps `θ`. Forgetting that accumulates
//!   across updates carrying little information along some coordinate (zero
//!   rows, sparse features) is bounded by the prior floor above,
//! - **numerical breakdown detection**: an observation whose update would
//!   produce a non-finite state or a negative variance is rejected with
//!   [`Error::NumericalBreakdown`] before any state is modified. Definiteness
//!   is only checked along the observed direction, which is why forgetting is
//!   bounded above,
//! - per-update diagnostics ([`UpdateReport`]): a-priori prediction, residual,
//!   `xᵀPx` and the innovation variance scale, usable for residual z-scoring
//!   and for rejecting outliers before they are applied
//!   ([`EwRls::update_weighted_if`]).
//!
//! # Time-based forgetting
//!
//! For irregularly spaced observations, keep `λ = 1` per update and forget
//! by elapsed time with [`EwRls::decay`] (half-life `τ`: factor
//! `2^(−Δt/τ)`). A long gap simply resets the confidence to the prior:
//!
//! ```
//! use ewrls::{EwRls, Ridge};
//!
//! let mut model = EwRls::builder(3)
//!     .lambda(1.0)
//!     .ridge(Ridge::Relative(0.1)) // shrinkage proportional to recent data
//!     .ridge_weights(&[0.0, 1.0, 1.0]) // column 0 is an unpenalized intercept
//!     .build()?;
//! let half_life = 3600.0; // seconds
//! let mut last = 0.0;
//! for (t, x, y) in [
//!     (0.0, [1.0, 0.2, -0.1], 0.3),
//!     (0.5, [1.0, 0.1, 0.4], -0.2),
//!     (86_400.0, [1.0, -0.3, 0.2], 0.1), // after a one-day gap
//! ] {
//!     model.decay(0.5_f64.powf((t - last) / half_life))?;
//!     last = t;
//!     model.update(&x, y)?;
//! }
//! # Ok::<(), ewrls::Error>(())
//! ```
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

/// Recompute the exact covariance bound every this many updates.
const BOUND_REFRESH_INTERVAL: u64 = 256;
/// Keep fused covariance intermediates comfortably below overflow.
const FUSED_UPDATE_LIMIT: f64 = f64::MAX / 4.0;

/// Smallest forgetting accepted between two updates.
///
/// Forgetting by a factor `f` inflates the covariance by `1/f`, and the
/// following updates must cancel that inflation, losing about `log₁₀(1/f)`
/// significant digits (more in ill-conditioned directions). Below about
/// `1e-8` the covariance form can silently lose positive definiteness and
/// corrupt `θ`; a factor of `1e-6` keeps a millionth of the old information.
/// [`EwRls::set_lambda`] and [`Builder::lambda`] therefore reject smaller
/// values, and [`EwRls::decay`] treats a smaller product of decays since the
/// last update as complete forgetting.
///
/// This bounds one step only. Forgetting accumulated over updates that carry
/// little information along some coordinate (zero rows, sparse features, a
/// decay followed by a small `λ`) is bounded separately: no coordinate's
/// variance stays above twice the prior `δ` (see the crate-level
/// safeguards).
pub const MIN_LAMBDA: f64 = 1e-6;

/// Errors returned by [`EwRls`] construction and updates.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The model must have at least one feature dimension.
    #[error("dimensions must be greater than 0")]
    ZeroDimensions,
    /// The forgetting factor must satisfy [`MIN_LAMBDA`] `<= lambda <= 1`.
    #[error("forgetting factor lambda must be in [{min}, 1], got {0}", min = MIN_LAMBDA)]
    InvalidLambda(f64),
    /// A [`EwRls::decay`] factor must satisfy `0 <= factor <= 1`.
    #[error("decay factor must be in [0, 1], got {0}")]
    InvalidDecay(f64),
    /// The initial covariance scale must be positive and finite, and so must
    /// the trace of the prior it gives with the ridge at zero data.
    #[error("initial covariance must be positive and finite, got {0}")]
    InvalidInitialCovariance(f64),
    /// The covariance trace cap must be positive.
    #[error("max covariance trace must be positive, got {0}")]
    InvalidMaxTrace(f64),
    /// The ridge strength and every ridge weight must be non-negative and
    /// finite, as must their products.
    #[error("regularization must be non-negative and finite, got {0}")]
    InvalidRegularization(f64),
    /// The input vector (or ridge weight vector) length does not match the
    /// model dimensions.
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
    /// Applying the observation would produce a non-finite parameter,
    /// covariance entry or trace, or a negative variance; the observation is
    /// rejected and the model state is unchanged.
    ///
    /// Definiteness is only checked along the observed direction, so a
    /// streak of these errors means the covariance is already unhealthy:
    /// recover with [`EwRls::reset_covariance`] (keeps `θ`) or
    /// [`EwRls::reset`].
    #[error(
        "numerical breakdown: the update would produce a non-finite or indefinite state; observation rejected"
    )]
    NumericalBreakdown,
}

/// Error from a batch fit or prediction operation.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum BatchError {
    /// Batch `fit` requires at least one observation.
    #[error("fit requires at least one observation")]
    EmptyBatch,
    /// The target slice length does not match the number of feature rows.
    #[error("batch has {expected} feature rows but {got} targets")]
    TargetLengthMismatch {
        /// Number of feature rows.
        expected: usize,
        /// Number of supplied targets.
        got: usize,
    },
    /// The weight slice length does not match the number of feature rows.
    #[error("batch has {expected} feature rows but {got} sample weights")]
    WeightLengthMismatch {
        /// Number of feature rows.
        expected: usize,
        /// Number of supplied weights.
        got: usize,
    },
    /// The output slice length does not match the number of feature rows.
    #[error("batch has {expected} feature rows but output has length {got}")]
    OutputLengthMismatch {
        /// Number of feature rows.
        expected: usize,
        /// Length of the supplied output buffer.
        got: usize,
    },
    /// A particular observation failed validation or numerical updating.
    #[error("batch sample {index}: {source}")]
    Sample {
        /// Zero-based index of the failing observation.
        index: usize,
        /// Underlying single-observation error.
        #[source]
        source: Error,
    },
}

/// Diagnostics produced by a single update, all computed *before* the
/// parameters were adjusted (a-priori quantities).
///
/// With observation noise variance `σ²` (per unit weight), the a-priori
/// residual has variance `σ²·innovation_scale` when `θ` is modelled as a
/// random walk (the Kalman reading of forgetting). For a fixed `θ` it is
/// `σ²·(1/w + c·predictive_variance)`, with `c = 1` at `λ = 1` and
/// `c ≈ 1/(1 + λ)` in the steady state of a constant `λ < 1` with stationary
/// inputs. For residual z-scoring use `residual / sqrt(σ̂²·innovation_scale)`
/// or the fixed-`θ` variance.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct UpdateReport {
    /// A-priori prediction `xᵀθ` using the parameters before this update.
    pub prediction: f64,
    /// A-priori residual `y − xᵀθ` (the innovation).
    pub residual: f64,
    /// `xᵀPx` before this update: the parameter uncertainty in the direction
    /// of `x`, in units of the noise variance `σ²`.
    pub predictive_variance: f64,
    /// `1/w + xᵀPx/λ`: the innovation variance in units of `σ²` under the
    /// random-walk reading of forgetting.
    pub innovation_scale: f64,
}

/// Persistent ridge penalty on `θ`, kept up to date under forgetting.
///
/// Coordinate `j`'s penalty is topped up to its target once every `d`
/// updates, where the target is scaled by the ridge weight `s_j`
/// ([`Builder::ridge_weights`], default `1`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Ridge {
    /// Target `γ·s_j`, fixed in absolute terms (`γ ≥ 0`). With `λ = 1` this is
    /// the classic ridge `(XᵀX + (δ⁻¹ + γ)I)⁻¹Xᵀy`.
    Absolute(f64),
    /// Target `κ·W_t·s_j`, proportional to the discounted sample mass `W_t`
    /// ([`EwRls::weight_mass`]), so the shrinkage per unit of data stays at
    /// `κ` when activity or the forgetting rate changes (`κ ≥ 0`). The target
    /// follows `W_t` within `d` updates.
    Relative(f64),
}

impl Ridge {
    /// The configured strength (`γ` or `κ`).
    #[must_use]
    pub fn strength(self) -> f64 {
        match self {
            Ridge::Absolute(strength) | Ridge::Relative(strength) => strength,
        }
    }

    fn validate(self) -> Result<Self, Error> {
        let strength = self.strength();
        if strength >= 0.0 && strength.is_finite() {
            Ok(self)
        } else {
            Err(Error::InvalidRegularization(strength))
        }
    }
}

/// What [`EwRls::decay`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecayOutcome {
    /// The information was scaled by the factor (`P ← P/factor`).
    Applied,
    /// The product of the decays since the last update fell below
    /// [`MIN_LAMBDA`] (or `P/factor` would overflow): the covariance was reset to the prior, keeping `θ`, as by
    /// [`EwRls::reset_covariance`].
    Reset,
}

/// Builder for [`EwRls`] with optional knobs beyond [`EwRls::new`].
#[derive(Debug, Clone)]
pub struct Builder {
    dimensions: usize,
    lambda: f64,
    initial_covariance: f64,
    max_covariance_trace: Option<f64>,
    ridge: Ridge,
    ridge_weights: Option<Vec<f64>>,
}

impl Builder {
    /// Forgetting factor `λ ∈ [MIN_LAMBDA, 1]` applied by every update.
    /// Default `0.99`.
    ///
    /// The effective sample window is roughly `1 / (1 − λ)`; `λ = 1` weights
    /// all history equally (plain recursive least squares). For irregularly
    /// spaced observations use `λ = 1` with time-based [`EwRls::decay`].
    #[must_use]
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Initial covariance scale `δ`. Default `1e3`.
    ///
    /// Equivalent to a ridge penalty `δ⁻¹‖θ‖²` on the discounted cost; large
    /// values mean low confidence in the zero initialization and fast initial
    /// adaptation. The prior covariance is `P₀ = diag((δ⁻¹ + g_j)⁻¹)`, where
    /// `g_j` is the ridge penalty on coordinate `j` at zero data (`γ·s_j` for
    /// an absolute ridge, `0` for a relative one), rescaled if it exceeds
    /// [`Builder::max_covariance_trace`].
    ///
    /// `δ` also sets the stabilized-forgetting floor: forgetting never leaves
    /// a coordinate's variance above `2δ`; it is pulled back to `δ`.
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
    /// with the objective. A binding cap also suspends forgetting, since it
    /// shrinks `P` in informed directions too, while the prior floor
    /// ([`Builder::initial_covariance`]) already bounds wind-up per
    /// coordinate.
    #[must_use]
    pub fn max_covariance_trace(mut self, max_trace: f64) -> Self {
        self.max_covariance_trace = Some(max_trace);
        self
    }

    /// Persistent absolute ridge penalty `γ ≥ 0` on `‖θ‖²`; shorthand for
    /// `ridge(Ridge::Absolute(gamma))`. Default `0`.
    ///
    /// Unlike the `δ⁻¹` prior, this penalty is kept up to date under
    /// forgetting. It is exactly `γ·diag(s)` at `λ = 1`; with forgetting,
    /// each coordinate is topped back up to `γ·s_j` once every `d` updates,
    /// so it ranges from `γ·s_j` down to `γ·s_j` times the forgetting of the
    /// last `d − 1` updates (`γλ^(d−1)` for a constant `λ`), and its mean is
    /// `ργ` with `ρ = (1 − λᵈ)/(d(1 − λ))`. Besides shrinking `θ`, it bounds
    /// the covariance in unexcited directions, so it also acts as wind-up
    /// protection.
    #[must_use]
    pub fn regularization(self, gamma: f64) -> Self {
        self.ridge(Ridge::Absolute(gamma))
    }

    /// Persistent ridge penalty, absolute or relative to the discounted
    /// sample mass. Default `Ridge::Absolute(0.0)`.
    #[must_use]
    pub fn ridge(mut self, ridge: Ridge) -> Self {
        self.ridge = ridge;
        self
    }

    /// Per-coordinate ridge weights `s_j ≥ 0`, one per feature. Default all
    /// `1`. A weight of `0` leaves that coordinate unpenalized (e.g. an
    /// intercept column); it still gets the `δ⁻¹` prior.
    #[must_use]
    pub fn ridge_weights(mut self, weights: &[f64]) -> Self {
        self.ridge_weights = Some(weights.to_vec());
        self
    }

    /// Validate the configuration and build the model.
    pub fn build(self) -> Result<EwRls, Error> {
        let d = self.dimensions;
        if d == 0 {
            return Err(Error::ZeroDimensions);
        }
        validate_lambda(self.lambda)?;
        if !(self.initial_covariance > 0.0 && self.initial_covariance.is_finite()) {
            return Err(Error::InvalidInitialCovariance(self.initial_covariance));
        }
        let max_trace = self.max_covariance_trace.unwrap_or(f64::INFINITY);
        if max_trace.is_nan() || max_trace <= 0.0 {
            return Err(Error::InvalidMaxTrace(max_trace));
        }
        let ridge_weights = match self.ridge_weights {
            Some(weights) if weights.len() != d => {
                return Err(Error::DimensionMismatch {
                    expected: d,
                    got: weights.len(),
                });
            }
            Some(weights) => DVector::from_vec(weights),
            None => DVector::from_element(d, 1.0),
        };
        let ridge = validate_ridge(self.ridge, &ridge_weights)?;

        let mut model = EwRls {
            theta: DVector::zeros(d),
            p: DMatrix::zeros(d, d),
            p_x: DVector::zeros(d),
            candidate_diagonal: DVector::zeros(d),
            p_abs_bound: 0.0,
            lambda: self.lambda,
            dimensions: d,
            initial_covariance: self.initial_covariance,
            max_trace,
            ridge,
            ridge_weights,
            ridge_information: DVector::zeros(d),
            weight_mass: 0.0,
            pending_decay: 1.0,
            updates: 0,
        };
        // The ridge starts in the prior information matrix,
        // P₀ = diag((δ⁻¹ + g_j)⁻¹). The trace cap applies from the start.
        model.reset_covariance();
        // A per-entry scale can be finite while summing the stored diagonal
        // overflows. Check the matrix using the same order as `trace()`.
        if !model.p.trace().is_finite() {
            return Err(Error::InvalidInitialCovariance(self.initial_covariance));
        }
        Ok(model)
    }
}

/// Exponentially-weighted recursive least squares regressor.
///
/// See the [crate-level documentation](crate) for the model definition,
/// numerical safeguards and usage examples.
///
/// With the `serde` feature the model (de)serializes as a validated,
/// versioned checkpoint (format version 2) that restores bit-exactly: the
/// restored model continues exactly like the original. With `serde_json`,
/// enable its `float_roundtrip` feature (or use a binary format), since the
/// default float parser does not round-trip every `f64`. Version 1
/// checkpoints (ewrls 0.2) still load; their ridge state and sample mass are
/// reconstructed assuming unit weights and a constant `λ`.
#[derive(Debug, Clone)]
pub struct EwRls {
    /// Parameter vector `θ` (`d × 1`).
    theta: DVector<f64>,
    /// Inverse information (covariance) matrix `P` (`d × d`).
    p: DMatrix<f64>,
    /// Workspace for `P·x`, kept to make updates allocation-free.
    p_x: DVector<f64>,
    /// Validated candidate covariance diagonal, reused by the fused update.
    candidate_diagonal: DVector<f64>,
    /// Conservative upper bound on every `|P[i, j]|`.
    p_abs_bound: f64,
    lambda: f64,
    dimensions: usize,
    initial_covariance: f64,
    /// Serialized as `null` when infinite — JSON has no Inf literal.
    max_trace: f64,
    ridge: Ridge,
    /// Per-coordinate ridge weights `s_j`.
    ridge_weights: DVector<f64>,
    /// Ridge penalty currently held by each coordinate (the diagonal of
    /// `Γ_t`), forgotten with the data and topped up by the refresh.
    ridge_information: DVector<f64>,
    /// Discounted sample mass `W_t = Σ Λ_{i,t} w_i`, saturating at `f64::MAX`.
    weight_mass: f64,
    /// Product of the [`EwRls::decay`] factors applied since the last
    /// committed update (1 right after one), bounded below by [`MIN_LAMBDA`].
    pending_decay: f64,
    updates: u64,
}

/// Checkpoint format written by this version. Version 1 checkpoints
/// (ewrls ≤ 0.2.0) are still accepted, see [`EwRls`]'s `Deserialize` notes.
#[cfg(feature = "serde")]
const CHECKPOINT_VERSION: u32 = 2;

#[cfg(feature = "serde")]
#[derive(serde::Serialize)]
struct EwRlsCheckpointRef<'a> {
    version: u32,
    theta: &'a DVector<f64>,
    p: &'a DMatrix<f64>,
    lambda: f64,
    dimensions: usize,
    initial_covariance: f64,
    #[serde(serialize_with = "serde_inf::serialize")]
    max_trace: f64,
    ridge: Ridge,
    ridge_weights: &'a DVector<f64>,
    ridge_information: &'a DVector<f64>,
    weight_mass: f64,
    pending_decay: f64,
    p_abs_bound: f64,
    updates: u64,
}

/// Serializes a versioned checkpoint that restores bit-exactly: a restored
/// model continues exactly like the original. With `serde_json`, enable its
/// `float_roundtrip` feature (or use a binary format): the default float
/// parser does not round-trip every `f64`.
#[cfg(feature = "serde")]
impl serde::Serialize for EwRls {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(
            &EwRlsCheckpointRef {
                version: CHECKPOINT_VERSION,
                theta: &self.theta,
                p: &self.p,
                lambda: self.lambda,
                dimensions: self.dimensions,
                initial_covariance: self.initial_covariance,
                max_trace: self.max_trace,
                ridge: self.ridge,
                ridge_weights: &self.ridge_weights,
                ridge_information: &self.ridge_information,
                weight_mass: self.weight_mass,
                pending_decay: self.pending_decay,
                p_abs_bound: self.p_abs_bound,
                updates: self.updates,
            },
            serializer,
        )
    }
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
    /// `lambda`, using default initial covariance (`1e3`) and no trace cap.
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
            ridge: Ridge::Absolute(0.0),
            ridge_weights: None,
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
    /// The parameter uncertainty in the direction of `x`, in units of the
    /// observation noise variance `σ²`. For a fixed `θ` the variance of the
    /// prediction `xᵀθ` is `σ²·xᵀPx` at `λ = 1`, and about `σ²·xᵀPx/(1 + λ)`
    /// in the steady state of a constant `λ < 1` with stationary inputs;
    /// under the random-walk reading of forgetting, the one-step-ahead
    /// variance is `σ²·xᵀPx/λ`. Clamped to be non-negative.
    pub fn prediction_variance(&self, x: &[f64]) -> Result<f64, Error> {
        self.validate_features(x)?;
        Ok(self.prediction_variance_validated(x))
    }

    fn prediction_variance_validated(&self, x: &[f64]) -> f64 {
        let xv = DVectorView::from_slice(x, self.dimensions);
        // Walk columns so accesses stay contiguous in the column-major P.
        let mut acc = 0.0;
        for (j, &xj) in x.iter().enumerate() {
            acc += xj * self.p.column(j).dot(&xv);
        }
        acc.max(0.0)
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
    pub fn update_weighted(
        &mut self,
        x: &[f64],
        y: f64,
        weight: f64,
    ) -> Result<UpdateReport, Error> {
        self.validate_observation(x, y, weight)?;
        self.update_weighted_validated(x, y, weight)
    }

    /// Like [`EwRls::update`], but `accept` sees the a-priori
    /// [`UpdateReport`] first and can veto the observation.
    ///
    /// See [`EwRls::update_weighted_if`].
    pub fn update_if<F: FnOnce(&UpdateReport) -> bool>(
        &mut self,
        x: &[f64],
        y: f64,
        accept: F,
    ) -> Result<Option<UpdateReport>, Error> {
        self.update_weighted_if(x, y, 1.0, accept)
    }

    /// Like [`EwRls::update_weighted`], but `accept` sees the a-priori
    /// [`UpdateReport`] before anything is committed and can veto the
    /// observation, e.g. a bad tick whose residual z-score is too large.
    ///
    /// Returns `Ok(None)` when vetoed; the model state is then unchanged
    /// (the update's forgetting is not applied either). The report costs
    /// nothing extra: it is computed by every update anyway.
    pub fn update_weighted_if<F: FnOnce(&UpdateReport) -> bool>(
        &mut self,
        x: &[f64],
        y: f64,
        weight: f64,
        accept: F,
    ) -> Result<Option<UpdateReport>, Error> {
        self.validate_observation(x, y, weight)?;
        let innovation = self.innovation(x, y, weight)?;
        if !accept(&innovation.report) {
            return Ok(None);
        }
        self.commit(&innovation)?;
        Ok(Some(innovation.report))
    }

    fn update_weighted_validated(
        &mut self,
        x: &[f64],
        y: f64,
        weight: f64,
    ) -> Result<UpdateReport, Error> {
        let innovation = self.innovation(x, y, weight)?;
        self.commit(&innovation)?;
        Ok(innovation.report)
    }

    /// The a-priori half of an update: computes `P·x` into the workspace and
    /// the innovation. Modifies no model state.
    fn innovation(&mut self, x: &[f64], y: f64, weight: f64) -> Result<Innovation, Error> {
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

        Ok(Innovation {
            report: UpdateReport {
                prediction,
                residual,
                predictive_variance: quad,
                innovation_scale: denominator / self.lambda,
            },
            inv_den: 1.0 / denominator,
            gain_scale,
            weight,
        })
    }

    /// The committing half of an update, using the `P·x` workspace left by
    /// [`EwRls::innovation`]. Validates the whole candidate state first, so
    /// an error leaves the model unchanged.
    fn commit(&mut self, innovation: &Innovation) -> Result<(), Error> {
        let Innovation {
            inv_den,
            gain_scale,
            weight,
            ..
        } = *innovation;

        // Validate the full candidate state before committing anything: every
        // new parameter and covariance entry must stay finite, every new
        // covariance diagonal must be non-negative, and its trace must not
        // overflow.
        // Otherwise the observation has exhausted f64 precision (or
        // overflowed 1/λ) and is rejected with the state untouched.
        let mut new_trace = 0.0;
        for j in 0..self.dimensions {
            let tj = self.theta[j] + gain_scale * self.p_x[j];
            let dj = (self.p[(j, j)] - self.p_x[j] * self.p_x[j] * inv_den) / self.lambda;
            if !tj.is_finite() || !dj.is_finite() || dj < 0.0 {
                return Err(Error::NumericalBreakdown);
            }
            self.candidate_diagonal[j] = dj;
            new_trace += dj;
        }
        if !new_trace.is_finite() {
            return Err(Error::NumericalBreakdown);
        }

        // Prove nalgebra's fused evaluation safe using a cached bound on all
        // covariance entries. This avoids an O(d²) validation pass for normal
        // inputs without relying on P being exactly positive semidefinite.
        let fused = fused_covariance_bound(self.p_abs_bound, self.p_x.amax(), inv_den, self.lambda);
        let mut exact_candidate_bound = 0.0_f64;
        if fused.is_none() {
            for j in 0..self.dimensions {
                for i in 0..self.dimensions {
                    let pij = (self.p[(i, j)] - self.p_x[i] * self.p_x[j] * inv_den) / self.lambda;
                    if !pij.is_finite() {
                        return Err(Error::NumericalBreakdown);
                    }
                    exact_candidate_bound = exact_candidate_bound.max(pij.abs());
                }
            }
        }

        // θ ← θ + k·e with gain k = P·x / den, element-wise with exactly the
        // expression validated above so the committed values cannot differ.
        for j in 0..self.dimensions {
            self.theta[j] += gain_scale * self.p_x[j];
        }

        if let Some(fused) = fused {
            // P ← P/λ − (Px)(Px)ᵀ/(den·λ), one SIMD-friendly matrix pass,
            // written as β·P − s·sᵀ with s = √α·Px. Entry (i, j) then subtracts
            // s_j·s_i and entry (j, i) subtracts s_i·s_j, the same product, so
            // P stays bit-for-bit symmetric. (With α·(Px)_j·(Px)_i the two
            // triangles round differently, and every later update multiplies
            // that asymmetry by 1/λ until P is no longer definite.) The
            // direct, validated diagonal is restored to avoid cancellation
            // drift from the distributed fused expression.
            self.p_x.scale_mut(fused.alpha_abs.sqrt());
            self.p.ger(-1.0, &self.p_x, &self.p_x, fused.beta);
            for j in 0..self.dimensions {
                self.p[(j, j)] = self.candidate_diagonal[j];
            }
            self.p_abs_bound = fused.candidate_bound.max(self.candidate_diagonal.amax());
        } else {
            // Cancellation-heavy extreme: preserve the exact evaluation order
            // validated above, even though it costs a second covariance pass.
            for j in 0..self.dimensions {
                for i in 0..self.dimensions {
                    self.p[(i, j)] =
                        (self.p[(i, j)] - self.p_x[i] * self.p_x[j] * inv_den) / self.lambda;
                }
            }
            self.p_abs_bound = exact_candidate_bound;
        }

        // The data step forgot the ridge penalty and the sample mass by λ
        // along with everything else. The mass is informational for an
        // absolute ridge, so it saturates instead of rejecting the update.
        self.weight_mass = (self.lambda * self.weight_mass + weight).min(f64::MAX);
        if self.lambda < 1.0 {
            self.ridge_information.scale_mut(self.lambda);
        }
        self.pending_decay = 1.0;
        self.refresh_ridge();
        self.floor_information();
        self.apply_trace_cap();

        self.updates += 1;
        if self.updates.is_multiple_of(BOUND_REFRESH_INTERVAL) {
            // Tighten the conservative bound before repeated triangle bounds
            // make it unnecessarily force the exact fallback.
            self.p_abs_bound = self.p.amax();
        }
        Ok(())
    }

    /// Persistent ridge refresh: tops coordinate `j = updates mod d` back up
    /// to its target with a zero-target pseudo-observation `x = √c·e_j`,
    /// `y = 0` (no forgetting on this step), where `c` is the penalty the
    /// coordinate lost to forgetting since its last refresh, or gained in
    /// target. Right after it, the coordinate holds exactly its target for any
    /// sequence of forgetting factors. Shrinks `θ_j` toward 0 and bounds `P`
    /// in direction `j`. A lower target is reached only through forgetting.
    fn refresh_ridge(&mut self) {
        let j = (self.updates % self.dimensions as u64) as usize;
        let target = self.ridge_target(j);
        let c = target - self.ridge_information[j];
        if !(c > 0.0 && c.is_finite()) {
            return;
        }
        // den ≥ 1 since c > 0 and P[jj] ≥ 0; skip the refresh on overflow and
        // retry on the coordinate's next turn.
        let den = 1.0 + c * self.p[(j, j)];
        if den.is_finite()
            && self.add_coordinate_information(j, 1.0 / den, (c / den).sqrt(), -self.theta[j])
        {
            self.ridge_information[j] = target;
        }
    }

    /// Stabilized forgetting: where forgetting has left coordinate `j` with
    /// `P_jj > 2δ`, adds information `1/δ − 1/P_jj` on it as a
    /// pseudo-observation at the current `θ_j` (so `θ` is unchanged), which
    /// brings `P_jj` back to `δ`. This bounds the inflation later updates
    /// must cancel for any forgetting history. Informed directions change by
    /// a relative `δ⁻¹/Ω`, and with `λ = 1` and no [`EwRls::decay`] it never
    /// fires, since `P` stays below the prior.
    fn floor_information(&mut self) {
        let delta = self.initial_covariance;
        for j in 0..self.dimensions {
            let pjj = self.p[(j, j)];
            if pjj > 2.0 * delta {
                // `1 + c·P_jj = P_jj/δ` and `c/(1 + c·P_jj) = (1 − δ/P_jj)/P_jj`,
                // formed without `1/δ`, which overflows for subnormal δ.
                let shrink = delta / pjj;
                let root = (1.0 - shrink).sqrt() / pjj.sqrt();
                self.add_coordinate_information(j, shrink, root, 0.0);
            }
        }
    }

    /// Adds information `c > 0` on coordinate `j` without forgetting: a
    /// pseudo-observation `x = √c·e_j` with residual `residual` (target
    /// `θ_j + residual`), given as `shrink = 1/(1 + c·P_jj)` and
    /// `root = √(c·shrink)`. It is the rank-1 downdate `P ← P − s·sᵀ` with
    /// `s = root·P[:, j]`, except that row and column `j` are written as the
    /// exact `shrink·P[:, j]`: the downdate `P_ij − s_i·s_j` cancels once
    /// `c·P_jj` is large. The other diagonal entries are the Schur complement
    /// `P_ii − s_i²`, which rounding can push below zero for a coordinate
    /// (nearly) collinear with `j`, so the whole candidate is validated
    /// first. Returns `false`, leaving the state untouched, if it is not
    /// usable.
    fn add_coordinate_information(
        &mut self,
        j: usize,
        shrink: f64,
        root: f64,
        residual: f64,
    ) -> bool {
        if !(shrink >= 0.0 && root > 0.0 && root.is_finite()) {
            return false;
        }
        self.p_x.copy_from(&self.p.column(j));
        // Bound the largest θ and P corrections before committing,
        // multiplying root into the (often tiny) column first so a huge `c`
        // cannot overflow an intermediate. The cached absolute covariance
        // bound covers every element.
        let s_max = root * self.p_x.amax();
        let theta_bound = self.theta.amax() + (root * s_max) * residual.abs();
        let p_bound = inflate_bound(self.p_abs_bound + s_max * s_max);
        if !(theta_bound.is_finite() && p_bound.is_finite()) {
            return false;
        }
        for i in 0..self.dimensions {
            let pij = self.p_x[i];
            let dii = if i == j {
                pij * shrink
            } else {
                let si = root * pij;
                self.p[(i, i)] - si * si
            };
            if !(dii >= 0.0 && dii.is_finite()) {
                return false;
            }
            self.candidate_diagonal[i] = dii;
        }
        self.p_x.scale_mut(root);
        if residual != 0.0 {
            // θ ← θ + (c·shrink)·P[:, j]·residual, with θ_j in exact form.
            let theta_j = self.theta[j];
            for i in 0..self.dimensions {
                self.theta[i] += (self.p_x[i] * root) * residual;
            }
            self.theta[j] = (theta_j + residual) - residual * shrink;
        }
        // Bit-symmetric rank-1 downdate, then the exact row and column j and
        // the validated diagonal.
        self.p.ger(-1.0, &self.p_x, &self.p_x, 1.0);
        for i in 0..self.dimensions {
            let exact = self.p_x[i] / root * shrink;
            self.p[(i, j)] = exact;
            self.p[(j, i)] = exact;
            self.p[(i, i)] = self.candidate_diagonal[i];
        }
        self.p_abs_bound = p_bound;
        true
    }

    /// Ridge penalty coordinate `j` should hold right after its refresh.
    fn ridge_target(&self, j: usize) -> f64 {
        let weight = self.ridge_weights[j];
        match self.ridge {
            Ridge::Absolute(gamma) => gamma * weight,
            Ridge::Relative(kappa) => kappa * self.weight_mass * weight,
        }
    }

    /// Optional wind-up cap: rescale `P` if its trace exceeds the cap.
    fn apply_trace_cap(&mut self) {
        let trace = self.p.trace();
        if trace > self.max_trace {
            let scale = self.max_trace / trace;
            self.p.scale_mut(scale);
            self.p_abs_bound = inflate_bound(self.p_abs_bound * scale);
        }
    }

    /// Forget the accumulated information by `factor ∈ [0, 1]` without an
    /// observation: `P ← P/factor`, while the ridge penalty and
    /// [`EwRls::weight_mass`] shrink by `factor`. `θ` is unchanged.
    ///
    /// This is time-based forgetting for irregularly spaced observations:
    /// keep `λ = 1` and call `decay(2^(−Δt/τ))` with half-life `τ` before each
    /// update, or once per time bucket (also for buckets without updates).
    /// Calls compose multiplicatively, and `decay(f)` followed by an update
    /// with `λ = 1` minimizes the same objective as an update with `λ = f`.
    /// It costs one `O(d²)` pass; when every decay is followed by an update
    /// and a gap never forgets more than [`MIN_LAMBDA`], `set_lambda(f)`
    /// before the update is equivalent and about 20% cheaper overall.
    ///
    /// When the product of the decays since the last update would fall below
    /// [`MIN_LAMBDA`] (for `2^(−Δt/τ)`, a gap of more than about 20
    /// half-lives, however it is split into calls), the remaining information
    /// cannot be represented accurately in covariance form. It is then treated
    /// as complete forgetting: the covariance is reset to the prior as by
    /// [`EwRls::reset_covariance`], keeping `θ`, and [`DecayOutcome::Reset`] is
    /// returned. The same happens if `P/factor` would overflow. Either way the
    /// ridge is back at its target within `d` updates. A decay that leaves a
    /// coordinate's variance above `2δ` pulls it back to `δ`, as described
    /// in the crate-level safeguards.
    ///
    /// Does not count as an update ([`EwRls::updates`] is unchanged).
    pub fn decay(&mut self, factor: f64) -> Result<DecayOutcome, Error> {
        if !(0.0..=1.0).contains(&factor) {
            return Err(Error::InvalidDecay(factor));
        }
        if factor == 1.0 {
            return Ok(DecayOutcome::Applied);
        }
        let pending_decay = self.pending_decay * factor;
        if pending_decay < MIN_LAMBDA {
            self.reset_covariance();
            return Ok(DecayOutcome::Reset);
        }
        // Scaling carries the cached bound along. Rescan for the exact
        // largest entry only when the scaled bound nears overflow, so that
        // repeated decays cannot inflate a stale bound into a spurious
        // overflow. If anything overflows, P is replaced below.
        let beta = 1.0 / factor;
        self.p.scale_mut(beta);
        let mut bound = inflate_bound(self.p_abs_bound * beta);
        if bound > FUSED_UPDATE_LIMIT {
            bound = self.p.amax();
        }
        if !bound.is_finite() || !self.p.trace().is_finite() {
            self.reset_covariance();
            return Ok(DecayOutcome::Reset);
        }
        self.p_abs_bound = bound;
        self.pending_decay = pending_decay;
        self.ridge_information.scale_mut(factor);
        self.weight_mass *= factor;
        self.floor_information();
        self.apply_trace_cap();
        Ok(DecayOutcome::Applied)
    }

    /// Reset the model and fit a batch of observations.
    ///
    /// Each element of `x` is one contiguous feature row, so this accepts
    /// idiomatic Rust containers such as `Vec<Vec<f64>>`, `Vec<[f64; N]>`,
    /// and slices of feature slices without conversion. All inputs are
    /// validated before the model is reset. If a later numerical breakdown
    /// occurs, the successfully processed prefix remains fitted.
    pub fn fit<X: AsRef<[f64]>>(&mut self, x: &[X], y: &[f64]) -> Result<&mut Self, BatchError> {
        self.validate_batch(x, y, None)?;
        if x.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        self.reset();
        self.apply_validated_batch(x, y, None)
    }

    /// Incrementally fit a batch without resetting the current model.
    ///
    /// On failure, observations before the reported sample index remain
    /// applied; the failing observation itself leaves model state unchanged.
    pub fn partial_fit<X: AsRef<[f64]>>(
        &mut self,
        x: &[X],
        y: &[f64],
    ) -> Result<&mut Self, BatchError> {
        self.validate_batch(x, y, None)?;
        self.apply_validated_batch(x, y, None)
    }

    /// Reset the model and fit a batch with one positive weight per sample.
    pub fn fit_weighted<X: AsRef<[f64]>>(
        &mut self,
        x: &[X],
        y: &[f64],
        sample_weight: &[f64],
    ) -> Result<&mut Self, BatchError> {
        self.validate_batch(x, y, Some(sample_weight))?;
        if x.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        self.reset();
        self.apply_validated_batch(x, y, Some(sample_weight))
    }

    /// Incrementally fit a weighted batch without resetting the model.
    pub fn partial_fit_weighted<X: AsRef<[f64]>>(
        &mut self,
        x: &[X],
        y: &[f64],
        sample_weight: &[f64],
    ) -> Result<&mut Self, BatchError> {
        self.validate_batch(x, y, Some(sample_weight))?;
        self.apply_validated_batch(x, y, Some(sample_weight))
    }

    /// Predict one value for every feature row, allocating the result vector.
    ///
    /// Use [`EwRls::predict_batch_into`] to reuse an output buffer in a hot
    /// path.
    pub fn predict_batch<X: AsRef<[f64]>>(&self, x: &[X]) -> Result<Vec<f64>, BatchError> {
        let mut output = vec![0.0; x.len()];
        self.predict_batch_into(x, &mut output)?;
        Ok(output)
    }

    /// Predict one value per feature row into a caller-provided buffer.
    ///
    /// The output remains unchanged if a row or the output length is invalid.
    pub fn predict_batch_into<X: AsRef<[f64]>>(
        &self,
        x: &[X],
        output: &mut [f64],
    ) -> Result<(), BatchError> {
        if output.len() != x.len() {
            return Err(BatchError::OutputLengthMismatch {
                expected: x.len(),
                got: output.len(),
            });
        }
        self.validate_feature_rows(x)?;
        for (row, prediction) in x.iter().zip(output) {
            let row = DVectorView::from_slice(row.as_ref(), self.dimensions);
            *prediction = row.dot(&self.theta);
        }
        Ok(())
    }

    /// Compute predictive variance for every feature row, allocating output.
    ///
    /// Use [`EwRls::prediction_variance_batch_into`] to reuse an output buffer
    /// in a hot path.
    pub fn prediction_variance_batch<X: AsRef<[f64]>>(
        &self,
        x: &[X],
    ) -> Result<Vec<f64>, BatchError> {
        let mut output = vec![0.0; x.len()];
        self.prediction_variance_batch_into(x, &mut output)?;
        Ok(output)
    }

    /// Compute predictive variance into a caller-provided output buffer.
    ///
    /// The output remains unchanged if a row or the output length is invalid.
    pub fn prediction_variance_batch_into<X: AsRef<[f64]>>(
        &self,
        x: &[X],
        output: &mut [f64],
    ) -> Result<(), BatchError> {
        if output.len() != x.len() {
            return Err(BatchError::OutputLengthMismatch {
                expected: x.len(),
                got: output.len(),
            });
        }
        self.validate_feature_rows(x)?;
        for (row, variance) in x.iter().zip(output) {
            *variance = self.prediction_variance_validated(row.as_ref());
        }
        Ok(())
    }

    /// Reset the covariance to the prior, keeping the learned parameters.
    ///
    /// The standard recovery from [`Error::NumericalBreakdown`], and a common
    /// deliberate move after a known regime change to let the model re-adapt
    /// quickly without discarding `θ`.
    ///
    /// The prior is `P = diag((δ⁻¹ + g_j)⁻¹)`, where `g_j` is the ridge
    /// penalty at zero data (`γ·s_j` for [`Ridge::Absolute`], `0` for
    /// [`Ridge::Relative`]), rescaled if its trace exceeds the cap; the
    /// [`EwRls::weight_mass`] restarts at `0`. Because `θ` is kept, this prior
    /// (including its ridge part, until forgotten) is centred on the current
    /// `θ` rather than on `0`; [`EwRls::reset`] also zeroes `θ`.
    /// Allocation-free.
    pub fn reset_covariance(&mut self) {
        self.weight_mass = 0.0;
        self.pending_decay = 1.0;
        for j in 0..self.dimensions {
            self.ridge_information[j] = self.ridge_target(j);
        }
        self.p_abs_bound = fill_prior(
            &mut self.p,
            self.initial_covariance,
            &self.ridge_information,
            self.max_trace,
        );
    }

    /// Reset the model to its initial state: `θ = 0`, the prior covariance
    /// (see [`EwRls::reset_covariance`]), zero sample mass and update count.
    ///
    /// The configuration is kept, including a forgetting factor changed by
    /// [`EwRls::set_lambda`] and a ridge changed by [`EwRls::set_ridge`].
    pub fn reset(&mut self) {
        self.theta.fill(0.0);
        self.reset_covariance();
        self.updates = 0;
    }

    /// Change the forgetting factor applied by subsequent updates
    /// (`MIN_LAMBDA <= lambda <= 1`), e.g. per-regime adaptation.
    ///
    /// For forgetting by elapsed time, prefer `λ = 1` with [`EwRls::decay`],
    /// which also handles long gaps.
    pub fn set_lambda(&mut self, lambda: f64) -> Result<(), Error> {
        self.lambda = validate_lambda(lambda)?;
        Ok(())
    }

    /// Change the persistent ridge at runtime.
    ///
    /// A higher target is reached by the refresh within `d` updates. A lower
    /// target is reached only as forgetting (or
    /// [`EwRls::reset_covariance`]) removes the existing penalty, since
    /// information cannot be removed stably; with `λ = 1` and no
    /// [`EwRls::decay`] it is never lowered.
    ///
    /// Rejected, like at build time, if a target `γ·s_j` overflows
    /// ([`Error::InvalidRegularization`]) or the prior that
    /// [`EwRls::reset_covariance`] would install under the new ridge has a
    /// non-finite trace ([`Error::InvalidInitialCovariance`]).
    pub fn set_ridge(&mut self, ridge: Ridge) -> Result<(), Error> {
        let ridge = validate_ridge(ridge, &self.ridge_weights)?;
        let trace = prior_trace(
            self.initial_covariance,
            self.max_trace,
            self.dimensions,
            |j| zero_mass_ridge(ridge, self.ridge_weights[j]),
        );
        if !trace.is_finite() {
            return Err(Error::InvalidInitialCovariance(self.initial_covariance));
        }
        self.ridge = ridge;
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

    /// The persistent ridge strength: `γ` for [`Ridge::Absolute`], `κ` for
    /// [`Ridge::Relative`].
    #[must_use]
    pub fn regularization(&self) -> f64 {
        self.ridge.strength()
    }

    /// The persistent ridge configuration.
    #[must_use]
    pub fn ridge(&self) -> Ridge {
        self.ridge
    }

    /// The per-coordinate ridge weights `s_j`.
    #[must_use]
    pub fn ridge_weights(&self) -> &[f64] {
        self.ridge_weights.as_slice()
    }

    /// The discounted sample mass `W_t = Σ Λ_{i,t} w_i`: the total weight of
    /// the observations the model currently remembers (`t` for unit weights
    /// and `λ = 1`, about `1/(1 − λ)` in steady state). Reset to `0` by
    /// [`EwRls::reset_covariance`] and by a [`DecayOutcome::Reset`];
    /// saturates at `f64::MAX`.
    #[must_use]
    pub fn weight_mass(&self) -> f64 {
        self.weight_mass
    }

    /// Effective sample window `1 / (1 − λ)` of the per-update forgetting;
    /// `f64::INFINITY` when `λ = 1`. [`EwRls::decay`] is not included.
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

    fn validate_observation(&self, x: &[f64], y: f64, weight: f64) -> Result<(), Error> {
        self.validate_features(x)?;
        if !y.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        if !(weight > 0.0 && weight.is_finite()) {
            return Err(Error::InvalidWeight(weight));
        }
        Ok(())
    }

    fn validate_feature_rows<X: AsRef<[f64]>>(&self, x: &[X]) -> Result<(), BatchError> {
        for (index, row) in x.iter().enumerate() {
            self.validate_features(row.as_ref())
                .map_err(|source| BatchError::Sample { index, source })?;
        }
        Ok(())
    }

    fn validate_batch<X: AsRef<[f64]>>(
        &self,
        x: &[X],
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<(), BatchError> {
        if y.len() != x.len() {
            return Err(BatchError::TargetLengthMismatch {
                expected: x.len(),
                got: y.len(),
            });
        }
        if let Some(weights) = sample_weight
            && weights.len() != x.len()
        {
            return Err(BatchError::WeightLengthMismatch {
                expected: x.len(),
                got: weights.len(),
            });
        }

        for (index, (row, &target)) in x.iter().zip(y).enumerate() {
            let weight = sample_weight.map_or(1.0, |weights| weights[index]);
            self.validate_observation(row.as_ref(), target, weight)
                .map_err(|source| BatchError::Sample { index, source })?;
        }
        Ok(())
    }

    fn apply_validated_batch<'a, X: AsRef<[f64]>>(
        &'a mut self,
        x: &[X],
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<&'a mut Self, BatchError> {
        for (index, (row, &target)) in x.iter().zip(y).enumerate() {
            let weight = sample_weight.map_or(1.0, |weights| weights[index]);
            self.update_weighted_validated(row.as_ref(), target, weight)
                .map_err(|source| BatchError::Sample { index, source })?;
        }
        Ok(self)
    }

    /// Averages the entry pairs that differ. The updates keep `P` exactly
    /// symmetric, so only restoring an asymmetric checkpoint needs this; it
    /// is a bitwise no-op otherwise (touching equal pairs could round
    /// subnormal entries).
    #[cfg(feature = "serde")]
    fn symmetrize(&mut self) {
        for i in 0..self.dimensions {
            for j in (i + 1)..self.dimensions {
                let (a, b) = (self.p[(i, j)], self.p[(j, i)]);
                if a != b {
                    // 0.5a + 0.5b, not (a+b)/2: the sum can overflow for
                    // finite same-sign entries near f64::MAX.
                    let avg = 0.5 * a + 0.5 * b;
                    self.p[(i, j)] = avg;
                    self.p[(j, i)] = avg;
                }
            }
        }
    }
}

/// A-priori quantities shared by the two halves of an update.
#[derive(Clone, Copy)]
struct Innovation {
    report: UpdateReport,
    inv_den: f64,
    gain_scale: f64,
    weight: f64,
}

fn validate_lambda(lambda: f64) -> Result<f64, Error> {
    if (MIN_LAMBDA..=1.0).contains(&lambda) {
        Ok(lambda)
    } else {
        Err(Error::InvalidLambda(lambda))
    }
}

/// Prior variance `1/(δ⁻¹ + g)` of one coordinate with ridge penalty `g`.
/// Computed in a form that survives subnormal `δ` (where `1/δ` overflows)
/// via the algebraically equal `δ/(1 + gδ)`.
fn prior_scale(delta: f64, g: f64) -> f64 {
    let inv = 1.0 / delta + g;
    if inv.is_finite() {
        1.0 / inv
    } else {
        delta / (1.0 + g * delta)
    }
}

/// Validates a ridge against the per-coordinate weights: the strength and
/// every weight must be non-negative and finite, and so must every target
/// `strength·s_j` at unit mass.
fn validate_ridge(ridge: Ridge, weights: &DVector<f64>) -> Result<Ridge, Error> {
    let strength = ridge.validate()?.strength();
    for &weight in weights.iter() {
        if !(weight >= 0.0 && weight.is_finite()) {
            return Err(Error::InvalidRegularization(weight));
        }
        let target = strength * weight;
        if !target.is_finite() {
            return Err(Error::InvalidRegularization(target));
        }
    }
    Ok(ridge)
}

/// Ridge penalty of a coordinate with weight `weight` at zero sample mass.
fn zero_mass_ridge(ridge: Ridge, weight: f64) -> f64 {
    match ridge {
        Ridge::Absolute(gamma) => gamma * weight,
        Ridge::Relative(_) => 0.0,
    }
}

/// The prior covariance diagonal `(δ⁻¹ + g_j)⁻¹` for ridge penalties `g(j)`,
/// rescaled to meet `max_trace`: passes each final entry to `write` in index
/// order and returns the largest. A uniform diagonal is capped entry-wise at
/// `max_trace / d`, the same rescaling without its rounding. Allocation-free.
fn prior_diagonal(
    delta: f64,
    max_trace: f64,
    d: usize,
    g: impl Fn(usize) -> f64,
    mut write: impl FnMut(usize, f64),
) -> f64 {
    let first = g(0);
    if (1..d).all(|j| g(j) == first) {
        let p0 = prior_scale(delta, first).min(max_trace / d as f64);
        (0..d).for_each(|j| write(j, p0));
        return p0;
    }
    let largest = (0..d).map(|j| prior_scale(delta, g(j))).fold(0.0, f64::max);
    // Compare against the cap without forming a trace that may overflow. The
    // capped largest entry is `budget`; scaling each entry by `v/largest ≤ 1`
    // keeps a far smaller cap from passing through a subnormal factor.
    let mut budget = f64::INFINITY;
    if largest > 0.0 {
        let relative_trace: f64 = (0..d).map(|j| prior_scale(delta, g(j)) / largest).sum();
        budget = max_trace / relative_trace;
    }
    let capped = budget < largest;
    for j in 0..d {
        let v = prior_scale(delta, g(j));
        write(j, if capped { v / largest * budget } else { v });
    }
    if capped { budget } else { largest }
}

/// Writes the prior covariance `diag((δ⁻¹ + g_j)⁻¹)` into `p` (see
/// [`prior_diagonal`]) and returns its largest entry. Allocation-free.
fn fill_prior(
    p: &mut DMatrix<f64>,
    delta: f64,
    ridge_information: &DVector<f64>,
    max_trace: f64,
) -> f64 {
    p.fill(0.0);
    prior_diagonal(
        delta,
        max_trace,
        ridge_information.len(),
        |j| ridge_information[j],
        |j, v| p[(j, j)] = v,
    )
}

/// Trace of the prior [`fill_prior`] would install, summed in the order of
/// `DMatrix::trace`, without building the matrix.
fn prior_trace(delta: f64, max_trace: f64, d: usize, g: impl Fn(usize) -> f64) -> f64 {
    let mut trace = 0.0;
    prior_diagonal(delta, max_trace, d, g, |_, v| trace += v);
    trace
}

#[derive(Clone, Copy)]
struct FusedCovarianceUpdate {
    alpha_abs: f64,
    beta: f64,
    candidate_bound: f64,
}

/// Return the fused GER coefficients only when every intermediate in
/// `beta·P + (-alpha_abs·p_x)·p_xᵀ` is comfortably finite.
fn fused_covariance_bound(
    p_abs_bound: f64,
    p_x_abs_max: f64,
    inv_den: f64,
    lambda: f64,
) -> Option<FusedCovarianceUpdate> {
    let beta = 1.0 / lambda;
    let alpha_abs = inv_den / lambda;
    let column_factor_bound = alpha_abs * p_x_abs_max;
    let rank_bound = column_factor_bound * p_x_abs_max;
    let scaled_p_bound = beta * p_abs_bound;
    let candidate_bound = scaled_p_bound + rank_bound;

    (beta <= FUSED_UPDATE_LIMIT
        && alpha_abs <= FUSED_UPDATE_LIMIT
        && column_factor_bound <= FUSED_UPDATE_LIMIT
        && rank_bound <= FUSED_UPDATE_LIMIT
        && scaled_p_bound <= FUSED_UPDATE_LIMIT
        && candidate_bound <= FUSED_UPDATE_LIMIT)
        .then_some(FusedCovarianceUpdate {
            alpha_abs,
            beta,
            candidate_bound: inflate_bound(candidate_bound),
        })
}

/// Round a computed magnitude bound outward by several ulps.
fn inflate_bound(bound: f64) -> f64 {
    if bound == 0.0 {
        0.0
    } else {
        bound * (1.0 + 8.0 * f64::EPSILON)
    }
}

/// Allow only the rounding error accumulated while summing a live matrix's
/// diagonal. This keeps serialized capped models round-trippable.
#[cfg(feature = "serde")]
fn trace_within_cap(trace: f64, max_trace: f64, dimensions: usize) -> bool {
    if trace <= max_trace {
        return true;
    }
    let tolerance = max_trace * (4.0 * f64::EPSILON * dimensions as f64);
    trace - max_trace <= tolerance
}

/// Untrusted mirror of [`EwRls`] used to validate deserialized checkpoints
/// before they become a live model. Version 1 (ewrls ≤ 0.2.0) stored
/// `gamma`; version 2 stores the ridge state, the sample mass, the pending
/// decay and the covariance bound, so that a restore is bit-exact.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct EwRlsCheckpoint {
    version: u32,
    theta: DVector<f64>,
    p: DMatrix<f64>,
    lambda: f64,
    dimensions: usize,
    initial_covariance: f64,
    #[serde(deserialize_with = "serde_inf::deserialize")]
    max_trace: f64,
    updates: u64,
    /// Version 1 only.
    #[serde(default)]
    gamma: Option<f64>,
    /// Version 2 only.
    #[serde(default)]
    ridge: Option<Ridge>,
    #[serde(default)]
    ridge_weights: Option<DVector<f64>>,
    #[serde(default)]
    ridge_information: Option<DVector<f64>>,
    #[serde(default)]
    weight_mass: Option<f64>,
    #[serde(default)]
    pending_decay: Option<f64>,
    #[serde(default)]
    p_abs_bound: Option<f64>,
}

/// Every field name either checkpoint version may carry, in version 2 order
/// with version 1's `gamma` last. Positional formats read the fields in the
/// writer's order ([`EwRlsCheckpointRef`], or 0.2's layout for version 1);
/// self-describing ones match them by name.
#[cfg(feature = "serde")]
const CHECKPOINT_FIELDS: &[&str] = &[
    "version",
    "theta",
    "p",
    "lambda",
    "dimensions",
    "initial_covariance",
    "max_trace",
    "ridge",
    "ridge_weights",
    "ridge_information",
    "weight_mass",
    "pending_decay",
    "p_abs_bound",
    "updates",
    "gamma",
];

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for EwRls {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_struct("EwRlsCheckpoint", CHECKPOINT_FIELDS, CheckpointVisitor)
    }
}

#[cfg(feature = "serde")]
struct CheckpointVisitor;

#[cfg(feature = "serde")]
impl<'de> serde::de::Visitor<'de> for CheckpointVisitor {
    type Value = EwRls;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("an ewrls checkpoint")
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, map: A) -> Result<EwRls, A::Error> {
        use serde::Deserialize;
        let checkpoint =
            EwRlsCheckpoint::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
        EwRls::try_from(checkpoint).map_err(serde::de::Error::custom)
    }

    /// Positional formats (bincode, postcard) cannot skip absent fields, so
    /// the layout is chosen by the leading version.
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<EwRls, A::Error> {
        fn next<'de, T: serde::Deserialize<'de>, A: serde::de::SeqAccess<'de>>(
            seq: &mut A,
            index: usize,
        ) -> Result<T, A::Error> {
            seq.next_element()?
                .ok_or_else(|| serde::de::Error::invalid_length(index, &"a complete checkpoint"))
        }
        let version: u32 = next(&mut seq, 0)?;
        if version != 1 && version != CHECKPOINT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported checkpoint version {version}; expected 1 or {CHECKPOINT_VERSION}"
            )));
        }
        let mut checkpoint = EwRlsCheckpoint {
            version,
            theta: next(&mut seq, 1)?,
            p: next(&mut seq, 2)?,
            lambda: next(&mut seq, 3)?,
            dimensions: next(&mut seq, 4)?,
            initial_covariance: next(&mut seq, 5)?,
            max_trace: next::<Option<f64>, _>(&mut seq, 6)?.unwrap_or(f64::INFINITY),
            updates: 0,
            gamma: None,
            ridge: None,
            ridge_weights: None,
            ridge_information: None,
            weight_mass: None,
            pending_decay: None,
            p_abs_bound: None,
        };
        if version == 1 {
            checkpoint.gamma = Some(next(&mut seq, 7)?);
            checkpoint.updates = next(&mut seq, 8)?;
        } else {
            checkpoint.ridge = Some(next(&mut seq, 7)?);
            checkpoint.ridge_weights = Some(next(&mut seq, 8)?);
            checkpoint.ridge_information = Some(next(&mut seq, 9)?);
            checkpoint.weight_mass = Some(next(&mut seq, 10)?);
            checkpoint.pending_decay = Some(next(&mut seq, 11)?);
            checkpoint.p_abs_bound = Some(next(&mut seq, 12)?);
            checkpoint.updates = next(&mut seq, 13)?;
        }
        EwRls::try_from(checkpoint).map_err(serde::de::Error::custom)
    }
}

/// Ridge, sample-mass and decay state of a checkpoint, per format version.
#[cfg(feature = "serde")]
struct RidgeState {
    ridge: Ridge,
    ridge_weights: DVector<f64>,
    ridge_information: DVector<f64>,
    weight_mass: f64,
    pending_decay: f64,
    p_abs_bound: Option<f64>,
}

#[cfg(feature = "serde")]
impl EwRlsCheckpoint {
    /// Called only after `dimensions` was checked against the stored `θ`, so
    /// it never allocates by an untrusted size.
    fn ridge_state(&mut self) -> Result<RidgeState, String> {
        let d = self.dimensions;
        if self.version == 1 {
            let gamma = self.gamma.ok_or("version 1 checkpoint lacks gamma")?;
            if self.ridge.is_some()
                || self.ridge_weights.is_some()
                || self.ridge_information.is_some()
                || self.weight_mass.is_some()
                || self.pending_decay.is_some()
                || self.p_abs_bound.is_some()
            {
                return Err("version 1 checkpoint has version 2 fields".into());
            }
            // The penalty each coordinate held under 0.2's constant-λ schedule:
            // Γ_0 = γ, every update forgets it by λ, and update u adds
            // γ(1 − λᵈ) to coordinate u mod d. After n updates, coordinate j
            // was refreshed k times, the last one s updates ago.
            let (lambda, n, du) = (self.lambda, self.updates, d as u64);
            let power = |e: u64| lambda.powf(e as f64);
            let ridge_information = DVector::from_fn(d, |j, _| {
                let j = j as u64;
                if n <= j {
                    return gamma * power(n);
                }
                let (s, k) = ((n - 1 - j) % du, (n - 1 - j) / du + 1);
                gamma * power(s) * (1.0 + power((k - 1) * du) * (power(j + 1) - power(du)))
            });
            let weight_mass = if lambda < 1.0 {
                (1.0 - power(n)) / (1.0 - lambda)
            } else {
                n as f64
            };
            return Ok(RidgeState {
                ridge: Ridge::Absolute(gamma),
                ridge_weights: DVector::from_element(d, 1.0),
                ridge_information,
                weight_mass,
                pending_decay: 1.0,
                p_abs_bound: None,
            });
        }
        if self.gamma.is_some() {
            return Err("version 2 checkpoint has the version 1 gamma field".into());
        }
        let missing = |field: &str| format!("version 2 checkpoint lacks {field}");
        Ok(RidgeState {
            ridge: self.ridge.ok_or_else(|| missing("ridge"))?,
            ridge_weights: self
                .ridge_weights
                .take()
                .ok_or_else(|| missing("ridge_weights"))?,
            ridge_information: self
                .ridge_information
                .take()
                .ok_or_else(|| missing("ridge_information"))?,
            weight_mass: self.weight_mass.ok_or_else(|| missing("weight_mass"))?,
            pending_decay: self.pending_decay.ok_or_else(|| missing("pending_decay"))?,
            p_abs_bound: Some(self.p_abs_bound.ok_or_else(|| missing("p_abs_bound"))?),
        })
    }
}

#[cfg(feature = "serde")]
impl TryFrom<EwRlsCheckpoint> for EwRls {
    type Error = String;

    fn try_from(mut c: EwRlsCheckpoint) -> Result<Self, String> {
        if c.version != 1 && c.version != CHECKPOINT_VERSION {
            return Err(format!(
                "unsupported checkpoint version {}; expected 1 or {CHECKPOINT_VERSION}",
                c.version
            ));
        }
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
        let state = c.ridge_state()?;
        if validate_lambda(c.lambda).is_err() {
            return Err(format!("lambda {} outside [{MIN_LAMBDA}, 1]", c.lambda));
        }
        if !(c.initial_covariance > 0.0 && c.initial_covariance.is_finite()) {
            return Err(format!(
                "invalid initial covariance {}",
                c.initial_covariance
            ));
        }
        if c.max_trace.is_nan() || c.max_trace <= 0.0 {
            return Err(format!("invalid max covariance trace {}", c.max_trace));
        }
        if state.ridge_weights.nrows() != d {
            return Err(format!("ridge weights must have {d} entries"));
        }
        let ridge = validate_ridge(state.ridge, &state.ridge_weights)
            .map_err(|e| format!("invalid ridge: {e}"))?;
        let valid = |v: &f64| *v >= 0.0 && v.is_finite();
        if state.ridge_information.nrows() != d || !state.ridge_information.iter().all(valid) {
            return Err(format!(
                "ridge information must be {d} non-negative finite values"
            ));
        }
        if !valid(&state.weight_mass) {
            return Err(format!("invalid weight mass {}", state.weight_mass));
        }
        if !(MIN_LAMBDA..=1.0).contains(&state.pending_decay) {
            return Err(format!("invalid pending decay {}", state.pending_decay));
        }
        // The prior that reset_covariance would install must be usable too.
        let reset_trace = prior_trace(c.initial_covariance, c.max_trace, d, |j| {
            zero_mass_ridge(ridge, state.ridge_weights[j])
        });
        if !reset_trace.is_finite() {
            return Err("initial covariance produces a non-finite reset trace".into());
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
        // the rank-1 product). Reject gross asymmetry; rounding-level
        // asymmetry (from 0.2, which let it drift between symmetrizations) is
        // averaged away below, since forgetting would amplify it.
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
        if !trace_within_cap(trace, c.max_trace, d) {
            return Err(format!(
                "covariance trace {trace} exceeds the stored max trace {}",
                c.max_trace
            ));
        }

        // Test definiteness on a symmetrized, scaled copy.
        let largest = c.p.amax();
        if largest > 0.0 {
            let half = c.p.map(|v| (v / largest) * 0.5);
            let eigenvalues = (&half + half.transpose()).symmetric_eigenvalues();
            if eigenvalues.iter().any(|&v| !v.is_finite() || v < -1e-12) {
                return Err("covariance is not positive semidefinite".into());
            }
        }
        let p_abs_bound = match state.p_abs_bound {
            Some(bound) if bound.is_finite() && bound >= largest => bound,
            Some(bound) => {
                return Err(format!(
                    "covariance bound {bound} is below the largest covariance entry {largest}"
                ));
            }
            None => largest,
        };

        let mut model = EwRls {
            theta: c.theta,
            p: c.p,
            // Workspace is scratch state; always rebuild it.
            p_x: DVector::zeros(d),
            candidate_diagonal: DVector::zeros(d),
            p_abs_bound,
            lambda: c.lambda,
            dimensions: d,
            initial_covariance: c.initial_covariance,
            max_trace: c.max_trace,
            ridge,
            ridge_weights: state.ridge_weights,
            ridge_information: state.ridge_information,
            weight_mass: state.weight_mass,
            pending_decay: state.pending_decay,
            updates: c.updates,
        };
        // Averages only differing pairs: a bitwise no-op for the exactly
        // symmetric states this version writes, so their restore stays exact.
        model.symmetrize();
        Ok(model)
    }
}

/// Compiles and runs the README examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

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
            EwRls::builder(3)
                .initial_covariance(-1.0)
                .build()
                .unwrap_err(),
            Error::InvalidInitialCovariance(-1.0)
        );
        assert_eq!(
            EwRls::builder(3)
                .max_covariance_trace(0.0)
                .build()
                .unwrap_err(),
            Error::InvalidMaxTrace(0.0)
        );
        assert!(EwRls::new(3, 1.0).is_ok());
    }

    #[test]
    fn input_validation() {
        let mut model = EwRls::new(2, 0.99).unwrap();
        assert_eq!(
            model.predict(&[1.0]).unwrap_err(),
            Error::DimensionMismatch {
                expected: 2,
                got: 1
            }
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
    fn batch_fit_and_partial_fit_match_streaming_updates() {
        let x = [[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0], [2.0, -0.5]];
        let y = [2.0, 1.5, -0.25, 4.0];
        let extra_x = [[0.25, -1.0], [1.5, 0.75]];
        let extra_y = [-1.0, 2.5];

        let mut batch = EwRls::new(2, 0.99).unwrap();
        batch.fit(&x, &y).unwrap();
        batch.partial_fit(&extra_x, &extra_y).unwrap();

        let mut streaming = EwRls::new(2, 0.99).unwrap();
        for (row, &target) in x.iter().zip(&y).chain(extra_x.iter().zip(&extra_y)) {
            streaming.update(row, target).unwrap();
        }

        assert_eq!(batch.params(), streaming.params());
        assert_eq!(batch.covariance(), streaming.covariance());
        assert_eq!(batch.updates(), 6);

        // `fit` starts a fresh training run.
        batch.fit(&extra_x, &extra_y).unwrap();
        let mut fresh = EwRls::new(2, 0.99).unwrap();
        fresh.partial_fit(&extra_x, &extra_y).unwrap();
        assert_eq!(batch.params(), fresh.params());
        assert_eq!(batch.covariance(), fresh.covariance());
    }

    #[test]
    fn weighted_batch_matches_weighted_streaming() {
        let x = vec![vec![1.0, 2.0], vec![-0.5, 1.0], vec![3.0, -1.0]];
        let y = [3.0, -1.0, 2.0];
        let weights = [0.5, 2.0, 4.0];

        let mut batch = EwRls::new(2, 1.0).unwrap();
        batch.fit_weighted(&x, &y, &weights).unwrap();

        let mut streaming = EwRls::new(2, 1.0).unwrap();
        for ((row, &target), &weight) in x.iter().zip(&y).zip(&weights) {
            streaming.update_weighted(row, target, weight).unwrap();
        }

        assert_eq!(batch.params(), streaming.params());
        assert_eq!(batch.covariance(), streaming.covariance());
    }

    #[test]
    fn batch_prediction_supports_owned_and_reused_outputs() {
        let mut model = EwRls::new(2, 1.0).unwrap();
        model.theta = DVector::from_vec(vec![2.0, -1.0]);
        let x = [[1.0, 3.0], [-2.0, 0.5], [0.0, -4.0]];

        assert_eq!(model.predict_batch(&x).unwrap(), vec![-1.0, -4.5, 4.0]);
        let mut output = [f64::NAN; 3];
        model.predict_batch_into(&x, &mut output).unwrap();
        assert_eq!(output, [-1.0, -4.5, 4.0]);

        let expected_variance: Vec<f64> = x
            .iter()
            .map(|row| model.prediction_variance(row).unwrap())
            .collect();
        assert_eq!(
            model.prediction_variance_batch(&x).unwrap(),
            expected_variance
        );
        let mut variance_output = [f64::NAN; 3];
        model
            .prediction_variance_batch_into(&x, &mut variance_output)
            .unwrap();
        assert_eq!(variance_output.as_slice(), expected_variance);
    }

    #[test]
    fn batch_errors_are_indexed_and_prevalidation_preserves_state() {
        let mut model = EwRls::new(2, 0.99).unwrap();
        model.update(&[1.0, 1.0], 2.0).unwrap();
        let theta_before = model.params().to_vec();
        let covariance_before = model.covariance().clone();
        let x = [[1.0, 0.0], [f64::NAN, 1.0]];

        assert_eq!(
            model.fit::<[f64; 2]>(&[], &[]).unwrap_err(),
            BatchError::EmptyBatch
        );
        assert_eq!(model.params(), theta_before);
        assert_eq!(model.covariance(), &covariance_before);

        assert_eq!(
            model.fit(&x, &[1.0, 2.0]).unwrap_err(),
            BatchError::Sample {
                index: 1,
                source: Error::NonFiniteInput,
            }
        );
        assert_eq!(model.params(), theta_before);
        assert_eq!(model.covariance(), &covariance_before);
        assert_eq!(
            model.partial_fit(&[[1.0, 2.0]], &[]).unwrap_err(),
            BatchError::TargetLengthMismatch {
                expected: 1,
                got: 0,
            }
        );
        assert_eq!(
            model
                .partial_fit_weighted(&[[1.0, 2.0]], &[1.0], &[])
                .unwrap_err(),
            BatchError::WeightLengthMismatch {
                expected: 1,
                got: 0,
            }
        );

        let mut output = [7.0];
        assert_eq!(
            model
                .predict_batch_into(&[[1.0, 2.0], [3.0, 4.0]], &mut output)
                .unwrap_err(),
            BatchError::OutputLengthMismatch {
                expected: 2,
                got: 1,
            }
        );
        assert_eq!(output, [7.0]);
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
        assert!(EwRls::builder(2).regularization(f64::NAN).build().is_err());
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
        // P[1,1] grows by 1/λ per step up to the prior floor at 2δ, which a
        // weak prior puts far above the cap.
        let mut model = EwRls::builder(2)
            .lambda(0.9)
            .initial_covariance(1e6)
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
    fn unexcited_direction_saturates_at_the_prior() {
        // No trace cap: under degenerate excitation P[1,1] grows by 1/λ per
        // step until it would pass 2δ, and is then pulled back to δ within
        // the same update. θ is untouched by the floor.
        let mut model = EwRls::builder(2)
            .lambda(0.9)
            .initial_covariance(100.0)
            .build()
            .unwrap();
        let mut largest: f64 = 0.0;
        for _ in 0..500 {
            model.update(&[1.0, 0.0], 1.0).unwrap();
            largest = largest.max(model.covariance()[(1, 1)]);
        }
        assert!((0.9 * 200.0..=200.0).contains(&largest), "{largest}");
        assert_eq!(model.params()[1], 0.0);
        assert!((model.params()[0] - 1.0).abs() < 1e-12);
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
        // λ at the bottom of its valid range with a huge prior: the 1/λ
        // forgetting overflows the covariance. Must be detected before any
        // state is modified.
        let mut model = EwRls::builder(1)
            .lambda(MIN_LAMBDA)
            .initial_covariance(1e305)
            .build()
            .unwrap();
        let theta_before = model.params().to_vec();
        let trace_before = model.covariance_trace();

        assert_eq!(
            model.update(&[1e-200], 1.0).unwrap_err(),
            Error::NumericalBreakdown
        );
        assert_eq!(model.params(), theta_before.as_slice());
        assert_eq!(model.covariance_trace(), trace_before);
        assert_eq!(model.updates(), 0);
        assert_eq!(model.weight_mass(), 0.0);
    }

    #[test]
    fn exact_fallback_commit_matches_validation_order() {
        // P/λ overflows, so the fused path is refused and the exact fallback
        // runs. The textbook expression cancels to a finite value;
        // distributing 1/λ across the subtraction would give +Inf - Inf = NaN.
        assert_eq!(1e303 / MIN_LAMBDA, f64::INFINITY);
        let mut model = EwRls::builder(1)
            .lambda(MIN_LAMBDA)
            .initial_covariance(1e303)
            .build()
            .unwrap();
        model.update(&[1e-150], 1.0).unwrap();
        let p = model.covariance()[(0, 0)];
        assert!(p.is_finite() && p >= 0.0, "committed variance {p}");
        assert!(model.covariance_trace().is_finite());
    }

    #[test]
    fn fused_covariance_guard_selects_only_safe_updates() {
        assert!(fused_covariance_bound(1000.0, 1000.0, 1e-3, 0.99).is_some());
        assert!(fused_covariance_bound(1000.0, 1000.0, 1e-3, 1e-308).is_none());
        assert!(fused_covariance_bound(f64::MAX, 1.0, 1.0, 1.0).is_none());
    }

    #[test]
    fn fused_covariance_matches_direct_formula() {
        let mut model = EwRls::builder(3)
            .lambda(0.99)
            .initial_covariance(100.0)
            .build()
            .unwrap();
        let x = [0.2, -0.4, 0.7];
        let before = model.covariance().clone();
        let xv = DVectorView::from_slice(&x, 3);
        let p_x = &before * xv;
        let inv_den = 1.0 / (0.99 + xv.dot(&p_x));
        assert!(fused_covariance_bound(before.amax(), p_x.amax(), inv_den, 0.99).is_some());

        let expected = DMatrix::from_fn(3, 3, |i, j| {
            (before[(i, j)] - p_x[i] * p_x[j] * inv_den) / 0.99
        });
        model.update(&x, 2.0).unwrap();

        let error = (model.covariance() - expected).amax();
        assert!(error < 1e-12, "fused covariance error {error}");
        assert!(model.p_abs_bound >= model.covariance().amax());
    }

    #[test]
    fn covariance_absolute_bound_remains_conservative() {
        let mut model = EwRls::builder(8)
            .lambda(0.97)
            .regularization(0.2)
            .max_covariance_trace(1e4)
            .build()
            .unwrap();
        let mut rng = rand::rng();

        for _ in 0..2000 {
            let x: Vec<f64> = (0..8).map(|_| rng.random_range(-2.0..2.0)).collect();
            model.update(&x, rng.random_range(-10.0..10.0)).unwrap();
            assert!(
                model.covariance().amax() <= model.p_abs_bound,
                "actual {} exceeds cached bound {}",
                model.covariance().amax(),
                model.p_abs_bound
            );
        }
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

        // The format is explicit and rejects unknown schema versions.
        assert!(json.contains("\"version\":2"));
        let unsupported = json.replace("\"version\":2", "\"version\":3");
        let err = serde_json::from_str::<EwRls>(&unsupported).unwrap_err();
        assert!(err.to_string().contains("checkpoint version"));
        let unversioned = json.replacen("\"version\":2,", "", 1);
        assert!(serde_json::from_str::<EwRls>(&unversioned).is_err());

        // Dimensions inconsistent with the stored matrices must be rejected,
        // as must out-of-range configuration.
        let tampered = json.replace("\"dimensions\":2", "\"dimensions\":3");
        assert!(serde_json::from_str::<EwRls>(&tampered).is_err());
        let tampered = json.replace("\"lambda\":0.99", "\"lambda\":1.5");
        assert!(serde_json::from_str::<EwRls>(&tampered).is_err());
        let tampered = json.replace("\"lambda\":0.99", "\"lambda\":1e-9");
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
        let model = EwRls::builder(1)
            .initial_covariance(1e-320)
            .build()
            .unwrap();
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
        assert!(
            err.to_string().contains("symmetric"),
            "unexpected error: {err}"
        );
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
            EwRls::builder(2)
                .initial_covariance(1e308)
                .build()
                .unwrap_err(),
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

        // Multiplication rounds to MAX, but summing the eleven stored
        // diagonal entries in matrix-trace order overflows.
        let delta = f64::MAX / 11.0;
        assert_eq!(
            EwRls::builder(11)
                .initial_covariance(delta)
                .build()
                .unwrap_err(),
            Error::InvalidInitialCovariance(delta)
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
        let err = serde_json::from_str::<EwRls>(&tamper("1000.0,9e307,1e308,1000.0")).unwrap_err();
        assert!(
            err.to_string().contains("symmetric"),
            "unexpected error: {err}"
        );

        // Same-sign off-diagonals near f64::MAX must not overflow while
        // symmetrizing, and the resulting indefinite matrix is rejected.
        let err = serde_json::from_str::<EwRls>(&tamper("1000.0,1e308,1e308,1000.0")).unwrap_err();
        assert!(
            err.to_string().contains("positive semidefinite"),
            "unexpected error: {err}"
        );

        // Diagonals whose sum overflows: the trace invariant must reject.
        let err = serde_json::from_str::<EwRls>(&tamper("1e308,0.0,0.0,1e308")).unwrap_err();
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
        assert!(
            err.to_string().contains("max trace"),
            "unexpected error: {err}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_capped_model_roundtrips_with_trace_rounding() {
        let model = EwRls::builder(9).max_covariance_trace(1.0).build().unwrap();
        assert!(model.covariance_trace() >= 1.0);

        let json = serde_json::to_string(&model).unwrap();
        let restored: EwRls = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.covariance(), model.covariance());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_validates_reset_covariance_trace() {
        let model = EwRls::new(2, 0.99).unwrap();
        let json = serde_json::to_string(&model).unwrap();
        let tampered = json.replace(
            "\"initial_covariance\":1000.0",
            "\"initial_covariance\":1e308",
        );
        assert_ne!(tampered, json, "initial covariance field not found");
        let err = serde_json::from_str::<EwRls>(&tampered).unwrap_err();
        assert!(
            err.to_string().contains("reset trace"),
            "unexpected error: {err}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_rejects_indefinite_covariance() {
        let model = EwRls::new(2, 0.99).unwrap();
        let json = serde_json::to_string(&model).unwrap();
        let tampered = json.replace("1000.0,0.0,0.0,1000.0", "1.0,2.0,2.0,1.0");
        assert_ne!(tampered, json, "covariance data pattern not found");
        let err = serde_json::from_str::<EwRls>(&tampered).unwrap_err();
        assert!(
            err.to_string().contains("positive semidefinite"),
            "unexpected error: {err}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_omits_non_finite_scratch_workspace() {
        let mut model = EwRls::new(1, 1.0).unwrap();
        assert_eq!(
            model.update(&[1e308], 1.0).unwrap_err(),
            Error::NumericalBreakdown
        );

        let json = serde_json::to_string(&model).unwrap();
        assert!(!json.contains("p_x"));
        let restored: EwRls = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.params(), model.params());
        assert_eq!(restored.covariance(), model.covariance());
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

    /// Deterministic uniform(-0.5, 0.5) stream (splitmix64).
    fn uniform(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// Row with `y = Σ j·x_j + 0.1·noise`.
    fn linear_row(state: &mut u64, d: usize) -> (Vec<f64>, f64) {
        let x: Vec<f64> = (0..d).map(|_| uniform(state)).collect();
        let y = x.iter().enumerate().map(|(j, v)| v * j as f64).sum::<f64>() + 0.1 * uniform(state);
        (x, y)
    }

    fn max_relative_diff(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        (a - b).amax() / b.amax().max(1e-300)
    }

    #[test]
    fn covariance_is_exactly_symmetric_under_strong_forgetting() {
        // Regression: the rank-1 update rounded the two triangles differently
        // and every 1/λ amplified the asymmetry until the next symmetrization
        // 256 updates later. At λ ≤ 0.85 that broke down within ~200 updates
        // of well-excited data; at λ = 0.9 the asymmetry reached 3%.
        for lambda in [0.9, 0.8, 0.7] {
            let (d, mut s) = (5, 21u64);
            let mut model = EwRls::builder(d)
                .lambda(lambda)
                .regularization(0.01)
                .build()
                .unwrap();
            for _ in 0..2000 {
                let (x, y) = linear_row(&mut s, d);
                model.update(&x, y).unwrap();
                let p = model.covariance();
                assert_eq!(p, &p.transpose(), "λ = {lambda}");
            }
        }
    }

    #[test]
    fn lambda_floor_is_enforced() {
        assert_eq!(EwRls::new(2, 1e-7).unwrap_err(), Error::InvalidLambda(1e-7));
        assert!(EwRls::new(2, MIN_LAMBDA).is_ok());
        let mut model = EwRls::new(2, 0.99).unwrap();
        assert_eq!(
            model.set_lambda(1e-18).unwrap_err(),
            Error::InvalidLambda(1e-18)
        );
        assert_eq!(model.lambda(), 0.99);
    }

    #[test]
    fn decay_validation_and_identity() {
        let mut model = EwRls::new(2, 1.0).unwrap();
        model.update(&[1.0, 2.0], 3.0).unwrap();
        for bad in [1.5, -0.1, f64::NAN, f64::INFINITY] {
            assert!(matches!(model.decay(bad), Err(Error::InvalidDecay(_))));
        }
        let before = model.clone();
        assert_eq!(model.decay(1.0).unwrap(), DecayOutcome::Applied);
        assert_eq!(model.covariance(), before.covariance());
        assert_eq!(model.weight_mass(), before.weight_mass());
        assert_eq!(model.decay(0.0).unwrap(), DecayOutcome::Reset);
        assert_eq!(model.params(), before.params());
        assert_eq!(model.weight_mass(), 0.0);
        assert_eq!(model.updates(), before.updates());
    }

    #[test]
    fn long_gap_decay_resets_instead_of_corrupting() {
        // Regression: one step at λ = 1e-18 used to be accepted, leave P
        // indefinite and then reject almost every later update.
        let (d, mut s) = (6, 1u64);
        let mut model = EwRls::builder(d).lambda(0.999).build().unwrap();
        for _ in 0..20_000 {
            let (x, y) = linear_row(&mut s, d);
            model.update(&x, y).unwrap();
        }
        let theta_before = model.params().to_vec();
        for gap in [1e-18, 2f64.powi(-144), 0.0] {
            assert_eq!(model.decay(gap).unwrap(), DecayOutcome::Reset);
            assert_eq!(model.params(), theta_before.as_slice());
        }
        for _ in 0..1000 {
            let (x, y) = linear_row(&mut s, d);
            model.update(&x, y).unwrap();
        }
        for (j, &v) in model.params().iter().enumerate() {
            assert!((v - j as f64).abs() < 0.02, "theta[{j}] = {v}");
        }
        let eigenvalues = model.covariance().clone().symmetric_eigenvalues();
        assert!(eigenvalues.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn decay_then_unit_update_matches_lambda_update() {
        let (d, mut s) = (4, 7u64);
        let mut per_update = EwRls::builder(d)
            .lambda(0.97)
            .regularization(0.5)
            .build()
            .unwrap();
        let mut decayed = EwRls::builder(d)
            .lambda(1.0)
            .regularization(0.5)
            .build()
            .unwrap();
        for _ in 0..2000 {
            let (x, y) = linear_row(&mut s, d);
            per_update.update(&x, y).unwrap();
            decayed.decay(0.97).unwrap();
            decayed.update(&x, y).unwrap();
        }
        assert!(max_relative_diff(decayed.theta(), per_update.theta()) < 1e-10);
        let p_diff = (decayed.covariance() - per_update.covariance()).amax()
            / per_update.covariance().amax();
        assert!(p_diff < 1e-10, "covariance differs by {p_diff}");
        assert!((decayed.weight_mass() - per_update.weight_mass()).abs() < 1e-9);
        assert!((per_update.weight_mass() - 1.0 / 0.03).abs() < 1e-6);
    }

    #[test]
    fn time_varying_forgetting_matches_exact_objective() {
        // Information-form oracle: R = Λ₀δ⁻¹I + Σ Λᵢwᵢxᵢxᵢᵀ, b = Σ Λᵢwᵢyᵢxᵢ.
        let (d, mut s) = (5, 11u64);
        let delta = 1e3;
        let mut model = EwRls::builder(d)
            .lambda(1.0)
            .initial_covariance(delta)
            .build()
            .unwrap();
        let mut info = DMatrix::identity(d, d) / delta;
        let mut rhs = DVector::zeros(d);
        let mut mass = 0.0;
        for t in 0..5000 {
            let (x, y) = linear_row(&mut s, d);
            let w = 1.0 + uniform(&mut s); // in (0.5, 1.5)
            if t % 10 == 0 {
                // Bursty clock: a decay step in (0.1, 1] between some updates.
                let f = 10f64.powf(-(uniform(&mut s) + 0.5));
                model.decay(f).unwrap();
                info *= f;
                rhs *= f;
                mass *= f;
            }
            let lambda = 0.99 + 0.01 * (uniform(&mut s) + 0.5); // in [0.99, 1)
            model.set_lambda(lambda).unwrap();
            model.update_weighted(&x, y, w).unwrap();
            let xv = DVector::from_vec(x);
            info = info * lambda + &xv * xv.transpose() * w;
            rhs = rhs * lambda + &xv * (w * y);
            mass = mass * lambda + w;
        }
        let exact = info.cholesky().unwrap().solve(&rhs);
        let error = max_relative_diff(model.theta(), &exact);
        assert!(
            error < 1e-8,
            "theta deviates from the exact minimizer by {error}"
        );
        assert!((model.weight_mass() - mass).abs() < 1e-9 * mass);
    }

    #[test]
    fn ridge_band_holds_under_time_varying_forgetting() {
        // Regression: the refresh used only the current λ, so with one decay
        // every K rows most coordinates starved (Γ/γ down to 1e-8).
        let d = 10;
        for k in [1, 3, 5, 10] {
            let mut model = EwRls::builder(d)
                .lambda(1.0)
                .initial_covariance(1e12)
                .regularization(1.0)
                .build()
                .unwrap();
            for t in 0..3000 {
                if t % k == 0 {
                    model.decay(0.97).unwrap();
                }
                // x = 0 rows only forget and refresh, so P stays diagonal
                // with P[jj] = 1/(Λδ⁻¹ + Γ[jj]).
                model.update(&[0.0; 10], 0.0).unwrap();
            }
            for j in 0..d {
                let gamma_jj = 1.0 / model.covariance()[(j, j)];
                assert!(
                    (0.97f64.powi(9) - 1e-9..=1.0 + 1e-9).contains(&gamma_jj),
                    "K = {k}: Γ[{j}{j}]/γ = {gamma_jj}"
                );
                let tracked = model.ridge_information[j];
                assert!(
                    (tracked - gamma_jj).abs() < 1e-9,
                    "tracked {tracked} vs {gamma_jj}"
                );
            }
        }
    }

    #[test]
    fn relative_ridge_follows_mass_and_spares_the_intercept() {
        let run = |weights: [f64; 3]| {
            let mut s = 5u64;
            let mut model = EwRls::builder(3)
                .lambda(1.0)
                .ridge(Ridge::Relative(0.5))
                .ridge_weights(&weights)
                .build()
                .unwrap();
            for _ in 0..40_000 {
                model.decay(0.9995).unwrap();
                let x = [1.0, 2.0 * uniform(&mut s), 2.0 * uniform(&mut s)];
                let y = 5.0 + 0.3 * x[1] - 0.2 * x[2] + 0.05 * uniform(&mut s);
                model.update(&x, y).unwrap();
            }
            // Right after its refresh a coordinate holds exactly κ·W·s_j.
            let j = ((model.updates() - 1) % 3) as usize;
            let target = 0.5 * model.weight_mass() * weights[j];
            assert!((model.ridge_information[j] - target).abs() <= 1e-12 * target.max(1.0));
            model
        };
        // E[x²] = 1/3 per slope, penalty κ = 0.5 per unit mass: shrink 0.4.
        let free = run([0.0, 1.0, 1.0]);
        let theta = free.params();
        assert!((theta[0] - 5.0).abs() < 0.02, "intercept {}", theta[0]);
        assert!((theta[1] - 0.12).abs() < 0.02, "slope {}", theta[1]);
        assert!((theta[2] + 0.08).abs() < 0.02, "slope {}", theta[2]);
        assert!((free.weight_mass() - 1.0 / 0.0005).abs() < 1.0);
        // Penalizing the intercept too shrinks it by 1/(1 + κ).
        let penalized = run([1.0, 1.0, 1.0]);
        assert!((penalized.params()[0] - 5.0 / 1.5).abs() < 0.05);
    }

    #[test]
    fn ridge_configuration_validation_and_prior() {
        assert_eq!(
            EwRls::builder(3).ridge_weights(&[1.0]).build().unwrap_err(),
            Error::DimensionMismatch {
                expected: 3,
                got: 1
            }
        );
        assert_eq!(
            EwRls::builder(2)
                .ridge_weights(&[1.0, -1.0])
                .build()
                .unwrap_err(),
            Error::InvalidRegularization(-1.0)
        );
        assert_eq!(
            EwRls::builder(2)
                .ridge(Ridge::Relative(f64::INFINITY))
                .build()
                .unwrap_err(),
            Error::InvalidRegularization(f64::INFINITY)
        );

        // Per-coordinate prior diag((δ⁻¹ + γ·s_j)⁻¹), then the trace cap.
        let model = EwRls::builder(3)
            .initial_covariance(1e3)
            .regularization(3.0)
            .ridge_weights(&[0.0, 1.0, 2.0])
            .build()
            .unwrap();
        let expected = [1e3, 1.0 / (1e-3 + 3.0), 1.0 / (1e-3 + 6.0)];
        for (j, want) in expected.iter().enumerate() {
            assert!((model.covariance()[(j, j)] - want).abs() < 1e-12 * want);
        }
        let capped = EwRls::builder(3)
            .regularization(3.0)
            .ridge_weights(&[0.0, 1.0, 2.0])
            .max_covariance_trace(10.0)
            .build()
            .unwrap();
        assert!(capped.covariance_trace() <= 10.0 * (1.0 + 1e-12));
        assert!(capped.p_abs_bound >= capped.covariance().amax());

        let mut model = EwRls::new(2, 1.0).unwrap();
        assert_eq!(
            model.set_ridge(Ridge::Absolute(-1.0)).unwrap_err(),
            Error::InvalidRegularization(-1.0)
        );
        model.set_ridge(Ridge::Absolute(2.0)).unwrap();
        model.update(&[1.0, 0.0], 1.0).unwrap();
        model.update(&[0.0, 1.0], 1.0).unwrap();
        assert_eq!(model.ridge_information.as_slice(), &[2.0, 2.0]);
        assert_eq!(model.regularization(), 2.0);
    }

    #[test]
    fn gated_update_can_veto() {
        let (d, mut s) = (3, 9u64);
        let mut model = EwRls::builder(d)
            .lambda(0.98)
            .regularization(0.1)
            .build()
            .unwrap();
        for _ in 0..50 {
            let (x, y) = linear_row(&mut s, d);
            model.update(&x, y).unwrap();
        }
        let (x, y) = linear_row(&mut s, d);
        let before = model.clone();
        let mut seen = None;
        let vetoed = model
            .update_weighted_if(&x, y + 1e3, 2.0, |report| {
                seen = Some(*report);
                false
            })
            .unwrap();
        assert!(vetoed.is_none());
        assert_eq!(model.params(), before.params());
        assert_eq!(model.covariance(), before.covariance());
        assert_eq!(model.updates(), before.updates());
        assert_eq!(model.weight_mass(), before.weight_mass());
        let seen = seen.unwrap();
        let expected_scale = 1.0 / 2.0 + seen.predictive_variance / 0.98;
        assert!((seen.innovation_scale - expected_scale).abs() < 1e-12 * expected_scale);

        let mut reference = model.clone();
        let accepted = model.update_if(&x, y, |_| true).unwrap().unwrap();
        let direct = reference.update(&x, y).unwrap();
        assert_eq!(accepted, direct);
        assert_eq!(model.params(), reference.params());
        assert_eq!(model.covariance(), reference.covariance());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn checkpoint_restore_is_bit_exact() {
        // Regression: restoring used to re-symmetrize P, so a restore at any
        // update count not divisible by 256 diverged from the live model.
        let (d, mut s) = (12, 3u64);
        let rows: Vec<_> = (0..6000).map(|_| linear_row(&mut s, d)).collect();
        let builders = [
            EwRls::builder(d).lambda(0.995).regularization(0.5),
            EwRls::builder(d)
                .lambda(1.0)
                .ridge(Ridge::Relative(0.01))
                .ridge_weights(&[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]),
        ];
        for builder in builders {
            for cut in [1000, 1023, 1024] {
                let mut live = builder.clone().build().unwrap();
                for (x, y) in &rows[..cut] {
                    live.decay(0.999).unwrap();
                    live.update(x, *y).unwrap();
                }
                let json = serde_json::to_string(&live).unwrap();
                let mut restored: EwRls = serde_json::from_str(&json).unwrap();
                assert_eq!(restored.covariance(), live.covariance());
                for (x, y) in &rows[cut..] {
                    for model in [&mut live, &mut restored] {
                        model.decay(0.999).unwrap();
                        model.update(x, *y).unwrap();
                    }
                }
                assert_eq!(restored.params(), live.params(), "cut {cut}");
                assert_eq!(restored.covariance(), live.covariance(), "cut {cut}");
                assert_eq!(restored.weight_mass(), live.weight_mass());
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn version_1_checkpoint_still_loads() {
        let mut model = EwRls::builder(2)
            .lambda(0.99)
            .regularization(0.3)
            .build()
            .unwrap();
        for i in 0..7 {
            model.update(&[1.0, f64::from(i)], 2.0).unwrap();
        }
        // Rewrite the current checkpoint into the 0.2 layout.
        let mut value: serde_json::Value = serde_json::to_value(&model).unwrap();
        let fields = value.as_object_mut().unwrap();
        for key in [
            "ridge",
            "ridge_weights",
            "ridge_information",
            "weight_mass",
            "pending_decay",
            "p_abs_bound",
        ] {
            fields.remove(key).unwrap();
        }
        fields.insert("version".into(), 1.into());
        fields.insert("gamma".into(), 0.3.into());
        let v1 = value.to_string();

        let restored: EwRls = serde_json::from_str(&v1).unwrap();
        assert_eq!(restored.params(), model.params());
        assert_eq!(restored.covariance(), model.covariance());
        assert_eq!(restored.ridge(), Ridge::Absolute(0.3));
        let mass = (1.0 - 0.99f64.powi(7)) / 0.01;
        assert!((restored.weight_mass() - mass).abs() < 1e-12);

        // A version 1 checkpoint must not carry version 2 fields.
        value
            .as_object_mut()
            .unwrap()
            .insert("weight_mass".into(), 1.0.into());
        let mixed = value.to_string();
        let err = serde_json::from_str::<EwRls>(&mixed).unwrap_err();
        assert!(err.to_string().contains("version 2 fields"), "{err}");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn checkpoint_rejects_an_understated_covariance_bound() {
        let model = EwRls::new(2, 0.99).unwrap();
        let json = serde_json::to_string(&model).unwrap();
        let tampered = json.replace("\"p_abs_bound\":1000.0", "\"p_abs_bound\":1.0");
        assert_ne!(tampered, json, "bound field not found");
        let err = serde_json::from_str::<EwRls>(&tampered).unwrap_err();
        assert!(err.to_string().contains("covariance bound"), "{err}");
    }

    /// Rewrites a live model's checkpoint into the 0.2 (version 1) layout.
    #[cfg(feature = "serde")]
    fn as_version_1(model: &EwRls, gamma: f64) -> serde_json::Value {
        let mut value = serde_json::to_value(model).unwrap();
        let fields = value.as_object_mut().unwrap();
        for key in [
            "ridge",
            "ridge_weights",
            "ridge_information",
            "weight_mass",
            "pending_decay",
            "p_abs_bound",
        ] {
            fields.remove(key).unwrap();
        }
        fields.insert("version".into(), 1.into());
        fields.insert("gamma".into(), gamma.into());
        value
    }

    #[test]
    fn chained_decays_cannot_bypass_the_floor() {
        // Regression: the floor was checked per call, so a gap delivered as
        // several decays (e.g. a per-second timer) still wedged the model.
        let (d, mut s) = (6, 1u64);
        let mut model = EwRls::builder(d).lambda(0.999).build().unwrap();
        for _ in 0..20_000 {
            let (x, y) = linear_row(&mut s, d);
            model.update(&x, y).unwrap();
        }
        let outcomes: Vec<_> = (0..4)
            .map(|_| model.decay(2f64.powi(-15)).unwrap())
            .collect();
        use DecayOutcome::{Applied, Reset};
        assert_eq!(outcomes, [Applied, Reset, Applied, Reset]);
        // A 10 h gap at τ = 10 min delivered as one decay per second.
        model.set_lambda(1.0).unwrap();
        let resets = (0..36_000)
            .filter(|_| model.decay(2f64.powf(-1.0 / 600.0)).unwrap() == Reset)
            .count();
        assert!(resets > 0);
        model.set_lambda(0.999).unwrap();
        for _ in 0..1000 {
            let (x, y) = linear_row(&mut s, d);
            model.update(&x, y).unwrap();
        }
        for (j, &v) in model.params().iter().enumerate() {
            assert!((v - j as f64).abs() < 0.02, "theta[{j}] = {v}");
        }
    }

    #[test]
    fn decay_before_every_update_has_no_spurious_resets() {
        // Regression: the cached bound grew by 1/f per decay until it
        // overflowed and forced a reset although P itself was small.
        let (d, mut s) = (4, 13u64);
        let mut model = EwRls::builder(d)
            .lambda(1.0)
            .regularization(0.1)
            .build()
            .unwrap();
        for _ in 0..2000 {
            assert_eq!(model.decay(0.05).unwrap(), DecayOutcome::Applied);
            let (x, y) = linear_row(&mut s, d);
            model.update(&x, y).unwrap();
            assert!(model.p_abs_bound >= model.covariance().amax());
        }
    }

    #[test]
    fn ridge_products_and_prior_are_validated() {
        // γ·s_j must be finite (it used to store an infinite target).
        assert_eq!(
            EwRls::builder(2)
                .regularization(1e300)
                .ridge_weights(&[1e10, 1.0])
                .build()
                .unwrap_err(),
            Error::InvalidRegularization(f64::INFINITY)
        );
        // set_ridge applies build's prior check: a relative ridge has no
        // penalty at zero data, so δ = 1e308 would give an infinite trace.
        let mut model = EwRls::builder(2)
            .initial_covariance(1e308)
            .regularization(1.0)
            .build()
            .unwrap();
        assert_eq!(
            model.set_ridge(Ridge::Relative(0.1)).unwrap_err(),
            Error::InvalidInitialCovariance(1e308)
        );
        assert_eq!(model.ridge(), Ridge::Absolute(1.0));
        model.reset_covariance();
        assert!(model.covariance_trace().is_finite());
    }

    #[test]
    fn weight_mass_saturates_instead_of_rejecting() {
        let mut model = EwRls::new(1, 1.0).unwrap();
        model.update_weighted(&[1.0], 1.0, 1e308).unwrap();
        model.update_weighted(&[1.0], 1.0, 1e308).unwrap();
        assert_eq!(model.weight_mass(), f64::MAX);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn version_1_ridge_reconstruction_matches_the_old_schedule() {
        // 0.2's recursion: Γ_0 = γ; each update forgets by λ, then update u
        // adds γ(1 − λᵈ) to coordinate u mod d (including its first-cycle
        // overshoot above γ).
        let (d, lambda, gamma) = (3usize, 0.97_f64, 0.4);
        for n in 0..12u64 {
            let mut old = vec![gamma; d];
            for u in 0..n {
                old.iter_mut().for_each(|g| *g *= lambda);
                old[(u % d as u64) as usize] += gamma * (1.0 - lambda.powi(d as i32));
            }
            let mut model = EwRls::builder(d)
                .lambda(lambda)
                .regularization(gamma)
                .build()
                .unwrap();
            for _ in 0..n {
                model.update(&[0.0; 3], 0.0).unwrap();
            }
            let restored: EwRls = serde_json::from_value(as_version_1(&model, gamma)).unwrap();
            for (j, &want) in old.iter().enumerate() {
                let got = restored.ridge_information[j];
                assert!(
                    (got - want).abs() <= 1e-14 * want,
                    "n {n}, j {j}: {got} vs {want}"
                );
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn corrupt_version_1_dimensions_are_rejected_without_allocating() {
        // Regression: the version 1 reconstruction allocated by `dimensions`
        // before the shape checks, so a huge value panicked or aborted.
        let model = EwRls::builder(2)
            .lambda(0.99)
            .regularization(0.3)
            .build()
            .unwrap();
        let mut value = as_version_1(&model, 0.3);
        value["dimensions"] = serde_json::json!(u64::MAX);
        let err = serde_json::from_value::<EwRls>(value).unwrap_err();
        assert!(err.to_string().contains("theta has 2 rows"), "{err}");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn asymmetric_restored_state_is_symmetrized() {
        // A 0.2 checkpoint may carry rounding-level asymmetry, which
        // forgetting would amplify by 1/λ per update; restore averages it.
        let (d, mut s) = (4, 17u64);
        let mut model = EwRls::builder(d).lambda(0.7).build().unwrap();
        for _ in 0..50 {
            let (x, y) = linear_row(&mut s, d);
            model.update(&x, y).unwrap();
        }
        let mut value = as_version_1(&model, 0.0);
        // nalgebra serializes the matrix data column-major, then the shape.
        let entry = &mut value["p"][0][1]; // P[1, 0]
        *entry = serde_json::json!(entry.as_f64().unwrap() * (1.0 + 1e-10));
        let mut restored: EwRls = serde_json::from_value(value).unwrap();
        assert_eq!(restored.covariance(), &restored.covariance().transpose());
        for _ in 0..254 {
            let (x, y) = linear_row(&mut s, d);
            restored.update(&x, y).unwrap();
        }
    }

    /// Warm model: `n` dense rows at `λ = 1`.
    fn warm_model(builder: Builder, d: usize, n: usize, s: &mut u64) -> EwRls {
        let mut model = builder.lambda(1.0).build().unwrap();
        for _ in 0..n {
            let (x, y) = linear_row(s, d);
            model.update(&x, y).unwrap();
        }
        model
    }

    /// Every one of `n` dense rows must be accepted, and `θ` must track the
    /// truth `θ_j = j` again.
    fn assert_recovers(model: &mut EwRls, d: usize, n: usize, s: &mut u64) {
        for t in 0..n {
            let (x, y) = linear_row(s, d);
            model
                .update(&x, y)
                .unwrap_or_else(|e| panic!("row {t}: {e}"));
        }
        for (j, &v) in model.params().iter().enumerate() {
            assert!((v - j as f64).abs() < 0.05, "θ[{j}] = {v}");
        }
    }

    #[test]
    fn zero_rows_between_decays_cannot_bypass_the_floor() {
        // Regression: every commit reset the pending decay, so decays just
        // above MIN_LAMBDA separated by x = 0 rows forgot 1e-18 in total,
        // after which most updates were rejected.
        let (d, mut s) = (6, 1u64);
        let mut model = warm_model(EwRls::builder(d), d, 20_000, &mut s);
        for _ in 0..2 {
            assert_eq!(model.decay(1.2e-6).unwrap(), DecayOutcome::Applied);
            model.update(&[0.0; 6], 0.0).unwrap();
        }
        assert_eq!(model.decay(1.2e-6).unwrap(), DecayOutcome::Applied);
        assert!((0..d).all(|j| model.covariance()[(j, j)] <= 2e3));
        assert_recovers(&mut model, d, 2000, &mut s);
    }

    #[test]
    fn sparse_rows_between_gaps_keep_the_model_healthy() {
        // Regression: one row after each 12-half-life gap re-anchored only
        // the observed direction; the others inflated until P was indefinite.
        let d = 12;
        for ridge in [
            Ridge::Absolute(0.0),
            Ridge::Absolute(0.01),
            Ridge::Relative(0.01),
        ] {
            let mut s = 1u64;
            let mut model = warm_model(EwRls::builder(d).ridge(ridge), d, 20_000, &mut s);
            for _ in 0..6 {
                model.decay(2f64.powi(-12)).unwrap();
                let (x, y) = linear_row(&mut s, d);
                model.update(&x, y).unwrap();
            }
            let p = model.covariance();
            let eigenvalues = ((p + p.transpose()) * (0.5 / p.amax())).symmetric_eigenvalues();
            assert!(eigenvalues.min() > 0.0, "{ridge:?}: {}", eigenvalues.min());
            for _ in 0..3000 {
                let (x, y) = linear_row(&mut s, d);
                model.update(&x, y).unwrap();
            }
        }
    }

    #[test]
    fn decay_before_a_small_lambda_is_bounded() {
        // Regression: the decay floor ignored the λ of the next update, so
        // one interval could forget MIN_LAMBDA².
        let (d, mut s) = (6, 1u64);
        let mut model = warm_model(EwRls::builder(d), d, 20_000, &mut s);
        for _ in 0..2 {
            assert_eq!(model.decay(1.01e-6).unwrap(), DecayOutcome::Applied);
            model.set_lambda(1e-6).unwrap();
            let (x, y) = linear_row(&mut s, d);
            model.update(&x, y).unwrap();
        }
        model.set_lambda(0.999).unwrap();
        assert_recovers(&mut model, d, 2000, &mut s);
    }

    #[test]
    fn split_gap_leaves_the_prior() {
        // Regression: after a Reset inside a gap split into calls, the rest
        // of the gap inflated the fresh prior (39 × decay(0.5) gave 5e8).
        let mut model = EwRls::builder(2).lambda(1.0).build().unwrap();
        model.update(&[1.0, 2.0], 1.0).unwrap();
        for _ in 0..39 {
            model.decay(0.5).unwrap();
        }
        assert!((0..2).all(|j| model.covariance()[(j, j)] <= 2e3));
    }

    #[test]
    fn ridge_refresh_is_exact_on_a_vague_prior() {
        // Regression: the refresh downdate P_jj − (√f·P_jj)² cancelled to 0
        // once c·P_jj ≳ 1e15, freezing the coordinate.
        let mut model = EwRls::builder(2)
            .lambda(1.0)
            .initial_covariance(1e12)
            .ridge(Ridge::Relative(1.0))
            .build()
            .unwrap();
        model.update_weighted(&[1.0, 0.0], 1.0, 1e4).unwrap();
        model.update_weighted(&[1.0, 0.0], 1.0, 1e4).unwrap();
        let p11 = model.covariance()[(1, 1)];
        assert!((p11 - 5e-5).abs() < 1e-12 * 5e-5, "{p11}");
    }

    #[test]
    fn floor_handles_a_subnormal_prior() {
        // 1/δ overflows here, so the floor must not form it.
        let delta = 1e-310;
        let mut model = EwRls::builder(1)
            .lambda(0.5)
            .initial_covariance(delta)
            .build()
            .unwrap();
        for _ in 0..50 {
            model.update(&[0.0], 0.0).unwrap();
            assert!(model.covariance()[(0, 0)] <= 2.0 * delta);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn capped_prior_far_below_delta_restores() {
        // Regression: the cap's scale factor went subnormal, so the model
        // exceeded its own cap and failed to restore from its checkpoint.
        let model = EwRls::builder(2)
            .initial_covariance(4.3e110)
            .regularization(1.0)
            .ridge_weights(&[0.0, 1.0])
            .max_covariance_trace(1e-200)
            .build()
            .unwrap();
        let json = serde_json::to_string(&model).unwrap();
        let restored: EwRls = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.covariance(), model.covariance());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn checkpoints_round_trip_through_positional_formats() {
        // Regression: the reader's field order and types differed from the
        // writer's, which only self-describing formats tolerate.
        #[derive(serde::Serialize)]
        struct Version1<'a> {
            version: u32,
            theta: &'a DVector<f64>,
            p: &'a DMatrix<f64>,
            lambda: f64,
            dimensions: usize,
            initial_covariance: f64,
            max_trace: Option<f64>,
            gamma: f64,
            updates: u64,
        }
        let (d, mut s) = (3, 5u64);
        let mut model = EwRls::builder(d)
            .lambda(0.99)
            .regularization(0.5)
            .build()
            .unwrap();
        for _ in 0..7 {
            let (x, y) = linear_row(&mut s, d);
            model.update(&x, y).unwrap();
        }
        let bytes = bincode::serialize(&model).unwrap();
        let restored: EwRls = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.params(), model.params());
        assert_eq!(restored.covariance(), model.covariance());
        assert_eq!(restored.weight_mass(), model.weight_mass());

        // The 0.2 layout, in 0.2's field order.
        let v1 = Version1 {
            version: 1,
            theta: model.theta(),
            p: model.covariance(),
            lambda: 0.99,
            dimensions: d,
            initial_covariance: 1e3,
            max_trace: None,
            gamma: 0.5,
            updates: model.updates(),
        };
        let restored: EwRls = bincode::deserialize(&bincode::serialize(&v1).unwrap()).unwrap();
        assert_eq!(restored.covariance(), model.covariance());
        assert_eq!(restored.ridge(), Ridge::Absolute(0.5));
    }
}
