use nalgebra::{DMatrix, DVector};

/// A Weighted Recursive Least Squares (WRLS) model.
///
/// This structure holds the state of the adaptive filter, which includes
/// the model parameters (theta) and the covariance matrix (P).
#[derive(Debug, Clone)]
pub struct WeightedRLS {
    /// The parameter vector (theta) of size `d x 1`, where `d` is the number of features.
    pub theta: DVector<f64>,
    /// The covariance matrix (P) of size `d x d`.
    covariance: DMatrix<f64>,
    /// The forgetting factor (lambda), typically between 0.0 and 1.0.
    lambda: f64,
    /// The number of dimensions (features) of the model.
    dimensions: usize,
}

impl WeightedRLS {
    /// Creates a new WRLS model.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The number of features in the input vector `x`.
    /// * `lambda` - The forgetting factor (0 < lambda <= 1). A smaller value adapts faster
    ///   but is more sensitive to noise. A value of 1.0 gives equal weight to all past data.
    /// * `initial_covariance_val` - A large positive value to initialize the diagonal of the
    ///   covariance matrix `P`. A large value (e.g., 1000.0) signifies low confidence in the
    ///   initial `theta` parameters, allowing for rapid initial adaptation.
    ///
    /// # Panics
    ///
    /// Panics if `dimensions` is 0 or if `lambda` is not in the range `(0, 1]`.
    pub fn new(dimensions: usize, lambda: f64, initial_covariance_val: f64) -> Self {
        if dimensions == 0 {
            panic!("WeightedRLS model dimensions must be greater than 0.");
        }
        if !(lambda > 0.0 && lambda <= 1.0) {
            panic!("Forgetting factor lambda must be in the range (0, 1].");
        }
        if initial_covariance_val <= 0.0 {
            panic!("Initial covariance value must be positive.");
        }

        Self {
            // Initialize parameters to zero.
            theta: DVector::zeros(dimensions),
            // Initialize covariance as a diagonal matrix. P = initial_val * I.
            // This represents high initial uncertainty.
            covariance: DMatrix::identity(dimensions, dimensions) * initial_covariance_val,
            lambda,
            dimensions,
        }
    }

    /// Predicts the output value for a given input vector.
    ///
    /// y_pred = x^T * theta
    ///
    /// # Arguments
    ///
    /// * `x` - The input feature vector of size `d x 1`.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `x` do not match the model's dimensions.
    pub fn predict(&self, x: &DVector<f64>) -> f64 {
        assert_eq!(
            x.nrows(),
            self.dimensions,
            "Input vector dimensions do not match model dimensions."
        );
        // Performance: dot product is a highly optimized O(d) operation.
        x.dot(&self.theta)
    }

    /// Updates the model parameters with a new data point (x, y).
    ///
    /// This is the core of the WRLS algorithm.
    ///
    /// # Arguments
    ///
    /// * `x` - The input feature vector of size `d x 1`.
    /// * `y` - The true measured output (a scalar).
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `x` do not match the model's dimensions.
    pub fn update(&mut self, x: &DVector<f64>, y: f64) {
        assert_eq!(
            x.nrows(),
            self.dimensions,
            "Input vector dimensions do not match model dimensions."
        );

        // --- Performance Critical Section ---
        // The following operations determine the overall performance of the model update.
        // We use nalgebra's optimized routines.

        // 1. Pre-calculate P * x. This is a matrix-vector multiplication: O(d^2).
        let p_x = &self.covariance * x;

        // 2. Calculate the denominator for the gain vector: lambda + x^T * P * x.
        // We reuse p_x from the previous step. x.dot(&p_x) is O(d).
        let denominator = self.lambda + x.dot(&p_x);

        // 3. Calculate the gain vector k = (P * x) / denominator. O(d).
        let gain = p_x / denominator;

        // 4. Calculate the prediction error: e = y - y_pred = y - x^T * theta. O(d).
        let error = y - x.dot(&self.theta);

        // 5. Update the parameter vector: theta_new = theta_old + k * e. O(d).
        // `nalgebra` overloads `+=` for efficient in-place addition.
        self.theta += &gain * error;

        // 6. Update the covariance matrix: P_new = (1/lambda) * (P_old - k * (x^T * P_old))
        // This uses the Sherman-Morrison formula for O(d^2) efficiency.
        //   - Compute x^T * P_old: vector-matrix multiplication, O(d^2).
        //   - Compute k * (x^T * P_old): outer product, O(d^2).
        //   - Subtract and divide: O(d^2).
        let x_t_p = x.transpose() * &self.covariance;
        let outer = &gain * &x_t_p;
        self.covariance -= &outer;
        self.covariance /= self.lambda;
    }

    /// Returns the slope `m` of the model, assuming a 2D model for a line (`y = m*x + b`).
    ///
    /// This method assumes the first element of the `theta` parameter vector corresponds
    /// to the slope. It will return `None` if the model does not have exactly 2 dimensions,
    /// as the concept of a single "slope" is ambiguous in other cases.
    pub fn slope(&self) -> Option<f64> {
        if self.dimensions == 2 {
            Some(self.theta[0])
        } else {
            None
        }
    }

    /// Returns the y-intercept `b` of the model, assuming a 2D model for a line (`y = m*x + b`).
    ///
    /// This method assumes the second element of the `theta` parameter vector corresponds
    /// to the intercept. It will return `None` if the model does not have exactly 2 dimensions.
    pub fn intercept(&self) -> Option<f64> {
        if self.dimensions == 2 {
            Some(self.theta[1])
        } else {
            None
        }
    }

    /// Returns a slice of the full parameter vector `theta`.
    pub fn params(&self) -> &[f64] {
        self.theta.as_slice()
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use rand::Rng;

    const A_TOL: f64 = 1e-9; // Absolute tolerance for float comparisons

    #[test]
    fn test_new_wrls() {
        let model = WeightedRLS::new(3, 0.99, 1000.0);
        assert_eq!(model.dimensions, 3);
        assert_eq!(model.lambda, 0.99);
        assert_eq!(model.theta, DVector::from_vec(vec![0.0, 0.0, 0.0]));
        assert_eq!(
            model.covariance,
            DMatrix::from_diagonal(&DVector::from_vec(vec![1000.0, 1000.0, 1000.0]))
        );
    }

    #[test]
    #[should_panic]
    fn test_new_wrls_invalid_lambda() {
        WeightedRLS::new(3, 1.1, 1000.0);
    }

    #[test]
    #[should_panic]
    fn test_new_wrls_zero_dimensions() {
        WeightedRLS::new(0, 0.99, 1000.0);
    }

    #[test]
    fn test_predict() {
        let mut model = WeightedRLS::new(2, 0.98, 100.0);
        model.theta = DVector::from_vec(vec![1.5, -2.0]);
        let x = DVector::from_vec(vec![10.0, 3.0]);
        // prediction = 10.0 * 1.5 + 3.0 * -2.0 = 15.0 - 6.0 = 9.0
        assert!((model.predict(&x) - 9.0).abs() < A_TOL);
    }

    #[test]
    fn test_single_update() {
        // Test a single update step and verify against manually calculated values.
        let mut model = WeightedRLS::new(2, 1.0, 100.0); // lambda=1 for simplicity
        let x = DVector::from_vec(vec![2.0, 3.0]);
        let y = 10.0;

        // Initial state:
        // theta = [0, 0]^T
        // P = [[100, 0], [0, 100]]

        // Manual calculation:
        // y_pred = x^T * theta = 0
        // error = 10 - 0 = 10
        // P*x = [[100, 0], [0, 100]] * [2, 3]^T = [200, 300]^T
        // den = lambda + x^T*P*x = 1 + [2, 3] * [200, 300]^T = 1 + 400 + 900 = 1301
        // k = [200, 300]^T / 1301 = [0.1537, 0.2306]^T
        // theta_new = [0, 0]^T + k * 10 = [1.537, 2.306]^T

        model.update(&x, y);

        let expected_theta = DVector::from_vec(vec![1.53727901614143, 2.3059185242121445]);
        assert!((model.theta - expected_theta).norm() < A_TOL);

        // Check if covariance matrix was updated (it should no longer be diagonal)
        assert!(model.covariance[(0, 1)].abs() > A_TOL);
    }

    #[test]
    fn test_convergence() {
        // See if the model can learn a simple linear relationship over time.
        let true_theta = DVector::from_vec(vec![2.5, -3.0, 1.2]);
        let dimensions = true_theta.len();
        let mut model = WeightedRLS::new(dimensions, 0.99, 1000.0);

        let mut rng = rand::rng();
        let num_updates = 2000;

        for _ in 0..num_updates {
            // Generate random input data
            let x: DVector<f64> = DVector::from_fn(dimensions, |_, _| rng.random_range(-5.0..5.0));
            // Calculate the "true" output with some noise
            let noise = rng.random_range(-0.1..0.1);
            let y = x.dot(&true_theta) + noise;

            model.update(&x, y);
        }

        // After many updates, the model's theta should be close to the true theta.
        let difference = (&model.theta - &true_theta).norm();
        println!("Final theta: {}", model.theta.transpose());
        println!("True theta:  {}", true_theta.transpose());
        println!("Norm of difference: {}", difference);

        // We expect the parameters to be very close.
        assert!(
            difference < 0.1,
            "Model did not converge to the true parameters."
        );
    }
}
