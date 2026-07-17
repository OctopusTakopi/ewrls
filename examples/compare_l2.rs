//! Scratch comparison: γ = 0 (no L2) vs persistent L2 across three regimes.

use ewrls::EwRls;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn build(gamma: f64, lambda: f64) -> EwRls {
    EwRls::builder(2)
        .lambda(lambda)
        .regularization(gamma)
        .build()
        .unwrap()
}

fn main() {
    // A. Noisy stream, weak true signal: one-step-ahead prediction MSE.
    //    The a-priori residual is out-of-sample, so this is honest test error.
    println!("A. noisy stream (sigma = 1.0, theta* = [0.3, -0.2], lambda = 0.99, 5000 steps)");
    for gamma in [0.0, 0.5, 1.0, 5.0, 20.0] {
        let mut rng = StdRng::seed_from_u64(42);
        let mut model = build(gamma, 0.99);
        let (mut sse, mut n) = (0.0, 0u32);
        for t in 0..5000 {
            let x = [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)];
            let noise: f64 = rng.random_range(-1.732..1.732); // var ~= 1
            let y = 0.3 * x[0] - 0.2 * x[1] + noise;
            let report = model.update(&x, y).unwrap();
            if t >= 500 {
                sse += report.residual * report.residual;
                n += 1;
            }
        }
        println!("   gamma = {gamma:>4}: pred MSE = {:.4}", sse / n as f64);
    }

    // B. Wind-up: feature 2 silent for 2000 steps, then one informative
    //    sample. How violently does theta_2 jump on a single observation?
    println!("\nB. wind-up (lambda = 0.99, feature 2 unexcited 2000 steps, then one sample)");
    for gamma in [0.0, 0.01, 0.1] {
        let mut rng = StdRng::seed_from_u64(7);
        let mut model = build(gamma, 0.99);
        for _ in 0..2000 {
            let x = [rng.random_range(-1.0..1.0), 0.0];
            model.update(&x, 2.0 * x[0]).unwrap();
        }
        let trace = model.covariance_trace();
        model.update(&[0.0, 0.001], 0.5).unwrap();
        println!(
            "   gamma = {gamma:>4}: trace(P) before = {trace:>9.3e}, theta after one sample = [{:>7.4}, {:>10.4}]",
            model.params()[0],
            model.params()[1]
        );
    }

    // C. Clean, well-excited data (Norris-like): the cost of shrinkage bias.
    println!("\nC. clean data (no noise, theta* = [2.0, 1.0], lambda = 0.99, 2000 steps)");
    for gamma in [0.0, 1.0, 5.0] {
        let mut rng = StdRng::seed_from_u64(3);
        let mut model = build(gamma, 0.99);
        for _ in 0..2000 {
            let x = [rng.random_range(-1.0..1.0), 1.0];
            model.update(&x, 2.0 * x[0] + 1.0).unwrap();
        }
        println!(
            "   gamma = {gamma:>4}: theta = [{:.4}, {:.4}]  (true [2.0, 1.0])",
            model.params()[0],
            model.params()[1]
        );
    }
}
