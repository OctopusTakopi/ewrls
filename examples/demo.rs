use nalgebra::DVector;
use rand::Rng;
use wrls::WeightedRLS;

fn main() {
    // --- 1. Setup ---
    let dimensions = 2; // We are fitting a line: y = m*x + b, so theta = [m, b]
    let lambda = 0.96;
    let initial_covariance = 1000.0;

    let mut model = WeightedRLS::new(dimensions, lambda, initial_covariance);

    let num_points = 1000;
    let x_vals: Vec<f64> = (0..num_points)
        .map(|i| 10.0 * (i as f64) / (num_points as f64))
        .collect();

    let mut rng = rand::rng();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| x.sin() + rng.random_range(-0.2..0.2))
        .collect();

    for t in 0..num_points {
        let xt = DVector::from_vec(vec![x_vals[t], 1.0]); // Feature vector [x, 1]
        let yt = y_vals[t];

        model.update(&xt, yt);

        if (t + 1) % 100 == 0 {
            println!(
                "Step {}: theta = [m: {:.4}, b: {:.4}]",
                t + 1,
                model.theta[0],
                model.theta[1]
            );
        }
    }

    println!("\n--- Final Learned Parameters ---");
    println!("Theta (m, b): {}", model.theta.transpose());
    println!("The model learned a linear approximation of the sine wave.");
    println!(
        "Since the true model is non-linear, these parameters represent the 'best fit' line at the end of the data stream."
    );
}
