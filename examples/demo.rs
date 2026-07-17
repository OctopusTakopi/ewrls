//! Streaming regression demo: track a linear relationship through a regime
//! switch, using the per-update diagnostics for residual monitoring.

use ewrls::EwRls;
use rand::RngExt;

fn main() {
    // Fit y = m*x + b online, features are [x, 1].
    let mut model = EwRls::builder(2)
        .lambda(0.97)
        .initial_covariance(1000.0)
        .build()
        .expect("valid config");

    println!(
        "lambda = {}, effective window ≈ {:.0} samples\n",
        model.lambda(),
        model.effective_window()
    );

    let mut rng = rand::rng();
    let regimes = [(2.0, 1.0), (-4.0, 0.5)]; // (m, b) switches halfway

    for t in 0..1000 {
        let (m, b) = if t < 500 { regimes[0] } else { regimes[1] };
        let x = [rng.random_range(-1.0..1.0), 1.0];
        let y = m * x[0] + b + rng.random_range(-0.1..0.1);

        let report = model.update(&x, y).expect("finite inputs");

        if (t + 1) % 100 == 0 {
            println!(
                "step {:4}: theta = [m: {:7.4}, b: {:7.4}]  residual = {:8.4}  xᵀPx = {:.2e}",
                t + 1,
                model.params()[0],
                model.params()[1],
                report.residual,
                report.predictive_variance,
            );
        }
    }

    println!("\ntrue final regime: m = -4.0, b = 0.5");
    println!(
        "learned:           m = {:.4}, b = {:.4}",
        model.params()[0],
        model.params()[1]
    );
}
