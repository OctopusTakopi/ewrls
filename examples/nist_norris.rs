//! Verify the model against the NIST StRD "Norris" linear regression
//! benchmark (ozone monitor calibration, 36 observations).
//!
//! With λ = 1 and a weak prior (large δ), EW-RLS computes ordinary least
//! squares, so the learned parameters must reproduce NIST's certified
//! values. https://www.itl.nist.gov/div898/strd/lls/data/norris.shtml

use ewrls::EwRls;

/// (y, x) pairs, verbatim from Norris.dat.
const DATA: [(f64, f64); 36] = [
    (0.1, 0.2),
    (338.8, 337.4),
    (118.1, 118.2),
    (888.0, 884.6),
    (9.2, 10.1),
    (228.1, 226.5),
    (668.5, 666.3),
    (998.5, 996.3),
    (449.1, 448.6),
    (778.9, 777.0),
    (559.2, 558.2),
    (0.3, 0.4),
    (0.1, 0.6),
    (778.1, 775.5),
    (668.8, 666.9),
    (339.3, 338.0),
    (448.9, 447.5),
    (10.8, 11.6),
    (557.7, 556.0),
    (228.3, 228.1),
    (998.0, 995.8),
    (888.8, 887.6),
    (119.6, 120.2),
    (0.3, 0.3),
    (0.6, 0.3),
    (557.6, 556.8),
    (339.3, 339.1),
    (888.0, 887.2),
    (998.5, 999.0),
    (778.9, 779.0),
    (10.2, 11.1),
    (117.6, 118.3),
    (228.9, 229.2),
    (668.4, 669.1),
    (449.2, 448.9),
    (0.2, 0.5),
];

// Certified values from Norris.dat.
const CERT_INTERCEPT: f64 = -0.262323073774029;
const CERT_SLOPE: f64 = 1.00211681802045;
const CERT_RESIDUAL_SD: f64 = 0.884796396144373;
const CERT_R_SQUARED: f64 = 0.999993745883712;

fn main() {
    // λ = 1 (no forgetting). δ = 1e6 keeps the ridge bias below 1e-9 on this
    // data while staying small enough that Sherman–Morrison cancellation
    // doesn't erode the covariance (much larger δ costs digits, and by 1e12
    // the model reports NumericalBreakdown).
    let mut model = EwRls::builder(2)
        .lambda(1.0)
        .initial_covariance(1e6)
        .build()
        .unwrap();

    for (y, x) in DATA {
        model.update(&[x, 1.0], y).unwrap();
    }
    let (slope, intercept) = (model.params()[0], model.params()[1]);

    let n = DATA.len() as f64;
    let mean_y = DATA.iter().map(|(y, _)| y).sum::<f64>() / n;
    let ss_res: f64 = DATA
        .iter()
        .map(|(y, x)| (y - (slope * x + intercept)).powi(2))
        .sum();
    let ss_tot: f64 = DATA.iter().map(|(y, _)| (y - mean_y).powi(2)).sum();
    let residual_sd = (ss_res / (n - 2.0)).sqrt();
    let r_squared = 1.0 - ss_res / ss_tot;

    println!("                 learned              certified");
    println!("slope        {slope:>18.12}  {CERT_SLOPE:>18.12}");
    println!("intercept    {intercept:>18.12}  {CERT_INTERCEPT:>18.12}");
    println!("residual sd  {residual_sd:>18.12}  {CERT_RESIDUAL_SD:>18.12}");
    println!("R²           {r_squared:>18.12}  {CERT_R_SQUARED:>18.12}");

    assert!((slope - CERT_SLOPE).abs() < 1e-9);
    assert!((intercept - CERT_INTERCEPT).abs() < 1e-6);
    assert!((residual_sd - CERT_RESIDUAL_SD).abs() < 1e-9);
    assert!((r_squared - CERT_R_SQUARED).abs() < 1e-12);
    println!("\nall values match the NIST certified results");
}
