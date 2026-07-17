use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::RngExt;
use std::hint::black_box;
use ewrls::EwRls;

/// Pre-generate a pool of random feature vectors so the timed loop measures
/// the model, not the RNG.
fn random_inputs(dims: usize, count: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect()
}

fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("EwRls/update");
    for &dims in &[5, 10, 25, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, &dims| {
            let mut model = EwRls::new(dims, 0.99).unwrap();
            // At least dims vectors so the cycled excitation stays full-rank;
            // fewer would leave directions winding up until breakdown.
            let inputs = random_inputs(dims, dims.max(64));
            let mut i = 0usize;
            b.iter(|| {
                let x = &inputs[i % inputs.len()];
                i += 1;
                model.update(black_box(x), black_box(0.5)).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("EwRls/predict");
    for &dims in &[5, 10, 25, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, &dims| {
            let mut model = EwRls::new(dims, 0.99).unwrap();
            let inputs = random_inputs(dims, dims.max(64));
            for x in &inputs {
                model.update(x, 0.5).unwrap();
            }
            let mut i = 0usize;
            b.iter(|| {
                let x = &inputs[i % inputs.len()];
                i += 1;
                model.predict(black_box(x)).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_update, bench_predict);
criterion_main!(benches);
