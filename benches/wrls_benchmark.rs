use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use rand::Rng;
use std::hint::black_box;
use wrls::WeightedRLS;

fn bench_wrls_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("WeightedRLS Update Performance");

    // We will benchmark the update function for different numbers of dimensions
    // to see how performance scales.
    for dimensions in [5, 10, 25, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(dimensions),
            dimensions,
            |b, &dims| {
                // --- Setup for the benchmark (not timed) ---
                let mut rng = rand::rng();
                // Create a model instance to be used in the benchmark
                let model = WeightedRLS::new(dims, 0.99, 1000.0);
                // Create a random input vector and output value
                let x: DVector<f64> = DVector::from_fn(dims, |_, _| rng.random());
                let y: f64 = rng.random();

                // --- The actual benchmark loop (timed) ---
                b.iter(|| {
                    // We clone the model in each iteration to ensure that the benchmark
                    // is not affected by the changing state of the model.
                    // The `black_box` calls prevent the compiler from optimizing
                    // away the function call we are trying to measure.
                    let mut model_clone = model.clone();
                    model_clone.update(black_box(&x), black_box(y));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_wrls_update);
criterion_main!(benches);
