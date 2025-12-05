use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_bn254::{Fr, G1, G2};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn bench_g1_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("G1");

    let g1 = G1::generator();
    let mut rng = SmallRng::seed_from_u64(42);
    let scalar: Fr = rng.random();

    group.bench_function("scalar_mul", |b| {
        b.iter(|| black_box(g1).mul_scalar(black_box(scalar)));
    });

    group.finish();
}

fn bench_g2_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("G2");

    let g2 = G2::generator();
    let mut rng = SmallRng::seed_from_u64(42);
    let scalar: Fr = rng.random();

    group.bench_function("scalar_mul", |b| {
        b.iter(|| black_box(g2).mul_scalar(black_box(scalar)));
    });

    group.finish();
}

fn bench_g1_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("G1_MSM");

    let g1 = G1::generator();
    let mut rng = SmallRng::seed_from_u64(42);

    // Benchmark different sizes of MSM
    for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].iter() {
        let points: Vec<G1> = (0..*size).map(|_| g1.mul_scalar(rng.random())).collect();
        let scalars: Vec<Fr> = (0..*size).map(|_| rng.random()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| G1::multi_exp(black_box(&points), black_box(&scalars)));
        });
    }

    group.finish();
}

fn bench_g2_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("G2_MSM");

    let g2 = G2::generator();
    let mut rng = SmallRng::seed_from_u64(42);

    // Benchmark different sizes of MSM
    for size in [1, 2, 4, 8, 16, 32, 64, 128, 256].iter() {
        let points: Vec<G2> = (0..*size).map(|_| g2.mul_scalar(rng.random())).collect();
        let scalars: Vec<Fr> = (0..*size).map(|_| rng.random()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| G2::multi_exp(black_box(&points), black_box(&scalars)));
        });
    }

    group.finish();
}

fn bench_pairing(c: &mut Criterion) {
    use p3_bn254::{multi_pairing, pairing};

    let mut group = c.benchmark_group("Pairing");

    let g1 = G1::generator();
    let g2 = G2::generator();
    let mut rng = SmallRng::seed_from_u64(42);

    group.bench_function("single_pairing", |b| {
        b.iter(|| pairing(black_box(g1), black_box(g2)));
    });

    // Benchmark multi-pairing with different sizes
    for size in [2, 4, 8, 16].iter() {
        let pairs: Vec<(G1, G2)> = (0..*size)
            .map(|_| {
                let g1_i = g1.mul_scalar(rng.random());
                let g2_i = g2.mul_scalar(rng.random());
                (g1_i, g2_i)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("multi_pairing", size), size, |b, _| {
            b.iter(|| multi_pairing(black_box(&pairs)));
        });
    }

    group.finish();
}

criterion_group!(
    curve_benches,
    bench_g1_operations,
    bench_g2_operations,
    bench_g1_msm,
    bench_g2_msm,
    bench_pairing
);
criterion_main!(curve_benches);
