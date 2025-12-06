use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_bn254::{Fr, Poseidon2Bn254};
use p3_challenger::DuplexChallenger;
use p3_commit::Pcs;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::PrimeCharacteristicRing;
use p3_kzg::KzgPcs;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type Perm = Poseidon2Bn254<3>;
type Chal = DuplexChallenger<Fr, Perm, 3, 2>;

fn make_random_matrix(height: usize, width: usize, rng: &mut SmallRng) -> RowMajorMatrix<Fr> {
    let mut values = Vec::with_capacity(height * width);
    for _ in 0..height * width {
        values.push(Fr::new(rng.random::<u64>()));
    }
    RowMajorMatrix::new(values, width)
}

fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_commit");
    let rng = &mut SmallRng::seed_from_u64(1);
    let pcs = KzgPcs::new(1 << 12, Fr::new(7));

    for &log_size in &[8usize, 10] {
        let height = 1 << log_size;
        let domain = TwoAdicMultiplicativeCoset::new(Fr::ONE, log_size).unwrap();
        let evals = make_random_matrix(height, 1, rng);

        group.bench_function(BenchmarkId::from_parameter(log_size), |b| {
            b.iter(|| {
                let _ = <KzgPcs as Pcs<Fr, Chal>>::commit(&pcs, [(domain, evals.clone())]);
            });
        });
    }
    group.finish();
}

fn bench_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_open");
    let rng = &mut SmallRng::seed_from_u64(2);
    let pcs = KzgPcs::new(1 << 12, Fr::new(9));

    for &log_size in &[8usize, 10] {
        let height = 1 << log_size;
        let domain = TwoAdicMultiplicativeCoset::new(Fr::ONE, log_size).unwrap();
        let evals = make_random_matrix(height, 1, rng);
        let (_commit, prover_data) = <KzgPcs as Pcs<Fr, Chal>>::commit(&pcs, [(domain, evals)]);
        let open_point = Fr::new(3);
        let perm = Perm::new_from_rng(8, 22, &mut SmallRng::seed_from_u64(3));

        group.bench_function(BenchmarkId::from_parameter(log_size), |b| {
            b.iter_batched(
                || DuplexChallenger::<Fr, Perm, 3, 2>::new(perm.clone()),
                |mut chal| {
                    let _ = pcs.open(vec![(&prover_data, vec![vec![open_point]])], &mut chal);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_commit, bench_open);
criterion_main!(benches);
