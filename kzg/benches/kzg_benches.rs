use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_bn254::{Fr, Poseidon2Bn254, G1};
use p3_challenger::DuplexChallenger;
use p3_commit::Pcs;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::PrimeCharacteristicRing;
use p3_kzg::util::{verify_batch, verify_single, OpeningInfo};
use p3_kzg::{init_srs_unsafe, KzgPcs};
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

fn bench_verify_individual(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_verify_individual");
    let rng = &mut SmallRng::seed_from_u64(4);

    // Setup: create multiple polynomial openings
    let alpha = Fr::new(42);
    let params = init_srs_unsafe(256, alpha);

    // Generate random polynomials and their openings
    for &num_openings in &[1usize, 5, 10, 20, 50] {
        let mut openings = Vec::new();

        for _ in 0..num_openings {
            // Random polynomial of degree 3
            let coeffs: Vec<Fr> = (0..4).map(|_| Fr::new(rng.random::<u64>())).collect();
            let commitment = G1::multi_exp(&params.g1_powers[..coeffs.len()], &coeffs);

            // Open at a random point
            let point = Fr::new(rng.random::<u64>());

            // Compute evaluation using Horner's method
            let mut value = Fr::ZERO;
            for &coeff in coeffs.iter().rev() {
                value = value * point + coeff;
            }

            // Compute quotient polynomial: q(X) = (f(X) - f(point)) / (X - point)
            let mut quotient = vec![Fr::ZERO; coeffs.len() - 1];
            let mut carry = *coeffs.last().unwrap();
            for i in (0..coeffs.len() - 1).rev() {
                quotient[i] = carry;
                carry = coeffs[i] + carry * point;
            }
            let witness = G1::multi_exp(&params.g1_powers[..quotient.len()], &quotient);

            openings.push(OpeningInfo {
                commitment,
                witness,
                value,
                point,
            });
        }

        group.bench_function(BenchmarkId::new("individual", num_openings), |b| {
            b.iter(|| {
                for opening in &openings {
                    verify_single(
                        &opening.commitment,
                        &opening.witness,
                        opening.value,
                        opening.point,
                        &params,
                    )
                    .unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_verify_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_verify_batch");
    let rng = &mut SmallRng::seed_from_u64(5);

    // Setup: create multiple polynomial openings
    let alpha = Fr::new(42);
    let params = init_srs_unsafe(256, alpha);

    // Generate random polynomials and their openings
    for &num_openings in &[1usize, 5, 10, 20, 50] {
        let mut openings = Vec::new();

        for _ in 0..num_openings {
            // Random polynomial of degree 3
            let coeffs: Vec<Fr> = (0..4).map(|_| Fr::new(rng.random::<u64>())).collect();
            let commitment = G1::multi_exp(&params.g1_powers[..coeffs.len()], &coeffs);

            // Open at a random point
            let point = Fr::new(rng.random::<u64>());

            // Compute evaluation using Horner's method
            let mut value = Fr::ZERO;
            for &coeff in coeffs.iter().rev() {
                value = value * point + coeff;
            }

            // Compute quotient polynomial
            let mut quotient = vec![Fr::ZERO; coeffs.len() - 1];
            let mut carry = *coeffs.last().unwrap();
            for i in (0..coeffs.len() - 1).rev() {
                quotient[i] = carry;
                carry = coeffs[i] + carry * point;
            }
            let witness = G1::multi_exp(&params.g1_powers[..quotient.len()], &quotient);

            openings.push(OpeningInfo {
                commitment,
                witness,
                value,
                point,
            });
        }

        group.bench_function(BenchmarkId::new("batch", num_openings), |b| {
            b.iter(|| {
                verify_batch(&openings, &params).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_commit,
    bench_open,
    bench_verify_individual,
    bench_verify_batch
);
criterion_main!(benches);
