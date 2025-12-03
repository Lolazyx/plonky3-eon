use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_bn254::Bn254;
use p3_dft::{Radix2Bowers, Radix2DFTSmallBatch, Radix2Dit, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::pretty_name;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

fn bench_fft(c: &mut Criterion) {
    // log_sizes correspond to the sizes of DFT we want to benchmark;
    let log_sizes = &[14, 16, 18, 20, 22];

    const BATCH_SIZE: usize = 256;

    fft::<Bn254, Radix2DFTSmallBatch<_>, BATCH_SIZE>(c, log_sizes);
    fft::<Bn254, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    fft::<Bn254, Radix2Bowers, BATCH_SIZE>(c, log_sizes);
    fft::<Bn254, Radix2DitParallel<_>, BATCH_SIZE>(c, log_sizes);

    ifft::<Bn254, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);

    coset_lde::<Bn254, Radix2Dit<_>, BATCH_SIZE>(c, log_sizes);
    coset_lde::<Bn254, Radix2Bowers, BATCH_SIZE>(c, log_sizes);
    coset_lde::<Bn254, Radix2DitParallel<_>, BATCH_SIZE>(c, log_sizes);
}

fn fft<F, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "fft/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter(|| {
                dft.dft_batch(messages.clone());
            });
        });
    }
}

fn ifft<F, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "ifft/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter(|| {
                dft.idft_batch(messages.clone());
            });
        });
    }
}

fn coset_lde<F, Dft, const BATCH_SIZE: usize>(c: &mut Criterion, log_sizes: &[usize])
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!(
        "coset_lde/{}/{}/ncols={}",
        pretty_name::<F>(),
        pretty_name::<Dft>(),
        BATCH_SIZE
    ));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for n_log in log_sizes {
        let n = 1 << n_log;

        let messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        let dft = Dft::default();
        group.bench_with_input(BenchmarkId::from_parameter(n), &dft, |b, dft| {
            b.iter(|| {
                dft.coset_lde_batch(messages.clone(), 1, F::GENERATOR);
            });
        });
    }
}

criterion_group!(benches, bench_fft);
criterion_main!(benches);
