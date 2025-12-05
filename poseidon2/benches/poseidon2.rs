use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_bn254::{Fr, Poseidon2Bn254};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_symmetric::Permutation;
use p3_util::pretty_name;
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn bench_poseidon2(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);

    // We hard code the round numbers for Bn254Fr.
    let poseidon2_bn254 = Poseidon2Bn254::<3>::new_from_rng(8, 22, &mut rng);
    poseidon2::<Fr, Poseidon2Bn254<3>, 3>(c, &poseidon2_bn254);
}

fn poseidon2<F, Perm, const WIDTH: usize>(c: &mut Criterion, poseidon2: &Perm)
where
    F: Field,
    Perm: Permutation<[F::Packing; WIDTH]>,
{
    let input = [F::Packing::ZERO; WIDTH];
    let name = format!("poseidon2::<{}, {}>", pretty_name::<F::Packing>(), WIDTH);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon2.permute(input)));
}

criterion_group!(benches, bench_poseidon2);
criterion_main!(benches);
