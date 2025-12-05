use super::*;
use crate::mmcs::KzgMmcs;
use crate::pcs::KzgPcs;
use p3_bn254::{Fr, Poseidon2Bn254};
use p3_challenger::DuplexChallenger;
use p3_commit::{BatchOpeningRef, Pcs};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[test]
fn pcs_roundtrip() {
    let alpha = Fr::new(7);
    let pcs = KzgPcs::new(8, alpha);

    let domain = TwoAdicMultiplicativeCoset::new(Fr::ONE, 3).unwrap();
    let points: Vec<_> = domain.iter().collect();
    let evals: Vec<_> = points.iter().map(|&x| x + Fr::ONE).collect();
    let eval_matrix = RowMajorMatrix::new(evals, 1);

    type Perm = Poseidon2Bn254<3>;
    type Chal = DuplexChallenger<Fr, Perm, 3, 2>;
    let perm = Perm::new_from_rng(8, 22, &mut SmallRng::seed_from_u64(1));
    let mut challenger = Chal::new(perm);

    let (commit, prover_data) = <KzgPcs as Pcs<Fr, Chal>>::commit(&pcs, [(domain, eval_matrix)]);
    let open_point = Fr::new(2);
    let (opened_values, proof) = pcs.open(
        vec![(&prover_data, vec![vec![open_point]])],
        &mut challenger,
    );

    let verify_inputs = vec![(
        commit,
        vec![(domain, vec![(open_point, opened_values[0][0][0].clone())])],
    )];

    pcs.verify(verify_inputs, &proof, &mut challenger)
        .expect("PCS verification should succeed");
}

#[test]
fn mmcs_roundtrip() {
    let alpha = Fr::new(5);
    let mmcs = KzgMmcs::new(4, alpha);

    let matrix = RowMajorMatrix::new(vec![Fr::ONE, Fr::TWO, Fr::new(3), Fr::new(4)], 2);
    let dims = [Dimensions {
        height: matrix.height(),
        width: matrix.width(),
    }];

    let (commit, prover_data) = mmcs.commit(vec![matrix]);
    let opening = mmcs.open_batch(0, &prover_data);
    let opening_ref = BatchOpeningRef {
        opened_values: &opening.opened_values,
        opening_proof: &opening.opening_proof,
    };

    mmcs.verify_batch(&commit, &dims, 0, opening_ref)
        .expect("MMCS verification should succeed");
}
