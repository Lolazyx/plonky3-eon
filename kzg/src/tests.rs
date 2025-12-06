use alloc::vec;
use alloc::vec::Vec;

use p3_bn254::{Fr, Poseidon2Bn254, G1};
use p3_challenger::DuplexChallenger;
use p3_commit::{BatchOpeningRef, Mmcs, Pcs};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use super::*;
use crate::mmcs::KzgMmcs;
use crate::pcs::KzgPcs;
use crate::util::{verify_batch, OpeningInfo};

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

#[test]
fn test_batch_verification() {
    // Test that batch verification works correctly with multiple openings
    let alpha = Fr::new(42);
    let params = init_srs_unsafe(16, alpha);

    // Create multiple polynomial commitments
    let poly1_coeffs = vec![Fr::ONE, Fr::TWO, Fr::new(3)];
    let poly2_coeffs = vec![Fr::new(5), Fr::new(7), Fr::new(11)];

    let commit1 = G1::multi_exp(&params.g1_powers[..poly1_coeffs.len()], &poly1_coeffs);
    let commit2 = G1::multi_exp(&params.g1_powers[..poly2_coeffs.len()], &poly2_coeffs);

    // Evaluate at different points and create witnesses
    let point1 = Fr::new(2);
    let point2 = Fr::new(3);

    // Poly1 at point1: 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
    let eval1_1 = Fr::ONE + Fr::TWO * point1 + Fr::new(3) * point1 * point1;
    let quotient1_1 = vec![Fr::TWO + Fr::new(3) * point1, Fr::new(3)];
    let witness1_1 = G1::multi_exp(&params.g1_powers[..quotient1_1.len()], &quotient1_1);

    // Poly2 at point2: 5 + 7*3 + 11*9 = 5 + 21 + 99 = 125
    let eval2_2 = Fr::new(5) + Fr::new(7) * point2 + Fr::new(11) * point2 * point2;
    let quotient2_2 = vec![Fr::new(7) + Fr::new(11) * point2, Fr::new(11)];
    let witness2_2 = G1::multi_exp(&params.g1_powers[..quotient2_2.len()], &quotient2_2);

    // Test batch verification with multiple openings
    let openings = vec![
        OpeningInfo {
            commitment: commit1,
            witness: witness1_1,
            value: eval1_1,
            point: point1,
        },
        OpeningInfo {
            commitment: commit2,
            witness: witness2_2,
            value: eval2_2,
            point: point2,
        },
    ];

    verify_batch(&openings, &params).expect("Batch verification should succeed");

    // Test that batch verification fails with wrong value
    let bad_openings = vec![
        OpeningInfo {
            commitment: commit1,
            witness: witness1_1,
            value: eval1_1 + Fr::ONE, // Wrong value
            point: point1,
        },
        OpeningInfo {
            commitment: commit2,
            witness: witness2_2,
            value: eval2_2,
            point: point2,
        },
    ];

    assert!(
        verify_batch(&bad_openings, &params).is_err(),
        "Batch verification should fail with incorrect value"
    );
}

#[test]
fn test_batch_verification_empty() {
    // Empty batch should succeed trivially
    let alpha = Fr::new(123);
    let params = init_srs_unsafe(8, alpha);

    let empty_batch: Vec<OpeningInfo> = vec![];
    verify_batch(&empty_batch, &params).expect("Empty batch should verify");
}

#[test]
fn test_batch_verification_single() {
    // Single opening should use verify_single path
    let alpha = Fr::new(999);
    let params = init_srs_unsafe(8, alpha);

    let poly_coeffs = vec![Fr::ONE, Fr::TWO];
    let commitment = G1::multi_exp(&params.g1_powers[..poly_coeffs.len()], &poly_coeffs);

    let point = Fr::new(5);
    let value = Fr::ONE + Fr::TWO * point; // 1 + 10 = 11
    let quotient = vec![Fr::TWO];
    let witness = G1::multi_exp(&params.g1_powers[..quotient.len()], &quotient);

    let single_batch = vec![OpeningInfo {
        commitment,
        witness,
        value,
        point,
    }];

    verify_batch(&single_batch, &params).expect("Single opening should verify");
}
