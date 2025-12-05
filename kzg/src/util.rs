use alloc::vec;
use alloc::vec::Vec;

use p3_bn254::{Fr, G1, G2, pairing};
use p3_field::PrimeCharacteristicRing;

use crate::params::{KzgError, KzgParams};

pub(crate) fn commit_column(params: &KzgParams, coeffs: &[Fr]) -> Result<G1, KzgError> {
    params.ensure_supported(coeffs.len().saturating_sub(1))?;
    Ok(G1::multi_exp(&params.g1_powers[..coeffs.len()], coeffs))
}

pub(crate) fn eval_poly(coeffs: &[Fr], point: Fr) -> Fr {
    coeffs
        .iter()
        .rev()
        .fold(Fr::ZERO, |acc, &c| acc * point + c)
}

pub(crate) fn quotient_and_eval(coeffs: &[Fr], point: Fr) -> (Vec<Fr>, Fr) {
    if coeffs.is_empty() {
        return (Vec::new(), Fr::ZERO);
    }
    let mut quotient = vec![Fr::ZERO; coeffs.len() - 1];
    let mut carry = *coeffs.last().unwrap();
    for i in (0..coeffs.len() - 1).rev() {
        quotient[i] = carry;
        carry = coeffs[i] + carry * point;
    }
    (quotient, carry)
}

pub(crate) fn verify_single(
    commitment: &G1,
    witness: &G1,
    value: Fr,
    point: Fr,
    params: &KzgParams,
) -> Result<(), KzgError> {
    let g1 = G1::generator();
    let g2 = G2::generator();

    let left = pairing(*commitment - g1.mul_scalar(value), g2);
    let right = pairing(*witness, params.g2_alpha - g2.mul_scalar(point));

    if left == right {
        Ok(())
    } else {
        Err(KzgError::ProofShapeMismatch)
    }
}
