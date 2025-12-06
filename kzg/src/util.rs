//! Utility functions for KZG commitment operations.
//!
//! This module contains internal helper functions used by both the PCS and MMCS
//! implementations for polynomial operations and verification.

use alloc::vec;
use alloc::vec::Vec;

use p3_bn254::{multi_pairing, pairing, Fr, G1, G2};
use p3_field::PrimeCharacteristicRing;

use crate::params::{KzgError, KzgParams};

/// Commits to a polynomial in coefficient form using KZG.
///
/// Given a polynomial `f(X) = c₀ + c₁X + c₂X² + ... + cₙXⁿ` represented by
/// coefficients `[c₀, c₁, ..., cₙ]`, this computes the KZG commitment:
///
/// `C = c₀·g₁ + c₁·(g₁^α) + c₂·(g₁^α²) + ... + cₙ·(g₁^αⁿ)`
///
/// This is equivalent to evaluating the polynomial at the secret value α
/// and encoding the result in G1: `C = f(α)·g₁`.
///
/// # Arguments
///
/// * `params` - The KZG parameters containing powers of α in G1
/// * `coeffs` - Polynomial coefficients in increasing degree order
///
/// # Returns
///
/// * `Ok(G1)` - The KZG commitment to the polynomial
/// * `Err(KzgError::DegreeTooLarge)` - If the polynomial degree exceeds the SRS capacity
///
/// # Complexity
///
/// Uses multi-scalar multiplication (MSM) which is O(n log n) for n coefficients.
pub(crate) fn commit_column(params: &KzgParams, coeffs: &[Fr]) -> Result<G1, KzgError> {
    params.ensure_supported(coeffs.len().saturating_sub(1))?;
    Ok(G1::multi_exp(&params.g1_powers[..coeffs.len()], coeffs))
}

/// Evaluates a polynomial at a given point using Horner's method.
///
/// For a polynomial `f(X) = c₀ + c₁X + c₂X² + ... + cₙXⁿ`, computes `f(point)`.
///
/// # Arguments
///
/// * `coeffs` - Polynomial coefficients `[c₀, c₁, ..., cₙ]` in increasing degree order
/// * `point` - The field element at which to evaluate the polynomial
///
/// # Returns
///
/// The value `f(point)` in the field Fr.
///
/// # Algorithm
///
/// Uses Horner's method: `f(x) = c₀ + x(c₁ + x(c₂ + ... + x·cₙ)))`
/// This requires only n-1 multiplications and n-1 additions.
///
/// # Complexity
///
/// O(n) where n is the number of coefficients.
pub(crate) fn eval_poly(coeffs: &[Fr], point: Fr) -> Fr {
    coeffs
        .iter()
        .rev()
        .fold(Fr::ZERO, |acc, &c| acc * point + c)
}

/// Computes the quotient polynomial and evaluation for KZG opening.
///
/// Given a polynomial `f(X)` with coefficients and an evaluation point `z`,
/// this computes:
/// - The quotient polynomial `q(X) = (f(X) - f(z)) / (X - z)`
/// - The evaluation `v = f(z)`
///
/// This is the core operation for generating KZG opening proofs. The quotient
/// polynomial is guaranteed to exist (i.e., `X - z` divides `f(X) - f(z)`) by
/// the polynomial remainder theorem.
///
/// # Arguments
///
/// * `coeffs` - Polynomial coefficients in increasing degree order
/// * `point` - The evaluation point z
///
/// # Returns
///
/// A tuple `(quotient_coeffs, evaluation)` where:
/// - `quotient_coeffs` - Coefficients of the quotient polynomial q(X)
/// - `evaluation` - The value f(z)
///
/// # Algorithm
///
/// Uses synthetic division to simultaneously compute the quotient and remainder.
/// The remainder equals f(z) by the polynomial remainder theorem.
///
/// # Complexity
///
/// O(n) where n is the degree of the polynomial.
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

/// Verifies a KZG opening proof using a pairing check.
///
/// Verifies that a commitment `C` opens to value `v` at point `z` by checking
/// the pairing equation:
///
/// `e(C - v·g₁, g₂) = e(W, α·g₂ - z·g₂)`
///
/// where W is the witness (commitment to the quotient polynomial).
///
/// This equation holds if and only if `W = q(α)·g₁` where `q(X) = (f(X) - v)/(X - z)`,
/// which is true if and only if `f(z) = v`.
///
/// # Arguments
///
/// * `commitment` - The KZG commitment C to the polynomial f(X)
/// * `witness` - The witness W (commitment to the quotient polynomial)
/// * `value` - The claimed evaluation v = f(z)
/// * `point` - The evaluation point z
/// * `params` - The KZG parameters containing α·g₂
///
/// # Returns
///
/// * `Ok(())` - If the verification succeeds (the opening is valid)
/// * `Err(KzgError::ProofShapeMismatch)` - If verification fails (invalid proof)
///
/// # Security
///
/// Under the KZG security assumption (a variant of the q-SDH assumption), it is
/// computationally infeasible to produce a valid-looking proof for an incorrect
/// evaluation without knowing the secret α.
///
/// # Note
///
/// For verifying multiple openings, consider using [`verify_batch`] which is more
/// efficient as it performs only a single multi-pairing instead of multiple individual
/// pairing checks.
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

/// Batch opening information for a single proof.
///
/// Contains all the necessary data to verify a single KZG opening:
/// the polynomial commitment, opening witness, claimed value, and evaluation point.
#[derive(Clone, Debug)]
pub(crate) struct OpeningInfo {
    /// The commitment to the polynomial
    pub commitment: G1,
    /// The opening witness (quotient polynomial commitment)
    pub witness: G1,
    /// The claimed evaluation value
    pub value: Fr,
    /// The evaluation point
    pub point: Fr,
}

/// Verifies multiple KZG opening proofs using a single batch pairing check.
///
/// This is significantly more efficient than verifying each proof individually,
/// as it requires only one multi-pairing computation instead of 2n individual
/// pairings for n openings.
///
/// # How It Works
///
/// Instead of checking each equation individually:
/// - `e(C₁ - v₁·g₁, g₂) = e(W₁, α·g₂ - z₁·g₂)`
/// - `e(C₂ - v₂·g₁, g₂) = e(W₂, α·g₂ - z₂·g₂)`
/// - ...
///
/// We rearrange and combine into a single check:
/// `e(C₁ - v₁·g₁, g₂) · e(-W₁, α·g₂ - z₁·g₂) · e(C₂ - v₂·g₁, g₂) · e(-W₂, α·g₂ - z₂·g₂) ... = 1`
///
/// This is equivalent but requires only one multi-pairing instead of 2n pairings.
///
/// # Arguments
///
/// * `openings` - Slice of opening information to verify
/// * `params` - The KZG parameters containing α·g₂
///
/// # Returns
///
/// * `Ok(())` - If all openings verify successfully
/// * `Err(KzgError::ProofShapeMismatch)` - If any opening fails verification
///
/// # Performance
///
/// For n openings:
/// - Individual verification: 2n pairings
/// - Batch verification: 1 multi-pairing (with 2n pairs)
///
/// Multi-pairing is typically 1.5-2x faster than computing individual pairings,
/// resulting in significant speedup for large batches.
///
/// # Example
///
/// ```ignore
/// let openings = vec![
///     OpeningInfo {
///         commitment: commitment1,
///         witness: witness1,
///         value: value1,
///         point: point1,
///     },
///     OpeningInfo {
///         commitment: commitment2,
///         witness: witness2,
///         value: value2,
///         point: point2,
///     },
/// ];
///
/// verify_batch(&openings, &params)?;
/// ```
pub(crate) fn verify_batch(openings: &[OpeningInfo], params: &KzgParams) -> Result<(), KzgError> {
    if openings.is_empty() {
        return Ok(());
    }

    // Special case: single opening can use the simpler verify_single
    if openings.len() == 1 {
        let opening = &openings[0];
        return verify_single(
            &opening.commitment,
            &opening.witness,
            opening.value,
            opening.point,
            params,
        );
    }

    let g1 = G1::generator();
    let g2 = G2::generator();

    // Build the multi-pairing inputs
    // We want to check: for all i, e(C_i - v_i·g₁, g₂) = e(W_i, α·g₂ - z_i·g₂)
    // Rearranging: e(C_i - v_i·g₁, g₂) · e(W_i, -(α·g₂ - z_i·g₂)) = 1
    // Which means: e(C_i - v_i·g₁, g₂) · e(-W_i, α·g₂ - z_i·g₂) = 1
    //
    // We combine all these: Π_i [e(C_i - v_i·g₁, g₂) · e(-W_i, α·g₂ - z_i·g₂)] = 1
    let mut pairs = Vec::with_capacity(2 * openings.len());

    for opening in openings {
        // First pair: (C_i - v_i·g₁, g₂)
        let commitment_adjusted = opening.commitment - g1.mul_scalar(opening.value);
        pairs.push((commitment_adjusted, g2));

        // Second pair: (-W_i, α·g₂ - z_i·g₂)
        let g2_adjusted = params.g2_alpha - g2.mul_scalar(opening.point);
        pairs.push((-opening.witness, g2_adjusted));
    }

    // Compute the multi-pairing
    let result = multi_pairing(&pairs);

    // Check if the result equals the identity in Gt
    if result == p3_bn254::Gt::identity() {
        Ok(())
    } else {
        Err(KzgError::ProofShapeMismatch)
    }
}
