//! BN254 curve group operations and pairing functionality
//!
//! This module provides wrapper types for the BN254 elliptic curve groups, wrapping the
//! battle-tested `halo2curves` library with a Plonky3-compatible interface.
//!
//! # Curve Groups
//!
//! - **G1**: Points on the base curve E(Fq) where the curve equation is `y² = x³ + 3`
//!   - Defined over the prime field Fq
//!   - Used for commitments and proofs in most cryptographic protocols
//!   - Supports efficient multi-scalar multiplication (MSM)
//!
//! - **G2**: Points on the twisted curve E'(Fq2)
//!   - Defined over the quadratic extension field Fq2
//!   - Used in pairing-based cryptography for verification keys
//!   - Required for KZG verification
//!
//! - **Gt**: Elements in the target group (multiplicative group of Fq12)
//!   - The output of the pairing operation e: G1 × G2 → Gt
//!   - Elements live in the 12th-degree extension field
//!   - Group operation is multiplication (corresponding to addition in the exponent)
//!
//! # Pairing Operations
//!
//! The module provides efficient bilinear pairing operations:
//!
//! - `pairing(P, Q)`: Computes the pairing e(P, Q) for P ∈ G1, Q ∈ G2
//! - `multi_pairing(pairs)`: Efficiently computes a product of pairings
//!
//! The pairing satisfies the bilinearity property:
//! - e(aP, bQ) = e(P, Q)^(ab) for scalars a, b
//! - e(P₁ + P₂, Q) = e(P₁, Q) · e(P₂, Q)
//!
//! # Examples
//!
//! ```rust
//! use p3_bn254::{Fr, G1, G2, pairing};
//! use p3_field::PrimeCharacteristicRing;
//!
//! // Work with G1 points
//! let g1_gen = G1::generator();
//! let scalar = Fr::from_u64(42);
//! let point = g1_gen.mul_scalar(scalar);
//!
//! // Multi-scalar multiplication
//! let points = vec![g1_gen, g1_gen];
//! let scalars = vec![Fr::from_u64(2), Fr::from_u64(3)];
//! let result = G1::multi_exp(&points, &scalars); // 2*G + 3*G = 5*G
//!
//! // Pairing operations
//! let g2_gen = G2::generator();
//! let gt_element = pairing(g1_gen, g2_gen);
//! ```

extern crate alloc;
use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use halo2curves::bn256::{
    G1 as Halo2G1, G1Affine as Halo2G1Affine, G2 as Halo2G2, G2Affine as Halo2G2Affine,
    Gt as Halo2Gt,
};
use halo2curves::group::GroupEncoding;
use halo2curves::group::{Curve, Group};
use halo2curves::msm::msm_best;
use halo2curves::pairing::{MillerLoopResult, MultiMillerLoop};

use crate::Fr;
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A point on the BN254 G1 curve (base curve over Fq)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct G1(pub(crate) Halo2G1);

/// A point on the BN254 G2 curve (twisted curve over Fq2)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct G2(pub(crate) Halo2G2);

/// An element in the BN254 Gt group (target group of the pairing, elements in Fq12)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Gt(pub(crate) Halo2Gt);

impl Serialize for G1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0.to_affine().to_bytes().as_ref())
    }
}

impl<'de> Deserialize<'de> for G1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let repr: <Halo2G1Affine as GroupEncoding>::Repr = bytes.as_slice().into();
        let affine = Option::<Halo2G1Affine>::from(Halo2G1Affine::from_bytes(&repr))
            .ok_or_else(|| DeError::custom("Invalid G1 point"))?;
        Ok(Self(Halo2G1::from(affine)))
    }
}

impl Serialize for G2 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0.to_affine().to_bytes().as_ref())
    }
}

impl<'de> Deserialize<'de> for G2 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let repr: <Halo2G2Affine as GroupEncoding>::Repr = bytes.as_slice().into();
        let affine = Option::<Halo2G2Affine>::from(Halo2G2Affine::from_bytes(&repr))
            .ok_or_else(|| DeError::custom("Invalid G2 point"))?;
        Ok(Self(Halo2G2::from(affine)))
    }
}

// ================================
// G1 Implementation
// ================================

impl G1 {
    /// Returns the identity element (point at infinity)
    pub fn identity() -> Self {
        Self(Halo2G1::identity())
    }

    /// Returns the generator point of G1
    pub fn generator() -> Self {
        Self(Halo2G1::generator())
    }

    /// Checks if this point is the identity
    pub fn is_identity(&self) -> bool {
        bool::from(self.0.is_identity())
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: Fr) -> Self {
        // Convert our Fr to halo2's Fr using unsafe transmute
        let halo2_fr = fr_to_halo2(scalar);
        Self(self.0 * halo2_fr)
    }

    /// Double this point
    pub fn double(&self) -> Self {
        Self(self.0.double())
    }

    /// Multi-scalar multiplication (MSM)
    /// Computes sum(scalars[i] * points[i]) efficiently
    ///
    /// # Panics
    /// Panics if `points` and `scalars` have different lengths
    pub fn multi_exp(points: &[Self], scalars: &[Fr]) -> Self {
        assert_eq!(
            points.len(),
            scalars.len(),
            "points and scalars must have the same length"
        );

        if points.is_empty() {
            return Self::identity();
        }

        // Convert points to affine coordinates
        let affine_points: Vec<Halo2G1Affine> = points.iter().map(|p| p.0.to_affine()).collect();

        // Convert scalars to halo2 format
        let halo2_scalars: Vec<halo2curves::bn256::Fr> =
            scalars.iter().map(|&s| fr_to_halo2(s)).collect();

        // Use halo2's optimized MSM
        let result = msm_best(&halo2_scalars, &affine_points);
        Self(result)
    }
}

impl Add for G1 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for G1 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for G1 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for G1 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Neg for G1 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Mul<Fr> for G1 {
    type Output = Self;

    fn mul(self, rhs: Fr) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl MulAssign<Fr> for G1 {
    fn mul_assign(&mut self, rhs: Fr) {
        *self = self.mul_scalar(rhs);
    }
}

// ================================
// G2 Implementation
// ================================

impl G2 {
    /// Returns the identity element (point at infinity)
    pub fn identity() -> Self {
        Self(Halo2G2::identity())
    }

    /// Returns the generator point of G2
    pub fn generator() -> Self {
        Self(Halo2G2::generator())
    }

    /// Checks if this point is the identity
    pub fn is_identity(&self) -> bool {
        bool::from(self.0.is_identity())
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: Fr) -> Self {
        // Convert our Fr to halo2's Fr using unsafe transmute
        let halo2_fr = fr_to_halo2(scalar);
        Self(self.0 * halo2_fr)
    }

    /// Double this point
    pub fn double(&self) -> Self {
        Self(self.0.double())
    }

    /// Multi-scalar multiplication (MSM)
    /// Computes sum(scalars[i] * points[i]) efficiently
    ///
    /// # Panics
    /// Panics if `points` and `scalars` have different lengths
    pub fn multi_exp(points: &[Self], scalars: &[Fr]) -> Self {
        assert_eq!(
            points.len(),
            scalars.len(),
            "points and scalars must have the same length"
        );

        if points.is_empty() {
            return Self::identity();
        }

        // Convert points to affine coordinates
        let affine_points: Vec<Halo2G2Affine> = points.iter().map(|p| p.0.to_affine()).collect();

        // Convert scalars to halo2 format
        let halo2_scalars: Vec<halo2curves::bn256::Fr> =
            scalars.iter().map(|&s| fr_to_halo2(s)).collect();

        // Use halo2's optimized MSM
        let result = msm_best(&halo2_scalars, &affine_points);
        Self(result)
    }
}

impl Add for G2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for G2 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for G2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for G2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Neg for G2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Mul<Fr> for G2 {
    type Output = Self;

    fn mul(self, rhs: Fr) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl MulAssign<Fr> for G2 {
    fn mul_assign(&mut self, rhs: Fr) {
        *self = self.mul_scalar(rhs);
    }
}

// ================================
// Gt Implementation
// ================================

impl Gt {
    /// Returns the identity element
    pub fn identity() -> Self {
        Self(Halo2Gt::identity())
    }

    /// Checks if this element is the identity
    pub fn is_identity(&self) -> bool {
        bool::from(self.0.is_identity())
    }

    /// Returns the generator element of Gt
    pub fn generator() -> Self {
        // The generator is e(G1::generator(), G2::generator())
        use halo2curves::bn256::Bn256;
        let g1_affine = Halo2G1::generator().to_affine();
        let g2_affine = Halo2G2::generator().to_affine();
        let miller_loop = Bn256::multi_miller_loop(&[(&g1_affine, &g2_affine)]);
        Self(miller_loop.final_exponentiation())
    }
}

impl Add for Gt {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Gt {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Gt {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Gt {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Neg for Gt {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Mul for Gt {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        // In the target group, multiplication is addition in the exponent
        // which corresponds to field multiplication in Fq12
        Self(self.0 + rhs.0)
    }
}

impl MulAssign for Gt {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn mul_assign(&mut self, rhs: Self) {
        // In the target group, multiplication is addition in the exponent
        self.0 += rhs.0;
    }
}

// ================================
// Pairing Operations
// ================================

/// Compute the pairing e(P, Q) where P ∈ G1 and Q ∈ G2
pub fn pairing(p: G1, q: G2) -> Gt {
    use halo2curves::bn256::Bn256;
    let p_affine = p.0.to_affine();
    let q_affine = q.0.to_affine();
    let miller_loop = Bn256::multi_miller_loop(&[(&p_affine, &q_affine)]);
    Gt(miller_loop.final_exponentiation())
}

/// Compute a multi-pairing (product of pairings)
/// Returns e(P1, Q1) * e(P2, Q2) * ... * e(Pn, Qn)
pub fn multi_pairing(pairs: &[(G1, G2)]) -> Gt {
    use halo2curves::bn256::Bn256;
    let halo2_pairs: Vec<_> = pairs
        .iter()
        .map(|(p, q)| {
            let p_affine = p.0.to_affine();
            let q_affine = q.0.to_affine();
            (p_affine, q_affine)
        })
        .collect();
    let refs: Vec<_> = halo2_pairs.iter().map(|(p, q)| (p, q)).collect();
    let miller_loop = Bn256::multi_miller_loop(&refs[..]);
    Gt(miller_loop.final_exponentiation())
}

// ================================
// Field Conversions
// ================================

/// Convert our Fr to halo2's Fr using unsafe transmute
///
/// # Safety
/// This is safe because both types have the same representation:
/// - Both are 4 u64 limbs in Montgomery form
/// - Both represent the same field with the same prime
/// - Both use the same Montgomery constant R = 2^256
#[inline]
fn fr_to_halo2(fr: Fr) -> halo2curves::bn256::Fr {
    // SAFETY: Both Fr types have the same memory layout (4 u64 limbs in Montgomery form)
    unsafe { core::mem::transmute(fr) }
}

/// Convert halo2's Fr to our Fr using unsafe transmute
///
/// # Safety
/// This is safe because both types have the same representation:
/// - Both are 4 u64 limbs in Montgomery form
/// - Both represent the same field with the same prime
/// - Both use the same Montgomery constant R = 2^256
#[inline]
pub fn fr_from_halo2(fr: halo2curves::bn256::Fr) -> Fr {
    // SAFETY: Both Fr types have the same memory layout (4 u64 limbs in Montgomery form)
    unsafe { core::mem::transmute(fr) }
}

// Helper for Gt scalar multiplication (exponentiation in the multiplicative group)
#[cfg(test)]
impl Gt {
    /// Scalar multiplication in Gt (exponentiation)
    /// Computes self^scalar in the multiplicative group
    fn mul_scalar_gt(&self, scalar: Fr) -> Self {
        // Gt multiplication is done by exponentiation in the multiplicative group
        // We need to compute self^scalar using halo2's built-in scalar multiplication
        let halo2_fr = fr_to_halo2(scalar);
        Self(self.0 * halo2_fr)
    }
}

// ================================
// Tests
// ================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_g1_identity() {
        let id = G1::identity();
        assert!(id.is_identity());
        assert_eq!(id + id, id);
    }

    #[test]
    fn test_g1_generator() {
        let g = G1::generator();
        assert!(!g.is_identity());

        // G + (-G) = identity
        assert_eq!(g + (-g), G1::identity());
    }

    #[test]
    fn test_g1_scalar_mul() {
        let g = G1::generator();
        let two = Fr::from_u8(2);
        let three = Fr::from_u8(3);

        // 2G + 3G = 5G
        let five_g = g.mul_scalar(Fr::from_u8(5));
        let two_g_plus_three_g = g.mul_scalar(two) + g.mul_scalar(three);
        assert_eq!(five_g, two_g_plus_three_g);
    }

    #[test]
    fn test_g2_identity() {
        let id = G2::identity();
        assert!(id.is_identity());
        assert_eq!(id + id, id);
    }

    #[test]
    fn test_g2_generator() {
        let g = G2::generator();
        assert!(!g.is_identity());

        // G + (-G) = identity
        assert_eq!(g + (-g), G2::identity());
    }

    #[test]
    fn test_gt_identity() {
        let id = Gt::identity();
        assert!(id.is_identity());
    }

    #[test]
    fn test_pairing_bilinearity() {
        let g1 = G1::generator();
        let g2 = G2::generator();

        let a = Fr::from_u8(3);
        let b = Fr::from_u8(5);

        // e(aG1, bG2) = e(G1, G2)^(ab)
        let left = pairing(g1.mul_scalar(a), g2.mul_scalar(b));
        let right = pairing(g1, g2.mul_scalar(b)).mul_scalar_gt(a);

        assert_eq!(left, right);
    }

    #[test]
    fn test_multi_pairing() {
        let g1 = G1::generator();
        let g2 = G2::generator();

        let a = Fr::from_u8(2);
        let b = Fr::from_u8(3);

        // e(aG1, G2) * e(G1, bG2) = e(G1, G2)^(a+b)
        let pairs = vec![(g1.mul_scalar(a), g2), (g1, g2.mul_scalar(b))];
        let left = multi_pairing(&pairs);

        let right = pairing(g1, g2).mul_scalar_gt(a + b);
        assert_eq!(left, right);
    }

    #[test]
    fn test_fr_conversion() {
        let our_fr = Fr::new(12345);
        let halo2_fr = fr_to_halo2(our_fr);
        let back = fr_from_halo2(halo2_fr);

        assert_eq!(our_fr, back);
    }

    #[test]
    fn test_g1_multi_exp() {
        let g1 = G1::generator();

        // Test empty input
        let empty: Vec<G1> = vec![];
        let empty_scalars: Vec<Fr> = vec![];
        assert_eq!(G1::multi_exp(&empty, &empty_scalars), G1::identity());

        // Test single point
        let scalar = Fr::from_u8(5);
        let result = G1::multi_exp(&[g1], &[scalar]);
        assert_eq!(result, g1.mul_scalar(scalar));

        // Test multiple points: 2G + 3G = 5G
        let two = Fr::from_u8(2);
        let three = Fr::from_u8(3);
        let result = G1::multi_exp(&[g1, g1], &[two, three]);
        let expected = g1.mul_scalar(Fr::from_u8(5));
        assert_eq!(result, expected);

        // Test with different points
        let g1_2 = g1.mul_scalar(Fr::from_u8(7));
        let g1_3 = g1.mul_scalar(Fr::from_u8(11));
        let a = Fr::from_u8(3);
        let b = Fr::from_u8(5);

        // MSM: 3*(7G) + 5*(11G) = 21G + 55G = 76G
        let result = G1::multi_exp(&[g1_2, g1_3], &[a, b]);
        let expected = g1.mul_scalar(Fr::from_u8(76));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_g2_multi_exp() {
        let g2 = G2::generator();

        // Test empty input
        let empty: Vec<G2> = vec![];
        let empty_scalars: Vec<Fr> = vec![];
        assert_eq!(G2::multi_exp(&empty, &empty_scalars), G2::identity());

        // Test single point
        let scalar = Fr::from_u8(5);
        let result = G2::multi_exp(&[g2], &[scalar]);
        assert_eq!(result, g2.mul_scalar(scalar));

        // Test multiple points: 2G + 3G = 5G
        let two = Fr::from_u8(2);
        let three = Fr::from_u8(3);
        let result = G2::multi_exp(&[g2, g2], &[two, three]);
        let expected = g2.mul_scalar(Fr::from_u8(5));
        assert_eq!(result, expected);

        // Test with different points
        let g2_2 = g2.mul_scalar(Fr::from_u8(7));
        let g2_3 = g2.mul_scalar(Fr::from_u8(11));
        let a = Fr::from_u8(3);
        let b = Fr::from_u8(5);

        // MSM: 3*(7G) + 5*(11G) = 21G + 55G = 76G
        let result = G2::multi_exp(&[g2_2, g2_3], &[a, b]);
        let expected = g2.mul_scalar(Fr::from_u8(76));
        assert_eq!(result, expected);
    }
}
