//! BN254 elliptic curve implementation for Plonky3
//!
//! This crate provides a comprehensive implementation of the BN254 (also known as BN128 or alt_bn128)
//! elliptic curve, optimized for use within the Plonky3 zero-knowledge proof framework.
//!
//! # Overview
//!
//! BN254 is a pairing-friendly elliptic curve widely used in zero-knowledge proof systems due to its
//! efficient pairing computation and widespread support in blockchain ecosystems (notably Ethereum).
//!
//! ## Core Components
//!
//! - **`Fr`**: The scalar field of the BN254 curve
//!   - Prime order: `P = 21888242871839275222246405745257275088548364400416034343698204186575808495617`
//!   - ~254 bits (2^254 + small correction)
//!   - Implemented using Montgomery representation for efficient arithmetic
//!   - Two-adic subgroup of size 2^28 for efficient FFTs
//!
//! - **`G1`**: Points on the base curve E(Fq)
//!   - The curve equation: `y^2 = x^3 + 3` over the base field Fq
//!   - Uses affine and projective coordinates for point operations
//!   - Supports efficient multi-scalar multiplication (MSM)
//!
//! - **`G2`**: Points on the twisted curve E'(Fq2)
//!   - Defined over the quadratic extension field Fq2
//!   - Used in pairing operations
//!
//! - **`Gt`**: Elements in the target group (Fq12)
//!   - The output of the pairing operation
//!   - Elements of the 12th-degree extension field Fq12
//!
//! - **Pairing operations**: Bilinear map e: G1 × G2 → Gt
//!   - Supports single and multi-pairings
//!   - Essential for KZG commitments and verification
//!
//! ## Features
//!
//! - **No-std compatible**: Can be used in embedded and constrained environments
//! - **Montgomery arithmetic**: Optimized field operations using Montgomery representation
//! - **Halo2 integration**: Wraps halo2curves for curve operations with a Plonky3-compatible interface
//! - **Serialization**: Serde support for `Fr`, `G1`, and `G2`
//! - **Poseidon2 hash**: Native Poseidon2 hash function for BN254's Fr field
//!
//! # Examples
//!
//! ```rust
//! use p3_bn254::{Fr, G1, G2, pairing};
//! use p3_field::PrimeCharacteristicRing;
//!
//! // Create scalar field elements
//! let scalar = Fr::from_u64(42);
//!
//! // Work with curve points
//! let g1_gen = G1::generator();
//! let point = g1_gen.mul_scalar(scalar);
//!
//! // Pairing operations
//! let g2_gen = G2::generator();
//! let gt_element = pairing(g1_gen, g2_gen);
//! ```
//!
//! # Implementation Notes
//!
//! The field arithmetic for `Fr` is implemented from scratch using Montgomery multiplication,
//! while the curve operations for G1, G2, and Gt wrap the battle-tested `halo2curves` library
//! to ensure correctness and performance. This hybrid approach provides:
//!
//! - Native integration with Plonky3's field trait system
//! - Proven cryptographic security from halo2curves
//! - Optimal performance for both field and curve operations
#![no_std]

extern crate alloc;

mod curve;
mod field;
mod helpers;
mod poseidon2;

pub use curve::{G1, G2, Gt, fr_from_halo2, multi_pairing, pairing};
pub use field::*;
pub use poseidon2::Poseidon2Bn254;
