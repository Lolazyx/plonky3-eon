//! KZG-based STARK proof example for Fibonacci sequence using BN254 field
//!
//! This crate demonstrates how to use Plonky3's KZG commitment scheme with the BN254
//! elliptic curve to generate and verify STARK proofs for Fibonacci sequence computation.
//!
//! # Overview
//!
//! The example proves the correct computation of Fibonacci numbers using:
//! - **Field**: BN254 Fr (scalar field of the BN254 curve)
//! - **Commitment Scheme**: KZG polynomial commitments
//! - **Hash Function**: Poseidon2 optimized for BN254
//!
//! # Components
//!
//! - [`FibonacciAir`]: The AIR (Algebraic Intermediate Representation) that defines
//!   the constraints for the Fibonacci recurrence relation
//! - [`prove_fibonacci`]: Generates a STARK proof for a Fibonacci sequence
//! - [`verify_fibonacci`]: Verifies a STARK proof
//!
//! # Example Usage
//!
//! ```no_run
//! use p3_kzg_example::{FibonacciAir, prove_fibonacci};
//! use p3_bn254::{Fr as Bn254Fr, Poseidon2Bn254};
//! use p3_field::PrimeCharacteristicRing;
//! use rand::SeedableRng;
//! use rand::rngs::SmallRng;
//!
//! let mut rng = SmallRng::seed_from_u64(42);
//! let perm = Poseidon2Bn254::<3>::new_from_rng(8, 22, &mut rng);
//!
//! let air = FibonacciAir::new(16);
//! let alpha = Bn254Fr::from_u64(12345);
//! let max_degree = 1024;
//!
//! let result = prove_fibonacci(&air, perm, max_degree, alpha);
//! assert!(result.is_ok());
//! ```

#![no_std]

extern crate alloc;

mod fibonacci_air;
mod proof;

pub use fibonacci_air::FibonacciAir;
pub use proof::{prove_fibonacci, verify_fibonacci};
