//! KZG polynomial commitment scheme for Plonky3 using BN254
//!
//! This crate implements the Kate-Zaverucha-Goldberg (KZG) polynomial commitment scheme,
//! a cryptographic primitive that allows a prover to commit to a polynomial and later
//! prove evaluations of that polynomial at arbitrary points.
//!
//! # Overview
//!
//! KZG commitments are a fundamental building block in modern zero-knowledge proof systems,
//! providing:
//!
//! - **Succinct commitments**: A polynomial of any degree is committed to a single group element
//! - **Efficient verification**: Polynomial evaluations can be verified using a single pairing check
//! - **Batch opening**: Multiple polynomials can be opened at multiple points with aggregated proofs
//!
//! This implementation uses the BN254 elliptic curve (via `p3-bn254`) for efficient pairing
//! operations and is designed to integrate seamlessly with the Plonky3 proving framework.
//!
//! # Main Components
//!
//! ## Polynomial Commitment Scheme (PCS)
//!
//! [`KzgPcs`] implements the [`Pcs`] trait from `p3-commit`, providing:
//!
//! - **Commitment**: Commit to polynomial evaluations on cosets
//! - **Opening**: Generate proofs for polynomial evaluations at specific points
//! - **Verification**: Verify opening proofs using pairing checks
//!
//! ## Merkle-Tree-like Commitment Scheme (MMCS)
//!
//! [`KzgMmcs`] implements the [`Mmcs`] trait, providing a vector commitment scheme:
//!
//! - Commit to matrices of field elements
//! - Open individual rows or entries with KZG proofs
//! - Useful for committing to execution traces in STARKs
//!
//! ## Setup Parameters
//!
//! [`KzgParams`] contains the structured reference string (SRS) or "trusted setup":
//!
//! - Powers of a secret value α in G1: [g₁, g₁^α, g₁^α², ..., g₁^α^n]
//! - α in G2: g₂^α
//! - Generated once and reused for all proofs (up to a maximum degree)
//!
//! # Security Considerations
//!
//! **Trusted Setup**: KZG requires a trusted setup ceremony to generate the SRS. The secret
//! value α must be securely discarded after setup. This implementation allows creating an
//! SRS from any field element for testing, but production use requires a proper multi-party
//! computation (MPC) ceremony.
//!
//! **Important**: The `KzgParams::new` method is for testing only. In production, use an
//! SRS from a trusted setup ceremony (e.g., Ethereum's KZG ceremony for EIP-4844).
//!
//! # Examples
//!
//! ## Basic PCS Usage
//!
//! ```no_run
//! extern crate alloc;
//! use alloc::vec;
//!
//! use p3_bn254::{Fr, Poseidon2Bn254};
//! use p3_challenger::DuplexChallenger;
//! use p3_commit::Pcs;
//! use p3_kzg::KzgPcs;
//! use p3_matrix::dense::RowMajorMatrix;
//! use p3_field::PrimeCharacteristicRing;
//!
//! type Perm = Poseidon2Bn254<3>;
//! type Chal = DuplexChallenger<Fr, Perm, 3, 2>;
//!
//! let pcs = KzgPcs::new(8, Fr::new(7));
//! let domain = <KzgPcs as Pcs<Fr, Chal>>::natural_domain_for_degree(&pcs, 4);
//! let values = vec![Fr::new(1), Fr::new(2), Fr::new(3), Fr::new(4)];
//! let matrix = RowMajorMatrix::new(values, 1);
//! let (commitment, prover_data) = <KzgPcs as Pcs<Fr, Chal>>::commit(&pcs, [(domain, matrix)]);
//! ```
//!
//! ## MMCS Usage
//!
//! ```no_run
//! use p3_bn254::Fr;
//! use p3_kzg::KzgMmcs;
//! use p3_matrix::dense::RowMajorMatrix;
//! use p3_field::PrimeCharacteristicRing;
//! use p3_commit::Mmcs;
//!
//! let mmcs = KzgMmcs::new(8, Fr::new(5));
//! let matrix = RowMajorMatrix::new(vec![Fr::new(1), Fr::new(2), Fr::new(3), Fr::new(4)], 2);
//! let (commitment, prover_data) = mmcs.commit(vec![matrix]);
//! let _opening = mmcs.open_batch(0, &prover_data);
//! ```
//!
//! # References
//!
//! - Original KZG paper: "Constant-Size Commitments to Polynomials and Their Applications"
//!   by Kate, Zaverucha, and Goldberg (2010)
//! - [EIP-4844](https://eips.ethereum.org/EIPS/eip-4844): Proto-Danksharding using KZG commitments

#![no_std]

extern crate alloc;

pub mod mmcs;
pub mod params;
pub mod pcs;

// Re-export util for benchmarking purposes
#[doc(hidden)]
pub mod util;

pub use mmcs::{KzgMmcs, KzgMmcsCommitment, KzgMmcsProof};
pub use params::{init_srs_unsafe, KzgError, KzgParams, StructuredReferenceString};
pub use pcs::{
    KzgCommitment, KzgPcs, KzgProof, MatrixCommitment, MatrixProof, PointProof,
    ProverData as PcsProverData,
};

#[cfg(test)]
mod tests;
