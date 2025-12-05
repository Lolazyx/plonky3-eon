//! BN254 elliptic curve implementation
//!
//! This crate provides:
//! - `Fr`: The scalar field of the BN254 curve, where `P = 21888242871839275222246405745257275088548364400416034343698204186575808495617`
//! - `G1`: Points on the base curve E(Fq)
//! - `G2`: Points on the twisted curve E'(Fq2)
//! - `Gt`: Elements in the target group (Fq12)
//! - Pairing operations
#![no_std]

extern crate alloc;

mod bn254;
mod curve;
mod helpers;
mod poseidon2;

pub use bn254::*;
pub use curve::{G1, G2, Gt, fr_from_halo2, multi_pairing, pairing};
pub use poseidon2::Poseidon2Bn254;
