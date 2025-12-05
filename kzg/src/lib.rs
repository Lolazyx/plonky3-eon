#![no_std]

extern crate alloc;

pub mod mmcs;
pub mod params;
pub mod pcs;
mod util;

pub use mmcs::{KzgMmcs, KzgMmcsCommitment, KzgMmcsProof};
pub use params::{KzgError, KzgParams};
pub use pcs::{
    KzgCommitment, KzgPcs, KzgProof, MatrixCommitment, MatrixProof, PointProof,
    ProverData as PcsProverData,
};

#[cfg(test)]
mod tests;
