//! A framework for various (not necessarily hiding) cryptographic commitment schemes.

#![no_std]

extern crate alloc;

// mod adapters;
mod domain;
mod dummy_mmcs;
mod dummy_pcs;
mod mmcs;
mod pcs;

#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

// pub use adapters::*;
pub use domain::*;
pub use dummy_mmcs::*;
pub use dummy_pcs::*;
pub use mmcs::*;
pub use pcs::*;
