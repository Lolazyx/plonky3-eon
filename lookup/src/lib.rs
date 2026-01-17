//! Lookup Arguments for STARKs

#![no_std]

extern crate alloc;

mod error;
pub mod folder;
pub mod logup;
pub mod lookup_traits;
pub use error::LookupError;

#[cfg(test)]
mod tests;
