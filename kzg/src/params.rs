use alloc::vec::Vec;

use p3_bn254::{Fr, G1, G2};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

/// The trusted setup for KZG.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct KzgParams {
    /// Powers of `alpha` in G1: `[g1, g1^{alpha}, g1^{alpha^2}, ...]`.
    pub g1_powers: Vec<G1>,
    /// `g2^{alpha}` used by the verifier.
    pub g2_alpha: G2,
    /// Maximum supported degree (inclusive).
    pub max_degree: usize,
}

impl KzgParams {
    /// Create a new SRS using the provided toxic waste `alpha`.
    #[must_use]
    pub fn new(max_degree: usize, alpha: Fr) -> Self {
        let g1 = G1::generator();
        let g2 = G2::generator();

        let mut g1_powers = Vec::with_capacity(max_degree + 1);
        let mut power = Fr::ONE;
        for _ in 0..=max_degree {
            g1_powers.push(g1.mul_scalar(power));
            power *= alpha;
        }

        Self {
            g1_powers,
            g2_alpha: g2.mul_scalar(alpha),
            max_degree,
        }
    }

    pub(crate) fn ensure_supported(&self, degree: usize) -> Result<(), KzgError> {
        if degree > self.max_degree {
            Err(KzgError::DegreeTooLarge {
                degree,
                max: self.max_degree,
            })
        } else {
            Ok(())
        }
    }
}

/// Errors surfaced by the KZG PCS/MMCS.
#[derive(Debug, Error, Clone)]
pub enum KzgError {
    #[error("proof has unexpected shape")]
    ProofShapeMismatch,
    #[error("domain height exceeds SRS degree bound")]
    DegreeTooLarge { degree: usize, max: usize },
}
