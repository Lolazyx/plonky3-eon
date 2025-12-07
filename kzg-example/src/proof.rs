//! Proof generation and verification for Fibonacci using KZG commitments

use p3_bn254::Fr as Bn254Fr;
use p3_challenger::DuplexChallenger;
use p3_field::Field;
use p3_kzg::KzgPcs;
use p3_symmetric::CryptographicPermutation;
use p3_uni_stark::{PcsError, Proof, StarkConfig, VerificationError, prove, verify};

use crate::FibonacciAir;

/// Type alias for the KZG-based STARK configuration with BN254
pub type KzgStarkConfig<Perm> = StarkConfig<KzgPcs, Bn254Fr, DuplexChallenger<Bn254Fr, Perm, 3, 2>>;

/// Result type for KZG-based Fibonacci proofs
pub type FibonacciProofResult<Perm> = Result<(), VerificationError<PcsError<KzgStarkConfig<Perm>>>>;

/// Generate and verify a STARK proof for the Fibonacci sequence using KZG commitments
///
/// # Arguments
///
/// * `air` - The Fibonacci AIR instance defining the computation
/// * `perm` - The Poseidon2 permutation for the Fiat-Shamir challenger
/// * `max_degree` - Maximum polynomial degree for KZG setup (should be >= trace size)
/// * `alpha` - The secret value for KZG setup (use proper trusted setup in production!)
///
/// # Returns
///
/// Returns `Ok(())` if the proof is generated and verified successfully, otherwise returns
/// a verification error.
///
/// # Security Warning
///
/// This function uses `alpha` directly for KZG setup, which is only suitable for testing.
/// In production, you must use a properly generated SRS from a trusted setup ceremony.
pub fn prove_fibonacci<Perm>(
    air: &FibonacciAir,
    perm: Perm,
    max_degree: usize,
    alpha: Bn254Fr,
) -> FibonacciProofResult<Perm>
where
    Perm: CryptographicPermutation<[Bn254Fr; 3]>
        + CryptographicPermutation<[<Bn254Fr as Field>::Packing; 3]>
        + Clone,
{
    // Create KZG PCS with the given parameters
    let pcs = KzgPcs::new(max_degree, alpha);

    // Generate the Fibonacci trace
    let trace = air.generate_trace();

    // Create the challenger for Fiat-Shamir
    let challenger = DuplexChallenger::new(perm.clone());

    // Create the STARK configuration
    let config = KzgStarkConfig::new(pcs, challenger);

    // Generate the proof
    let proof = prove(&config, air, trace, &[]);

    // Verify the proof
    verify_fibonacci_with_proof(&config, air, &proof)
}

/// Verify a STARK proof for the Fibonacci sequence
///
/// # Arguments
///
/// * `config` - The STARK configuration (PCS + Challenger)
/// * `air` - The Fibonacci AIR instance
/// * `proof` - The proof to verify
///
/// # Returns
///
/// Returns `Ok(())` if the proof is valid, otherwise returns a verification error.
pub fn verify_fibonacci_with_proof<Perm>(
    config: &KzgStarkConfig<Perm>,
    air: &FibonacciAir,
    proof: &Proof<KzgStarkConfig<Perm>>,
) -> FibonacciProofResult<Perm>
where
    Perm: CryptographicPermutation<[Bn254Fr; 3]>
        + CryptographicPermutation<[<Bn254Fr as Field>::Packing; 3]>,
{
    verify(config, air, proof, &[])
}

/// Verify a Fibonacci proof (convenience function that combines generation and verification)
///
/// This is a wrapper around [`prove_fibonacci`] for testing purposes.
pub fn verify_fibonacci<Perm>(
    air: &FibonacciAir,
    perm: Perm,
    max_degree: usize,
    alpha: Bn254Fr,
) -> FibonacciProofResult<Perm>
where
    Perm: CryptographicPermutation<[Bn254Fr; 3]>
        + CryptographicPermutation<[<Bn254Fr as Field>::Packing; 3]>
        + Clone,
{
    prove_fibonacci(air, perm, max_degree, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_bn254::Poseidon2Bn254;
    use p3_field::PrimeCharacteristicRing;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_fibonacci_proof() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Poseidon2Bn254::<3>::new_from_rng(8, 22, &mut rng);

        let air = FibonacciAir::new(16);
        let alpha = Bn254Fr::from_u64(12345);
        let max_degree = 1 << 10; // 1024

        let result = prove_fibonacci(&air, perm, max_degree, alpha);
        assert!(result.is_ok(), "Fibonacci proof should verify");
    }

    #[test]
    fn test_fibonacci_proof_larger() {
        let mut rng = SmallRng::seed_from_u64(123);
        let perm = Poseidon2Bn254::<3>::new_from_rng(8, 22, &mut rng);

        let air = FibonacciAir::new(64);
        let alpha = Bn254Fr::from_u64(67890);
        let max_degree = 1 << 12; // 4096

        let result = prove_fibonacci(&air, perm, max_degree, alpha);
        assert!(result.is_ok(), "Larger Fibonacci proof should verify");
    }
}
