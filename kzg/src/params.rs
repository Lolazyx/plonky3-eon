use alloc::vec::Vec;

use p3_bn254::{Fr, G1, G2};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

/// Structured Reference String (SRS) for KZG commitments, also known as the "trusted setup".
///
/// The KZG commitment scheme requires a one-time setup that generates public parameters
/// encoding powers of a secret value α. This secret must be discarded after setup to ensure
/// security - if an adversary learns α, they can create fraudulent proofs.
///
/// # Structure
///
/// The SRS consists of:
///
/// - **g1_powers**: Encodings of powers of α in the G1 group: [g₁, g₁^α, g₁^α², ..., g₁^α^n]
///   - Used by the prover to create polynomial commitments
///   - The number of powers determines the maximum polynomial degree that can be committed
///
/// - **g2_alpha**: Encoding of α in the G2 group: g₂^α
///   - Used by the verifier in pairing checks
///   - Only a single G2 element is needed (unlike the many G1 powers)
///
/// - **max_degree**: Maximum degree of polynomials that can be committed
///   - Equal to `g1_powers.len() - 1`
///   - Attempting to commit to higher degree polynomials will fail
///
/// # Security
///
/// **Critical**: The secret value α used to generate this SRS must be securely destroyed
/// after generation. In production systems, the SRS should be generated via a multi-party
/// computation (MPC) ceremony where no single party learns α.
///
/// # Production vs Testing
///
/// - **Testing**: Use `init_srs_unsafe(max_degree, alpha)` to generate an SRS from any field element
/// - **Production**: Load an SRS from a trusted setup ceremony (e.g., Ethereum's KZG ceremony)
///
/// # Example
///
/// ```rust
/// use p3_bn254::Fr;
/// use p3_kzg::init_srs_unsafe;
/// use p3_field::PrimeCharacteristicRing;
///
/// // For testing only - in production, use a trusted ceremony
/// let max_degree = 1024;
/// let secret_alpha = Fr::from_u64(999); // This must be discarded!
/// let srs = init_srs_unsafe(max_degree, secret_alpha);
///
/// // The SRS can now commit to polynomials up to degree 1024
/// assert_eq!(srs.max_degree, 1024);
/// assert_eq!(srs.g1_powers.len(), 1025); // Degree 0 to 1024
/// ```
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct StructuredReferenceString {
    /// Powers of the secret α in G1: [g₁, g₁^α, g₁^α², ..., g₁^α^max_degree]
    ///
    /// These are used to commit to polynomials by computing:
    /// `C(f) = f(α) * g₁ = Σᵢ fᵢ * g₁^(α^i)`
    ///
    /// The length of this vector is `max_degree + 1` to include powers from 0 to max_degree.
    pub g1_powers: Vec<G1>,

    /// The secret α encoded in G2: g₂^α
    ///
    /// Used by verifiers in pairing checks to verify opening proofs. The verification
    /// equation uses this to check that a commitment matches a claimed evaluation.
    pub g2_alpha: G2,

    /// Maximum polynomial degree supported by this SRS (inclusive)
    ///
    /// Polynomials of degree up to and including this value can be committed.
    /// Attempting to commit to a polynomial of higher degree will return an error.
    pub max_degree: usize,
}

/// Type alias for backward compatibility
pub type KzgParams = StructuredReferenceString;

/// Initialize a Structured Reference String (SRS) for testing purposes.
///
/// **WARNING**: This function is UNSAFE for production use! It generates an SRS from a
/// known secret value α, which compromises security. In production, use an SRS from a
/// trusted setup ceremony where α is generated via multi-party computation and destroyed.
///
/// # Arguments
///
/// * `max_degree` - Maximum degree of polynomials that can be committed (inclusive)
/// * `alpha` - The "toxic waste" secret value used to generate the SRS
///
/// # Returns
///
/// A `StructuredReferenceString` containing:
/// - `g1_powers`: [g₁, g₁^α, g₁^α², ..., g₁^α^max_degree] (max_degree + 1 elements)
/// - `g2_alpha`: g₂^α
/// - `max_degree`: The maximum degree parameter
///
/// # Security
///
/// The `alpha` parameter is called "toxic waste" because knowledge of this value
/// allows creating fraudulent proofs. In a real system:
///
/// 1. α should be generated via a multi-party computation (MPC) ceremony
/// 2. All participants' secret shares should be destroyed immediately after setup
/// 3. The final SRS should be verified for correctness
///
/// # Example
///
/// ```rust
/// use p3_bn254::Fr;
/// use p3_kzg::init_srs_unsafe;
/// use p3_field::PrimeCharacteristicRing;
///
/// // Generate test parameters (DO NOT use in production)
/// let srs = init_srs_unsafe(16, Fr::from_u64(12345));
///
/// // Can commit to polynomials of degree 0 to 16
/// assert_eq!(srs.max_degree, 16);
/// ```
#[must_use]
pub fn init_srs_unsafe(max_degree: usize, alpha: Fr) -> StructuredReferenceString {
    let g1 = G1::generator();
    let g2 = G2::generator();

    let mut g1_powers = Vec::with_capacity(max_degree + 1);
    let mut power = Fr::ONE;
    for _ in 0..=max_degree {
        g1_powers.push(g1.mul_scalar(power));
        power *= alpha;
    }

    StructuredReferenceString {
        g1_powers,
        g2_alpha: g2.mul_scalar(alpha),
        max_degree,
    }
}

impl StructuredReferenceString {
    /// Checks whether a polynomial of the given degree can be committed with this SRS.
    ///
    /// # Arguments
    ///
    /// * `degree` - The degree of the polynomial to check
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the degree is supported (degree ≤ max_degree)
    /// * `Err(KzgError::DegreeTooLarge)` if the degree exceeds the SRS capacity
    ///
    /// # Example
    ///
    /// ```rust
    /// use p3_bn254::Fr;
    /// use p3_kzg::init_srs_unsafe;
    ///
    /// let srs = init_srs_unsafe(100, Fr::new(999));
    /// assert!(srs.ensure_supported(50).is_ok());
    /// assert!(srs.ensure_supported(100).is_ok());
    /// assert!(srs.ensure_supported(101).is_err());
    /// ```
    pub fn ensure_supported(&self, degree: usize) -> Result<(), KzgError> {
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

/// Errors that can occur during KZG commitment and verification operations.
#[derive(Debug, Error, Clone)]
pub enum KzgError {
    /// The proof structure doesn't match expected format or verification failed.
    ///
    /// This can occur when:
    /// - The proof has the wrong number of elements
    /// - The pairing check fails during verification (indicating an invalid proof)
    /// - The commitment and proof structures are incompatible
    #[error("proof has unexpected shape")]
    ProofShapeMismatch,

    /// Attempted to commit to a polynomial whose degree exceeds the SRS capacity.
    ///
    /// The KZG parameters (SRS) support polynomials up to a maximum degree determined
    /// during the trusted setup. This error occurs when trying to commit to a polynomial
    /// of higher degree.
    ///
    /// # Solution
    ///
    /// Either:
    /// - Use a larger SRS with higher `max_degree`
    /// - Reduce the polynomial degree (e.g., by working with smaller domains)
    #[error(
        "domain height exceeds SRS degree bound: tried to commit to degree {degree}, but max is {max}"
    )]
    DegreeTooLarge {
        /// The degree that was attempted
        degree: usize,
        /// The maximum degree supported by the SRS
        max: usize,
    },
}
