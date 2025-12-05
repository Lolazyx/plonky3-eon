use alloc::vec::Vec;

use p3_bn254::{Fr, G1};
use p3_commit::{OpenedValues, Pcs};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use crate::params::{KzgError, KzgParams};
use crate::util::{commit_column, eval_poly, quotient_and_eval, verify_single};

/// KZG commitment for a single matrix.
///
/// Each column of the matrix is treated as a polynomial in coefficient form,
/// and each polynomial is committed separately using KZG. This allows for
/// efficient column-wise opening of the matrix.
#[derive(Clone, Serialize, Deserialize)]
pub struct MatrixCommitment {
    /// KZG commitments to each column polynomial.
    ///
    /// For a matrix with `w` columns, this vector contains `w` G1 elements,
    /// where `columns[i]` is the KZG commitment to the polynomial represented
    /// by the i-th column.
    pub columns: Vec<G1>,
}

/// Commitment to a batch of matrices.
///
/// In the PCS protocol, multiple matrices may be committed in a single round.
/// This structure holds all matrix commitments for one commitment phase.
#[derive(Clone, Serialize, Deserialize)]
pub struct KzgCommitment {
    /// Commitments to each matrix in the batch.
    pub matrices: Vec<MatrixCommitment>,
}

/// Prover-side data for a committed matrix.
///
/// Contains all the information the prover needs to generate opening proofs
/// for a committed matrix, including both evaluation and coefficient forms
/// of the polynomials.
#[derive(Clone)]
pub struct MatrixProverData {
    /// The two-adic coset domain over which the polynomials are evaluated.
    pub domain: TwoAdicMultiplicativeCoset<Fr>,

    /// The matrix in evaluation form (values at domain points).
    ///
    /// Each row corresponds to an evaluation point in the domain,
    /// and each column represents a polynomial's values.
    pub evals: RowMajorMatrix<Fr>,

    /// The matrix in coefficient form (polynomial coefficients).
    ///
    /// Obtained by applying iDFT to the evaluation form. Used to
    /// compute polynomial evaluations at arbitrary points and to
    /// generate opening proofs.
    pub coeffs: RowMajorMatrix<Fr>,
}

/// Prover data for all committed matrices.
pub type ProverData = Vec<MatrixProverData>;

/// Opening proof for multiple points on a single matrix.
///
/// Contains opening proofs for potentially multiple evaluation points,
/// where each point may open multiple columns of the matrix.
#[derive(Clone, Serialize, Deserialize)]
pub struct MatrixProof {
    /// Proofs for each evaluation point.
    pub points: Vec<PointProof>,
}

/// Opening proof for a single evaluation point.
///
/// In the KZG scheme, to prove that a polynomial `f(X)` evaluates to `v` at point `z`,
/// the prover provides a witness `w = (f(X) - v) / (X - z)`, which is itself a polynomial
/// commitment. This structure contains one such witness per opened column.
#[derive(Clone, Serialize, Deserialize)]
pub struct PointProof {
    /// KZG witnesses (quotient polynomial commitments) for each opened column.
    ///
    /// For each column polynomial `f_i(X)` being opened at point `z` with claimed
    /// value `v_i`, this contains the commitment to the quotient polynomial:
    /// `q_i(X) = (f_i(X) - v_i) / (X - z)`
    pub witnesses: Vec<G1>,
}

/// Complete opening proof matching the [`OpenedValues`] structure.
///
/// The PCS protocol may involve multiple rounds of commitment and opening.
/// This structure organizes all opening proofs hierarchically to match the
/// structure of opened values.
#[derive(Clone, Serialize, Deserialize)]
pub struct KzgProof {
    /// Proofs organized by round, then by batch within each round.
    ///
    /// `rounds[i][j]` contains the proof for the j-th batch in the i-th round.
    pub rounds: Vec<Vec<MatrixProof>>,
}

/// KZG polynomial commitment scheme implementation.
///
/// This implements the [`Pcs`] trait from `p3-commit`, providing a complete
/// polynomial commitment scheme using KZG commitments over the BN254 curve.
///
/// # How It Works
///
/// 1. **Commitment**: Polynomials are represented as matrices where each column
///    is a polynomial in coefficient form. The scheme:
///    - Takes matrices in evaluation form on two-adic cosets
///    - Converts to coefficient form using iDFT
///    - Commits to each column polynomial using KZG
///
/// 2. **Opening**: To prove evaluations at specific points:
///    - Evaluates each column polynomial at the requested point
///    - Computes quotient polynomials `q(X) = (f(X) - v) / (X - z)`
///    - Commits to the quotient polynomials as witnesses
///
/// 3. **Verification**: Uses a pairing check to verify that the commitment,
///    witness, and claimed evaluation are consistent:
///    `e(C - v·G₁, G₂) = e(W, α·G₂ - z·G₂)`
///
/// # Example
///
/// ```rust
/// use p3_bn254::Fr;
/// use p3_kzg::KzgPcs;
/// use p3_field::PrimeCharacteristicRing;
///
/// // Create a KZG PCS instance
/// let max_degree = 1024;
/// let alpha = Fr::from_u64(12345); // Testing only!
/// let pcs = KzgPcs::new(max_degree, alpha);
///
/// // Use with Plonky3's PCS trait...
/// ```
#[derive(Clone)]
pub struct KzgPcs {
    /// The KZG parameters (structured reference string).
    pub params: KzgParams,

    /// DFT implementation for converting between evaluation and coefficient forms.
    pub dft: Radix2Dit<Fr>,
}

impl KzgPcs {
    /// Creates a new KZG PCS instance with the given parameters.
    ///
    /// **Warning**: This method generates a trusted setup for testing purposes only.
    /// In production, use parameters from a proper trusted setup ceremony.
    ///
    /// # Arguments
    ///
    /// * `max_degree` - Maximum degree of polynomials that can be committed
    /// * `alpha` - Secret value for generating the SRS (toxic waste - discard after use!)
    ///
    /// # Example
    ///
    /// ```rust
    /// use p3_bn254::Fr;
    /// use p3_kzg::KzgPcs;
    /// use p3_field::PrimeCharacteristicRing;
    ///
    /// let pcs = KzgPcs::new(1024, Fr::from_u64(999));
    /// ```
    #[must_use]
    pub fn new(max_degree: usize, alpha: Fr) -> Self {
        Self {
            params: KzgParams::new(max_degree, alpha),
            dft: Radix2Dit::default(),
        }
    }

    fn commit_column(&self, coeffs: &[Fr]) -> Result<G1, KzgError> {
        commit_column(&self.params, coeffs)
    }
}

impl<Challenger> Pcs<Fr, Challenger> for KzgPcs {
    type Domain = TwoAdicMultiplicativeCoset<Fr>;
    type Commitment = KzgCommitment;
    type ProverData = ProverData;
    type EvaluationsOnDomain<'a> = RowMajorMatrix<Fr>;
    type Proof = KzgProof;
    type Error = KzgError;

    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        let log_n = log2_strict_usize(degree.next_power_of_two());
        TwoAdicMultiplicativeCoset::new(Fr::ONE, log_n).expect("valid domain")
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Fr>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let mut commitments = Vec::new();
        let mut prover_mats = Vec::new();

        for (domain, evals) in evaluations {
            let height = evals.height();
            let width = evals.width();
            assert_eq!(
                height,
                domain.size(),
                "evaluation height must match domain size"
            );
            self.params
                .ensure_supported(height.saturating_sub(1))
                .unwrap();

            let coeffs = self.dft.coset_idft_batch(evals.clone(), domain.shift());

            let columns = (0..width)
                .map(|col| {
                    let col_coeffs: Vec<_> = coeffs.row_slices().map(|row| row[col]).collect();
                    self.commit_column(&col_coeffs).unwrap()
                })
                .collect();

            commitments.push(MatrixCommitment { columns });
            prover_mats.push(MatrixProverData {
                domain,
                evals,
                coeffs,
            });
        }

        (
            KzgCommitment {
                matrices: commitments,
            },
            prover_mats,
        )
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let matrix = &prover_data[idx];
        if matrix.domain.shift() == domain.shift() && matrix.domain.size() == domain.size() {
            return matrix.evals.clone();
        }

        let width = matrix.coeffs.width();
        let mut values = Vec::with_capacity(domain.size() * width);
        for point in domain.iter() {
            for col in 0..width {
                let col_coeffs: Vec<_> = matrix.coeffs.row_slices().map(|row| row[col]).collect();
                values.push(eval_poly(&col_coeffs, point));
            }
        }
        RowMajorMatrix::new(values, width)
    }

    fn open(
        &self,
        commitment_data_with_opening_points: Vec<(&Self::ProverData, Vec<Vec<Fr>>)>,
        _fiat_shamir_challenger: &mut Challenger,
    ) -> (OpenedValues<Fr>, Self::Proof) {
        let mut opened_values = Vec::new();
        let mut rounds = Vec::new();

        for (prover_data, points_per_matrix) in commitment_data_with_opening_points {
            assert_eq!(prover_data.len(), points_per_matrix.len());
            let mut matrix_values = Vec::new();
            let mut matrix_proofs = Vec::new();

            for (matrix, points) in prover_data.iter().zip(points_per_matrix) {
                let mut values_for_matrix = Vec::new();
                let mut proofs_for_matrix = Vec::new();

                for point in points {
                    let base_point = point;

                    let mut evals = Vec::new();
                    let mut witnesses = Vec::new();
                    for col in 0..matrix.coeffs.width() {
                        let col_coeffs: Vec<_> =
                            matrix.coeffs.row_slices().map(|row| row[col]).collect();
                        let (quotient, value) = quotient_and_eval(&col_coeffs, base_point);
                        evals.push(value);
                        let witness = self.commit_column(&quotient).unwrap();
                        witnesses.push(witness);
                    }

                    values_for_matrix.push(evals);
                    proofs_for_matrix.push(PointProof { witnesses });
                }

                matrix_values.push(values_for_matrix);
                matrix_proofs.push(MatrixProof {
                    points: proofs_for_matrix,
                });
            }

            opened_values.push(matrix_values);
            rounds.push(matrix_proofs);
        }

        (opened_values, KzgProof { rounds })
    }

    fn verify(
        &self,
        commitments_with_opening_points: Vec<(
            Self::Commitment,
            Vec<(Self::Domain, Vec<(Fr, Vec<Fr>)>)>,
        )>,
        proof: &Self::Proof,
        _fiat_shamir_challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        if proof.rounds.len() != commitments_with_opening_points.len() {
            return Err(KzgError::ProofShapeMismatch);
        }

        for ((commitment, matrices), matrix_proofs) in commitments_with_opening_points
            .into_iter()
            .zip(&proof.rounds)
        {
            if matrix_proofs.len() != matrices.len() {
                return Err(KzgError::ProofShapeMismatch);
            }

            for (((domain, openings), matrix_proof), matrix_commitment) in matrices
                .into_iter()
                .zip(matrix_proofs)
                .zip(commitment.matrices)
            {
                self.params
                    .ensure_supported(domain.size().saturating_sub(1))?;

                if openings.len() != matrix_proof.points.len() {
                    return Err(KzgError::ProofShapeMismatch);
                }

                for ((point, values), point_proof) in openings.into_iter().zip(&matrix_proof.points)
                {
                    let base_point = point;

                    if values.len() != point_proof.witnesses.len()
                        || values.len() != matrix_commitment.columns.len()
                    {
                        return Err(KzgError::ProofShapeMismatch);
                    }

                    for ((value, witness), column_commitment) in values
                        .into_iter()
                        .zip(&point_proof.witnesses)
                        .zip(&matrix_commitment.columns)
                    {
                        verify_single(column_commitment, witness, value, base_point, &self.params)?;
                    }
                }
            }
        }

        Ok(())
    }
}
