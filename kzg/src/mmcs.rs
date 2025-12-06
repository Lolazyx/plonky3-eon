use alloc::vec::Vec;

use p3_bn254::{Fr, G1};
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_matrix::{Dimensions, Matrix};
use serde::{Deserialize, Serialize};

use crate::params::{KzgError, KzgParams, StructuredReferenceString};
use crate::util::{commit_column, quotient_and_eval, verify_batch, OpeningInfo};

/// KZG-based Merkle-tree-like commitment scheme (MMCS).
///
/// This structure implements the [`Mmcs`] trait from `p3-commit`, providing a vector
/// commitment scheme using KZG. Unlike traditional Merkle trees that use hashing,
/// this scheme uses polynomial commitments, offering algebraic properties useful
/// in zero-knowledge proofs.
///
/// # Use Case
///
/// MMCS is particularly useful for committing to execution traces in STARK proofs.
/// Each matrix represents a trace table, where:
/// - Rows correspond to execution steps
/// - Columns correspond to registers or state variables
///
/// # How It Works
///
/// 1. **Commitment**: Each column of a matrix is treated as a polynomial in
///    coefficient form and committed using KZG
/// 2. **Opening**: Individual rows can be opened by providing KZG opening proofs
///    for all column polynomials at the row index
/// 3. **Verification**: Uses pairing checks to verify the openings
///
/// # Example
///
/// ```rust
/// use p3_bn254::Fr;
/// use p3_kzg::KzgMmcs;
/// use p3_commit::Mmcs;
/// use p3_matrix::dense::RowMajorMatrix;
/// use p3_field::PrimeCharacteristicRing;
///
/// let mmcs = KzgMmcs::new(1024, Fr::from_u64(999));
///
/// // Commit to a 4x2 matrix (4 rows, 2 columns)
/// let values = vec![
///     Fr::from_u64(1), Fr::from_u64(2),
///     Fr::from_u64(3), Fr::from_u64(4),
///     Fr::from_u64(5), Fr::from_u64(6),
///     Fr::from_u64(7), Fr::from_u64(8),
/// ];
/// let matrix = RowMajorMatrix::new(values, 2);
/// let (commitment, prover_data) = mmcs.commit(vec![matrix]);
///
/// // Open row 0
/// let opening = mmcs.open_batch(0, &prover_data);
/// ```
#[derive(Clone)]
pub struct KzgMmcs {
    /// The KZG parameters (structured reference string).
    pub params: KzgParams,
}

/// Commitment to a batch of matrices in the MMCS scheme.
///
/// Contains KZG commitments for all columns of all committed matrices.
#[derive(Clone, Serialize, Deserialize)]
pub struct KzgMmcsCommitment {
    /// Commitments to each matrix in the batch.
    pub matrices: Vec<MatrixCommitment>,
}

/// Opening proof for a batch of matrices at a specific index.
///
/// Contains KZG witnesses proving that the opened values are correct
/// for all columns across all matrices.
#[derive(Clone, Serialize, Deserialize)]
pub struct KzgMmcsProof {
    /// Witnesses organized by matrix, then by column.
    ///
    /// `witnesses[i][j]` is the KZG witness for opening the j-th column
    /// of the i-th matrix at the specified row index.
    pub witnesses: Vec<Vec<G1>>,
}

/// Prover data for committed matrices.
///
/// Stores the original matrices so the prover can generate opening proofs.
#[derive(Clone)]
pub struct KzgMmcsProverData<M> {
    /// The committed matrices in their original form.
    matrices: Vec<M>,
}

/// KZG commitment to a single matrix.
///
/// Each column is committed as a separate polynomial.
#[derive(Clone, Serialize, Deserialize)]
pub struct MatrixCommitment {
    /// KZG commitments to each column polynomial.
    ///
    /// `columns[i]` is the commitment to the polynomial formed by
    /// the i-th column of the matrix.
    pub columns: Vec<G1>,
}

impl KzgMmcs {
    /// Creates a new KZG MMCS instance from a Structured Reference String.
    ///
    /// # Arguments
    ///
    /// * `srs` - The Structured Reference String (trusted setup parameters)
    ///
    /// # Example
    ///
    /// ```rust
    /// use p3_bn254::Fr;
    /// use p3_kzg::{KzgMmcs, init_srs_unsafe};
    /// use p3_field::PrimeCharacteristicRing;
    ///
    /// // For testing only - use a trusted setup in production
    /// let srs = init_srs_unsafe(1024, Fr::from_u64(999));
    /// let mmcs = KzgMmcs::from_srs(srs);
    /// ```
    #[must_use]
    pub fn from_srs(srs: StructuredReferenceString) -> Self {
        Self { params: srs }
    }

    /// Creates a new KZG MMCS instance by generating an unsafe SRS for testing.
    ///
    /// **WARNING**: This method is for testing purposes only! In production, use
    /// `from_srs()` with parameters from a proper trusted setup ceremony.
    ///
    /// # Arguments
    ///
    /// * `max_degree` - Maximum degree of polynomials (= maximum matrix height - 1)
    /// * `alpha` - Secret value for generating the SRS (toxic waste - discard after use!)
    ///
    /// # Example
    ///
    /// ```rust
    /// use p3_bn254::Fr;
    /// use p3_kzg::KzgMmcs;
    /// use p3_field::PrimeCharacteristicRing;
    ///
    /// // For testing only
    /// let mmcs = KzgMmcs::new(1024, Fr::from_u64(999));
    /// ```
    #[must_use]
    pub fn new(max_degree: usize, alpha: Fr) -> Self {
        use crate::init_srs_unsafe;
        Self::from_srs(init_srs_unsafe(max_degree, alpha))
    }

    fn commit_matrix<M: Matrix<Fr>>(&self, matrix: &M) -> MatrixCommitment {
        let columns = (0..matrix.width())
            .map(|col| {
                let coeffs: Vec<_> = (0..matrix.height())
                    .map(|r| matrix.row_slice(r).unwrap()[col])
                    .collect();
                commit_column(&self.params, &coeffs).unwrap()
            })
            .collect();
        MatrixCommitment { columns }
    }
}

impl Mmcs<Fr> for KzgMmcs {
    type ProverData<M> = KzgMmcsProverData<M>;
    type Commitment = KzgMmcsCommitment;
    type Proof = KzgMmcsProof;
    type Error = KzgError;

    fn commit<M: Matrix<Fr>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        let mut matrices = Vec::with_capacity(inputs.len());
        let mut commitments = Vec::with_capacity(inputs.len());

        for mat in inputs {
            let height = mat.height();
            self.params
                .ensure_supported(height.saturating_sub(1))
                .unwrap();

            commitments.push(self.commit_matrix(&mat));
            matrices.push(mat);
        }

        (
            KzgMmcsCommitment {
                matrices: commitments,
            },
            KzgMmcsProverData { matrices },
        )
    }

    fn open_batch<M: Matrix<Fr>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<Fr, Self> {
        let max_height = prover_data
            .matrices
            .iter()
            .map(Matrix::height)
            .max()
            .unwrap_or(0);
        let log2_max_height = max_height.next_power_of_two().trailing_zeros() as usize;

        let mut opened_values = Vec::new();
        let mut witnesses = Vec::new();

        for matrix in &prover_data.matrices {
            let log2_height = matrix.height().next_power_of_two().trailing_zeros() as usize;
            let local_index = if log2_max_height >= log2_height {
                index >> (log2_max_height - log2_height)
            } else {
                index
            } % matrix.height();

            let point = Fr::new(local_index as u64);
            let mut row = Vec::with_capacity(matrix.width());
            let mut matrix_witnesses = Vec::with_capacity(matrix.width());
            for col in 0..matrix.width() {
                let coeffs: Vec<_> = (0..matrix.height())
                    .map(|r| matrix.row_slice(r).unwrap()[col])
                    .collect();
                let (quotient, value) = quotient_and_eval(&coeffs, point);
                row.push(value);
                matrix_witnesses.push(commit_column(&self.params, &quotient).unwrap());
            }
            opened_values.push(row);
            witnesses.push(matrix_witnesses);
        }

        BatchOpening::new(opened_values, KzgMmcsProof { witnesses })
    }

    fn get_matrices<'a, M: Matrix<Fr>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        prover_data.matrices.iter().collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, Fr, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, proof) = batch_opening.unpack();
        if opened_values.len() != commit.matrices.len() {
            return Err(KzgError::ProofShapeMismatch);
        }

        let max_height = dimensions.iter().map(|d| d.height).max().unwrap_or(0);
        let log2_max_height = max_height.next_power_of_two().trailing_zeros() as usize;

        // Collect all openings to verify in a single batch
        let mut all_openings = Vec::new();

        for (((values, commitment), dims), witnesses) in opened_values
            .iter()
            .zip(&commit.matrices)
            .zip(dimensions)
            .zip(&proof.witnesses)
        {
            if values.len() != commitment.columns.len() || values.len() != witnesses.len() {
                return Err(KzgError::ProofShapeMismatch);
            }
            self.params
                .ensure_supported(dims.height.saturating_sub(1))?;

            let log2_height = dims.height.next_power_of_two().trailing_zeros() as usize;
            let local_index = if log2_max_height >= log2_height {
                index >> (log2_max_height - log2_height)
            } else {
                index
            } % dims.height;

            let point = Fr::new(local_index as u64);
            for ((value, commitment), witness) in
                values.iter().zip(&commitment.columns).zip(witnesses)
            {
                all_openings.push(OpeningInfo {
                    commitment: *commitment,
                    witness: *witness,
                    value: *value,
                    point,
                });
            }
        }

        // Verify all openings in a single batch
        verify_batch(&all_openings, &self.params)
    }
}
