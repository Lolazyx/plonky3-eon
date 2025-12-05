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

/// Commitment for a single matrix; contains one KZG commitment per column.
#[derive(Clone, Serialize, Deserialize)]
pub struct MatrixCommitment {
    pub columns: Vec<G1>,
}

/// Commitment for a batch of matrices.
#[derive(Clone, Serialize, Deserialize)]
pub struct KzgCommitment {
    pub matrices: Vec<MatrixCommitment>,
}

/// Prover data for a single committed matrix.
#[derive(Clone)]
pub struct MatrixProverData {
    pub domain: TwoAdicMultiplicativeCoset<Fr>,
    pub evals: RowMajorMatrix<Fr>,
    pub coeffs: RowMajorMatrix<Fr>,
}

pub type ProverData = Vec<MatrixProverData>;

/// Proof for opening a batch of matrices in a single PCS round.
#[derive(Clone, Serialize, Deserialize)]
pub struct MatrixProof {
    pub points: Vec<PointProof>,
}

/// Proof for a single point; contains one witness per opened column.
#[derive(Clone, Serialize, Deserialize)]
pub struct PointProof {
    pub witnesses: Vec<G1>,
}

/// Proof object matching the [`OpenedValues`] structure.
#[derive(Clone, Serialize, Deserialize)]
pub struct KzgProof {
    pub rounds: Vec<Vec<MatrixProof>>,
}

#[derive(Clone)]
pub struct KzgPcs {
    pub params: KzgParams,
    pub dft: Radix2Dit<Fr>,
}

impl KzgPcs {
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
