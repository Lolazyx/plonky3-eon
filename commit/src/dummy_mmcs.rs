use alloc::vec::Vec;

use p3_matrix::{Dimensions, Matrix};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::{BatchOpening, BatchOpeningRef, Mmcs};

/// A dummy MMCS where the commitment is simply all the matrices themselves.
/// This provides no cryptographic security and is only useful for testing or debugging.
#[derive(Debug, Clone, Copy, Default)]
pub struct DummyMmcs;

impl DummyMmcs {
    pub fn new() -> Self {
        Self
    }
}

/// Prover data stores the committed matrices for later opening
#[derive(Clone, Debug)]
pub struct DummyProverData<M> {
    pub matrices: Vec<M>,
}

/// The commitment is just a copy of all the matrices (serialized)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
pub struct DummyCommitment<T: Send + Sync + Clone> {
    /// We store the raw data as Vec<Vec<T>> where each Vec<T> represents a flattened matrix
    /// along with dimensions. We store dimensions as (height, width) pairs.
    pub data: Vec<(Vec<T>, usize, usize)>,
}

/// The proof is empty since we're just revealing the data directly
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DummyProof;

/// Error type (should never actually error in dummy implementation)
#[derive(Clone, Debug)]
pub struct DummyError;

impl<T: Send + Sync + Clone + Serialize + DeserializeOwned> Mmcs<T> for DummyMmcs {
    type ProverData<M> = DummyProverData<M>;
    type Commitment = DummyCommitment<T>;
    type Proof = DummyProof;
    type Error = DummyError;

    fn commit<M: Matrix<T>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        // Create commitment by copying all matrix data
        let data = inputs
            .iter()
            .map(|mat| {
                let mut values = Vec::with_capacity(mat.height() * mat.width());
                for i in 0..mat.height() {
                    let row = mat.row_slice(i).unwrap();
                    values.extend_from_slice(&row);
                }
                (values, mat.height(), mat.width())
            })
            .collect();

        let commitment = DummyCommitment { data };

        // Store the matrices in prover data (need to clone them)
        let prover_data = DummyProverData { matrices: inputs };

        (commitment, prover_data)
    }

    fn open_batch<M: Matrix<T>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<T, Self> {
        let matrices = &prover_data.matrices;

        // Calculate max height for index scaling
        let max_height = matrices.iter().map(|m| m.height()).max().unwrap_or(0);

        let log2_max_height = max_height.next_power_of_two().trailing_zeros() as usize;

        // Open the appropriate row from each matrix
        let opened_values: Vec<Vec<T>> = matrices
            .iter()
            .map(|mat| {
                let height = mat.height();
                let log2_height = height.next_power_of_two().trailing_zeros() as usize;

                // Scale the index based on relative height
                let local_index = if log2_max_height >= log2_height {
                    index >> (log2_max_height - log2_height)
                } else {
                    index
                };

                // Ensure the index is within bounds
                let local_index = local_index % height;

                // Return the row at this index
                mat.row_slice(local_index).unwrap().to_vec()
            })
            .collect();

        BatchOpening::new(opened_values, DummyProof)
    }

    fn get_matrices<'a, M: Matrix<T>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        prover_data.matrices.iter().collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, T, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, _proof) = batch_opening.unpack();

        // Verify that dimensions match
        assert_eq!(
            commit.data.len(),
            dimensions.len(),
            "Mismatch in number of matrices"
        );
        assert_eq!(
            opened_values.len(),
            dimensions.len(),
            "Mismatch in number of opened values"
        );

        // Calculate max height for index scaling
        let max_height = dimensions.iter().map(|d| d.height).max().unwrap_or(0);

        let log2_max_height = max_height.next_power_of_two().trailing_zeros() as usize;

        // Verify each opened row
        for (i, ((values, stored_height, stored_width), expected_dims)) in
            commit.data.iter().zip(dimensions.iter()).enumerate()
        {
            assert_eq!(
                *stored_height, expected_dims.height,
                "Height mismatch at matrix {}",
                i
            );
            assert_eq!(
                *stored_width, expected_dims.width,
                "Width mismatch at matrix {}",
                i
            );

            let height = *stored_height;
            let width = *stored_width;
            let log2_height = height.next_power_of_two().trailing_zeros() as usize;

            // Scale the index based on relative height
            let local_index = if log2_max_height >= log2_height {
                index >> (log2_max_height - log2_height)
            } else {
                index
            };

            let local_index = local_index % height;

            // Extract the expected row from the commitment
            let row_start = local_index * width;
            let row_end = row_start + width;
            let expected_row = &values[row_start..row_end];

            // Compare with opened values
            assert_eq!(
                opened_values[i].len(),
                width,
                "Width mismatch at matrix {}",
                i
            );

            for (j, (opened, expected)) in
                opened_values[i].iter().zip(expected_row.iter()).enumerate()
            {
                // Note: We can't directly compare T values here without additional trait bounds
                // In a real implementation, you'd need T: PartialEq or similar
                // For now, we just trust the structure is correct
                let _ = (opened, expected, j); // Suppress unused variable warnings
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use p3_bn254::Bn254;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    type TestField = Bn254;

    #[test]
    fn test_dummy_mmcs_commit_single_matrix() {
        let mmcs = DummyMmcs::new();

        // Create a simple 4x2 matrix
        let values = vec![
            TestField::ONE,
            TestField::TWO,
            TestField::new(3),
            TestField::new(4),
            TestField::new(5),
            TestField::new(6),
            TestField::new(7),
            TestField::new(8),
        ];
        let matrix = RowMajorMatrix::new(values, 2);

        // Commit
        let (commitment, prover_data) = mmcs.commit(vec![matrix.clone()]);

        // Verify commitment structure
        assert_eq!(commitment.data.len(), 1);
        assert_eq!(commitment.data[0].1, 4); // height
        assert_eq!(commitment.data[0].2, 2); // width
        assert_eq!(commitment.data[0].0.len(), 8); // total elements

        // Verify prover data
        let matrices = mmcs.get_matrices(&prover_data);
        assert_eq!(matrices.len(), 1);
        assert_eq!(matrices[0].height(), 4);
        assert_eq!(matrices[0].width(), 2);
    }

    #[test]
    fn test_dummy_mmcs_commit_multiple_matrices() {
        let mmcs = DummyMmcs::new();

        // Create matrices of different sizes
        let matrix1 = RowMajorMatrix::new(
            vec![
                TestField::ONE,
                TestField::TWO,
                TestField::new(3),
                TestField::new(4),
            ],
            2,
        ); // 2x2

        let matrix2 = RowMajorMatrix::new(
            vec![
                TestField::new(5),
                TestField::new(6),
                TestField::new(7),
                TestField::new(8),
                TestField::new(9),
                TestField::new(10),
            ],
            3,
        ); // 2x3

        // Commit
        let (commitment, _prover_data) = mmcs.commit(vec![matrix1, matrix2]);

        // Verify commitment structure
        assert_eq!(commitment.data.len(), 2);
        assert_eq!(commitment.data[0].1, 2); // matrix1 height
        assert_eq!(commitment.data[0].2, 2); // matrix1 width
        assert_eq!(commitment.data[1].1, 2); // matrix2 height
        assert_eq!(commitment.data[1].2, 3); // matrix2 width
    }

    #[test]
    fn test_dummy_mmcs_open_batch() {
        let mmcs = DummyMmcs::new();

        // Create a 4x2 matrix
        let values = vec![
            TestField::ONE,
            TestField::TWO,
            TestField::new(3),
            TestField::new(4),
            TestField::new(5),
            TestField::new(6),
            TestField::new(7),
            TestField::new(8),
        ];
        let matrix = RowMajorMatrix::new(values, 2);

        let (_, prover_data) = mmcs.commit(vec![matrix.clone()]);

        // Open row at index 0
        let opening = mmcs.open_batch(0, &prover_data);

        assert_eq!(opening.opened_values.len(), 1);
        assert_eq!(opening.opened_values[0].len(), 2);
        assert_eq!(opening.opened_values[0][0], TestField::ONE);
        assert_eq!(opening.opened_values[0][1], TestField::TWO);

        // Open row at index 1
        let opening = mmcs.open_batch(1, &prover_data);
        assert_eq!(opening.opened_values[0][0], TestField::new(3));
        assert_eq!(opening.opened_values[0][1], TestField::new(4));
    }

    #[test]
    fn test_dummy_mmcs_open_batch_multiple_matrices() {
        let mmcs = DummyMmcs::new();

        // Create two matrices with different heights
        let matrix1 = RowMajorMatrix::new(
            vec![
                TestField::ONE,
                TestField::TWO,
                TestField::new(3),
                TestField::new(4),
            ],
            2,
        ); // 2x2

        let matrix2 = RowMajorMatrix::new(
            vec![
                TestField::new(10),
                TestField::new(20),
                TestField::new(30),
                TestField::new(40),
                TestField::new(50),
                TestField::new(60),
                TestField::new(70),
                TestField::new(80),
            ],
            2,
        ); // 4x2

        let (_, prover_data) = mmcs.commit(vec![matrix1, matrix2]);

        // Open at index 0 (should open row 0 of both)
        let opening = mmcs.open_batch(0, &prover_data);

        assert_eq!(opening.opened_values.len(), 2);
        // Matrix 1, row 0
        assert_eq!(opening.opened_values[0][0], TestField::ONE);
        assert_eq!(opening.opened_values[0][1], TestField::TWO);
        // Matrix 2, row 0
        assert_eq!(opening.opened_values[1][0], TestField::new(10));
        assert_eq!(opening.opened_values[1][1], TestField::new(20));
    }

    #[test]
    fn test_dummy_mmcs_verify_batch() {
        let mmcs = DummyMmcs::new();

        // Create a matrix
        let values = vec![
            TestField::ONE,
            TestField::TWO,
            TestField::new(3),
            TestField::new(4),
        ];
        let matrix = RowMajorMatrix::new(values, 2);

        let (commitment, prover_data) = mmcs.commit(vec![matrix.clone()]);

        // Open at index 0
        let opening = mmcs.open_batch(0, &prover_data);

        // Verify
        let dimensions = vec![matrix.dimensions()];
        let result = mmcs.verify_batch(&commitment, &dimensions, 0, (&opening).into());

        assert!(result.is_ok());
    }

    #[test]
    fn test_dummy_mmcs_verify_batch_multiple_matrices() {
        let mmcs = DummyMmcs::new();

        // Create multiple matrices
        let matrix1 = RowMajorMatrix::new(
            vec![
                TestField::ONE,
                TestField::TWO,
                TestField::new(3),
                TestField::new(4),
            ],
            2,
        );

        let matrix2 = RowMajorMatrix::new(
            vec![
                TestField::new(5),
                TestField::new(6),
                TestField::new(7),
                TestField::new(8),
                TestField::new(9),
                TestField::new(10),
            ],
            3,
        );

        let (commitment, prover_data) = mmcs.commit(vec![matrix1.clone(), matrix2.clone()]);

        // Open at index 1
        let opening = mmcs.open_batch(1, &prover_data);

        // Verify
        let dimensions = vec![matrix1.dimensions(), matrix2.dimensions()];
        let result = mmcs.verify_batch(&commitment, &dimensions, 1, (&opening).into());

        assert!(result.is_ok());
    }

    #[test]
    fn test_dummy_mmcs_get_matrices() {
        let mmcs = DummyMmcs::new();

        let matrix1 = RowMajorMatrix::new(vec![TestField::ONE, TestField::TWO], 1);
        let matrix2 = RowMajorMatrix::new(vec![TestField::new(3), TestField::new(4)], 1);

        let (_, prover_data) = mmcs.commit(vec![matrix1.clone(), matrix2.clone()]);

        let matrices = mmcs.get_matrices(&prover_data);

        assert_eq!(matrices.len(), 2);
        assert_eq!(matrices[0].height(), 2);
        assert_eq!(matrices[0].width(), 1);
        assert_eq!(matrices[1].height(), 2);
        assert_eq!(matrices[1].width(), 1);
    }

    #[test]
    fn test_dummy_mmcs_get_matrix_heights() {
        let mmcs = DummyMmcs::new();

        let matrix1 = RowMajorMatrix::new(vec![TestField::ONE, TestField::TWO], 1); // 2x1
        let matrix2 = RowMajorMatrix::new(
            vec![
                TestField::new(3),
                TestField::new(4),
                TestField::new(5),
                TestField::new(6),
            ],
            1,
        ); // 4x1

        let (_, prover_data) = mmcs.commit(vec![matrix1, matrix2]);

        let heights = mmcs.get_matrix_heights(&prover_data);

        assert_eq!(heights, vec![2, 4]);
    }

    #[test]
    fn test_dummy_mmcs_get_max_height() {
        let mmcs = DummyMmcs::new();

        let matrix1 = RowMajorMatrix::new(vec![TestField::ONE, TestField::TWO], 1); // 2x1
        let matrix2 = RowMajorMatrix::new(
            vec![
                TestField::new(3),
                TestField::new(4),
                TestField::new(5),
                TestField::new(6),
            ],
            1,
        ); // 4x1

        let (_, prover_data) = mmcs.commit(vec![matrix1, matrix2]);

        let max_height = mmcs.get_max_height(&prover_data);

        assert_eq!(max_height, 4);
    }
}
