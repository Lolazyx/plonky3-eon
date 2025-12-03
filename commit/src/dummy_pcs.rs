use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::CanSample;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

use crate::{OpenedValues, Pcs, PolynomialSpace};

/// A dummy PCS where the commitment is the polynomial itself (stored as evaluations).
/// This provides no cryptographic security and is only useful for testing or debugging.
#[derive(Debug, Clone)]
pub struct DummyPcs<Domain> {
    pub _phantom: PhantomData<Domain>,
}

impl<Domain> DummyPcs<Domain> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<Domain> Default for DummyPcs<Domain> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to evaluate a polynomial (given as evaluations on a domain) at a point
/// using Lagrange interpolation.
pub fn eval_poly_at_point<F: Field, EF: ExtensionField<F>, D: PolynomialSpace<Val = F>>(
    domain: &D,
    evals: &RowMajorMatrix<F>,
    point: EF,
) -> Vec<EF> {
    let n = evals.height();
    assert_eq!(n, domain.size());

    let mut result = vec![EF::ZERO; evals.width()];

    // Compute all domain points by iterating using next_point
    let mut domain_points: Vec<EF> = Vec::with_capacity(n);
    let first = domain.first_point();
    let mut current = EF::from(first);
    domain_points.push(current);

    for _ in 1..n {
        current = domain
            .next_point(current)
            .expect("domain point should exist");
        domain_points.push(current);
    }

    // Lagrange interpolation: f(x) = sum_i f(x_i) * L_i(x)
    // where L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
    for i in 0..n {
        let x_i = domain_points[i];

        // Compute L_i(point)
        let mut lagrange_i = EF::ONE;
        for j in 0..n {
            if i != j {
                let x_j = domain_points[j];
                lagrange_i *= (point - x_j) / (x_i - x_j);
            }
        }

        // Add f(x_i) * L_i(point) to result
        let row = evals.row_slice(i).unwrap();
        for (res, &val) in result.iter_mut().zip(row.iter()) {
            *res += lagrange_i * EF::from(val);
        }
    }

    result
}

impl<Domain, Challenge, Challenger> Pcs<Challenge, Challenger> for DummyPcs<Domain>
where
    Domain: PolynomialSpace + Clone,
    Domain::Val: Field,
    Challenge: ExtensionField<Domain::Val>,
    Challenger: CanSample<Challenge>,
{
    type Domain = Domain;
    type Commitment = Vec<RowMajorMatrix<Domain::Val>>;
    type ProverData = Vec<(Domain, RowMajorMatrix<Domain::Val>)>;
    type EvaluationsOnDomain<'a> = RowMajorMatrix<Domain::Val>;
    type Proof = ();
    type Error = ();
    const ZK: bool = false;

    fn natural_domain_for_degree(&self, _degree: usize) -> Self::Domain {
        // For a dummy PCS, we don't have a way to construct a domain from a degree generically.
        // This would need to be implemented based on the specific domain type.
        // For now, panic - users should construct their own domains.
        panic!(
            "DummyPcs does not support natural_domain_for_degree. Please construct domains explicitly."
        );
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Domain::Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let data: Vec<_> = evaluations.into_iter().collect();
        let commitment = data.iter().map(|(_, evals)| evals.clone()).collect();
        (commitment, data)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        // For the dummy PCS, we assume the requested domain matches the stored domain
        // In a real implementation, you might need to interpolate to a different domain
        assert_eq!(
            prover_data[idx].0.size(),
            domain.size(),
            "Domain size mismatch in dummy PCS"
        );
        prover_data[idx].1.clone()
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        _challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        let opened_values = rounds
            .into_iter()
            .map(|(prover_data, points_for_round)| {
                assert_eq!(prover_data.len(), points_for_round.len());
                prover_data
                    .iter()
                    .zip(points_for_round)
                    .map(|((domain, evals), points_for_mat)| {
                        points_for_mat
                            .into_iter()
                            .map(|pt| eval_poly_at_point(domain, evals, pt))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        (opened_values, ())
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    Challenge,
                    // values at this point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        _proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // Verify by recomputing the evaluations at the given points
        for (commitment, matrices_data) in rounds {
            assert_eq!(commitment.len(), matrices_data.len());
            for (evals, (domain, points_and_values)) in commitment.iter().zip(matrices_data) {
                for (pt, claimed_values) in points_and_values {
                    let computed_values = eval_poly_at_point(&domain, evals, pt);
                    assert_eq!(
                        computed_values, claimed_values,
                        "Opened values do not match"
                    );
                }
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
    use p3_field::coset::TwoAdicMultiplicativeCoset;
    use p3_matrix::dense::RowMajorMatrix;

    // Use Bn254 field for testing
    type TestVal = Bn254;

    #[test]
    fn test_dummy_pcs_commit_and_open() {
        // Create a dummy PCS
        let pcs: DummyPcs<TwoAdicMultiplicativeCoset<TestVal>> = DummyPcs::new();

        // Create a small domain (size 4)
        let domain = TwoAdicMultiplicativeCoset::<TestVal>::new(TestVal::ONE, 2).unwrap();

        // Create a simple polynomial evaluation matrix (4 rows x 2 columns)
        // This represents 2 polynomials evaluated at 4 points
        let values = vec![
            TestVal::ONE,
            TestVal::TWO,
            TestVal::new(3),
            TestVal::new(4),
            TestVal::new(5),
            TestVal::new(6),
            TestVal::new(7),
            TestVal::new(8),
        ];
        let evals = RowMajorMatrix::new(values, 2);

        // Commit to the polynomial
        let (commitment, prover_data) = pcs.commit(vec![(domain, evals.clone())]);

        // Verify the commitment contains the evaluations
        assert_eq!(commitment.len(), 1);
        assert_eq!(commitment[0].height(), 4);
        assert_eq!(commitment[0].width(), 2);
    }

    #[test]
    fn test_dummy_pcs_multiple_matrices() {
        let pcs: DummyPcs<TwoAdicMultiplicativeCoset<TestVal>> = DummyPcs::new();

        // Create two domains
        let domain1 = TwoAdicMultiplicativeCoset::<TestVal>::new(TestVal::ONE, 2).unwrap(); // size 4
        let domain2 = TwoAdicMultiplicativeCoset::<TestVal>::new(TestVal::ONE, 3).unwrap(); // size 8

        // Create evaluation matrices
        let evals1 = RowMajorMatrix::new(
            vec![TestVal::ONE, TestVal::TWO, TestVal::new(3), TestVal::new(4)],
            1,
        );

        let evals2 = RowMajorMatrix::new(
            vec![
                TestVal::new(1),
                TestVal::new(2),
                TestVal::new(3),
                TestVal::new(4),
                TestVal::new(5),
                TestVal::new(6),
                TestVal::new(7),
                TestVal::new(8),
            ],
            1,
        );

        // Commit to both
        let (commitment, _prover_data) = pcs.commit(vec![(domain1, evals1), (domain2, evals2)]);

        assert_eq!(commitment.len(), 2);
        assert_eq!(commitment[0].height(), 4);
        assert_eq!(commitment[1].height(), 8);
    }

    #[test]
    fn test_get_evaluations_on_domain() {
        let pcs: DummyPcs<TwoAdicMultiplicativeCoset<TestVal>> = DummyPcs::new();
        let domain = TwoAdicMultiplicativeCoset::<TestVal>::new(TestVal::ONE, 2).unwrap();

        let evals = RowMajorMatrix::new(
            vec![TestVal::ONE, TestVal::TWO, TestVal::new(3), TestVal::new(4)],
            1,
        );

        let (_, prover_data) = pcs.commit(vec![(domain, evals.clone())]);

        // Get evaluations on the same domain
        let retrieved_evals = pcs.get_evaluations_on_domain(&prover_data, 0, domain);

        assert_eq!(retrieved_evals.height(), evals.height());
        assert_eq!(retrieved_evals.width(), evals.width());
    }
}
