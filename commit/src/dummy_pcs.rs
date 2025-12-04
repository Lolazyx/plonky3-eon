use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::CanSample;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_interpolation::interpolate_coset;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

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
///
/// For TwoAdicMultiplicativeCoset, this uses the optimized interpolation from p3-interpolation.
///
/// For KZG-based SNARKs (UltraPlonk), the domain is typically the multiplicative subgroup
/// (shift = 1), while for STARKs with FRI, cosets (shift ≠ 1) are used.
pub fn eval_poly_at_point_coset<F: TwoAdicField, EF: ExtensionField<F>>(
    coset: &TwoAdicMultiplicativeCoset<F>,
    evals: &RowMajorMatrix<F>,
    point: EF,
) -> Vec<EF> {
    let n = evals.height();
    assert_eq!(n, coset.size());

    // Use the optimized interpolation based on the domain's shift
    // For KZG (UltraPlonk): shift = 1 (multiplicative subgroup)
    // For FRI (STARKs): shift may be ≠ 1 (coset)
    interpolate_coset(evals, coset.shift(), point)
}

impl<F, Challenge, Challenger> Pcs<Challenge, Challenger>
    for DummyPcs<TwoAdicMultiplicativeCoset<F>>
where
    F: TwoAdicField,
    Challenge: ExtensionField<F>,
    Challenger: CanSample<Challenge>,
{
    type Domain = TwoAdicMultiplicativeCoset<F>;
    type Commitment = Vec<RowMajorMatrix<F>>;
    type ProverData = Vec<(TwoAdicMultiplicativeCoset<F>, RowMajorMatrix<F>)>;
    type EvaluationsOnDomain<'a> = RowMajorMatrix<F>;
    type Proof = ();
    type Error = ();
    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        let log_n = log2_strict_usize(degree.next_power_of_two());
        TwoAdicMultiplicativeCoset::new(F::ONE, log_n)
            .expect("log_n should be within two-adicity bounds")
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<F>)>,
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
                            .map(|pt| eval_poly_at_point_coset(domain, evals, pt))
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
        _rounds: Vec<(
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
        // DummyPcs provides no cryptographic security
        // We trust all opened values without verification
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use p3_bn254::Bn254;
    use p3_challenger::CanSample;
    use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
    use p3_field::coset::TwoAdicMultiplicativeCoset;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_strict_usize;

    // Use Bn254 field for testing
    type TestVal = Bn254;

    // Dummy challenger that satisfies trait bounds but does nothing
    struct DummyChallenger;
    impl CanSample<Bn254> for DummyChallenger {
        fn sample(&mut self) -> Bn254 {
            Bn254::ZERO
        }
    }

    /// Test polynomial: f(x) = 1 + 2x + 3x^2 + 4x^3
    /// Evaluating at points OUTSIDE the domain (for KZG-style opening)
    #[test]
    fn test_polynomial_evaluation() {
        // Define polynomial f(x) = 1 + 2x + 3x^2 + 4x^3
        let eval_poly = |x: TestVal| -> TestVal {
            TestVal::ONE +
            x * TestVal::TWO +
            x * x * TestVal::new(3) +
            x * x * x * TestVal::new(4)
        };

        // Create a domain of size 8 (multiplicative subgroup, shift=1)
        // Note: The domain contains powers of the 8th root of unity
        // For BN254, these are specific field elements, NOT including 0, 2, 100, etc.
        let n = 8;
        let log_n = log2_strict_usize(n);
        let domain = TwoAdicMultiplicativeCoset::<TestVal>::new(TestVal::ONE, log_n).unwrap();

        // Generate evaluations of the polynomial over the domain
        let subgroup: Vec<TestVal> = TestVal::two_adic_generator(log_n).powers().collect_n(n);
        let evals: Vec<TestVal> = subgroup.iter().map(|&x| eval_poly(x)).collect();
        let evals_mat = RowMajorMatrix::new(evals, 1);

        // Test evaluation at x=100 (outside the domain)
        // f(100) = 1 + 200 + 30000 + 4000000 = 4030201
        let point = TestVal::new(100);
        let result = eval_poly_at_point_coset(&domain, &evals_mat, point);
        let expected = eval_poly(point);
        assert_eq!(result, vec![expected], "f(100) should match direct evaluation");

        // Test evaluation at x=42 (outside the domain)
        // f(42) = 1 + 84 + 5292 + 296352 = 301729
        let point2 = TestVal::new(42);
        let result2 = eval_poly_at_point_coset(&domain, &evals_mat, point2);
        let expected2 = eval_poly(point2);
        assert_eq!(result2, vec![expected2], "f(42) should match direct evaluation");
    }

    #[test]
    fn test_dummy_pcs_polynomial_opening() {
        // Define polynomial f(x) = 1 + 2x + 3x^2 + 4x^3
        let eval_poly = |x: TestVal| -> TestVal {
            TestVal::ONE +
            x * TestVal::TWO +
            x * x * TestVal::new(3) +
            x * x * x * TestVal::new(4)
        };

        // Create domain and polynomial evaluations
        let n = 8;
        let log_n = log2_strict_usize(n);
        let domain = TwoAdicMultiplicativeCoset::<TestVal>::new(TestVal::ONE, log_n).unwrap();
        let subgroup: Vec<TestVal> = TestVal::two_adic_generator(log_n).powers().collect_n(n);
        let evals: Vec<TestVal> = subgroup.iter().map(|&x| eval_poly(x)).collect();
        let evals_mat = RowMajorMatrix::new(evals, 1);

        // Create PCS and commit
        let pcs: DummyPcs<TwoAdicMultiplicativeCoset<TestVal>> = DummyPcs::new();
        let (commitment, prover_data) = <DummyPcs<_> as Pcs<TestVal, DummyChallenger>>::commit(
            &pcs,
            vec![(domain, evals_mat.clone())],
        );

        assert_eq!(commitment.len(), 1);
        assert_eq!(commitment[0].height(), n);

        // Open at multiple points OUTSIDE the domain
        // Use points 100, 42, 7 which are not in the 8th roots of unity
        let point1 = TestVal::new(100);
        let point2 = TestVal::new(42);
        let point3 = TestVal::new(7);
        let points = vec![vec![point1, point2, point3]];
        let rounds = vec![(&prover_data, points)];

        let (opened_values, _proof) = <DummyPcs<_> as Pcs<TestVal, DummyChallenger>>::open(
            &pcs,
            rounds,
            &mut DummyChallenger,
        );

        // Verify the opened values match expected polynomial evaluations
        assert_eq!(opened_values.len(), 1); // 1 round
        assert_eq!(opened_values[0].len(), 1); // 1 matrix in this round
        assert_eq!(opened_values[0][0].len(), 3); // 3 points

        // Check f(100) = 1 + 200 + 30000 + 4000000
        assert_eq!(opened_values[0][0][0], vec![eval_poly(point1)]);
        // Check f(42) = 1 + 84 + 5292 + 296352
        assert_eq!(opened_values[0][0][1], vec![eval_poly(point2)]);
        // Check f(7) = 1 + 14 + 147 + 1372
        assert_eq!(opened_values[0][0][2], vec![eval_poly(point3)]);
    }

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
        // Use trait explicitly with Bn254 as Challenge
        let (commitment, _prover_data) = <DummyPcs<_> as Pcs<TestVal, DummyChallenger>>::commit(
            &pcs,
            vec![(domain, evals.clone())],
        );

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
        let (commitment, _prover_data) = <DummyPcs<_> as Pcs<TestVal, DummyChallenger>>::commit(
            &pcs,
            vec![(domain1, evals1), (domain2, evals2)],
        );

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

        let (_, prover_data) = <DummyPcs<_> as Pcs<TestVal, DummyChallenger>>::commit(
            &pcs,
            vec![(domain, evals.clone())],
        );

        // Get evaluations on the same domain
        let retrieved_evals =
            <DummyPcs<_> as Pcs<TestVal, DummyChallenger>>::get_evaluations_on_domain(
                &pcs,
                &prover_data,
                0,
                domain,
            );

        assert_eq!(retrieved_evals.height(), evals.height());
        assert_eq!(retrieved_evals.width(), evals.width());
    }
}
