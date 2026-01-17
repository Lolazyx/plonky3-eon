use eon_air::impl_p3_air_builder_traits;
use eon_air::{EonAir, EonAirBuilder, RowMajorMatrix};
use p3_field::{ExtensionField, Field};
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::AirLookupHandler;
use p3_lookup::lookup_traits::{Kind, LookupData};
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::{Entry, SymbolicExpression, SymbolicVariable};

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, EF, A>(
    air: &mut A,
    preprocessed_width: usize,
    num_public_values: usize,
    is_zk: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: EonAir<F, EF> + AirLookupHandler<SymbolicAirBuilder<F>>,
{
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (get_max_constraint_degree::<F, EF, A>(
        air,
        preprocessed_width,
        num_public_values,
        permutation_width,
        num_permutation_challenges,
    ) + is_zk)
        .max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the vanishing polynomial.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, EF, A>(
    air: &mut A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: EonAir<F, EF> + AirLookupHandler<SymbolicAirBuilder<F>>,
{
    get_symbolic_constraints::<F, EF, A>(
        air,
        preprocessed_width,
        num_public_values,
        permutation_width,
        num_permutation_challenges,
    )
    .iter()
    .map(|c| c.degree_multiple())
    .max()
    .unwrap_or(0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, EF, A>(
    air: &mut A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: EonAir<F, EF> + AirLookupHandler<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::<F>::new(
        preprocessed_width,
        EonAir::<F, EF>::width(&*air),
        num_public_values,
        permutation_width,
        num_permutation_challenges,
    );

    let lookups = <A as AirLookupHandler<SymbolicAirBuilder<F>>>::get_lookups(air);
    let mut lookup_data: Vec<LookupData<F>> = lookups
        .iter()
        .filter_map(|ctx| match &ctx.kind {
            Kind::Global(name) => Some(LookupData {
                name: name.clone(),
                aux_idx: ctx.columns[0],
                expected_cumulated: F::ZERO,
            }),
            Kind::Local => None,
        })
        .collect();
    lookup_data.sort_by_key(|d| d.aux_idx);
    let lookup_gadget = LogUpGadget;

    <A as AirLookupHandler<SymbolicAirBuilder<F>>>::eval(
        &*air,
        &mut builder,
        &lookups,
        &lookup_data,
        &lookup_gadget,
    );
    builder.constraints()
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    permutation: Option<RowMajorMatrix<SymbolicVariable<F>>>,
    permutation_challenges: Vec<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    constraints: Vec<SymbolicExpression<F>>,
}

impl<F: Field> SymbolicAirBuilder<F> {
    pub fn new(
        preprocessed_width: usize,
        width: usize,
        num_public_values: usize,
        permutation_width: usize,
        num_permutation_challenges: usize,
    ) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width)
                    .map(move |index| SymbolicVariable::new(Entry::Preprocessed { offset }, index))
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width).map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let permutation = if permutation_width > 0 {
            let perm_values = [0, 1] // Permutation trace also use consecutive rows for LogUp based permutation check
                .into_iter()
                .flat_map(|offset| {
                    (0..permutation_width).map(move |index| {
                        SymbolicVariable::new(Entry::Permutation { offset }, index)
                    })
                })
                .collect();
            Some(RowMajorMatrix::new(perm_values, permutation_width))
        } else {
            None
        };
        let permutation_challenges = Self::sample_randomness(num_permutation_challenges);
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            permutation,
            permutation_challenges,
            public_values,
            constraints: vec![],
        }
    }

    pub fn constraints(self) -> Vec<SymbolicExpression<F>> {
        self.constraints
    }

    pub(crate) fn sample_randomness(num_randomness: usize) -> Vec<SymbolicVariable<F>> {
        (0..num_randomness)
            .map(|index| SymbolicVariable::new(Entry::Challenge, index))
            .collect()
    }
}

impl<F: Field> EonAirBuilder for SymbolicAirBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;
    type EF = F;
    type ExprEF = SymbolicExpression<F>;
    type VarEF = SymbolicVariable<F>;

    type PublicVar = SymbolicVariable<F>;

    type MP = RowMajorMatrix<SymbolicVariable<F>>;
    type RandomVar = SymbolicVariable<F>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }

    fn permutation(&self) -> Self::MP {
        self.permutation.clone().expect("permutation called but aux trace is None - AIR should check num_randomness > 0 before using permutation columns")
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.permutation_challenges
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.constraints.push(x.into());
    }

    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl_p3_air_builder_traits!(SymbolicAirBuilder<F> where F: eon_air::Field + Sync);

#[cfg(test)]
mod tests {
    use eon_air::EonAir;
    use eon_air::{Air, BaseAir, BaseAirWithPublicValues, EonAirBuilder, RowMajorMatrix};
    use p3_lookup::lookup_traits::{AirLookupHandler, Kind, Lookup, LookupInput};

    use p3_bn254::Fr;
    use p3_matrix::Matrix;

    use super::*;

    #[derive(Debug)]
    struct MockAir {
        // Store (entry_type, index) pairs instead of SymbolicVariables
        constraint_specs: Vec<(Entry, usize)>,
        width: usize,
    }

    // impl EonAir<Fr, Fr> for MockAir {
    //     fn width(&self) -> usize {
    //         self.width
    //     }

    //     fn eval<AB: EonAirBuilder<F = Fr>>(&self, builder: &mut AB) {
    //         let main = builder.main();

    //         for (entry, index) in &self.constraint_specs {
    //             match entry {
    //                 Entry::Main { offset } => {
    //                     builder.assert_zero(main.row_slice(*offset).unwrap()[*index].clone());
    //                 }
    //                 _ => panic!("Test only supports Main entry"),
    //             }
    //         }
    //     }
    // }
    impl<F, EF> EonAir<F, EF> for MockAir
    where
        F: eon_air::Field,
        EF: eon_air::Field,
    {
        fn width(&self) -> usize {
            self.width
        }

        fn eval<AB: EonAirBuilder<F = F, EF = EF>>(&self, builder: &mut AB) {
            let main = builder.main();

            for (entry, index) in &self.constraint_specs {
                match entry {
                    Entry::Main { offset } => {
                        builder.assert_zero(main.row_slice(*offset).unwrap()[*index].clone());
                    }
                    _ => panic!("Test only supports Main entry"),
                }
            }
        }
    }

    impl BaseAir<Fr> for MockAir {
        fn width(&self) -> usize {
            <Self as EonAir<Fr, Fr>>::width(self)
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Fr>> {
            <Self as EonAir<Fr, Fr>>::preprocessed_trace(self)
        }
    }

    impl BaseAirWithPublicValues<Fr> for MockAir {
        fn num_public_values(&self) -> usize {
            <Self as EonAir<Fr, Fr>>::num_public_values(self)
        }
    }

    impl Air<SymbolicAirBuilder<Fr>> for MockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<Fr>) {
            <Self as EonAir<Fr, Fr>>::eval(self, builder)
        }
    }

    impl AirLookupHandler<SymbolicAirBuilder<Fr>> for MockAir {
        fn add_lookup_columns(&mut self) -> Vec<usize> {
            vec![]
        }

        fn get_lookups(&mut self) -> Vec<Lookup<Fr>> {
            vec![]
        }

        fn register_lookup(&mut self, kind: Kind, lookup_inputs: &[LookupInput<Fr>]) -> Lookup<Fr> {
            // 如果测试里不会触发 register_lookup，你也可以 unimplemented!()
            <Self as EonAir<Fr, Fr>>::register_lookup(self, kind, lookup_inputs)
        }
    }

    #[test]
    fn test_get_log_quotient_degree_no_constraints() {
        let mut air = MockAir {
            constraint_specs: vec![],
            width: 4,
        };
        let log_degree = get_log_quotient_degree::<Fr, Fr, _>(&mut air, 3, 2, 0, 0, 0);
        assert_eq!(log_degree, 0);
    }

    #[test]
    fn test_get_log_quotient_degree_single_constraint() {
        let mut air = MockAir {
            constraint_specs: vec![(Entry::Main { offset: 0 }, 0)],
            width: 4,
        };
        let log_degree = get_log_quotient_degree::<Fr, Fr, _>(&mut air, 3, 2, 0, 0, 0);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }

    #[test]
    fn test_get_log_quotient_degree_multiple_constraints() {
        let mut air = MockAir {
            constraint_specs: vec![
                (Entry::Main { offset: 0 }, 0),
                (Entry::Main { offset: 1 }, 1),
                (Entry::Main { offset: 0 }, 2),
            ],
            width: 4,
        };
        let log_degree = get_log_quotient_degree::<Fr, Fr, _>(&mut air, 3, 2, 0, 0, 0);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }

    #[test]
    fn test_get_max_constraint_degree_no_constraints() {
        let mut air = MockAir {
            constraint_specs: vec![],
            width: 4,
        };
        let max_degree = get_max_constraint_degree::<Fr, Fr, _>(&mut air, 3, 2, 0, 0);
        assert_eq!(
            max_degree, 0,
            "No constraints should result in a degree of 0"
        );
    }

    #[test]
    fn test_get_max_constraint_degree_multiple_constraints() {
        let mut air = MockAir {
            constraint_specs: vec![
                (Entry::Main { offset: 0 }, 0),
                (Entry::Main { offset: 1 }, 1),
                (Entry::Main { offset: 0 }, 2),
            ],
            width: 4,
        };
        let max_degree = get_max_constraint_degree::<Fr, Fr, _>(&mut air, 3, 2, 0, 0);
        assert_eq!(max_degree, 1, "Max constraint degree should be 1");
    }

    #[test]
    fn test_get_symbolic_constraints() {
        let c1: SymbolicVariable<Fr> = SymbolicVariable::new(Entry::Main { offset: 0 }, 0);
        let c2: SymbolicVariable<Fr> = SymbolicVariable::new(Entry::Main { offset: 1 }, 1);

        let mut air = MockAir {
            constraint_specs: vec![
                (Entry::Main { offset: 0 }, 0),
                (Entry::Main { offset: 1 }, 1),
            ],
            width: 4,
        };

        let constraints = get_symbolic_constraints::<Fr, Fr, _>(&mut air, 3, 2, 0, 0);

        assert_eq!(constraints.len(), 2, "Should return exactly 2 constraints");

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Variable(v) if v.index == c1.index && v.entry == c1.entry)),
            "Expected constraint {c1:?} was not found"
        );

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Variable(v) if v.index == c2.index && v.entry == c2.entry)),
            "Expected constraint {c2:?} was not found"
        );
    }

    #[test]
    fn test_symbolic_air_builder_initialization() {
        let builder = SymbolicAirBuilder::<Fr>::new(2, 4, 0, 0, 3);

        let expected_main = [
            SymbolicVariable::<Fr>::new(Entry::Main { offset: 0 }, 0),
            SymbolicVariable::<Fr>::new(Entry::Main { offset: 0 }, 1),
            SymbolicVariable::<Fr>::new(Entry::Main { offset: 0 }, 2),
            SymbolicVariable::<Fr>::new(Entry::Main { offset: 0 }, 3),
            SymbolicVariable::<Fr>::new(Entry::Main { offset: 1 }, 0),
            SymbolicVariable::<Fr>::new(Entry::Main { offset: 1 }, 1),
            SymbolicVariable::<Fr>::new(Entry::Main { offset: 1 }, 2),
            SymbolicVariable::<Fr>::new(Entry::Main { offset: 1 }, 3),
        ];

        let builder_main = builder.main.values;

        assert_eq!(
            builder_main.len(),
            expected_main.len(),
            "Main matrix should have the expected length"
        );

        for (expected, actual) in expected_main.iter().zip(builder_main.iter()) {
            assert_eq!(expected.index, actual.index, "Index mismatch");
            assert_eq!(expected.entry, actual.entry, "Entry mismatch");
        }
    }

    #[test]
    fn test_symbolic_air_builder_is_first_last_row() {
        let builder = SymbolicAirBuilder::<Fr>::new(2, 4, 0, 0, 3);

        assert!(
            matches!(builder.is_first_row(), SymbolicExpression::IsFirstRow),
            "First row condition did not match"
        );

        assert!(
            matches!(builder.is_last_row(), SymbolicExpression::IsLastRow),
            "Last row condition did not match"
        );
    }

    #[test]
    fn test_symbolic_air_builder_assert_zero() {
        let mut builder = SymbolicAirBuilder::<Fr>::new(2, 4, 0, 0, 3);
        let expr = SymbolicExpression::Constant(Fr::new(5));
        builder.assert_zero(expr);

        let constraints = builder.constraints();
        assert_eq!(constraints.len(), 1, "One constraint should be recorded");

        assert!(
            constraints
                .iter()
                .any(|x| matches!(x, SymbolicExpression::Constant(val) if *val == Fr::new(5))),
            "Constraint should match the asserted one"
        );
    }
}
