use eon_air::impl_p3_air_builder_traits;
use eon_air::{EonAir, EonAirBuilder};
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_lookup::lookup_traits::{AirLookupHandler, Lookup, LookupData};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::ViewPair;
use tracing::instrument;
/// Runs constraint checks using a given AIR definition and trace matrix.
///
/// Iterates over every row in `main`, providing both the current and next row
/// (with wraparound) to the AIR logic. Also injects public values into the builder
/// for first/last row assertions.
///
/// # Arguments
/// - `air`: The AIR logic to run
/// - `main`: The trace matrix (rows of witness values)
/// - `aux`: The aux trace matrix (if 2 phase proving)
/// - `aux_randomness`: The randomness values that are used to generate `aux` trace
/// - `public_values`: Public values provided to the builder
#[instrument(name = "check constraints", skip_all)]
pub(crate) fn check_constraints<F, EF, A, LG>(
    air: &mut A,
    main: &RowMajorMatrix<F>,
    permutation: Option<&RowMajorMatrix<EF>>,
    permutation_challenges: &[EF],
    public_values: &[F],
    lookups: &[Lookup<F>],
    lookup_data: &[LookupData<EF>],
    lookup_gadget: &LG,
) where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    A: EonAir<F, EF>,
    for<'a> A: AirLookupHandler<DebugConstraintBuilder<'a, F, EF>>,
    LG: p3_lookup::lookup_traits::LookupGadget,
{
    let height = main.height();
    let preprocessed = <A as EonAir<F, EF>>::preprocessed_trace(&*air);

    (0..height).for_each(|row_index| {
        let row_index_next = (row_index + 1) % height;

        // row_index < height so we can used unchecked indexing.
        let local = unsafe { main.row_slice_unchecked(row_index) };
        // row_index_next < height so we can used unchecked indexing.
        let next = unsafe { main.row_slice_unchecked(row_index_next) };
        let main = ViewPair::new(
            RowMajorMatrixView::new_row(&*local),
            RowMajorMatrixView::new_row(&*next),
        );

        // Keep these Vecs in the outer scope so their backing memory lives
        // long enough for the `RowMajorMatrixView` references stored in `aux`.
        let pp_local_guard = preprocessed
            .as_ref()
            .map(|pm| unsafe { pm.row_slice_unchecked(row_index) });
        let pp_next_guard = preprocessed
            .as_ref()
            .map(|pm| unsafe { pm.row_slice_unchecked(row_index_next) });

        let preprocessed_pair: Option<ViewPair<'_, F>> =
            match (pp_local_guard.as_deref(), pp_next_guard.as_deref()) {
                (Some(l), Some(n)) => Some(ViewPair::new(
                    RowMajorMatrixView::new_row(l),
                    RowMajorMatrixView::new_row(n),
                )),
                _ => None,
            };

        let perm_local_guard = permutation.map(|pm| unsafe { pm.row_slice_unchecked(row_index) });
        let perm_next_guard =
            permutation.map(|pm| unsafe { pm.row_slice_unchecked(row_index_next) });

        let empty: &[EF] = &[];
        let permutation_pair: ViewPair<'_, EF> = ViewPair::new(
            RowMajorMatrixView::new_row(perm_local_guard.as_deref().unwrap_or(empty)),
            RowMajorMatrixView::new_row(perm_next_guard.as_deref().unwrap_or(empty)),
        );

        let mut builder = DebugConstraintBuilder {
            row_index,
            main,
            preprocessed: preprocessed_pair,
            permutation: permutation_pair,
            permutation_challenges,
            public_values,
            is_first_row: F::from_bool(row_index == 0),
            is_last_row: F::from_bool(row_index == height - 1),
            is_transition: F::from_bool(row_index != height - 1),
        };

        AirLookupHandler::<DebugConstraintBuilder<'_, F, EF>>::eval(
            air,
            &mut builder,
            lookups,
            lookup_data,
            lookup_gadget,
        );
    });
}

use p3_air::Air;

pub(crate) fn check_constraints_without_lookups<F, EF, A>(
    air: &mut A,
    main: &RowMajorMatrix<F>,
    public_values: &[F],
) where
    F: Field,
    EF: ExtensionField<F>,
    // 注意：这里要求的是 p3_air::Air<DebugConstraintBuilder<...>>
    for<'a> A: Air<DebugConstraintBuilder<'a, F, EF>>,
{
    let height = main.height();

    // BaseAir 是 Air 的前置约束，所以可以拿到 preprocessed_trace（如果你需要）
    let preprocessed = air.preprocessed_trace();

    for row_index in 0..height {
        let row_index_next = (row_index + 1) % height;

        let local = unsafe { main.row_slice_unchecked(row_index) };
        let next = unsafe { main.row_slice_unchecked(row_index_next) };
        let main_pair = ViewPair::new(
            RowMajorMatrixView::new_row(&*local),
            RowMajorMatrixView::new_row(&*next),
        );

        // preprocessed（可选）
        let pp_local_guard = preprocessed
            .as_ref()
            .map(|pm| unsafe { pm.row_slice_unchecked(row_index) });
        let pp_next_guard = preprocessed
            .as_ref()
            .map(|pm| unsafe { pm.row_slice_unchecked(row_index_next) });

        let preprocessed_pair: Option<ViewPair<'_, F>> =
            match (pp_local_guard.as_deref(), pp_next_guard.as_deref()) {
                (Some(l), Some(n)) => Some(ViewPair::new(
                    RowMajorMatrixView::new_row(l),
                    RowMajorMatrixView::new_row(n),
                )),
                _ => None,
            };

        // permutation 先用空（因为你当前 prover 没接 permutation/lookup）
        let empty_ef: &[EF] = &[];
        let permutation_pair: ViewPair<'_, EF> = ViewPair::new(
            RowMajorMatrixView::new_row(empty_ef),
            RowMajorMatrixView::new_row(empty_ef),
        );
        let permutation_challenges: &[EF] = &[];

        let mut builder = DebugConstraintBuilder {
            row_index,
            main: main_pair,
            preprocessed: preprocessed_pair,
            permutation: permutation_pair,
            permutation_challenges,
            public_values,
            is_first_row: F::from_bool(row_index == 0),
            is_last_row: F::from_bool(row_index == height - 1),
            is_transition: F::from_bool(row_index != height - 1),
        };

        // 关键点：直接走 Air::eval
        air.eval(&mut builder);
    }
}

/// A builder that runs constraint assertions during testing.
///
/// Used in conjunction with [`check_constraints`] to simulate
/// an execution trace and verify that the AIR logic enforces all constraints.
#[derive(Debug)]
pub struct DebugConstraintBuilder<'a, F: Field, EF: ExtensionField<F>> {
    /// The index of the row currently being evaluated.
    row_index: usize,
    /// A view of the current and next row as a vertical pair.
    main: ViewPair<'a, F>,
    /// A view of the preprocessed current and next row as a vertical pair (if present).
    preprocessed: Option<ViewPair<'a, F>>,
    /// A view of the permutation current and next row as a vertical pair.
    permutation: ViewPair<'a, EF>,
    /// A view of the permutation randomness used for permutation arguments.
    permutation_challenges: &'a [EF],
    /// The public values provided for constraint validation (e.g. inputs or outputs).
    public_values: &'a [F],
    /// A flag indicating whether this is the first row.
    is_first_row: F,
    /// A flag indicating whether this is the last row.
    is_last_row: F,
    /// A flag indicating whether this is a transition row (not the last row).
    is_transition: F,
}

impl<'a, F, EF> EonAirBuilder for DebugConstraintBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = ViewPair<'a, F>;
    type PublicVar = F;
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;
    type MP = ViewPair<'a, EF>;
    type RandomVar = EF;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("DebugConstraintBuilder only supports transition window of size 2");
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let value = x.into();
        assert!(
            value == F::ZERO,
            "Constraint failed at row {}: expected zero, got {:?}",
            self.row_index,
            value
        );
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    fn preprocessed(&self) -> Self::M {
        self.preprocessed.unwrap_or_else(|| {
            // Return an empty ViewPair if there are no preprocessed columns
            let empty: &[F] = &[];
            ViewPair::new(
                RowMajorMatrixView::new_row(empty),
                RowMajorMatrixView::new_row(empty),
            )
        })
    }

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let value = x.into();
        assert!(
            value == EF::ZERO,
            "Extension field constraint failed at row {}: expected zero, got {:?}",
            self.row_index,
            value
        );
    }

    fn permutation(&self) -> Self::MP {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }
}

impl_p3_air_builder_traits!(DebugConstraintBuilder<'a, F, EF>
where
    F: eon_air::Field,
    EF: ExtensionField<F>
);

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use eon_air::Field;
    use eon_air::PrimeCharacteristicRing;
    use eon_air::impl_p3_air_traits;
    use p3_bn254::Fr;
    use p3_lookup::logup::LogUpGadget;
    use p3_lookup::lookup_traits::{Lookup, LookupData};

    type EF = Fr; // For these tests, we use Fr as both base and extension field.

    use super::*;

    /// A test AIR that enforces a simple linear transition logic:
    /// - Each cell in the next row must equal the current cell plus 1 (i.e., `next = current + 1`)
    /// - On the last row, the current row must match the provided public values.
    ///
    /// This is useful for validating constraint evaluation, transition logic,
    /// and row condition flags (first/last/transition).
    #[derive(Debug)]
    struct RowLogicAir {
        with_aux: bool,
    }

    impl<F, EF> EonAir<F, EF> for RowLogicAir
    where
        F: eon_air::Field,
        EF: eon_air::Field,
    {
        fn width(&self) -> usize {
            2
        }

        fn eval<AB: EonAirBuilder<F = F>>(&self, builder: &mut AB) {
            let main = builder.main();

            // ======================
            // main trace
            // ======================
            // | main1             | main2            |
            // | row[i]            | perm(main1)[i]   |
            // | row[i+1]=row[i]+1 | perm(main1)[i+1] |

            let a = main.get(0, 0).unwrap();
            let b = main.get(1, 0).unwrap();

            // New logic: enforce row[i+1] = row[i] + 1, only on transitions
            builder.when_transition().assert_eq(b, a + F::ONE);

            // ======================
            // public input
            // ======================
            // Add public value equality on last row for extra coverage
            let public_values = builder.public_values();
            let pv0 = public_values[0];
            let pv1 = public_values[1];

            let mut when_last = builder.when_last_row();
            when_last.assert_eq(main.get(0, 0).unwrap(), pv0);
            when_last.assert_eq(main.get(0, 1).unwrap(), pv1);
        }
    }

    impl_p3_air_traits!(RowLogicAir, base = Fr, challenge = Fr);

    // A very simple permutation
    fn permute<F: Field>(x: &[F]) -> Vec<F> {
        x.iter().rev().cloned().collect::<Vec<F>>()
    }

    // Generate a main trace.
    // The first column is incremental
    // The second column is the rev of the first column
    fn gen_main(main_col: &[Fr]) -> RowMajorMatrix<Fr> {
        let main_rev = permute(main_col);
        let main_values = main_col
            .iter()
            .zip(main_rev.iter())
            .flat_map(|(a, b)| vec![a, b])
            .cloned()
            .collect();
        RowMajorMatrix::new(main_values, 2)
    }

    fn empty_lookup_inputs() -> (Vec<Lookup<Fr>>, Vec<LookupData<Fr>>) {
        (vec![], vec![])
    }

    #[test]
    fn test_incremental_rows_with_last_row_check() {
        // Each row = previous + 1, with 4 rows total, 2 columns.
        // Last row must match public values [4, 4]
        let mut air = RowLogicAir { with_aux: false };
        let values = vec![
            Fr::ONE,    // Row 0
            Fr::new(2), // Row 1
            Fr::new(3), // Row 2
            Fr::new(4), // Row 3 (last)
        ];
        let main = gen_main(&values);

        let (lookups, lookup_data) = empty_lookup_inputs();
        let lookup_gadget = LogUpGadget;
        check_constraints(
            &mut air,
            &main,
            None,
            &[],
            &vec![Fr::new(4), Fr::new(1)],
            &lookups,
            &lookup_data,
            &lookup_gadget,
        );
    }

    #[test]
    #[should_panic]
    fn test_incorrect_increment_logic() {
        // Row 2 does not equal row 1 + 1 → should fail on transition from row 1 to 2.
        let mut air = RowLogicAir { with_aux: false };
        let values = vec![
            Fr::ONE,
            Fr::ONE, // Row 0
            Fr::new(2),
            Fr::new(2), // Row 1
            Fr::new(5),
            Fr::new(5), // Row 2 (wrong)
            Fr::new(6),
            Fr::new(6), // Row 3
        ];
        let main = RowMajorMatrix::new(values, 2);

        let (lookups, lookup_data) = empty_lookup_inputs();
        let lookup_gadget = LogUpGadget;

        check_constraints(
            &mut air,
            &main,
            None,
            &[],
            &vec![Fr::new(6); 2],
            &lookups,
            &lookup_data,
            &lookup_gadget,
        );
    }

    #[test]
    #[should_panic]
    fn test_wrong_last_row_public_value() {
        // The transition logic is fine, but public value check fails at the last row.
        let mut air = RowLogicAir { with_aux: false };
        let values = vec![
            Fr::ONE,
            Fr::ONE, // Row 0
            Fr::new(2),
            Fr::new(2), // Row 1
            Fr::new(3),
            Fr::new(3), // Row 2
            Fr::new(4),
            Fr::new(4), // Row 3
        ];
        let main = RowMajorMatrix::new(values, 2);

        let (lookups, lookup_data) = empty_lookup_inputs();
        let lookup_gadget = LogUpGadget;
        // Wrong public value on column 1
        check_constraints(
            &mut air,
            &main,
            None,
            &[],
            &vec![Fr::new(4), Fr::new(5)],
            &lookups,
            &lookup_data,
            &lookup_gadget,
        );
    }

    #[test]
    fn test_single_row_wraparound_logic() {
        // A single-row matrix still performs a wraparound check with itself.
        // row[0] == row[0] + 1 ⇒ fails unless handled properly by transition logic.
        // Here: is_transition == false ⇒ so no assertions are enforced.
        let mut air = RowLogicAir { with_aux: false };
        let values = vec![
            Fr::new(99),
            Fr::new(77), // Row 0
        ];
        let main = RowMajorMatrix::new(values, 2);
        let (lookups, lookup_data) = empty_lookup_inputs();
        let lookup_gadget = LogUpGadget;
        check_constraints(
            &mut air,
            &main,
            None,
            &[],
            &vec![Fr::new(99), Fr::new(77)],
            &lookups,
            &lookup_data,
            &lookup_gadget,
        );
    }
}
