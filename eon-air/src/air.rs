use crate::{EonAirBuilder, RowMajorMatrix};
use alloc::vec;
use alloc::vec::Vec;
use p3_field::Field;
use p3_lookup::lookup_traits::{Kind, Lookup, LookupInput};

/// Super trait for all AIR definitions in the Eon ecosystem.
///
/// This trait contains all methods from `BaseAir`, `BaseAirWithPublicValues`,
/// `BaseAirWithAuxTrace`, and `Air`. Implementers only need to implement this
/// single trait.
///
/// To use your AIR with the STARK prover/verifier, you'll need to also implement
/// the p3-air traits using the `impl_p3_air_traits!` macro.
///
/// # Type Parameters
///
/// - `F`: The base field type
/// - `EF`: The extension field type (used for auxiliary traces like LogUp)
///
/// # Required Methods
///
/// - [`width`](EonAir::width) - Number of columns in the main trace
/// - [`eval`](EonAir::eval) - Constraint evaluation logic
///
/// # Optional Methods (with default implementations)
///
/// All other methods have default implementations that can be overridden as needed.
pub trait EonAir<F: Field, EF>: Sync {
    // ==================== BaseAir Methods ====================

    /// The number of columns (a.k.a. registers) in this AIR.
    fn width(&self) -> usize;

    /// Return an optional preprocessed trace matrix to be included in the prover's trace.
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }

    // ==================== BaseAirWithPublicValues Methods ====================

    /// Return the number of expected public values.
    fn num_public_values(&self) -> usize {
        0
    }

    // ==================== Lookup (AirLookupHandler) Methods ====================

    /// Allocate (and return) column indices in the permutation trace for a lookup.
    ///
    /// Default: no lookup columns.
    ///
    /// When an AIR uses lookups, it should override this method to return freshly
    /// allocated permutation-trace columns (indices) and update any internal state
    /// needed for allocation.
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        vec![]
    }

    /// Return all lookups registered by this AIR.
    ///
    /// Default: empty (no lookups).
    ///
    /// When an AIR uses lookups, it should override this method to return the list
    /// of `Lookup<F>` descriptors (each containing expressions + multiplicities + columns).
    fn get_lookups(&mut self) -> Vec<Lookup<F>>
    where
        F: Field,
    {
        vec![]
    }

    /// Convenience helper to build a `Lookup<F>` from inputs and auto-fill `columns`
    /// using `add_lookup_columns()`.
    ///
    /// Default implementation matches the canonical Plonky3 lookup handler logic:
    /// - `Direction::Positive` keeps multiplicity as-is
    /// - `Direction::Negative` negates multiplicity
    fn register_lookup(&mut self, kind: Kind, lookup_inputs: &[LookupInput<F>]) -> Lookup<F>
    where
        F: Field,
    {
        let columns = self.add_lookup_columns();

        let mut element_exprs = Vec::with_capacity(lookup_inputs.len());
        let mut multiplicities_exprs = Vec::with_capacity(lookup_inputs.len());

        for (elements, multiplicity, direction) in lookup_inputs.iter() {
            element_exprs.push(elements.to_vec());
            multiplicities_exprs.push(direction.multiplicity((*multiplicity).clone()));
        }

        Lookup {
            kind,
            element_exprs,
            multiplicities_exprs,
            columns,
        }
    }

    // ==================== Air Methods ====================

    /// Evaluate all AIR constraints using the provided builder.
    ///
    /// The builder provides both the trace on which the constraints
    /// are evaluated on as well as the method of accumulating the
    /// constraint evaluations.
    ///
    /// # Arguments
    /// - `builder`: Mutable reference to a `EonAirBuilder` for defining constraints.
    fn eval<AB: EonAirBuilder<F = F, EF = EF>>(&self, builder: &mut AB);
}

/// Helper macro to implement p3-air traits by delegating to EonAir.
///
/// This macro generates the boilerplate implementations of `BaseAir`, `BaseAirWithPublicValues`,
/// `BaseAirWithAuxTrace`, and `Air` that simply delegate to your `EonAir` implementation.
///
/// # Usage
///
/// ```rust,ignore
/// use eon_air::{EonAir, EonAirBuilder, impl_p3_air_traits};
/// use p3_field::extension::BinomialExtensionField;
///
/// struct MyAir { width: usize }
///
/// impl<F: Field, EF: ExtensionField<F>> EonAir<F, EF> for MyAir {
///     fn width(&self) -> usize { self.width }
///     fn eval<AB: EonAirBuilder<F = F, EF = EF>>(&self, builder: &mut AB) {
///         // constraints...
///     }
/// }
///
/// // Generate all p3-air trait implementations
/// impl_p3_air_traits!(MyAir, BinomialExtensionField<_, 2>);
/// ```
#[macro_export]
macro_rules! impl_p3_air_traits {
    ($air_type:ty, $ef_type:ty) => {
        impl<F: $crate::Field> $crate::BaseAir<F> for $air_type
        where
            $ef_type: $crate::ExtensionField<F>,
        {
            fn width(&self) -> usize {
                <Self as $crate::EonAir<F, $ef_type>>::width(self)
            }

            fn preprocessed_trace(&self) -> Option<$crate::p3_matrix::dense::RowMajorMatrix<F>> {
                <Self as $crate::EonAir<F, $ef_type>>::preprocessed_trace(self)
            }
        }

        impl<F: $crate::Field> $crate::BaseAirWithPublicValues<F> for $air_type
        where
            $ef_type: $crate::ExtensionField<F>,
        {
            fn num_public_values(&self) -> usize {
                <Self as $crate::EonAir<F, $ef_type>>::num_public_values(self)
            }
        }

        // Bridge into Plonky3 lookup handler.
        impl<AB> p3_lookup::lookup_traits::AirLookupHandler<AB> for $air_type
        where
            $ef_type: $crate::ExtensionField<<AB as $crate::AirBuilder>::F>,
            AB: $crate::EonAirBuilder<F = <AB as $crate::AirBuilder>::F, EF = $ef_type>
                + $crate::AirBuilderWithPublicValues
                + $crate::PairBuilder
                + $crate::PermutationAirBuilder,
        {
            fn add_lookup_columns(&mut self) -> Vec<usize> {
                <Self as $crate::EonAir<<AB as $crate::AirBuilder>::F, $ef_type>>::add_lookup_columns(self)
            }

            fn get_lookups(
                &mut self,
            ) -> Vec<p3_lookup::lookup_traits::Lookup<<AB as $crate::AirBuilder>::F>> {
                let lookups: Vec<p3_lookup::lookup_traits::Lookup<<AB as $crate::AirBuilder>::F>> =
                    <Self as $crate::EonAir<<AB as $crate::AirBuilder>::F, $ef_type>>::get_lookups(self);
                lookups
            }

            fn register_lookup(
                &mut self,
                kind: p3_lookup::lookup_traits::Kind,
                lookup_inputs: &[p3_lookup::lookup_traits::LookupInput<
                    <AB as $crate::AirBuilder>::F,
                >],
            ) -> p3_lookup::lookup_traits::Lookup<<AB as $crate::AirBuilder>::F> {
                <Self as $crate::EonAir<<AB as $crate::AirBuilder>::F, $ef_type>>::register_lookup(self, kind, lookup_inputs)
            }
        }

        impl<AB> $crate::Air<AB> for $air_type
        where
            AB: $crate::EonAirBuilder + $crate::AirBuilder,
        {

            fn eval(&self, builder: &mut AB) {
                <Self as $crate::EonAir<<AB as $crate::EonAirBuilder>::F, AB::EF>>::eval(self, builder)
            }
        }
    };
}
