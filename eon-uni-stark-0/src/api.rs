use eon_air::EonAir;
use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;

use crate::{Proof, StarkGenericConfig, Val};

mod sealed {
    use super::*;

    pub trait ProveInternal<SC> {
        fn prove_internal(
            config: &SC,
            air: &Self,
            trace: RowMajorMatrix<Val<SC>>,
            public_values: &[Val<SC>],
        ) -> Proof<SC>;
    }

    pub trait VerifyInternal<SC> {
        type Error;

        fn verify_internal(
            config: &SC,
            air: &Self,
            proof: &Proof<SC>,
            public_values: &[Val<SC>],
        ) -> Result<(), Self::Error>;
    }

    #[cfg(not(debug_assertions))]
    impl<SC, A> ProveInternal<SC> for A
    where
        SC: StarkGenericConfig,
        A: p3_air::Air<crate::symbolic_builder::SymbolicAirBuilder<Val<SC>>>
            + for<'a> p3_air::Air<crate::folder::ProverConstraintFolder<'a, SC>>,
    {
        fn prove_internal(
            config: &SC,
            air: &Self,
            trace: RowMajorMatrix<Val<SC>>,
            public_values: &[Val<SC>],
        ) -> Proof<SC> {
            crate::prover::prove::<SC, A>(config, air, trace, public_values)
        }
    }

    #[cfg(debug_assertions)]
    impl<SC, A> ProveInternal<SC> for A
    where
        SC: StarkGenericConfig,
        A: p3_air::Air<crate::symbolic_builder::SymbolicAirBuilder<Val<SC>>>
            + for<'a> p3_air::Air<crate::folder::ProverConstraintFolder<'a, SC>>
            + for<'a> p3_air::Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    {
        fn prove_internal(
            config: &SC,
            air: &Self,
            trace: RowMajorMatrix<Val<SC>>,
            public_values: &[Val<SC>],
        ) -> Proof<SC> {
            crate::prover::prove::<SC, A>(config, air, trace, public_values)
        }
    }

    impl<SC, A> VerifyInternal<SC> for A
    where
        SC: StarkGenericConfig,
        A: p3_air::Air<crate::symbolic_builder::SymbolicAirBuilder<Val<SC>>>,
    {
        type Error = crate::verifier::VerificationError; // 如果你仓库里不是这个名字，改成实际的

        fn verify_internal(
            config: &SC,
            air: &Self,
            proof: &Proof<SC>,
            public_values: &[Val<SC>],
        ) -> Result<(), Self::Error> {
            crate::verifier::verify::<SC, A>(config, air, proof, public_values)
        }
    }
}

pub fn prove<SC, A>(
    config: &SC,
    air: &A,
    trace: RowMajorMatrix<Val<SC>>,
    public_values: &[Val<SC>],
) -> Proof<SC>
where
    SC: StarkGenericConfig + Sync,
    A: EonAir<Val<SC>, SC::Challenge>,
    Val<SC>: TwoAdicField,
{
    sealed::ProveInternal::<SC>::prove_internal(config, air, trace, public_values)
}

pub fn verify<SC, A>(
    config: &SC,
    air: &A,
    proof: &Proof<SC>,
    public_values: &[Val<SC>],
) -> Result<(), <A as sealed::VerifyInternal<SC>>::Error>
where
    SC: StarkGenericConfig + Sync,
    A: EonAir<Val<SC>, SC::Challenge>,
{
    sealed::VerifyInternal::<SC>::verify_internal(config, air, proof, public_values)
}
