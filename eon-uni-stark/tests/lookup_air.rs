use core::borrow::Borrow;

use eon_air::{EonAir, EonAirBuilder, impl_p3_air_traits};
use eon_uni_stark::{StarkConfig, prove, verify};

use p3_uni_stark::{Entry, SymbolicExpression, SymbolicVariable};

use p3_bn254::{Fr, Poseidon2Bn254};
use p3_challenger::DuplexChallenger;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_kzg::KzgPcs;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use p3_lookup::lookup_traits::{Direction, Kind, Lookup, LookupInput};

/// main trace:
///   col0 = val
///   col1 = table
///
/// lookup:  +table  -val  ==> multiset(table) == multiset(val)
pub struct LookupMultisetEqAir {
    next_lookup_col: usize,
}

impl Default for LookupMultisetEqAir {
    fn default() -> Self {
        Self { next_lookup_col: 0 }
    }
}

impl<F: Field, EF: ExtensionField<F>> EonAir<F, EF> for LookupMultisetEqAir {
    fn width(&self) -> usize {
        NUM_LOOKUP_COLS
    }

    fn eval<AB: EonAirBuilder<F = F, EF = EF>>(&self, _builder: &mut AB) {}

    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let col = self.next_lookup_col;
        self.next_lookup_col += 1;
        vec![col]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.next_lookup_col = 0;

        // val = main(0)[0]
        let val_var = SymbolicVariable::new(Entry::Main { offset: 0 }, 0);
        let val_expr = SymbolicExpression::Variable(val_var);

        // table = main(0)[1]
        let table_var = SymbolicVariable::new(Entry::Main { offset: 0 }, 1);
        let table_expr = SymbolicExpression::Variable(table_var);

        let one = SymbolicExpression::Constant(F::ONE);

        // Direction::Receive => +mult, Direction::Send => -mult
        // Objective: sum(+table) + sum(-val) == 0  <=> multiset(table) == multiset(val)
        let inputs: [LookupInput<F>; 2] = [
            (vec![table_expr], one.clone(), Direction::Receive),
            (vec![val_expr], one, Direction::Send),
        ];

        vec![<Self as EonAir<F, EF>>::register_lookup(
            self,
            Kind::Local,
            &inputs,
        )]
    }
}

const NUM_LOOKUP_COLS: usize = 2;

pub struct LookupRow<F> {
    pub val: F,
    pub table: F,
}

impl<F> LookupRow<F> {
    const fn new(val: F, table: F) -> Self {
        Self { val, table }
    }
}

impl<F> Borrow<LookupRow<F>> for [F] {
    fn borrow(&self) -> &LookupRow<F> {
        debug_assert_eq!(self.len(), NUM_LOOKUP_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<LookupRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

pub fn generate_trace_rows<F: PrimeCharacteristicRing + Copy + Send + Sync>(
    n: usize,
    bad: bool,
) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(F::zero_vec(n * NUM_LOOKUP_COLS), NUM_LOOKUP_COLS);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<LookupRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    for i in 0..n {
        let val = if bad && i == 0 { F::ONE } else { F::ZERO };
        rows[i] = LookupRow::new(val, F::ZERO);
    }

    trace
}

type Val = Fr;
type Perm = Poseidon2Bn254<3>;
type Challenge = Val; // BN254 base field, no extension
type Challenger = DuplexChallenger<Val, Perm, 3, 2>;
type Pcs = KzgPcs;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

impl_p3_air_traits!(LookupMultisetEqAir, base = Fr, challenge = Fr);

fn build_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng(4, 22, &mut rng);
    let pcs = Pcs::new(1024, Fr::from_u64(12345));
    let challenger = Challenger::new(perm);
    MyConfig::new(pcs, challenger)
}

#[test]
fn test_lookup_ok() {
    let config = build_config();
    let trace = generate_trace_rows::<Val>(1 << 3, false);
    let pis: Vec<Fr> = vec![];

    let mut air = LookupMultisetEqAir::default();
    let proof = prove(&config, &mut air, trace, &pis);
    verify(&config, &mut air, &proof, &pis).expect("verification failed");
}

#[cfg(debug_assertions)]
#[test]
#[should_panic]
fn test_lookup_bad_trace_should_fail_verifier() {
    let config = build_config();
    let trace = generate_trace_rows::<Val>(1 << 3, true); // 第 0 行 val=1，table=0
    let pis: Vec<Fr> = vec![];

    let mut air = LookupMultisetEqAir::default();
    let proof = prove(&config, &mut air, trace, &pis);

    assert!(verify(&config, &mut air, &proof, &pis).is_err());
}

#[cfg(not(debug_assertions))]
#[test]
fn test_lookup_bad_trace_should_fail_verifier() {
    let proof = prove(&config, &mut air, bad_trace, &public_values);
    assert!(!verify(&config, &air, &proof, &public_values));
}

#[test]
fn test_lookup_tamper_permutation_opening_should_fail() {
    let config = build_config();
    let trace = generate_trace_rows::<Val>(1 << 3, false);
    let pis: Vec<Fr> = vec![];

    let mut air = LookupMultisetEqAir::default();
    let mut proof = prove(&config, &mut air, trace, &pis);

    let perm_local = proof
        .opened_values
        .permutation_local
        .as_mut()
        .expect("expected permutation_local to exist (lookup enabled)");
    assert!(!perm_local.is_empty());

    perm_local[0] += Fr::ONE;

    assert!(verify(&config, &mut air, &proof, &pis).is_err());
}
