//! Fibonacci AIR (Algebraic Intermediate Representation)
//!
//! This module defines the AIR for proving Fibonacci sequence computation.
//! The AIR enforces the Fibonacci recurrence relation: f(n+2) = f(n+1) + f(n)

use alloc::vec::Vec;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

/// Number of columns in the Fibonacci trace
/// We use 2 columns to represent consecutive Fibonacci numbers
const NUM_FIBONACCI_COLS: usize = 2;

/// Fibonacci AIR
///
/// The trace has 2 columns representing consecutive Fibonacci numbers:
/// - Column 0: f(n)
/// - Column 1: f(n+1)
///
/// Constraints:
/// - Boundary: First row should have initial values (e.g., [1, 1])
/// - Transition: For each row i, the next row satisfies:
///   - next[0] = current[1]
///   - next[1] = current[0] + current[1]
#[derive(Clone, Debug)]
pub struct FibonacciAir {
    /// Number of Fibonacci numbers to compute (trace height)
    pub num_steps: usize,
}

impl FibonacciAir {
    /// Create a new Fibonacci AIR
    pub fn new(num_steps: usize) -> Self {
        Self { num_steps }
    }

    /// Generate a trace for the Fibonacci sequence
    ///
    /// The trace starts with [1, 1] and computes subsequent Fibonacci numbers.
    pub fn generate_trace<F: PrimeCharacteristicRing + Send + Sync>(&self) -> RowMajorMatrix<F> {
        let mut trace = Vec::with_capacity(self.num_steps * NUM_FIBONACCI_COLS);

        // Initial values: f(0) = 1, f(1) = 1
        let mut a = F::ONE;
        let mut b = F::ONE;

        for _ in 0..self.num_steps {
            trace.push(a.clone());
            trace.push(b.clone());

            // Compute next Fibonacci number
            let c = a.clone() + b.clone();
            a = b;
            b = c;
        }

        RowMajorMatrix::new(trace, NUM_FIBONACCI_COLS)
    }
}

impl<F: Field> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for FibonacciAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        let next = main.row_slice(1).unwrap();

        // Get current and next values
        let current_0 = local[0].clone();
        let current_1 = local[1].clone();
        let next_0 = next[0].clone();
        let next_1 = next[1].clone();

        // Transition constraints:
        // 1. next[0] should equal current[1]
        builder
            .when_transition()
            .assert_eq(next_0, current_1.clone());

        // 2. next[1] should equal current[0] + current[1]
        builder
            .when_transition()
            .assert_eq(next_1, current_0 + current_1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_bn254::Fr as Bn254Fr;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;

    #[test]
    fn test_fibonacci_trace_generation() {
        let air = FibonacciAir::new(10);
        let trace: RowMajorMatrix<Bn254Fr> = air.generate_trace();

        assert_eq!(trace.height(), 10);
        assert_eq!(trace.width(), 2);

        // Check first few Fibonacci numbers (1, 1, 2, 3, 5, 8, 13, 21, 34, 55)
        let expected = [
            (1u64, 1u64),
            (1, 2),
            (2, 3),
            (3, 5),
            (5, 8),
            (8, 13),
            (13, 21),
            (21, 34),
            (34, 55),
            (55, 89),
        ];

        for (i, (exp_a, exp_b)) in expected.iter().enumerate() {
            assert_eq!(
                trace.get(i, 0).unwrap(),
                Bn254Fr::from_u64(*exp_a),
                "Mismatch at row {}, col 0",
                i
            );
            assert_eq!(
                trace.get(i, 1).unwrap(),
                Bn254Fr::from_u64(*exp_b),
                "Mismatch at row {}, col 1",
                i
            );
        }
    }

    #[test]
    fn test_air_width() {
        let air = FibonacciAir::new(16);
        assert_eq!(<FibonacciAir as BaseAir<Bn254Fr>>::width(&air), 2);
    }
}
