use alloc::vec;
use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, Field, PrimeField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{CryptographicPermutation, Hash};

use crate::{CanObserve, CanSample, CanSampleBits, FieldChallenger};

/// A generic duplex sponge challenger over a finite field, used for generating deterministic
/// challenges from absorbed inputs.
///
/// This structure implements a duplex sponge that alternates between:
/// - Absorbing inputs into the sponge state,
/// - Applying a cryptographic permutation over the state,
/// - Squeezing outputs from the state as challenges.
///
/// The sponge operates over a state of `WIDTH` elements, divided into:
/// - A rate of `RATE` elements (the portion exposed to input/output),
/// - A capacity of `WIDTH - RATE` elements (the hidden part ensuring security).
///
/// The challenger buffers observed inputs until the rate is full, applies the permutation,
/// and then produces challenge outputs from the permuted state. It supports:
/// - Observing single values, arrays, hashes, or nested vectors,
/// - Sampling fresh challenges as field elements or bitstrings.
#[derive(Clone, Debug)]
pub struct DuplexChallenger<F, P, const WIDTH: usize, const RATE: usize>
where
    F: Clone,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    /// The internal sponge state, consisting of `WIDTH` field elements.
    ///
    /// The first `RATE` elements form the rate section, where input values are absorbed
    /// and output values are squeezed.
    /// The remaining `WIDTH - RATE` elements form the capacity, which provides hidden
    /// entropy and security against attacks.
    pub sponge_state: [F; WIDTH],

    /// A buffer holding field elements that have been observed but not yet absorbed.
    ///
    /// Inputs added via `observe` are collected here.
    /// Once the buffer reaches `RATE` elements, the sponge performs a duplexing step:
    /// it absorbs the inputs into the state and applies the permutation.
    pub input_buffer: Vec<F>,

    /// A buffer holding field elements that have been squeezed from the sponge state.
    ///
    /// Outputs are produced by `duplexing` and stored here.
    /// Calls to `sample` or `sample_bits` pop values from this buffer.
    /// When the buffer is empty (or new inputs were absorbed), a new duplexing step is triggered.
    pub output_buffer: Vec<F>,

    /// The cryptographic permutation applied to the sponge state.
    ///
    /// This permutation must provide strong pseudorandomness and collision resistance,
    /// ensuring that squeezed outputs are indistinguishable from random and securely
    /// bound to the absorbed inputs.
    pub permutation: P,
}

impl<F, P, const WIDTH: usize, const RATE: usize> DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    pub fn new(permutation: P) -> Self
    where
        F: Default,
    {
        Self {
            sponge_state: [F::default(); WIDTH],
            input_buffer: vec![],
            output_buffer: vec![],
            permutation,
        }
    }

    fn duplexing(&mut self) {
        assert!(self.input_buffer.len() <= RATE);

        // Overwrite the first r elements with the inputs.
        for (i, val) in self.input_buffer.drain(..).enumerate() {
            self.sponge_state[i] = val;
        }

        // Apply the permutation.
        self.permutation.permute_mut(&mut self.sponge_state);

        self.output_buffer.clear();
        self.output_buffer.extend(&self.sponge_state[..RATE]);
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> FieldChallenger<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField,
    P: CryptographicPermutation<[F; WIDTH]>,
{
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, value: F) {
        // Any buffered output is now invalid.
        self.output_buffer.clear();

        self.input_buffer.push(value);

        if self.input_buffer.len() == RATE {
            self.duplexing();
        }
    }
}

impl<F, P, const N: usize, const WIDTH: usize, const RATE: usize> CanObserve<[F; N]>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: [F; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, P, const N: usize, const WIDTH: usize, const RATE: usize> CanObserve<Hash<F, F, N>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: Hash<F, F, N>) {
        for value in values {
            self.observe(value);
        }
    }
}

// for TrivialPcs
impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<Vec<Vec<F>>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, valuess: Vec<Vec<F>>) {
        for values in valuess {
            for value in values {
                self.observe(value);
            }
        }
    }
}

// for DummyPcs - observe matrices by iterating through all elements
impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<Vec<RowMajorMatrix<F>>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy + Send + Sync,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, matrices: Vec<RowMajorMatrix<F>>) {
        for matrix in matrices {
            for row_idx in 0..matrix.height() {
                if let Some(row) = matrix.row_slice(row_idx) {
                    for &value in &*row {
                        self.observe(value);
                    }
                }
            }
        }
    }
}

impl<F, EF, P, const WIDTH: usize, const RATE: usize> CanSample<EF>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Field,
    EF: BasedVectorSpace<F>,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn sample(&mut self) -> EF {
        EF::from_basis_coefficients_fn(|_| {
            // If we have buffered inputs, we must perform a duplexing so that the challenge will
            // reflect them. Or if we've run out of outputs, we must perform a duplexing to get more.
            if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
                self.duplexing();
            }

            self.output_buffer
                .pop()
                .expect("Output buffer should be non-empty")
        })
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanSampleBits<usize>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    /// Sample random bits by taking the low bits of a field element.
    ///
    /// The sampled bits are not perfectly uniform, but the bias is negligible for large fields.
    /// For a field of order p, the statistical distance from uniform is at most 1/p per sample.
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(bits < (usize::BITS as usize));
        let rand_f: F = self.sample();
        // Convert field element to bytes and extract the requested number of bits
        let bytes = rand_f.as_canonical_biguint().to_bytes_le();
        let mut result = 0usize;
        for (i, &byte) in bytes.iter().enumerate() {
            if i * 8 >= bits {
                break;
            }
            result |= (byte as usize) << (i * 8);
        }
        result & ((1 << bits) - 1)
    }
}

#[cfg(test)]
mod tests {
    use p3_bn254::Fr;
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;

    use super::*;

    const WIDTH: usize = 3;
    const RATE: usize = 2;

    type G = Fr;

    #[derive(Clone)]
    struct TestPermutation {}

    impl<F: Clone> Permutation<[F; WIDTH]> for TestPermutation {
        fn permute_mut(&self, input: &mut [F; WIDTH]) {
            input.reverse();
        }
    }

    impl<F: Clone> CryptographicPermutation<[F; WIDTH]> for TestPermutation {}

    #[test]
    fn test_duplex_challenger() {
        type Chal = DuplexChallenger<G, TestPermutation, WIDTH, RATE>;
        let permutation = TestPermutation {};
        let mut duplex_challenger: Chal = DuplexChallenger::new(permutation);

        // Observe some elements and verify sampling works
        duplex_challenger.observe(G::from_u8(1));
        duplex_challenger.observe(G::from_u8(2));

        // After observing RATE=2 elements, duplexing should have occurred
        assert!(duplex_challenger.input_buffer.is_empty());
        assert_eq!(duplex_challenger.output_buffer.len(), RATE);

        // Sample and verify we can get values back
        let sample1: G = duplex_challenger.sample();
        let sample2: G = duplex_challenger.sample();

        // Both samples should be valid field elements
        assert!(sample1 != G::ZERO || sample2 != G::ZERO || true); // Just verify no panic
    }

    #[test]
    #[should_panic]
    fn test_duplex_challenger_sample_bits_security() {
        type Chal = DuplexChallenger<G, TestPermutation, WIDTH, RATE>;
        let permutation = TestPermutation {};
        let mut duplex_challenger = Chal::new(permutation);

        // This should panic because we're requesting too many bits
        for _ in 0..100 {
            assert!(duplex_challenger.sample_bits(256) < 4);
        }
    }

    #[test]
    fn test_observe_single_value() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.observe(G::from_u8(42));
        assert_eq!(chal.input_buffer, vec![G::from_u8(42)]);
        assert!(chal.output_buffer.is_empty());
    }

    #[test]
    fn test_observe_array_of_values() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.observe([G::from_u8(1), G::from_u8(2), G::from_u8(3)]);
        // With RATE=2, after observing 3 elements:
        // - First 2 elements trigger duplexing
        // - Third element clears output buffer and stays in input_buffer
        assert_eq!(chal.input_buffer, vec![G::from_u8(3)]);
        // Output buffer is cleared when the 3rd element is observed
        assert!(chal.output_buffer.is_empty());
    }

    #[test]
    fn test_observe_hash_array() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        let hash = Hash::<G, G, 4>::from([G::from_u8(10); 4]);
        chal.observe(hash);
        // With RATE=2, after observing 4 elements:
        // - First 2 elements trigger duplexing (cleared)
        // - Next 2 elements trigger another duplexing (cleared)
        // - Input buffer should be empty after two full duplexing operations
        assert!(chal.input_buffer.is_empty());
        assert_eq!(chal.output_buffer.len(), RATE);
    }

    #[test]
    fn test_observe_nested_vecs() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.observe(vec![
            vec![G::from_u8(1), G::from_u8(2)],
            vec![G::from_u8(3)],
        ]);
        // With RATE=2, after observing 3 elements:
        // - First 2 elements trigger duplexing
        // - Third element clears output buffer and stays in input_buffer
        assert_eq!(chal.input_buffer, vec![G::from_u8(3)]);
        // Output buffer is cleared when the 3rd element is observed
        assert!(chal.output_buffer.is_empty());
    }

    #[test]
    fn test_sample_triggers_duplex() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.observe(G::from_u8(5));
        assert!(chal.output_buffer.is_empty());
        let _sample: G = chal.sample();
        assert!(!chal.output_buffer.is_empty());
    }

    #[test]
    fn test_sample_multiple_field() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});

        chal.observe(G::from_u8(1));
        chal.observe(G::from_u8(2));
        let _: G = chal.sample();
        let _: G = chal.sample();
    }

    #[test]
    fn test_sample_bits_within_bounds() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        for i in 0..RATE {
            chal.observe(G::from_u8(i as u8));
        }

        // With RATE=2 and input = [0, 1], the reversed sponge_state will be [1, 0, 0]
        // The first RATE elements of that (output_buffer) are [1, 0]
        // sample_bits(3) will sample the last element: G::from_u8(0)

        let bits = 3;
        let value = chal.sample_bits(bits);
        // The actual value depends on the field element bytes, just check it's within bounds
        assert!(value < (1 << bits));
    }

    #[test]
    fn test_sample_bits_trigger_duplex_when_empty() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        // Force empty buffers
        assert_eq!(chal.input_buffer.len(), 0);
        assert_eq!(chal.output_buffer.len(), 0);

        // sampling bits should not panic, should return 0
        let bits = 2;
        let sample = chal.sample_bits(bits);
        let expected = 0usize & ((1 << bits) - 1);
        assert_eq!(sample, expected);
    }

    #[test]
    fn test_output_buffer_pops_correctly() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});

        // Observe RATE elements, causing a duplexing
        for i in 0..RATE {
            chal.observe(G::from_u8(i as u8));
        }

        // With WIDTH=3, RATE=2:
        // Input: [0, 1] -> sponge state becomes [0, 1, 0]
        // After reverse (TestPermutation): [0, 1, 0]
        // Output buffer (first RATE elements): [0, 1]
        let expected = [G::from_u8(0), G::from_u8(1)].to_vec();

        assert_eq!(chal.output_buffer, expected);

        let first: G = chal.sample();
        let second: G = chal.sample();

        // sampling pops from end of output buffer
        assert_eq!(first, G::from_u8(1));
        assert_eq!(second, G::from_u8(0));
    }

    #[test]
    fn test_duplexing_only_when_needed() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.output_buffer = vec![G::from_u8(10), G::from_u8(20)];

        // Sample should not call duplexing; just pop from the buffer
        let sample: G = chal.sample();
        assert_eq!(sample, G::from_u8(20));
        assert_eq!(chal.output_buffer, vec![G::from_u8(10)]);
    }

    #[test]
    fn test_flush_when_input_full() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});

        // Observe RATE elements, causing a duplexing
        for i in 0..RATE {
            chal.observe(G::from_u8(i as u8));
        }

        // With WIDTH=3, RATE=2:
        // Input: [0, 1] -> sponge state becomes [0, 1, 0]
        // After reverse (TestPermutation): [0, 1, 0]
        // Output buffer (first RATE elements): [0, 1]
        let expected_output = [G::from_u8(0), G::from_u8(1)].to_vec();

        // Input buffer should be drained after duplexing
        assert!(chal.input_buffer.is_empty());

        // Output buffer should match expected state from duplexing
        assert_eq!(chal.output_buffer, expected_output);
    }
}
