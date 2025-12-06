# KZG Example - Fibonacci STARK Proof

This crate demonstrates how to use Plonky3's KZG commitment scheme with the BN254 elliptic curve to generate and verify STARK proofs for Fibonacci sequence computation.

## Overview

The example proves the correct computation of Fibonacci numbers using:
- **Field**: BN254 Fr (scalar field of the BN254 curve)
- **Commitment Scheme**: KZG polynomial commitments
- **Hash Function**: Poseidon2 optimized for BN254

## Features

- **Fibonacci AIR**: Algebraic Intermediate Representation that defines constraints for the Fibonacci recurrence relation
- **KZG Commitments**: Uses the Kate-Zaverucha-Goldberg polynomial commitment scheme on BN254
- **Complete Example**: Includes proof generation, verification, and demonstration code

## Components

### FibonacciAir

The AIR (Algebraic Intermediate Representation) that defines the constraints for proving Fibonacci sequence computation. The trace has 2 columns:
- Column 0: f(n)
- Column 1: f(n+1)

The AIR enforces the Fibonacci recurrence relation:
- next[0] = current[1]
- next[1] = current[0] + current[1]

### Proof Functions

- `prove_fibonacci`: Generates and verifies a STARK proof for a Fibonacci sequence
- `verify_fibonacci`: Convenience wrapper for verification

## Usage

### Running the Example

```bash
cargo run --example fibonacci_kzg --release -p p3-kzg-example
```

### Using in Your Code

```rust
use p3_kzg_example::{FibonacciAir, prove_fibonacci};
use p3_bn254::{Fr as Bn254Fr, Poseidon2Bn254};
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// Setup
let mut rng = SmallRng::seed_from_u64(42);
let perm = Poseidon2Bn254::<3>::new_from_rng(8, 22, &mut rng);

// Create Fibonacci AIR for 16 steps
let air = FibonacciAir::new(16);

// KZG setup (testing only - use proper trusted setup in production!)
let alpha = Bn254Fr::from_u64(12345);
let max_degree = 1024;

// Generate and verify proof
let result = prove_fibonacci(&air, perm, max_degree, alpha);
assert!(result.is_ok());
```

## Running Tests

```bash
cargo test -p p3-kzg-example
```

## Security Warning

⚠️ **IMPORTANT**: The example uses a simple setup for KZG parameters which is only suitable for testing. In production, you must use a properly generated SRS (Structured Reference String) from a trusted setup ceremony, such as:

- Ethereum's KZG ceremony for EIP-4844
- A multi-party computation (MPC) ceremony
- Pre-computed SRS from a trusted source

## Dependencies

- `p3-air`: AIR framework
- `p3-bn254`: BN254 curve implementation
- `p3-kzg`: KZG commitment scheme
- `p3-uni-stark`: Univariate STARK prover
- `p3-challenger`: Fiat-Shamir challenger
- `p3-matrix`: Matrix operations
- `p3-symmetric`: Cryptographic permutations
- `p3-field`: Field arithmetic

## Example Output

```
=== Fibonacci KZG Example ===

Configuration:
  Field: BN254 Fr
  Commitment Scheme: KZG
  Fibonacci steps: 128
  Max polynomial degree: 4096

Setting up Poseidon2 permutation...
Creating Fibonacci AIR...

First 10 Fibonacci numbers in the trace:
  Step 0: f(n)=1, f(n+1)=1
  Step 1: f(n)=1, f(n+1)=2
  Step 2: f(n)=2, f(n+1)=3
  Step 3: f(n)=3, f(n+1)=5
  Step 4: f(n)=5, f(n+1)=8
  Step 5: f(n)=8, f(n+1)=13
  Step 6: f(n)=13, f(n+1)=21
  Step 7: f(n)=21, f(n+1)=34
  Step 8: f(n)=34, f(n+1)=55
  Step 9: f(n)=55, f(n+1)=89

Performing KZG setup (testing only - use trusted setup in production)...

Generating STARK proof...

✓ Proof generated and verified successfully!

The proof demonstrates that the Fibonacci sequence was computed correctly
using STARK constraints verified with KZG polynomial commitments on BN254.
```

## References

- [KZG Paper](https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf): "Constant-Size Commitments to Polynomials and Their Applications" by Kate, Zaverucha, and Goldberg (2010)
- [EIP-4844](https://eips.ethereum.org/EIPS/eip-4844): Proto-Danksharding using KZG commitments
- [Plonky3 Documentation](https://github.com/Plonky3/Plonky3)

## License

MIT OR Apache-2.0
