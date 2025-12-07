//! Example: Prove Fibonacci sequence using KZG commitments with BN254
//!
//! This example demonstrates how to generate and verify a STARK proof for
//! computing Fibonacci numbers using KZG polynomial commitments on the BN254 curve.
//!
//! Run with:
//! ```bash
//! cargo run --example fibonacci_kzg --release
//! ```

use p3_bn254::{Fr as Bn254Fr, Poseidon2Bn254};
use p3_field::PrimeCharacteristicRing;
use p3_kzg_example::{FibonacciAir, prove_fibonacci};
use p3_matrix::Matrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::time::Instant;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

fn main() {
    // Initialize tracing
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    println!("=== Fibonacci KZG Example ===\n");

    // Configuration
    let num_steps = 1024; // Compute 1024 Fibonacci numbers
    let max_degree = 1 << 12; // 4096 - should be larger than trace size

    println!("Configuration:");
    println!("  Field: BN254 Fr");
    println!("  Commitment Scheme: KZG");
    println!("  Fibonacci steps: {}", num_steps);
    println!("  Max polynomial degree: {}\n", max_degree);

    // WARNING: Use a real cryptographic PRNG in production!
    let mut rng = SmallRng::seed_from_u64(42);

    // Create Poseidon2 permutation for BN254
    println!("Setting up Poseidon2 permutation...");
    let perm = Poseidon2Bn254::<3>::new_from_rng(8, 22, &mut rng);

    // Create Fibonacci AIR
    println!("Creating Fibonacci AIR...");
    let air = FibonacciAir::new(num_steps);

    // Generate the trace to show some values
    println!("\nGenerating trace...");
    let start = Instant::now();
    let trace = air.generate_trace::<Bn254Fr>();
    let trace_gen_time = start.elapsed();

    println!("\nTrace dimensions:");
    println!("  Rows (height): {}", trace.height());
    println!("  Columns (width): {}", trace.width());
    println!("  Trace generation time: {:?}", trace_gen_time);

    println!("\nFirst 10 Fibonacci numbers in the trace:");
    for i in 0..10.min(num_steps) {
        let f_n = trace.get(i, 0).unwrap();
        let f_n1 = trace.get(i, 1).unwrap();
        println!("  Step {}: f(n)={}, f(n+1)={}", i, f_n, f_n1);
    }

    // KZG setup (WARNING: This is for testing only!)
    // In production, use a proper trusted setup ceremony
    println!("\nPerforming KZG setup (testing only - use trusted setup in production)...");
    let alpha = Bn254Fr::from_u64(12345);

    // Generate and verify proof
    println!("\nGenerating STARK proof...");
    let result = prove_fibonacci(&air, perm, max_degree, alpha);

    match result {
        Ok(()) => {
            println!("\n✓ Proof generated and verified successfully!");
            println!("\nThe proof demonstrates that the Fibonacci sequence was computed correctly");
            println!("using STARK constraints verified with KZG polynomial commitments on BN254.");
        }
        Err(e) => {
            eprintln!("\n✗ Proof verification failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
