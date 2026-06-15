//! Uniform random read simulator.
//!
//! `RandomReadSimulator` simulates sequencing by placing reads at
//! uniformly-random positions across the reference genome.
//!
//! ## Algorithm
//! 1. Compute `num_reads = ceil(genome_len × coverage / read_length)`.
//! 2. For each read, draw a random start position from [0, genome_len - read_length].
//! 3. Extract the subsequence and store it as a `DnaRead`.
//!
//! The PRNG seed is fixed by `SimulatorConfig::seed`, making every run
//! with the same parameters fully reproducible.
//!
//! ## Assumptions (Stage 1 — no error model)
//! • Reads are exact substrings of the reference (zero sequencing errors).
//! • Read length is constant (variable-length support deferred to Stage 2).
//! • Only the forward strand is sampled (reverse-complement deferred).

use anyhow::{bail, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{infra::config::SimulatorConfig, infra::io::DnaRead};
use super::traits::ReadSimulator;

/// Reads simulator that samples uniformly random positions from a reference genome.
pub struct RandomReadSimulator {
    read_length: usize,
    coverage:    usize,
    seed:        u64,
}

impl RandomReadSimulator {
    /// Construct from an assembled `SimulatorConfig`.
    pub fn from_config(cfg: &SimulatorConfig) -> Self {
        Self {
            read_length: cfg.read_length,
            coverage:    cfg.coverage,
            seed:        cfg.seed,
        }
    }

    /// Calculate the number of reads needed to achieve target coverage.
    ///
    /// Formula: `num_reads = ceil(genome_len × coverage / read_length)`
    fn num_reads(&self, genome_len: usize) -> usize {
        let total_bases = genome_len * self.coverage;
        (total_bases + self.read_length - 1) / self.read_length
    }
}

impl ReadSimulator for RandomReadSimulator {
    fn description(&self) -> &'static str {
        "Uniform random read simulator (zero error, forward strand only)"
    }

    fn simulate(&self, reference: &[u8]) -> Result<Vec<DnaRead>> {
        let genome_len = reference.len();

        if genome_len < self.read_length {
            bail!(
                "Genome length ({} bp) is shorter than read_length ({} bp).",
                genome_len,
                self.read_length
            );
        }

        let num_reads = self.num_reads(genome_len);
        let max_start = genome_len - self.read_length;

        let mut rng  = StdRng::seed_from_u64(self.seed);
        let mut reads = Vec::with_capacity(num_reads);

        for i in 0..num_reads {
            let start: usize = rng.gen_range(0..=max_start);
            let end = start + self.read_length;

            let seq = std::str::from_utf8(&reference[start..end])
                .expect("reference contains non-UTF8 bytes")
                .to_uppercase();

            let header = format!(
                "read_{} start={} end={} len={}",
                i, start, end, self.read_length
            );

            reads.push(DnaRead::new(header, seq));
        }

        println!(
            "Simulated {} reads × {} bp  →  {:.1}× coverage  (seed={})",
            num_reads,
            self.read_length,
            (num_reads * self.read_length) as f64 / genome_len as f64,
            self.seed
        );

        Ok(reads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REF: &[u8] = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGT"; // 36 bp

    fn make_sim(read_length: usize, coverage: usize, seed: u64) -> RandomReadSimulator {
        RandomReadSimulator { read_length, coverage, seed }
    }

    #[test]
    fn correct_read_length() {
        let sim = make_sim(10, 5, 0);
        let reads = sim.simulate(REF).unwrap();
        assert!(reads.iter().all(|r| r.len() == 10));
    }

    #[test]
    fn approximate_coverage() {
        let sim = make_sim(10, 10, 0);
        let reads = sim.simulate(REF).unwrap();
        let total_bases: usize = reads.iter().map(|r| r.len()).sum();
        let target = REF.len() * 10;
        assert!((total_bases as isize - target as isize).unsigned_abs() <= 10);
    }

    #[test]
    fn reproducible_with_same_seed() {
        let sim = make_sim(10, 5, 42);
        let r1 = sim.simulate(REF).unwrap();
        let r2 = sim.simulate(REF).unwrap();
        assert_eq!(r1.len(), r2.len());
        assert!(r1.iter().zip(r2.iter()).all(|(a, b)| a.sequence == b.sequence));
    }

    #[test]
    fn different_seeds_differ() {
        let r1 = make_sim(10, 5, 1).simulate(REF).unwrap();
        let r2 = make_sim(10, 5, 2).simulate(REF).unwrap();
        let all_same = r1.iter().zip(r2.iter()).all(|(a, b)| a.sequence == b.sequence);
        assert!(!all_same);
    }

    #[test]
    fn genome_shorter_than_read_errors() {
        let sim = make_sim(100, 5, 0);
        assert!(sim.simulate(b"ACGT").is_err());
    }

    #[test]
    fn reads_are_substrings_of_reference() {
        let sim = make_sim(8, 5, 99);
        let reads = sim.simulate(REF).unwrap();
        let ref_str = std::str::from_utf8(REF).unwrap().to_uppercase();
        for r in &reads {
            assert!(
                ref_str.contains(&r.sequence),
                "Read '{}' is not a substring of the reference",
                r.sequence
            );
        }
    }
}
