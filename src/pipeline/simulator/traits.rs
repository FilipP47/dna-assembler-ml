use anyhow::Result;
use crate::infra::io::DnaRead;

/// Strategy interface for read simulators.
pub trait ReadSimulator {
    /// Generate simulated reads from the given reference sequence.
    fn simulate(&self, reference: &[u8]) -> Result<Vec<DnaRead>>;

    /// Human-readable description of the simulation strategy.
    fn description(&self) -> &'static str;
}
