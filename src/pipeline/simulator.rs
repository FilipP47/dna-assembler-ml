//! Read simulators: generate synthetic DNA reads from a reference genome.
pub mod random;
pub mod traits;

pub use random::RandomReadSimulator;
pub use traits::ReadSimulator;