//! K-mer encoding: compact 2-bit DNA keys used as graph node and edge identifiers.
pub mod encoding;

pub use encoding::{KmerKey, NodeId, INDEX_TO_BASE};
