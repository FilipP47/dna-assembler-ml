//! 2-bit DNA encoding and the `DnaKey` compact sequence type.
//!
//! Each nucleotide is stored as 2 bits (A=00, C=01, G=10, T=11),
//! packing up to 64 bases into two `u64` words with no heap allocation.

use std::num::NonZeroU8;

/// Map a DNA byte to its 2-bit index (0–3).
///
/// # Panics
/// Panics on any byte that is not a valid ACGT base (case-insensitive).
pub fn base_to_index(b: u8) -> usize {
    match b {
        b'A' | b'a' => 0, b'C' | b'c' => 1,
        b'G' | b'g' => 2, b'T' | b't' => 3,
        _ => panic!("Invalid DNA base: '{}'", b as char),
    }
}

pub const INDEX_TO_BASE: [char; 4] = ['A', 'C', 'G', 'T'];

fn encode_base(b: u8) -> u64 {
    match b {
        b'A' | b'a' => 0, b'C' | b'c' => 1,
        b'G' | b'g' => 2, b'T' | b't' => 3,
        _ => panic!("Invalid DNA base: '{}'", b as char),
    }
}

/// Encode up to 32 bases into a u64 (MSB = first base).
fn pack_u64(seq: &[u8]) -> u64 {
    seq.iter().fold(0u64, |acc, &b| (acc << 2) | encode_base(b))
}

/// Decode `len` bases from a packed u64 back to a DNA string.
fn unpack_u64(val: u64, len: usize) -> String {
    (0..len).rev()
        .map(|i| INDEX_TO_BASE[((val >> (i * 2)) & 0b11) as usize])
        .collect()
}

/// Compact 2-bit encoding for a DNA sequence up to 64 bases.
///
/// Memory layout (24 bytes, 8-byte aligned):
///   lo:  u64  — bases 0–31
///   hi:  u64  — bases 32–63 (zero if len ≤ 32)
///   len: NonZeroU8 — sequence length (enables `Option<DnaKey>` niche, zero overhead)
///
/// `Option<DnaKey>` is the same size as `DnaKey` — no extra byte for discriminant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DnaKey {
    lo:  u64,
    hi:  u64,
    len: NonZeroU8,
}

/// Type alias for a k-mer key (an edge in the de Bruijn graph).
// One encoding, two semantic roles — the compiler sees one type; we see intent.
pub type KmerKey = DnaKey;

/// Type alias for a node identifier ((k-1)-mer) in the de Bruijn graph.
pub type NodeId  = DnaKey;

impl DnaKey {
    /// Encode a DNA sequence.
    ///
    /// # Panics
    /// Panics if `seq` is empty or longer than 64 bases.
    pub fn encode(seq: &[u8]) -> Self {
        assert!(!seq.is_empty(), "Cannot encode empty sequence");
        assert!(seq.len() <= 64, "k > 64 is not supported (got {})", seq.len());

        let (lo, hi) = if seq.len() <= 32 {
            (pack_u64(seq), 0u64)
        } else {
            (pack_u64(&seq[..32]), pack_u64(&seq[32..]))
        };

        DnaKey {
            lo,
            hi,
            // SAFETY: seq.len() ≥ 1, so len ≥ 1, NonZeroU8 is valid
            len: NonZeroU8::new(seq.len() as u8).unwrap(),
        }
    }

    /// Decode back to a DNA string (only for display / output).
    pub fn decode(&self) -> String {
        let n = self.len.get() as usize;
        if n <= 32 {
            unpack_u64(self.lo, n)
        } else {
            let mut s = unpack_u64(self.lo, 32);
            s.push_str(&unpack_u64(self.hi, n - 32));
            s
        }
    }

    /// Length of the encoded sequence in bases.
    pub fn len(&self) -> usize { self.len.get() as usize }

    /// First (len-1) bases → source NodeId of this edge.
    ///
    /// Decode + re-encode — called only at graph-build time (not hot path).
    pub fn prefix_node(&self) -> NodeId {
        let n = self.len.get() as usize;
        let s = self.decode();
        DnaKey::encode(&s.as_bytes()[..n - 1])
    }

    /// Last (len-1) bases → destination NodeId of this edge.
    pub fn suffix_node(&self) -> NodeId {
        let s = self.decode();
        let n = self.len.get() as usize;
        DnaKey::encode(&s.as_bytes()[1..n])
    }

    /// Index (0–3) of the last nucleotide — used to index into `node.out_edges`.
    pub fn last_base_idx(&self) -> usize {
        (self.lo & 0b11) as usize       // last 2 bits of lo always = last base
    }

    /// Index (0–3) of the first nucleotide — used to index into `node.in_edges`.
    pub fn first_base_idx(&self) -> usize {
        let n = self.len.get() as usize;
        if n <= 32 {
            ((self.lo >> ((n - 1) * 2)) & 0b11) as usize
        } else {
            ((self.hi >> ((n - 33) * 2)) & 0b11) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_24_bytes() {
        assert_eq!(std::mem::size_of::<DnaKey>(), 24);
    }

    #[test]
    fn option_niche_zero_overhead() {
        // Option<DnaKey> must be the same size as DnaKey thanks to NonZeroU8 niche
        assert_eq!(std::mem::size_of::<Option<DnaKey>>(), std::mem::size_of::<DnaKey>());
    }

    #[test]
    fn roundtrip_short() {
        let k = DnaKey::encode(b"ACGT");
        assert_eq!(k.decode(), "ACGT");
    }

    #[test]
    fn roundtrip_32_bases() {
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
        let k = DnaKey::encode(seq);
        assert_eq!(k.decode(), std::str::from_utf8(seq).unwrap());
    }

    #[test]
    fn roundtrip_long() {
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"; // 40 bases
        let k = DnaKey::encode(seq);
        assert_eq!(k.decode(), std::str::from_utf8(seq).unwrap());
    }

    #[test]
    fn prefix_suffix() {
        let k = DnaKey::encode(b"ACGT");
        assert_eq!(k.prefix_node().decode(), "ACG");
        assert_eq!(k.suffix_node().decode(), "CGT");
    }

    #[test]
    fn base_indices() {
        let k = DnaKey::encode(b"ACGT");
        assert_eq!(k.first_base_idx(), base_to_index(b'A')); // 0
        assert_eq!(k.last_base_idx(),  base_to_index(b'T')); // 3
    }
}
