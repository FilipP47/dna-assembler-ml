//! Ground-truth labeling for training data.
//!
//! Determines the correct branch at a junction by searching for the
//! junction context sequence in the reference genome.
//!
//! Strategy:
//!   1. Find all occurrences of `context_seq` in the reference.
//!   2. For each occurrence, read the nucleotide immediately following it.
//!   3. Also search the reverse complement of `context_seq` (captures the
//!      complementary strand).
//!   4. If every occurrence leads to the same branch → `Known`.
//!   5. If occurrences lead to different branches → `Ambiguous`
//!      (context too short or a repeat region).
//!   6. If the context is not found → `NotFound`.

/// The ground-truth label for one junction.
#[derive(Debug, Clone, PartialEq)]
pub enum GroundTruth {
    /// A unique correct branch was identified.
    Known { base: u8, base_char: char },
    /// The context appears in multiple places with different continuations.
    Ambiguous,
    /// The context was not found in the reference genome.
    NotFound,
}

impl GroundTruth {
    /// Serialize to a pair of CSV column values `(ground_truth, gt_ambiguous)`.
    pub fn to_csv_value(&self) -> (&str, &str) {
        match self {
            Self::Known { base_char, .. } => {
                let s = match base_char {
                    'A' => "A", 'C' => "C", 'G' => "G", 'T' => "T", _ => "NA",
                };
                (s, "false")
            }
            Self::Ambiguous => ("ambiguous", "true"),
            Self::NotFound  => ("NA", "false"),
        }
    }

    /// Returns `true` only for unambiguous `Known` labels.
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Known { .. })
    }
}

/// Search `context_seq` in `reference` and identify the correct outgoing branch.
///
/// Both the forward strand and the reverse complement of `context_seq` are searched
/// so that junctions covered only by the complementary strand are still labeled.
///
/// `available_branches` — ASCII bytes (A/C/G/T) of existing outgoing edges at the junction.
pub fn find_ground_truth(
    context_seq: &str,
    reference: &str,
    available_branches: &[u8],
) -> GroundTruth {
    if context_seq.is_empty() || reference.is_empty() {
        return GroundTruth::NotFound;
    }

    let positions: Vec<usize> = find_all_occurrences(context_seq, reference);

    let rc = reverse_complement(context_seq);
    let rc_positions: Vec<usize> = if rc != context_seq {
        find_all_occurrences(&rc, reference)
    } else {
        vec![]
    };

    let all_positions: Vec<(usize, bool)> = positions.iter().map(|&p| (p, false))
        .chain(rc_positions.iter().map(|&p| (p, true)))
        .collect();

    if all_positions.is_empty() {
        return GroundTruth::NotFound;
    }

    let mut found_base: Option<u8> = None;
    let ref_bytes = reference.as_bytes();
    let ctx_len = context_seq.len();

    for &(pos, is_rc) in &all_positions {
        let next_byte = if !is_rc {
            let next_pos = pos + ctx_len;
            if next_pos >= ref_bytes.len() { continue; }
            ref_bytes[next_pos].to_ascii_uppercase()
        } else {
            if pos == 0 { continue; }
            let prev_byte = ref_bytes[pos - 1].to_ascii_uppercase();
            match prev_byte {
                b'A' => b'T',
                b'T' => b'A',
                b'C' => b'G',
                b'G' => b'C',
                other => other,
            }
        };

        if !available_branches.contains(&next_byte) {
            continue;
        }

        match found_base {
            None => found_base = Some(next_byte),
            Some(fb) if fb == next_byte => {}
            Some(_) => {
                return GroundTruth::Ambiguous;
            }
        }
    }

    match found_base {
        Some(base) => GroundTruth::Known {
            base,
            base_char: base as char,
        },
        None => GroundTruth::NotFound,
    }
}

/// Return all start positions of `pattern` within `text`.
fn find_all_occurrences(pattern: &str, text: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let pat_bytes = pattern.as_bytes();
    let txt_bytes = text.as_bytes();

    if pat_bytes.is_empty() || pat_bytes.len() > txt_bytes.len() {
        return positions;
    }

    for i in 0..=(txt_bytes.len() - pat_bytes.len()) {
        if txt_bytes[i..i + pat_bytes.len()] == *pat_bytes {
            positions.push(i);
        }
    }
    positions
}

/// Compute the reverse complement of a DNA sequence.
pub fn reverse_complement(seq: &str) -> String {
    seq.chars()
        .rev()
        .map(|c| match c {
            'A' | 'a' => 'T',
            'T' | 't' => 'A',
            'C' | 'c' => 'G',
            'G' | 'g' => 'C',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_continuation() {
        let reference = "AAAACGTTTTT";
        let context   = "AAAACG";
        let branches  = &[b'T', b'A'];
        let gt = find_ground_truth(context, reference, branches);
        assert!(matches!(gt, GroundTruth::Known { base_char: 'T', .. }));
    }

    #[test]
    fn ambiguous_when_repeat() {
        let reference = "AAAACGTTTAAAACGAAA";
        let context   = "AAAACG";
        let branches  = &[b'T', b'A'];
        let gt = find_ground_truth(context, reference, branches);
        assert_eq!(gt, GroundTruth::Ambiguous);
    }

    #[test]
    fn not_found_when_missing() {
        let reference = "TTTTTTTTTT";
        let context   = "AAAACG";
        let branches  = &[b'T', b'A'];
        let gt = find_ground_truth(context, reference, branches);
        assert_eq!(gt, GroundTruth::NotFound);
    }

    #[test]
    fn reverse_complement_basic() {
        assert_eq!(reverse_complement("ACGT"), "ACGT");
        assert_eq!(reverse_complement("AAAA"), "TTTT");
        assert_eq!(reverse_complement("ACGT"), reverse_complement("ACGT"));
    }

    #[test]
    fn ground_truth_via_reverse_complement_strand() {
        // Forward reference: 5'-ATCG-3'
        // Reverse complement: 5'-CGAT-3'
        // Searching for context "GA": not present on forward strand, but on the RC
        // strand 5'-C[GA]T-3' the next base is 'T'.
        let reference = "ATCG";
        let context   = "GA";
        let branches  = &[b'T', b'C'];

        let gt = find_ground_truth(context, reference, branches);

        assert!(
            matches!(gt, GroundTruth::Known { base_char: 'T', .. }),
            "Expected GroundTruth::Known with 'T', got: {:?}", gt
        );
    }
}
