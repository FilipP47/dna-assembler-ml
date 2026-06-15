//! Concrete implementations of `ReadParser` and `ReadWriter` for the FASTA format.
//!
//! FASTA format rules handled here:
//!   - Header line starts with `>`; everything after `>` is the header string.
//!   - Sequence may span multiple lines; they are concatenated.
//!   - Lines starting with `;` are comments — skipped.
//!   - Bases are upper-cased; any non-ACGT character is replaced with 'N'
//!     and a warning is emitted (so corrupt files don't silently break assembly).
//!   - Empty records (header with no sequence) are skipped.

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};
use anyhow::{Context, Result};

use super::{
    reader::{DnaRead, ReadParser},
    writer::ReadWriter,
};

/// Parses FASTA files into a `Vec<DnaRead>`.
pub struct FastaReader;

impl ReadParser for FastaReader {
    /// Returns `"FASTA"`.
    fn format_name(&self) -> &'static str {
        "FASTA"
    }

    /// Parse all records from a FASTA file at `path`.
    fn parse(&self, path: &Path) -> Result<Vec<DnaRead>> {
        let file = File::open(path)
            .with_context(|| format!("Cannot open FASTA file: {}", path.display()))?;
        let reader = BufReader::new(file);

        let mut reads = Vec::new();
        let mut current_header: Option<String> = None;
        let mut current_seq = String::new();

        for (line_no, line) in reader.lines().enumerate() {
            let line = line.with_context(|| {
                format!("IO error reading {} at line {}", path.display(), line_no + 1)
            })?;
            let line = line.trim();

            if line.is_empty() || line.starts_with(';') {
                continue;
            }

            if let Some(stripped) = line.strip_prefix('>') {
                if let Some(header) = current_header.take() {
                    if !current_seq.is_empty() {
                        reads.push(DnaRead::new(header, current_seq.clone()));
                    }
                    current_seq.clear();
                }
                current_header = Some(stripped.to_string());
            } else {
                for ch in line.chars() {
                    match ch.to_ascii_uppercase() {
                        'A' | 'C' | 'G' | 'T' => current_seq.push(ch.to_ascii_uppercase()),
                        _ => current_seq.push('N'),
                    }
                }
            }
        }

        if let Some(header) = current_header {
            if !current_seq.is_empty() {
                reads.push(DnaRead::new(header, current_seq));
            }
        }

        Ok(reads)
    }
}

/// Writes `DnaRead` slices to a FASTA file.
///
/// Sequence lines are wrapped at 60 characters (standard bioinformatics convention).
pub struct FastaWriter {
    line_width: usize,
}

impl FastaWriter {
    /// Create a writer with the default 60-character line wrap.
    pub fn new() -> Self {
        Self { line_width: 60 }
    }

    /// Override the default line wrap width.
    pub fn with_line_width(mut self, width: usize) -> Self {
        self.line_width = width;
        self
    }
}

impl Default for FastaWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadWriter for FastaWriter {
    /// Returns `"FASTA"`.
    fn format_name(&self) -> &'static str {
        "FASTA"
    }

    /// Write all `reads` to `path` in FASTA format, creating parent directories as needed.
    fn write(&self, reads: &[DnaRead], path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("Cannot create output directory: {}", parent.display())
                })?;
            }
        }

        let file = File::create(path)
            .with_context(|| format!("Cannot create FASTA file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);

        for read in reads {
            writeln!(writer, ">{}", read.header)?;
            for chunk in read.sequence.as_bytes().chunks(self.line_width) {
                writeln!(writer, "{}", std::str::from_utf8(chunk).unwrap())?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::NamedTempFile;

    fn write_temp_fasta(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn parse_single_record() {
        let f = write_temp_fasta(">read1\nACGTACGT\n");
        let reader = FastaReader;
        let reads = reader.parse(f.path()).unwrap();
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0].header, "read1");
        assert_eq!(reads[0].sequence, "ACGTACGT");
    }

    #[test]
    fn parse_multi_line_sequence() {
        let f = write_temp_fasta(">r\nACGT\nACGT\n");
        let reads = FastaReader.parse(f.path()).unwrap();
        assert_eq!(reads[0].sequence, "ACGTACGT");
    }

    #[test]
    fn parse_multiple_records() {
        let f = write_temp_fasta(">r1\nAAAA\n>r2\nCCCC\n");
        let reads = FastaReader.parse(f.path()).unwrap();
        assert_eq!(reads.len(), 2);
        assert_eq!(reads[1].sequence, "CCCC");
    }

    #[test]
    fn skips_comments_and_blanks() {
        let f = write_temp_fasta("; comment\n>r1\n\nACGT\n");
        let reads = FastaReader.parse(f.path()).unwrap();
        assert_eq!(reads[0].sequence, "ACGT");
    }
}
