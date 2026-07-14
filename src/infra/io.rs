//! File I/O subsystem: reading and writing DNA reads in multiple formats.
//!
//! Public surface:
//!   - `DnaRead`          - the core data type
//!   - `ReadParser`       - trait for format readers
//!   - `ReadWriter`       - trait for format writers
//!   - `reader_for_path`  - factory: picks the right reader by file extension
//!   - `FastaReader`      - concrete FASTA reader
//!   - `FastaWriter`      - concrete FASTA writer

pub mod factory;
pub mod fasta;
pub mod reader;
pub mod writer;

pub use factory::reader_for_path;
pub use fasta::{FastaReader, FastaWriter};
pub use reader::{DnaRead, ReadParser};
pub use writer::ReadWriter;
