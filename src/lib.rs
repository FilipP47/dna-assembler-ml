//! DNA assembler library - re-exports pipeline, core, and infra modules.
pub mod core;
pub mod infra;
pub mod pipeline;

pub use core::graph::DeBruijnGraph;
pub use infra::config::{AssemblerConfig, SimulatorConfig};
pub use infra::io::{reader_for_path, FastaReader, FastaWriter, ReadParser, ReadWriter};
pub use pipeline::analysis::traits::JunctionAnalyzer;
pub use pipeline::analysis::{AnalysisOptions, MlDataCollector};
pub use pipeline::assembler::SimpleAssembler;
pub use pipeline::simulator::{RandomReadSimulator, ReadSimulator};
pub use pipeline::training_data::{GeneratorStats, TrainingDataConfig, TrainingDataGenerator};
