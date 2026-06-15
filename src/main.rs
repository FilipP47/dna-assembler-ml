//! CLI entry point for the DNA assembler.
//!
//! Three subcommands:
//!   simulate   — generate synthetic reads from a reference genome
//!   assemble   — build de Bruijn graph and extract contigs from reads
//!   pipeline   — simulate + assemble in one step
//!
//! Output layout:
//!   results/<genome>/   — assembly artefacts (contigs, optional reads)
//!   data/training/      — training CSVs accumulated across genomes

use std::path::PathBuf;
use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use dna_assembler::{
    AnalysisOptions, AssemblerConfig, DeBruijnGraph, FastaReader, FastaWriter,
    JunctionAnalyzer, MlDataCollector, RandomReadSimulator, ReadParser, ReadSimulator,
    ReadWriter, SimpleAssembler, SimulatorConfig, reader_for_path,
    TrainingDataConfig, TrainingDataGenerator,
};

/// Directory where training CSVs are always written, shared across all genomes.
const TRAINING_DATA_DIR: &str = "data/training";

/// Top-level CLI struct parsed by clap.
#[derive(Parser)]
#[command(
    name    = "dna-assembler",
    version = "0.1.0",
    about   = "De Bruijn graph DNA assembler — baseline for ML-guided assembly",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Shared simulation arguments reused by `simulate` and `pipeline`.
#[derive(Args)]
struct SimArgs {
    /// Reference genome FASTA file
    #[arg(short, long)]
    reference: PathBuf,

    /// Length of each simulated read (bp)
    #[arg(long, default_value = "150")]
    read_length: usize,

    /// Target sequencing coverage (X-fold)
    #[arg(long, default_value = "30")]
    coverage: usize,

    /// PRNG seed for reproducible runs
    #[arg(long, default_value = "42")]
    seed: u64,
}

/// Available subcommands.
#[derive(Subcommand)]
enum Commands {
    /// Simulate sequencing reads from a reference genome FASTA file.
    Simulate {
        #[command(flatten)]
        sim: SimArgs,

        /// Output FASTA file for simulated reads
        #[arg(short, long, default_value = "reads.fasta")]
        output: PathBuf,
    },

    /// Build a de Bruijn graph from reads and extract contigs.
    Assemble {
        /// Input reads file (FASTA: .fa / .fasta / .fna)
        #[arg(short, long)]
        input: PathBuf,

        /// k-mer length (nodes are (k-1)-mers)
        #[arg(short, long, default_value = "31")]
        k: usize,

        /// Directory for assembly results (contigs.fasta, junctions.csv)
        #[arg(short, long, default_value = "results")]
        results_dir: PathBuf,

        /// Enable junction analysis and write junctions.csv
        #[arg(long)]
        analyze: bool,
    },

    /// Run simulate then assemble in a single step.
    Pipeline {
        #[command(flatten)]
        sim: SimArgs,

        /// k-mer length
        #[arg(short, long, default_value = "31")]
        k: usize,

        /// Directory for assembly results (contigs.fasta, optional reads.fasta)
        #[arg(short, long, default_value = "results")]
        results_dir: PathBuf,

        /// Enable junction analysis
        #[arg(long)]
        analyze: bool,

        /// Save the simulated reads inside results_dir
        #[arg(long)]
        save_reads: bool,

        /// Generate ML training data; CSV written to data/training/<genome>_training.csv
        #[arg(long)]
        gen_train_data: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Simulate { sim, output } => cmd_simulate(sim, output),

        Commands::Assemble { input, k, results_dir, analyze } => {
            let mut builder = AssemblerConfig::builder().input(input).k(k).output_dir(results_dir);
            if analyze { builder = builder.analysis(AnalysisOptions::all()); }
            cmd_assemble(builder.build()?)
        }

        Commands::Pipeline { sim, k, results_dir, analyze, save_reads, gen_train_data } => {
            cmd_pipeline(sim, k, results_dir, analyze, save_reads, gen_train_data)
        }
    }
}

/// Load and decode a reference genome FASTA into a byte vector.
fn load_reference(path: &PathBuf) -> Result<Vec<u8>> {
    println!("Loading reference: {}", path.display());
    let ref_reads = FastaReader.parse(path)?;
    anyhow::ensure!(!ref_reads.is_empty(), "Reference FASTA is empty");
    let ref_seq: Vec<u8> = ref_reads.iter().flat_map(|r| r.sequence.bytes()).collect();
    println!("Reference length: {} bp", ref_seq.len());
    Ok(ref_seq)
}

/// Read a reference genome, simulate reads, and write them to a FASTA file.
fn cmd_simulate(sim: SimArgs, output: PathBuf) -> Result<()> {
    let cfg = SimulatorConfig::builder()
        .reference(sim.reference.clone())
        .read_length(sim.read_length)
        .coverage(sim.coverage)
        .seed(sim.seed)
        .output(output.clone())
        .build()?;

    let ref_seq = load_reference(&sim.reference)?;
    let reads = RandomReadSimulator::from_config(&cfg).simulate(&ref_seq)?;

    FastaWriter::new().write(&reads, &output)?;
    println!("Reads written → {}", output.display());
    Ok(())
}

/// Load reads, build the de Bruijn graph, traverse it, and write contigs.
fn cmd_assemble(cfg: AssemblerConfig) -> Result<()> {
    println!("Loading reads: {}", cfg.input_path.display());
    let reads = reader_for_path(&cfg.input_path)?.parse(&cfg.input_path)?;
    println!("Reads loaded: {}", reads.len());

    println!("Building de Bruijn graph (k={})…", cfg.k);
    let graph = DeBruijnGraph::build(&reads, cfg.k)?;

    let mut analyzer: Option<Box<dyn JunctionAnalyzer>> = cfg
        .analysis
        .map(|opts| Box::new(MlDataCollector::new(opts)) as Box<dyn JunctionAnalyzer>);

    SimpleAssembler.run(&graph, &cfg.output_dir, &mut analyzer)?;
    Ok(())
}

/// Simulate reads from a reference genome, then assemble them in a single step.
///
/// Assembly results go to `results_dir`.
/// Training data (when `--gen-train-data`) always goes to `data/training/`,
/// independent of `results_dir`, so CSVs accumulate across genome runs.
fn cmd_pipeline(
    sim: SimArgs,
    k: usize,
    results_dir: PathBuf,
    analyze: bool,
    save_reads: bool,
    gen_train_data: bool,
) -> Result<()> {
    println!("=== STEP 1: Simulate ===");
    let ref_seq = load_reference(&sim.reference)?;

    let sim_cfg = SimulatorConfig::builder()
        .reference(sim.reference.clone())
        .read_length(sim.read_length)
        .coverage(sim.coverage)
        .seed(sim.seed)
        .output(results_dir.join("reads.fasta"))
        .build()?;

    let reads = RandomReadSimulator::from_config(&sim_cfg).simulate(&ref_seq)?;

    if save_reads {
        std::fs::create_dir_all(&results_dir)?;
        FastaWriter::new().write(&reads, &sim_cfg.output_path)?;
        println!("Reads saved → {}", sim_cfg.output_path.display());
    }

    println!("\n=== STEP 2: Assemble ===");
    println!("Building de Bruijn graph (k={})…", k);
    let graph = DeBruijnGraph::build(&reads, k)?;

    let mut analyzer: Option<Box<dyn JunctionAnalyzer>> = if gen_train_data {
        let ref_str = String::from_utf8(ref_seq)
            .map_err(|_| anyhow::anyhow!("Reference sequence is not valid UTF-8"))?;
        let ref_stem = sim.reference.file_stem().and_then(|s| s.to_str()).unwrap_or("reference");
        let csv_path = PathBuf::from(TRAINING_DATA_DIR)
            .join(format!("{}_training.csv", ref_stem));
        println!("Training data → {}", csv_path.display());
        Some(Box::new(TrainingDataGenerator::new(TrainingDataConfig::default(), ref_str, &csv_path, &graph)?) as _)
    } else if analyze {
        Some(Box::new(MlDataCollector::new(AnalysisOptions::all())) as _)
    } else {
        None
    };

    SimpleAssembler.run(&graph, &results_dir, &mut analyzer)?;
    Ok(())
}
