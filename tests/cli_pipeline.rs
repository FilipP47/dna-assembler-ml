//! End-to-end tests of the `pipeline` subcommand on small synthetic genomes.
//!
//! - REQ001: `pipeline` writes `<results-dir>/contigs.fasta`; CLI default k is 31.
//! - REQ002: the training CSV has the 21-column schema, one row per junction
//!   with at least 2 branches of coverage >= 2.
//! - REQ027: `--gen-train-data` fails on a graph without junctions.

use assert_cmd::cargo::cargo_bin_cmd;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{fs, path::Path};

const CSV_COLUMNS: [&str; 21] = [
    "junction_id",
    "contig_id",
    "depth_in_contig",
    "out_degree",
    "in_degree",
    "node_seq",
    "context_seq",
    "branch_A_seq",
    "branch_A_cov",
    "branch_C_seq",
    "branch_C_cov",
    "branch_G_seq",
    "branch_G_cov",
    "branch_T_seq",
    "branch_T_cov",
    "ground_truth",
    "gt_ambiguous",
    "cov_ratio_max",
    "total_reads",
    "k",
    "repeat_len",
];

/// Seeded random ACGT string.
fn random_seq(rng: &mut StdRng, len: usize) -> String {
    (0..len)
        .map(|_| b"ACGT"[rng.gen_range(0..4)] as char)
        .collect()
}

/// Write a single-record FASTA file.
fn write_fasta(path: &Path, seq: &str) {
    fs::write(path, format!(">synthetic\n{seq}\n")).unwrap();
}

/// A 4 kb genome with a 60 bp repeat in three copies, each with its own
/// flanks, so the de Bruijn graph has junctions for any k <= 60.
fn genome_with_repeats() -> String {
    let mut rng = StdRng::seed_from_u64(7);
    let repeat = random_seq(&mut rng, 60);
    let mut genome = random_seq(&mut rng, 1000);
    for _ in 0..3 {
        genome.push_str(&repeat);
        genome.push_str(&random_seq(&mut rng, 1000));
    }
    genome
}

/// Run `pipeline --gen-train-data` and return (results_dir, csv rows as fields).
fn run_pipeline(dir: &Path, k: Option<&str>) -> Vec<Vec<String>> {
    let fasta = dir.join("synthetic.fna");
    write_fasta(&fasta, &genome_with_repeats());
    let results = dir.join("results");
    let train = dir.join("train");

    let mut cmd = cargo_bin_cmd!("dna-assembler");
    cmd.arg("pipeline")
        .arg("--reference")
        .arg(&fasta)
        .args(["--coverage", "30", "--gen-train-data"])
        .arg("--results-dir")
        .arg(&results)
        .arg("--train-data-dir")
        .arg(&train);
    if let Some(k) = k {
        cmd.args(["--k", k]);
    }
    cmd.assert().success();

    let contigs = fs::read_to_string(results.join("contigs.fasta")).unwrap();
    assert!(contigs.starts_with('>'), "contigs.fasta is not FASTA");
    let bases: String = contigs.lines().filter(|l| !l.starts_with('>')).collect();
    assert!(!bases.is_empty());
    assert!(bases.bytes().all(|b| b"ACGT".contains(&b)));

    let csv = fs::read_to_string(train.join("synthetic_training.csv")).unwrap();
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    assert_eq!(header, CSV_COLUMNS);
    lines
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect()
}

#[test]
fn pipeline_writes_contigs_and_training_csv_schema() {
    let dir = tempfile::tempdir().unwrap();

    let rows = run_pipeline(dir.path(), Some("21"));

    assert!(
        !rows.is_empty(),
        "expected junction rows for a genome with repeats"
    );
    for row in &rows {
        assert_eq!(row.len(), 21, "row has {} fields: {row:?}", row.len());
        let field = |name: &str| &row[CSV_COLUMNS.iter().position(|c| *c == name).unwrap()];
        assert_eq!(field("k"), "21");
        assert_eq!(field("node_seq").len(), 20);
        assert!(field("context_seq").len() <= 100);
        assert!(field("out_degree").parse::<u8>().unwrap() >= 2);

        let mut covered = 0;
        for base in ["A", "C", "G", "T"] {
            let seq = field(&format!("branch_{base}_seq"));
            let cov = field(&format!("branch_{base}_cov"));
            assert_eq!(seq == "NA", cov == "NA", "seq/cov NA mismatch in {row:?}");
            if seq != "NA" {
                assert!((1..=100).contains(&seq.len()));
                assert!(cov.parse::<u32>().unwrap() >= 2);
                covered += 1;
            }
        }
        assert!(covered >= 2, "row with fewer than 2 branches: {row:?}");
        assert!(["true", "false"].contains(&field("gt_ambiguous").as_str()));
        field("cov_ratio_max").parse::<f64>().unwrap();
        field("repeat_len").parse::<usize>().unwrap();
    }
}

#[test]
fn pipeline_default_k_is_31() {
    let dir = tempfile::tempdir().unwrap();

    let rows = run_pipeline(dir.path(), None);

    assert!(!rows.is_empty());
    assert!(rows.iter().all(|r| r[19] == "31" && r[5].len() == 30));
}

#[test]
fn gen_train_data_fails_without_junctions() {
    let dir = tempfile::tempdir().unwrap();
    let fasta = dir.path().join("unique.fna");
    let mut rng = StdRng::seed_from_u64(11);
    write_fasta(&fasta, &random_seq(&mut rng, 2000));

    let output = cargo_bin_cmd!("dna-assembler")
        .arg("pipeline")
        .arg("--reference")
        .arg(&fasta)
        .args(["--k", "21", "--coverage", "30", "--gen-train-data"])
        .arg("--results-dir")
        .arg(dir.path().join("results"))
        .arg("--train-data-dir")
        .arg(dir.path().join("train"))
        .assert()
        .failure()
        .get_output()
        .clone();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("No junctions found"), "stderr: {stderr}");
    assert!(stderr.contains("lowering k"), "stderr: {stderr}");
}
