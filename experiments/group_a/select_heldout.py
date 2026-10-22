"""Pick the 10 phylogenetically-farthest genomes as held-out via Mash distance.

Implements the "farthest" selection rule for the transfer experiment: the
held-out set is not a random split, it is the N genomes that maximize the
minimum distance to the rest of the pool. Taxonomy alone cannot do this for group A
(same-species E. coli - every candidate shares the same taxon), so this
uses real k-mer sketch distance (Mash) instead.

Requires the `mash` CLI (https://github.com/marbl/Mash) on PATH - install
with `sudo apt install mash` (not automated here, matches the project rule
of not installing new dependencies without the user's go-ahead).

Usage (from repo root):
    python3 experiments/group_a/select_heldout.py

Reads:
    genomes/ecoli_group_a/*.fna  (or --manifest for the accession list)
Writes:
    experiments/group_a/results/distances.csv          - long-form pairwise distances
    experiments/group_a/results/train_accessions.txt
    experiments/group_a/results/heldout_accessions.txt
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import statistics
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENOMES_DIR = REPO_ROOT / "genomes" / "ecoli_group_a"
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "group_a" / "results"


def require_mash() -> None:
    """Raise a clear error if the `mash` CLI is not on PATH."""
    if shutil.which("mash") is None:
        raise SystemExit(
            "mash CLI not found on PATH. Install it first, e.g.:\n"
            "  sudo apt install mash\n"
            "(new CLI dependency for this experiment, confirmed with the user)"
        )


def list_genomes(genomes_dir: Path) -> list[Path]:
    """List genome FASTA files, keyed by accession (filename stem)."""
    fastas = sorted(genomes_dir.glob("*.fna"))
    if not fastas:
        raise SystemExit(f"No .fna files found in {genomes_dir}")
    return fastas


def sketch_and_dist(fastas: list[Path], work_dir: Path, jobs: int) -> Path:
    """Run `mash sketch` then `mash dist` all-vs-all, returning the raw TSV path.

    Args:
        fastas: Genome FASTA files to sketch.
        work_dir: Scratch directory for the sketch file and raw distance TSV.
        jobs: Thread count passed to `mash sketch -p`.

    Returns:
        Path to the raw `mash dist` TSV output.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    file_list = work_dir / "genome_list.txt"
    file_list.write_text("\n".join(str(f) for f in fastas) + "\n")

    sketch_prefix = work_dir / "pool"
    logger.info("mash sketch: %d genomes", len(fastas))
    subprocess.run(
        [
            "mash",
            "sketch",
            "-p",
            str(jobs),
            "-o",
            str(sketch_prefix),
            "-l",
            str(file_list),
        ],
        check=True,
    )
    sketch_path = sketch_prefix.with_suffix(".msh")

    dist_path = work_dir / "mash_dist_raw.tsv"
    logger.info("mash dist (all-vs-all)")
    with open(dist_path, "w") as f:
        subprocess.run(
            ["mash", "dist", str(sketch_path), str(sketch_path)], check=True, stdout=f
        )
    return dist_path


def accession_from_path(path_str: str) -> str:
    """Extract the accession (filename stem) from a `mash dist` path column."""
    return Path(path_str).stem


def load_distance_matrix(dist_tsv: Path) -> dict[str, dict[str, float]]:
    """Parse `mash dist` output into a nested dict {a: {b: distance}}.

    Args:
        dist_tsv: Raw TSV from `mash dist ref.msh ref.msh`
            (columns: reference-ID, query-ID, mash-distance, p-value, matching-hashes).

    Returns:
        Distance matrix as nested dicts, symmetric by construction.
    """
    matrix: dict[str, dict[str, float]] = {}
    with open(dist_tsv) as f:
        for line in f:
            ref_path, query_path, distance, _p_value, _hashes = line.rstrip("\n").split(
                "\t"
            )
            a, b = accession_from_path(ref_path), accession_from_path(query_path)
            matrix.setdefault(a, {})[b] = float(distance)
    return matrix


def select_heldout(
    matrix: dict[str, dict[str, float]],
    n_heldout: int,
) -> tuple[list[str], list[str]]:
    """Select the N genomes maximizing minimum distance to the rest of the pool.

    Args:
        matrix: Pairwise distance matrix (see `load_distance_matrix`).
        n_heldout: Number of genomes to hold out.

    Returns:
        Tuple of ``(train_accessions, heldout_accessions)``.
    """
    accessions = sorted(matrix.keys())
    scores = {a: min(matrix[a][b] for b in accessions if b != a) for a in accessions}
    ranked = sorted(accessions, key=lambda a: scores[a], reverse=True)
    heldout = ranked[:n_heldout]
    train = ranked[n_heldout:]
    return train, heldout


def write_split(train: list[str], heldout: list[str], results_dir: Path) -> None:
    """Write the train/held-out accession lists, one accession per line."""
    (results_dir / "train_accessions.txt").write_text("\n".join(sorted(train)) + "\n")
    (results_dir / "heldout_accessions.txt").write_text(
        "\n".join(sorted(heldout)) + "\n"
    )
    logger.info("train=%d heldout=%d -> %s", len(train), len(heldout), results_dir)


def write_distances_csv(matrix: dict[str, dict[str, float]], out_path: Path) -> None:
    """Write the pairwise distance matrix in long form (a, b, distance)."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["genome_a", "genome_b", "mash_distance"])
        for a in sorted(matrix):
            for b in sorted(matrix[a]):
                writer.writerow([a, b, matrix[a][b]])


def report_distance_summary(
    matrix: dict[str, dict[str, float]], train: list[str], heldout: list[str]
) -> None:
    """Log median train<->train vs train<->heldout distance as a sanity check.

    Held-out genomes should sit farther from the training pool than training
    genomes sit from each other - this is what "farthest" is meant to buy us.
    """
    train_train = [matrix[a][b] for a in train for b in train if a != b]
    train_heldout = [matrix[a][b] for a in train for b in heldout]
    if train_train and train_heldout:
        logger.info(
            "median distance: train<->train=%.5f  train<->heldout=%.5f",
            statistics.median(train_train),
            statistics.median(train_heldout),
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--genomes-dir",
        default=str(DEFAULT_GENOMES_DIR),
        help="Directory of downloaded genome FASTA files",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Output directory for distances.csv and accession lists",
    )
    parser.add_argument(
        "--n-heldout",
        type=int,
        default=10,
        help="Number of genomes to hold out (the phylogenetically farthest)",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Threads for `mash sketch -p`"
    )
    return parser


def main() -> None:
    """Parse CLI arguments and run Mash-based held-out selection."""
    args = build_parser().parse_args()
    require_mash()

    genomes_dir = Path(args.genomes_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    fastas = list_genomes(genomes_dir)
    dist_tsv = sketch_and_dist(fastas, results_dir / "mash_work", jobs=args.jobs)
    matrix = load_distance_matrix(dist_tsv)

    train, heldout = select_heldout(matrix, n_heldout=args.n_heldout)
    write_split(train, heldout, results_dir)
    write_distances_csv(matrix, results_dir / "distances.csv")
    report_distance_summary(matrix, train, heldout)


if __name__ == "__main__":
    main()
