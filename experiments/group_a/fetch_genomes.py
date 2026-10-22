"""Download a reproducible random pool of complete E. coli genomes from NCBI RefSeq.

Uses the NCBI FTP per-species ``assembly_summary.txt`` (plain HTTP download,
no extra CLI dependency) instead of the NCBI Datasets CLI, since the summary
file already carries every column needed for filtering (assembly level,
exclusion flags, genome size) without querying anything else.

Usage (from repo root):
    cd ml && uv run python3 ../experiments/group_a/fetch_genomes.py
    (or: python3 experiments/group_a/fetch_genomes.py, stdlib only)

Output:
    genomes/ecoli_group_a/<accession>.fna       - one FASTA per genome
    experiments/group_a/results/manifest.csv    - accession, strain, size, path, status
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import random
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/Escherichia_coli/assembly_summary.txt"
REQUEST_TIMEOUT_S = 60


@dataclass
class Candidate:
    """One row of the filtered NCBI assembly summary."""

    accession: str
    organism_name: str
    strain: str
    genome_size: int
    ftp_path: str

    def genomic_fna_gz_url(self) -> str:
        """Build the ``*_genomic.fna.gz`` download URL from ``ftp_path``.

        ``ftp_path`` already ends with a trailing slash and points at a
        directory whose basename is the filename prefix, e.g.
        ``.../GCF_000157115.2_Escherichia_sp_3_2_53FAA_V2/``.
        """
        base = self.ftp_path.rstrip("/")
        name = base.rsplit("/", 1)[-1]
        return f"{base}/{name}_genomic.fna.gz"


def download_summary(url: str, dest: Path, refresh: bool) -> Path:
    """Download the NCBI assembly summary TSV, caching it locally.

    Args:
        url: Source URL of ``assembly_summary.txt``.
        dest: Local cache path.
        refresh: When ``True``, re-download even if ``dest`` already exists.

    Returns:
        Path to the local file.
    """
    if dest.exists() and not refresh:
        logger.info("Using cached assembly summary: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading assembly summary from %s", url)
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as resp:
        dest.write_bytes(resp.read())
    logger.info("Saved assembly summary -> %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def parse_summary(path: Path) -> list[dict[str, str]]:
    """Parse the NCBI assembly summary TSV into a list of row dicts.

    The file has a leading ``##`` comment line, then a header line starting
    with ``#assembly_accession`` (the ``#`` is stripped before parsing).

    Args:
        path: Local path to ``assembly_summary.txt``.

    Returns:
        List of rows as dicts keyed by column name.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = next(
        i for i, line in enumerate(lines) if line.startswith("#assembly_accession")
    )
    lines[header_idx] = lines[header_idx].lstrip("#")
    rows = list(csv.DictReader(lines[header_idx:], delimiter="\t"))
    logger.info("Parsed %d rows from %s", len(rows), path)
    return rows


def filter_candidates(
    rows: list[dict[str, str]],
    min_size_bp: int,
    max_size_bp: int,
) -> list[Candidate]:
    """Filter assembly summary rows to usable, appropriately-sized complete genomes.

    Args:
        rows: Parsed assembly summary rows.
        min_size_bp: Minimum genome size in base pairs.
        max_size_bp: Maximum genome size in base pairs.

    Returns:
        List of :class:`Candidate` passing all filters.
    """
    candidates = []
    for row in rows:
        if row.get("assembly_level") != "Complete Genome":
            continue
        # "na" means "not excluded" in this column; any other value is a real flag.
        if row.get("excluded_from_refseq", "na") != "na":
            continue
        try:
            size = int(row["genome_size"])
        except (KeyError, ValueError):
            continue
        if not (min_size_bp <= size <= max_size_bp):
            continue
        candidates.append(
            Candidate(
                accession=row["assembly_accession"],
                organism_name=row.get("organism_name", ""),
                strain=row.get("infraspecific_name", ""),
                genome_size=size,
                ftp_path=row["ftp_path"],
            )
        )
    logger.info(
        "Filtered to %d candidates (Complete Genome, not excluded, %d-%d bp)",
        len(candidates),
        min_size_bp,
        max_size_bp,
    )
    return candidates


def download_genome(
    candidate: Candidate, out_dir: Path
) -> tuple[Candidate, str | None]:
    """Download and decompress one genome FASTA.

    Args:
        candidate: The genome to fetch.
        out_dir: Directory to write ``<accession>.fna`` into.

    Returns:
        Tuple of ``(candidate, error)`` - ``error`` is ``None`` on success.
    """
    dest = out_dir / f"{candidate.accession}.fna"
    if dest.exists():
        return candidate, None

    gz_dest = dest.with_suffix(dest.suffix + ".gz")
    try:
        url = candidate.genomic_fna_gz_url()
        with (
            urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as resp,
            open(gz_dest, "wb") as f,
        ):
            shutil.copyfileobj(resp, f)
        with gzip.open(gz_dest, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_dest.unlink()
        return candidate, None
    except Exception as exc:  # noqa: BLE001 - record and continue, don't kill the batch
        gz_dest.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)
        return candidate, str(exc)


def write_manifest(
    results: list[tuple[Candidate, str | None, Path]],
    manifest_path: Path,
) -> None:
    """Write the download manifest CSV.

    Args:
        results: List of ``(candidate, error, local_path)`` tuples.
        manifest_path: Destination CSV path.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "accession",
                "organism_name",
                "strain",
                "genome_size",
                "local_path",
                "status",
            ]
        )
        for cand, error, local_path in results:
            writer.writerow(
                [
                    cand.accession,
                    cand.organism_name,
                    cand.strain,
                    cand.genome_size,
                    str(local_path) if error is None else "",
                    "ok" if error is None else f"FAILED: {error}",
                ]
            )
    n_ok = sum(1 for _, error, _ in results if error is None)
    logger.info("Manifest written -> %s (%d/%d ok)", manifest_path, n_ok, len(results))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-url",
        default=DEFAULT_SUMMARY_URL,
        help="NCBI assembly_summary.txt URL for the target species",
    )
    parser.add_argument(
        "--summary-cache",
        default=None,
        help="Local cache path for the summary TSV "
        "(default: experiments/group_a/results/assembly_summary.txt)",
    )
    parser.add_argument(
        "--refresh-summary",
        action="store_true",
        help="Re-download assembly_summary.txt even if cached",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for downloaded FASTA files (default: genomes/ecoli_group_a)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output manifest CSV path "
        "(default: experiments/group_a/results/manifest.csv)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=110,
        help="Number of genomes to sample and download "
        "(buffer above the 90+10 target for download failures)",
    )
    parser.add_argument(
        "--min-size-mbp", type=float, default=2.0, help="Min genome size (Mbp)"
    )
    parser.add_argument(
        "--max-size-mbp", type=float, default=8.0, help="Max genome size (Mbp)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for candidate sampling"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Concurrent downloads (network-bound, keep modest)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only download the first N sampled candidates "
        "(smoke-testing the pipeline without a full batch)",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and run the full fetch pipeline."""
    args = build_parser().parse_args()

    out_dir = (
        Path(args.out_dir) if args.out_dir else REPO_ROOT / "genomes" / "ecoli_group_a"
    )
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else REPO_ROOT / "experiments" / "group_a" / "results" / "manifest.csv"
    )
    summary_cache = (
        Path(args.summary_cache)
        if args.summary_cache
        else REPO_ROOT / "experiments" / "group_a" / "results" / "assembly_summary.txt"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = download_summary(
        args.summary_url, summary_cache, args.refresh_summary
    )
    rows = parse_summary(summary_path)
    candidates = filter_candidates(
        rows,
        min_size_bp=int(args.min_size_mbp * 1_000_000),
        max_size_bp=int(args.max_size_mbp * 1_000_000),
    )
    if not candidates:
        raise SystemExit(
            "No candidates survived filtering - check assembly_summary.txt filters."
        )

    n = min(args.n_candidates, len(candidates))
    sampled = random.Random(args.seed).sample(candidates, k=n)
    if args.limit is not None:
        sampled = sampled[: args.limit]
    logger.info("Sampled %d candidates (seed=%d)", len(sampled), args.seed)

    results: list[tuple[Candidate, str | None, Path]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(download_genome, cand, out_dir): cand for cand in sampled
        }
        for i, future in enumerate(as_completed(futures), start=1):
            cand = futures[future]
            _, error = future.result()
            dest = out_dir / f"{cand.accession}.fna"
            if error:
                logger.warning(
                    "[%d/%d] FAILED %s: %s", i, len(sampled), cand.accession, error
                )
            else:
                logger.info(
                    "[%d/%d] OK %s -> %s", i, len(sampled), cand.accession, dest
                )
            results.append((cand, error, dest))

    write_manifest(results, manifest_path)


if __name__ == "__main__":
    main()
