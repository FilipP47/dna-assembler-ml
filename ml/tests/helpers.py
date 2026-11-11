"""Shared test helpers: synthetic junction data in the Rust CSV schema."""

from __future__ import annotations

import random

import polars as pl

BASES = ["A", "C", "G", "T"]


def make_junction_df(n_junctions: int, seed: int = 0) -> pl.DataFrame:
    """Build a synthetic training DataFrame with every column the Rust pipeline writes.

    Each junction has two branches (the correct one and one wrong one); the
    other two branch columns are ``"NA"``.

    Args:
        n_junctions: Number of junction rows to generate.
        seed: Seed for the random sequence generator.

    Returns:
        Polars DataFrame that ``load_training_csv`` accepts once written to CSV
        and that ``JunctionDataset`` accepts directly.
    """
    rng = random.Random(seed)

    def seq(n: int) -> str:
        return "".join(rng.choice(BASES) for _ in range(n))

    rows = []
    for jid in range(n_junctions):
        gt = rng.choice(BASES)
        other = rng.choice([b for b in BASES if b != gt])
        row: dict[str, object] = {
            "junction_id": jid,
            "contig_id": jid,
            "depth_in_contig": 10,
            "out_degree": 2,
            "in_degree": 1,
            "node_seq": seq(20),
            "context_seq": seq(100),
        }
        for base in BASES:
            active = base in (gt, other)
            row[f"branch_{base}_seq"] = seq(100) if active else "NA"
            row[f"branch_{base}_cov"] = str(rng.randint(2, 50)) if active else "NA"
        row.update(
            {
                "ground_truth": gt,
                "gt_ambiguous": "false",
                "cov_ratio_max": 0.5,
                "total_reads": 40,
                "k": 21,
                "repeat_len": 3,
            }
        )
        rows.append(row)
    return pl.DataFrame(rows)
