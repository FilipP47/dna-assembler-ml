#!/usr/bin/env python3
"""
Liczy baseline max-coverage na tych samych 40 plikach testowych, replikujac
filtrowanie z loader.py (length filter + label filter), zeby n_junctions
bylo porownywalne 1:1 z wynikami main.py evaluate.

Nie wymaga polars - czysty csv ze standardowej biblioteki.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict

BASES = ["A", "C", "G", "T"]

MIN_CONTEXT_LEN = 10  # domyslne z main.py evaluate
MIN_BRANCH_LEN  = 5   # domyslne z main.py evaluate

BASE_DIR = Path("/scratch/fpietrew/dna-assembler/data/training")

VARIANTS = {
    "group_a_farthest": BASE_DIR / "group_a" / "farthest_test",
    "group_a_random":   BASE_DIR / "group_a" / "random_test",
    "group_b_farthest": BASE_DIR / "group_b" / "farthest_test",
    "group_b_random":   BASE_DIR / "group_b" / "random_test",
}


def evaluate_file(csv_path: Path) -> dict:
    total = 0
    length_filtered = 0
    usable = 0
    correct = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1

            context_len = len(row.get("context_seq") or "")
            branch_lens = [len(row.get(f"branch_{b}_seq") or "") for b in BASES]
            max_branch_len = max(branch_lens)

            if context_len < MIN_CONTEXT_LEN or max_branch_len < MIN_BRANCH_LEN:
                continue
            length_filtered += 1

            gt_ambig_raw = (row.get("gt_ambiguous") or "").strip().lower()
            if gt_ambig_raw != "false":
                continue

            gt = (row.get("ground_truth") or "").strip().upper()
            if gt not in BASES:
                continue

            correct_seq = row.get(f"branch_{gt}_seq") or ""
            if not correct_seq or correct_seq == "NA":
                continue

            usable += 1

            covs = {}
            for b in BASES:
                raw = row.get(f"branch_{b}_cov")
                try:
                    covs[b] = float(raw) if raw not in (None, "", "NA") else -1.0
                except ValueError:
                    covs[b] = -1.0

            pred = max(BASES, key=lambda b: covs[b])  # tie-break: kolejnosc A,C,G,T
            if pred == gt:
                correct += 1

    acc = correct / usable if usable > 0 else 0.0
    return {
        "total_rows": total,
        "after_length_filter": length_filtered,
        "usable": usable,
        "correct": correct,
        "baseline_junction_accuracy": acc,
    }


def main():
    rows = []
    for variant, test_dir in VARIANTS.items():
        if not test_dir.exists():
            print(f"UWAGA: brak katalogu {test_dir}")
            continue
        for csv_file in sorted(test_dir.glob("*.csv")):
            organism = csv_file.stem.replace("_training", "")
            result = evaluate_file(csv_file)
            result["variant"] = variant
            result["organism"] = organism
            rows.append(result)

    print(f"{'variant':<20} {'organism':<20} {'baseline_acc':>13} {'n':>6}")
    print("-" * 65)
    for r in rows:
        print(f"{r['variant']:<20} {r['organism']:<20} "
              f"{r['baseline_junction_accuracy']*100:>12.2f}% {r['usable']:>6}")

    print("\n" + "=" * 65)
    print("SREDNIE per wariant:")
    print("=" * 65)
    import statistics
    for variant in VARIANTS:
        vals = [r for r in rows if r["variant"] == variant]
        if not vals:
            continue
        mean_acc = statistics.mean(r["baseline_junction_accuracy"] for r in vals)
        std_acc = statistics.stdev(r["baseline_junction_accuracy"] for r in vals) if len(vals) > 1 else 0.0
        total_n = sum(r["usable"] for r in vals)
        weighted_acc = sum(r["baseline_junction_accuracy"] * r["usable"] for r in vals) / total_n
        print(f"{variant:<20} n_organisms={len(vals):2d}  "
              f"mean_acc={mean_acc*100:6.2f}% (+/-{std_acc*100:.2f}pp)  "
              f"weighted_acc={weighted_acc*100:6.2f}%  total_n={total_n}")

    out_path = Path("/scratch/fpietrew/dna-assembler/results/baseline_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nZapisano szczegoly -> {out_path}")


if __name__ == "__main__":
    main()
