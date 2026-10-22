#!/usr/bin/env python3
"""Zbiera wszystkie wyniki evaluate (JSON per organizm) w jedna tabele CSV."""
import json
from pathlib import Path
import sys

base = Path("/scratch/fpietrew/dna-assembler/results")
variants = ["group_a_farthest", "group_a_random", "group_b_farthest", "group_b_random"]

rows = []
for variant in variants:
    eval_dir = base / f"eval_{variant}"
    if not eval_dir.exists():
        print(f"UWAGA: brak katalogu {eval_dir}", file=sys.stderr)
        continue
    for jf in sorted(eval_dir.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        rows.append({
            "variant": variant,
            "organism": jf.stem,
            "junction_accuracy": data.get("junction_accuracy"),
            "pair_accuracy": data.get("pair_accuracy"),
            "f1": data.get("f1"),
            "n_junctions": data.get("n_junctions"),
        })

print(f"{'variant':<20} {'organism':<20} {'junc_acc':>9} {'pair_acc':>9} {'f1':>7} {'n':>6}")
print("-" * 75)
for r in rows:
    print(f"{r['variant']:<20} {r['organism']:<20} "
          f"{r['junction_accuracy']*100:>8.2f}% {r['pair_accuracy']*100:>8.2f}% "
          f"{r['f1']:>7.4f} {r['n_junctions']:>6}")

print("\n" + "=" * 75)
print("SREDNIE (nieważone po organizmach) per wariant:")
print("=" * 75)
import statistics
for variant in variants:
    vals = [r for r in rows if r["variant"] == variant]
    if not vals:
        continue
    mean_acc = statistics.mean(r["junction_accuracy"] for r in vals)
    std_acc = statistics.stdev(r["junction_accuracy"] for r in vals) if len(vals) > 1 else 0.0
    total_n = sum(r["n_junctions"] for r in vals)
    weighted_acc = sum(r["junction_accuracy"] * r["n_junctions"] for r in vals) / total_n
    print(f"{variant:<20} n_organisms={len(vals):2d}  "
          f"mean_junc_acc={mean_acc*100:6.2f}% (+/-{std_acc*100:.2f}pp)  "
          f"weighted_junc_acc={weighted_acc*100:6.2f}%  total_n={total_n}")
