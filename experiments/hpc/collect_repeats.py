#!/usr/bin/env python3
"""Zbiera wyniki 12 powtorzen (4 warianty x 3 seedy) w tabele mean+/-std."""
import json
from pathlib import Path
from collections import defaultdict
import statistics

base = Path("/scratch/fpietrew/dna-assembler/results/eval_repeats")
variants = ["group_a_farthest", "group_a_random", "group_b_farthest", "group_b_random"]
seeds = [42, 123, 7]

# Krok 1: per-seed, srednia wazona po 10 organizmach (jak w oryginalnej tabeli)
per_seed_acc = defaultdict(dict)  # variant -> {seed: weighted_acc}
per_organism = defaultdict(lambda: defaultdict(list))  # variant -> organism -> [acc_per_seed]

for variant in variants:
    for seed in seeds:
        run_dir = base / f"{variant}_seed{seed}"
        if not run_dir.exists():
            print(f"UWAGA: brak {run_dir}")
            continue
        total_correct_w = 0.0
        total_n = 0
        for jf in sorted(run_dir.glob("*.json")):
            with open(jf) as f:
                data = json.load(f)
            acc = data["junction_accuracy"]
            n = data["n_junctions"]
            total_correct_w += acc * n
            total_n += n
            per_organism[variant][jf.stem].append(acc)
        if total_n > 0:
            per_seed_acc[variant][seed] = total_correct_w / total_n

print(f"{'variant':<20} {'seed':>6} {'weighted_acc':>13}")
print("-" * 45)
for variant in variants:
    for seed in seeds:
        if seed in per_seed_acc[variant]:
            print(f"{variant:<20} {seed:>6} {per_seed_acc[variant][seed]*100:>12.2f}%")

print("\n" + "=" * 60)
print("FINALNE: mean +/- std PO SEEDACH (n=3), na wazonej sredniej z 10 organizmow")
print("=" * 60)
for variant in variants:
    vals = list(per_seed_acc[variant].values())
    if not vals:
        continue
    mean_v = statistics.mean(vals)
    std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{variant:<20} {mean_v*100:6.2f}% +/- {std_v*100:.2f}pp  (seeds: {[f'{v*100:.1f}%' for v in vals]})")

print("\n" + "=" * 60)
print("Rozbicie per organizm (srednia po 3 seedach) - pierwsze i ostatnie 3 na wariant")
print("=" * 60)
for variant in variants:
    orgs = per_organism[variant]
    org_means = [(o, statistics.mean(v)) for o, v in orgs.items()]
    org_means.sort(key=lambda x: x[1])
    print(f"\n{variant}:")
    for o, m in org_means:
        print(f"  {o:<25} {m*100:6.2f}%")
