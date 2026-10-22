#!/usr/bin/env python3
"""Zbiera wyniki solo baseline (val_junction_accuracy z metadanych checkpointow)."""
import os
import torch
from pathlib import Path
from collections import defaultdict
import statistics

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2]))
ckpt_dir = ROOT / "results/checkpoints_solo"

# nazwa pliku: mv1_siamese_cnn_solo_<grp>_<acc>_seed<seed>_best.pt
by_org = defaultdict(list)

for pt_file in sorted(ckpt_dir.glob("*_best.pt")):
    ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
    name = pt_file.stem.replace("mv1_siamese_cnn_solo_", "").replace("_best", "")
    # name = "<grp>_<acc>_seed<seed>"
    parts = name.rsplit("_seed", 1)
    org_key = parts[0]  # "<grp>_<acc>"
    val_acc = ckpt.get("val_junction_accuracy")
    by_org[org_key].append(val_acc)

print(f"{'organizm':<25} {'n_seeds':>8} {'mean_acc':>10} {'std':>8}")
print("-" * 55)
group_a_means = []
group_b_means = []
for org_key in sorted(by_org.keys()):
    vals = by_org[org_key]
    mean_v = statistics.mean(vals)
    std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{org_key:<25} {len(vals):>8} {mean_v*100:>9.2f}% {std_v*100:>7.2f}pp")
    if org_key.startswith("a_"):
        group_a_means.append(mean_v)
    else:
        group_b_means.append(mean_v)

print("\n" + "=" * 55)
print(f"Grupa A: srednia po organizmach = {statistics.mean(group_a_means)*100:.2f}% "
      f"(+/-{statistics.stdev(group_a_means)*100:.2f}pp), n_organizmow={len(group_a_means)}")
print(f"Grupa B: srednia po organizmach = {statistics.mean(group_b_means)*100:.2f}% "
      f"(+/-{statistics.stdev(group_b_means)*100:.2f}pp), n_organizmow={len(group_b_means)}")
