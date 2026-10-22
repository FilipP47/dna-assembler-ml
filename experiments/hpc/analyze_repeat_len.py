#!/usr/bin/env python3
"""
Liczy trafnosc modelu w rozbiciu po repeat_len i cov_ratio_max, na zbiorze
testowym grupy A (farthest). Analogiczne do sekcji RAPORT.md "Natura sufitu".

Wymaga torch + polars (odpalic w dzialajacym venv, np. venv-analysis).
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import polars as pl
import numpy as np

sys.path.insert(0, "/scratch/fpietrew/dna-assembler/ml")
from data.loader import load_training_csv
from data.dataset import JunctionDataset, collate_junction_batch
from models.mv1 import Mv1JunctionModel

TEST_DIR = Path("/scratch/fpietrew/dna-assembler/data/training/group_a/farthest_test")
CKPT_PATH = Path("/scratch/fpietrew/dna-assembler/results/checkpoints/mv1_siamese_cnn_group_a_farthest_best.pt")

device = torch.device("cpu")

checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
hparams = checkpoint.get("model_hparams", {})
model = Mv1JunctionModel(
    context_embed_dim=hparams.get("context_embed_dim", 64),
    branch_embed_dim=hparams.get("branch_embed_dim", 64),
    cov_feat_dim=hparams.get("cov_feat_dim", 6),
    hidden_dim=hparams.get("hidden_dim", 64),
)
model.load_state_dict(checkpoint["model_state"])
use_cov = hparams.get("cov_feat_dim", 6) > 0
model.to(device)
model.eval()

all_results = []  # (junction_id, organism, correct: bool)

for csv_file in sorted(TEST_DIR.glob("*.csv")):
    organism = csv_file.stem.replace("_training", "")
    df, stats = load_training_csv(
        csv_path=csv_file, min_context_len=10, min_branch_len=5, only_labeled=True,
    )
    dataset = JunctionDataset(
        df=df, context_len=150, branch_len=100,
        augment=False, use_cov_feat=use_cov, pair_seed=0,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_junction_batch)

    with torch.no_grad():
        for batch in loader:
            context  = batch["context"].to(device)
            b_a      = batch["branch_a"].to(device)
            b_b      = batch["branch_b"].to(device)
            cov_feat = batch["cov_feat"].to(device)
            labels   = batch["label"].to(device)
            j_ids    = batch["junction_id"].cpu().numpy()

            sim_a, sim_b = model(context, b_a, b_b, cov_feat)
            logits = torch.stack([sim_a, sim_b], dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            lbls  = labels.cpu().numpy()

            for jid, pred, lbl in zip(j_ids, preds, lbls):
                all_results.append((organism, int(jid), bool(pred == lbl)))

    print(f"  {organism}: {len(dataset)} pary przetworzone")

# junction-level correctness (wszystkie pary danego junctionu musza byc poprawne)
res_df = pl.DataFrame(all_results, schema=["organism", "junction_id_local", "correct"], orient="row")
junc_correct = (
    res_df.group_by(["organism", "junction_id_local"])
    .agg(pl.col("correct").all().alias("junction_correct"))
)

print(f"\nLaczna liczba unikalnych junctionow ocenionych: {junc_correct.height}")

# Teraz trzeba dolaczyc repeat_len / cov_ratio_max z oryginalnych CSV.
# junction_id w JunctionDataset to RAW junction_id z CSV (loader offsetuje go
# TYLKO przy laczeniu wielu plikow w jeden katalog - tu kazdy plik czytany
# osobno, wiec junction_id_local odpowiada oryginalnemu junction_id z tego
# jednego pliku CSV organizmu).
meta_rows = []
for csv_file in sorted(TEST_DIR.glob("*.csv")):
    organism = csv_file.stem.replace("_training", "")
    df = pl.read_csv(csv_file, infer_schema_length=0)
    df = df.select(["junction_id", "repeat_len", "cov_ratio_max", "gt_ambiguous"])
    df = df.filter(pl.col("gt_ambiguous").str.to_lowercase() == "false")
    df = df.with_columns([
        pl.col("junction_id").cast(pl.Int64),
        pl.col("repeat_len").cast(pl.Int64, strict=False),
        pl.col("cov_ratio_max").cast(pl.Float64, strict=False),
        pl.lit(organism).alias("organism"),
    ])
    meta_rows.append(df)

meta_df = pl.concat(meta_rows, how="vertical_relaxed")
meta_df = meta_df.rename({"junction_id": "junction_id_local"})

merged = junc_correct.join(meta_df, on=["organism", "junction_id_local"], how="inner")
print(f"Po polaczeniu z metadanymi: {merged.height} junctionow (z {junc_correct.height} ocenionych)")

# --- Rozbicie po repeat_len ---
print("\n=== Trafnosc wg repeat_len ===")
bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]
for lo, hi in bins:
    sub = merged.filter((pl.col("repeat_len") >= lo) & (pl.col("repeat_len") <= hi))
    n = sub.height
    if n == 0:
        continue
    acc = sub["junction_correct"].mean()
    label = f"{lo}" if lo == hi else (f">={lo}" if hi >= 10**9 else f"{lo}-{hi}")
    print(f"  repeat_len {label:<8} n={n:5d}  model_acc={acc*100:6.2f}%")

# --- Rozbicie po cov_ratio_max ---
print("\n=== Trafnosc wg cov_ratio_max ===")
cov_bins = [(0.0, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]
for lo, hi in cov_bins:
    sub = merged.filter((pl.col("cov_ratio_max") >= lo) & (pl.col("cov_ratio_max") < hi))
    n = sub.height
    if n == 0:
        continue
    acc = sub["junction_correct"].mean()
    print(f"  cov_ratio_max [{lo:.2f}-{hi:.2f}) n={n:5d}  model_acc={acc*100:6.2f}%")

print(f"\nOgolem: n={merged.height}  model_acc={merged['junction_correct'].mean()*100:.2f}%")
