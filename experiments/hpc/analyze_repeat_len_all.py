#!/usr/bin/env python3
"""
Jak analyze_repeat_len.py, ale przechodzi przez wszystkie 4 warianty
(grupa A/B x farthest/random) i drukuje porownawcza tabele.
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import polars as pl

sys.path.insert(0, "/scratch/fpietrew/dna-assembler/ml")
from data.loader import load_training_csv
from data.dataset import JunctionDataset, collate_junction_batch
from models.mv1 import Mv1JunctionModel

BASE = Path("/scratch/fpietrew/dna-assembler")

VARIANTS = {
    "group_a_farthest": (BASE / "data/training/group_a/farthest_test",
                          BASE / "results/checkpoints/mv1_siamese_cnn_group_a_farthest_best.pt"),
    "group_a_random":   (BASE / "data/training/group_a/random_test",
                          BASE / "results/checkpoints/mv1_siamese_cnn_group_a_random_best.pt"),
    "group_b_farthest": (BASE / "data/training/group_b/farthest_test",
                          BASE / "results/checkpoints/mv1_siamese_cnn_group_b_farthest_best.pt"),
    "group_b_random":   (BASE / "data/training/group_b/random_test",
                          BASE / "results/checkpoints/mv1_siamese_cnn_group_b_random_best.pt"),
}

REPEAT_BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]
COV_BINS = [(0.0, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]

device = torch.device("cpu")


def evaluate_variant(test_dir: Path, ckpt_path: Path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

    all_results = []
    for csv_file in sorted(test_dir.glob("*.csv")):
        organism = csv_file.stem.replace("_training", "")
        df, _ = load_training_csv(csv_path=csv_file, min_context_len=10, min_branch_len=5, only_labeled=True)
        dataset = JunctionDataset(df=df, context_len=150, branch_len=100, augment=False, use_cov_feat=use_cov, pair_seed=0)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_junction_batch)
        with torch.no_grad():
            for batch in loader:
                context, b_a, b_b, cov_feat, labels = (
                    batch["context"].to(device), batch["branch_a"].to(device),
                    batch["branch_b"].to(device), batch["cov_feat"].to(device), batch["label"].to(device),
                )
                j_ids = batch["junction_id"].cpu().numpy()
                sim_a, sim_b = model(context, b_a, b_b, cov_feat)
                logits = torch.stack([sim_a, sim_b], dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
                lbls = labels.cpu().numpy()
                for jid, pred, lbl in zip(j_ids, preds, lbls):
                    all_results.append((organism, int(jid), bool(pred == lbl)))

    res_df = pl.DataFrame(all_results, schema=["organism", "junction_id_local", "correct"], orient="row")
    junc_correct = res_df.group_by(["organism", "junction_id_local"]).agg(pl.col("correct").all().alias("junction_correct"))

    meta_rows = []
    for csv_file in sorted(test_dir.glob("*.csv")):
        organism = csv_file.stem.replace("_training", "")
        d = pl.read_csv(csv_file, infer_schema_length=0)
        d = d.select(["junction_id", "repeat_len", "cov_ratio_max", "gt_ambiguous"])
        d = d.filter(pl.col("gt_ambiguous").str.to_lowercase() == "false")
        d = d.with_columns([
            pl.col("junction_id").cast(pl.Int64),
            pl.col("repeat_len").cast(pl.Int64, strict=False),
            pl.col("cov_ratio_max").cast(pl.Float64, strict=False),
            pl.lit(organism).alias("organism"),
        ])
        meta_rows.append(d)
    meta_df = pl.concat(meta_rows, how="vertical_relaxed").rename({"junction_id": "junction_id_local"})

    merged = junc_correct.join(meta_df, on=["organism", "junction_id_local"], how="inner")
    return merged


all_merged = {}
for variant, (test_dir, ckpt_path) in VARIANTS.items():
    print(f"\n### Przetwarzanie: {variant} ###")
    merged = evaluate_variant(test_dir, ckpt_path)
    all_merged[variant] = merged
    print(f"  n={merged.height}, ogolna trafnosc={merged['junction_correct'].mean()*100:.2f}%")

print("\n" + "=" * 90)
print("TABELA: trafnosc wg repeat_len, wszystkie 4 warianty")
print("=" * 90)
header = f"{'repeat_len':<12}" + "".join(f"{v:>20}" for v in VARIANTS)
print(header)
for lo, hi in REPEAT_BINS:
    label = f"{lo}" if lo == hi else (f">={lo}" if hi >= 10**9 else f"{lo}-{hi}")
    row = f"{label:<12}"
    for variant in VARIANTS:
        sub = all_merged[variant].filter((pl.col("repeat_len") >= lo) & (pl.col("repeat_len") <= hi))
        if sub.height == 0:
            row += f"{'--':>20}"
        else:
            acc = sub["junction_correct"].mean()
            row += f"{acc*100:>15.2f}%(n={sub.height:<3})" if sub.height < 1000 else f"{acc*100:>13.2f}%(n={sub.height})"
    print(row)

print("\n" + "=" * 90)
print("TABELA: trafnosc wg cov_ratio_max, wszystkie 4 warianty")
print("=" * 90)
print(header)
for lo, hi in COV_BINS:
    row = f"[{lo:.2f}-{hi:.2f})".ljust(12)
    for variant in VARIANTS:
        sub = all_merged[variant].filter((pl.col("cov_ratio_max") >= lo) & (pl.col("cov_ratio_max") < hi))
        if sub.height == 0:
            row += f"{'--':>20}"
        else:
            acc = sub["junction_correct"].mean()
            row += f"{acc*100:>15.2f}%(n={sub.height:<3})" if sub.height < 1000 else f"{acc*100:>13.2f}%(n={sub.height})"
    print(row)
