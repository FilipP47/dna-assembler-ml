"""Out-of-fold analysis: when does the max-coverage heuristic beat the model, and vice versa.

Trains one model per fold (5-fold CV), predicts on the held-out fold, then:
  1. bins every junction by cov_ratio_max (heuristic confidence) - model vs heuristic per bin,
  2. reports the agreement breakdown (both/only-model/only-heuristic/neither),
  3. computes the oracle upper bound (a perfect router picking the right method per junction),
  4. tests whether a learnable router (logistic regression on per-junction features, incl. the
     model's own confidence margin) can approach that oracle - i.e. whether the disagreements
     are predictable or just noise.

Uses a pure-sequence model (use_cov=False) to isolate sequence signal.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import JunctionDataset, collate_junction_batch  # noqa: E402
from data.loader import kfold_splits, load_training_csv  # noqa: E402
from models.mv1 import Mv1JunctionModel  # noqa: E402
from training.trainer import Trainer  # noqa: E402

DATA = (
    Path(__file__).resolve().parent.parent.parent
    / "data/training/cerevisiae_k21_training.csv"
)
OUT_CSV = Path(__file__).resolve().parent / "oof_predictions.csv"
BASES = ["A", "C", "G", "T"]
N_SPLITS, SEED = 5, 42
CTX, USE_COV = 150, False


def train_and_predict(df_train, df_val, seed: int = 42) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cov_dim = 6 if USE_COV else 0
    tr = JunctionDataset(
        df_train, context_len=CTX, branch_len=100, augment=True, use_cov_feat=USE_COV
    )
    vl = JunctionDataset(
        df_val, context_len=CTX, branch_len=100, augment=False, use_cov_feat=USE_COV
    )
    trl = DataLoader(
        tr,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_junction_batch,
        num_workers=0,
    )
    vll = DataLoader(
        vl,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_junction_batch,
        num_workers=0,
    )

    model = Mv1JunctionModel(
        context_embed_dim=64,
        branch_embed_dim=64,
        cov_feat_dim=cov_dim,
        hidden_dim=64,
        dropout=0.3,
    )
    tmp = tempfile.mkdtemp()
    trainer = Trainer(
        model,
        checkpoint_dir=Path(tmp),
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=100,
        early_stop=15,
    )
    trainer.fit(trl, vll)
    ckpt = torch.load(
        Path(tmp) / "mv1_siamese_cnn_best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Per junction: list of (pair_correct, pair_margin). Margin = |score_a - score_b| (model confidence).
    junc_pairs: dict[int, list[tuple[bool, float]]] = {}
    with torch.no_grad():
        for b in vll:
            sa, sb = model(b["context"], b["branch_a"], b["branch_b"], b["cov_feat"])
            preds = torch.stack([sa, sb], dim=1).argmax(dim=1)
            margins = (sa - sb).abs()
            for jid_t, p, lbl, mg in zip(
                b["junction_id"], preds.tolist(), b["label"].tolist(), margins.tolist()
            ):
                junc_pairs.setdefault(jid_t.item(), []).append((p == lbl, mg))

    out = []
    cols = [
        "junction_id",
        "ground_truth",
        "cov_ratio_max",
        "out_degree",
        "context_len",
        "total_reads",
        "repeat_len",
    ] + [f"branch_{x}_cov" for x in BASES]
    for row in df_val.select(cols).iter_rows(named=True):
        jid = int(row["junction_id"])
        cov = {x: (row[f"branch_{x}_cov"] or 0.0) for x in BASES}
        if not any(v > 0 for v in cov.values()):
            continue
        base_ok = int(max(cov, key=cov.get) == row["ground_truth"])
        pairs = junc_pairs.get(jid, [])
        model_ok = int(all(c for c, _ in pairs)) if pairs else 0
        # Junction confidence = weakest pair's margin (the bottleneck for "all pairs correct").
        margin_min = min((m for _, m in pairs), default=0.0)
        margin_mean = float(np.mean([m for _, m in pairs])) if pairs else 0.0
        out.append(
            {
                "junction_id": jid,
                "cov_ratio": float(row["cov_ratio_max"] or 0.0),
                "out_degree": int(row["out_degree"]),
                "context_len": int(row["context_len"] or 0),
                "total_reads": int(row["total_reads"] or 0),
                "repeat_len": int(row["repeat_len"] or 0),
                "margin_min": margin_min,
                "margin_mean": margin_mean,
                "model_ok": model_ok,
                "base_ok": base_ok,
            }
        )
    return out


def print_bins(all_rows: list[dict]) -> None:
    edges = [0.0, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]
    lbls = ["<0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.80", ">=0.80"]

    def which(r: float) -> str:
        for lo, hi, lb in zip(edges[:-1], edges[1:], lbls):
            if lo <= r < hi:
                return lb
        return lbls[-1]

    bins = {lb: {"n": 0, "m": 0, "b": 0} for lb in lbls}
    for r in all_rows:
        lb = which(r["cov_ratio"])
        bins[lb]["n"] += 1
        bins[lb]["m"] += r["model_ok"]
        bins[lb]["b"] += r["base_ok"]

    print(f"\n{'cov_ratio':>12}  {'n':>5}  {'Model':>7}  {'Heur':>7}  {'delta':>8}")
    print("-" * 48)
    for lb in lbls:
        d = bins[lb]
        if d["n"] == 0:
            continue
        m = d["m"] / d["n"] * 100
        b = d["b"] / d["n"] * 100
        print(f"{lb:>12}  {d['n']:>5}  {m:>6.1f}%  {b:>6.1f}%  {m - b:>+7.1f}pp")

    tm = sum(r["model_ok"] for r in all_rows) / len(all_rows) * 100
    tb = sum(r["base_ok"] for r in all_rows) / len(all_rows) * 100
    print("-" * 48)
    print(
        f"{'OGOLEM':>12}  {len(all_rows):>5}  {tm:>6.1f}%  {tb:>6.1f}%  {tm - tb:>+7.1f}pp"
    )


def print_repeat_bins(all_rows: list[dict]) -> None:
    edges = [0, 1, 3, 6, 11, 21, 10_000]
    lbls = ["0", "1-2", "3-5", "6-10", "11-20", ">=21"]

    def which(rl: int) -> str:
        for lo, hi, lb in zip(edges[:-1], edges[1:], lbls):
            if lo <= rl < hi:
                return lb
        return lbls[-1]

    bins = {lb: {"n": 0, "m": 0, "b": 0, "cov": 0.0} for lb in lbls}
    for r in all_rows:
        lb = which(r["repeat_len"])
        bins[lb]["n"] += 1
        bins[lb]["m"] += r["model_ok"]
        bins[lb]["b"] += r["base_ok"]
        bins[lb]["cov"] += r["cov_ratio"]

    print(
        f"\n{'repeat_len':>12}  {'n':>5}  {'Model':>7}  {'Heur':>7}  {'delta':>8}  {'avg cov_ratio':>13}"
    )
    print("-" * 62)
    for lb in lbls:
        d = bins[lb]
        if d["n"] == 0:
            continue
        m = d["m"] / d["n"] * 100
        b = d["b"] / d["n"] * 100
        c = d["cov"] / d["n"]
        print(
            f"{lb:>12}  {d['n']:>5}  {m:>6.1f}%  {b:>6.1f}%  {m - b:>+7.1f}pp  {c:>13.3f}"
        )


def print_agreement_and_router(all_rows: list[dict]) -> None:
    n = len(all_rows)
    both = sum(1 for r in all_rows if r["model_ok"] and r["base_ok"])
    only_m = sum(1 for r in all_rows if r["model_ok"] and not r["base_ok"])
    only_b = sum(1 for r in all_rows if not r["model_ok"] and r["base_ok"])
    neither = sum(1 for r in all_rows if not r["model_ok"] and not r["base_ok"])
    print("\nZgodnosc model vs heurystyka:")
    print(f"  oba dobrze:        {both:>5} ({both / n * 100:.1f}%)")
    print(f"  tylko model:       {only_m:>5} ({only_m / n * 100:.1f}%)")
    print(f"  tylko heurystyka:  {only_b:>5} ({only_b / n * 100:.1f}%)")
    print(f"  oba zle:           {neither:>5} ({neither / n * 100:.1f}%)")

    oracle = (both + only_m + only_b) / n * 100
    base_acc = sum(r["base_ok"] for r in all_rows) / n * 100
    model_acc = sum(r["model_ok"] for r in all_rows) / n * 100
    print("\nGorne granice:")
    print(f"  zawsze heurystyka:      {base_acc:.1f}%")
    print(f"  zawsze model:           {model_acc:.1f}%")
    print(f"  ORACLE (idealny router):{oracle:>6.1f}%")

    # --- Learnable router: can features predict which method is right on disagreements? ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    feat_names = [
        "cov_ratio",
        "out_degree",
        "context_len",
        "total_reads",
        "margin_min",
        "margin_mean",
    ]

    def feats(r: dict) -> list[float]:
        return [
            r["cov_ratio"],
            float(r["out_degree"]),
            float(r["context_len"]),
            float(np.log1p(r["total_reads"])),
            r["margin_min"],
            r["margin_mean"],
        ]

    # (a) Predictability of disagreements: train only on rows where methods disagree,
    #     target = 1 if only_model_right, 0 if only_heuristic_right. AUC ~ 0.5 => unpredictable.
    dis = [r for r in all_rows if r["model_ok"] != r["base_ok"]]
    Xd = np.array([feats(r) for r in dis])
    yd = np.array([1 if r["model_ok"] else 0 for r in dis])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_prob = np.zeros(len(dis))
    for tr_i, va_i in skf.split(Xd, yd):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xd[tr_i], yd[tr_i])
        oof_prob[va_i] = clf.predict_proba(Xd[va_i])[:, 1]
    auc = roc_auc_score(yd, oof_prob)
    print("\nCzy niezgodnosci sa przewidywalne? (klasyfikator na przypadkach spornych)")
    print(f"  n spornych: {len(dis)}  |  target: 1=tylko model, 0=tylko heurystyka")
    print(f"  AUC (5-fold): {auc:.3f}   (0.5 = nieprzewidywalne, szum)")

    # (b) Full router: predict per junction whether to trust the model; realise the accuracy.
    #     Target y=1 when picking model is at least as good as heuristic (model_ok >= base_ok).
    Xa = np.array([feats(r) for r in all_rows])
    ya = np.array([1 if r["model_ok"] >= r["base_ok"] else 0 for r in all_rows])
    picked_correct = np.zeros(len(all_rows))
    for tr_i, va_i in skf.split(Xa, ya):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xa[tr_i], ya[tr_i])
        route_model = clf.predict(Xa[va_i]) == 1
        for local_idx, j in enumerate(va_i):
            r = all_rows[j]
            picked_correct[j] = (
                r["model_ok"] if route_model[local_idx] else r["base_ok"]
            )
    router_acc = picked_correct.mean() * 100
    print("\nWyuczony router (regresja logistyczna, 5-fold):")
    print(
        f"  dokladnosc: {router_acc:.1f}%  (heurystyka {base_acc:.1f}%, oracle {oracle:.1f}%)"
    )
    coef = (
        LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xa, ya).coef_[0]
    )
    print("  wagi cech (znormalizowane wejscie surowe):")
    for name, w in sorted(zip(feat_names, coef), key=lambda t: -abs(t[1])):
        print(f"    {name:>12}: {w:+.3f}")


def main() -> None:
    labeled, stats = load_training_csv(DATA, only_labeled=True)
    print(f"Labeled: {labeled.height}, k={stats.k_value}", flush=True)
    print(
        f"Out-of-fold, {N_SPLITS}-fold, ctx={CTX}, use_cov={USE_COV}, seed={SEED}",
        flush=True,
    )

    all_rows: list[dict] = []
    for i, (dtr, dvl) in enumerate(kfold_splits(labeled, n_splits=N_SPLITS, seed=SEED)):
        print(f"  fold {i + 1}/{N_SPLITS} ...", flush=True)
        all_rows += train_and_predict(dtr, dvl, seed=SEED)

    print(f"\nTotal junctions with OOF preds: {len(all_rows)}", flush=True)
    pl.DataFrame(all_rows).write_csv(OUT_CSV)
    print(f"Per-junction predictions -> {OUT_CSV}")

    print_bins(all_rows)
    print_repeat_bins(all_rows)
    print_agreement_and_router(all_rows)


if __name__ == "__main__":
    main()
