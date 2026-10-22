#!/usr/bin/env python3
"""
Sprawdza co odroznia najlepsze organizmy (GCF_041005995.1, GCF_041316455.1)
od reszty w group_a_farthest: statystyki repeat_len/cov_ratio_max z CSV,
oraz odleglosc Mash do najblizszego sasiada w zbiorze treningowym (90 org.).
"""
from pathlib import Path
import polars as pl

BASE = Path("/scratch/fpietrew/dna-assembler")
TEST_DIR = BASE / "data/training/group_a/farthest_test"
TRAIN_LIST = BASE / "data/genomes_ecoli/output/train_90_for_farthest_split.txt"
DIST_MATRIX = BASE / "data/genomes_ecoli/distance_matrix_triangle.txt"

# Znana srednia celnosc per organizm (z collect_repeats.py, 3 seedy, farthest)
KNOWN_ACCURACY = {
    "GCF_000618905.3": 65.92,
    "GCF_017639185.1": 68.97,
    "GCF_037094575.1": 69.95,
    "GCF_053102695.1": 70.12,
    "GCF_040500455.1": 70.47,
    "GCF_002587005.1": 70.93,
    "GCF_001886895.1": 72.27,
    "GCF_022494785.1": 73.24,
    "GCF_041316455.1": 79.37,
    "GCF_041005995.1": 80.60,
}

# --- 1. Statystyki repeat_len / cov_ratio_max per organizm z CSV ---
print("=== Statystyki danych per organizm (z CSV testowych) ===")
csv_stats = {}
for csv_file in sorted(TEST_DIR.glob("*.csv")):
    acc = csv_file.stem.replace("_training", "")
    df = pl.read_csv(csv_file, infer_schema_length=0)
    df = df.filter(pl.col("gt_ambiguous").str.to_lowercase() == "false")
    df = df.with_columns([
        pl.col("repeat_len").cast(pl.Int64, strict=False),
        pl.col("cov_ratio_max").cast(pl.Float64, strict=False),
        pl.col("context_seq").fill_null("").str.len_chars().alias("context_len_actual"),
    ])
    n = df.height
    pct_repeat0 = (df["repeat_len"] == 0).mean() * 100
    mean_repeat = df["repeat_len"].mean()
    mean_cov_ratio = df["cov_ratio_max"].mean()
    pct_cov_low = (df["cov_ratio_max"] < 0.55).mean() * 100
    mean_ctx_len = df["context_len_actual"].mean()
    csv_stats[acc] = {
        "n": n, "pct_repeat0": pct_repeat0, "mean_repeat": mean_repeat,
        "mean_cov_ratio": mean_cov_ratio, "pct_cov_low": pct_cov_low,
        "mean_ctx_len": mean_ctx_len,
    }

# --- 2. Odleglosc Mash do najblizszego sasiada w train_90 ---
print("=== Wczytywanie macierzy odleglosci Mash ===")
train_accs = set(TRAIN_LIST.read_text().split())

def parse_mash_triangle(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    n = int(lines[0].strip())
    names = []
    mat = [[0.0] * n for _ in range(n)]
    for i, line in enumerate(lines[1:1 + n]):
        parts = line.split("\t")
        # fix: mash embeduje ORYGINALNA sciezke wejsciowa (fasta/ACC.fna), nie
        # prefix -o (sketches/ACC) - trzeba wziac .stem, nie .name
        name = Path(parts[0]).stem
        names.append(name)
        dists = [float(x) for x in parts[1:]]
        for j, d in enumerate(dists):
            mat[i][j] = d
            mat[j][i] = d
    matrix = {a: {} for a in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix[a][b] = mat[i][j]
    return matrix

matrix = parse_mash_triangle(DIST_MATRIX)

min_dist = {}
for acc in KNOWN_ACCURACY:
    if acc not in matrix:
        print(f"  UWAGA: {acc} nie znaleziono w macierzy odleglosci")
        continue
    dists_to_train = [matrix[acc][t] for t in train_accs if t in matrix[acc]]
    min_dist[acc] = min(dists_to_train) if dists_to_train else None

# --- 3. Polaczona tabela, posortowana po celnosci ---
print("\n" + "=" * 100)
print(f"{'organizm':<20} {'accuracy':>9} {'n':>6} {'%rep=0':>8} {'mean_rep':>9} {'mean_cov':>9} {'%cov<0.55':>10} {'min_dist_train':>15}")
print("-" * 100)
for acc, accuracy in sorted(KNOWN_ACCURACY.items(), key=lambda x: x[1]):
    s = csv_stats.get(acc, {})
    md = min_dist.get(acc)
    print(f"{acc:<20} {accuracy:>8.2f}% {s.get('n', '?'):>6} "
          f"{s.get('pct_repeat0', 0):>7.1f}% {s.get('mean_repeat', 0):>9.2f} "
          f"{s.get('mean_cov_ratio', 0):>9.3f} {s.get('pct_cov_low', 0):>9.1f}% "
          f"{md if md is not None else '?':>15}")

# --- 4. Korelacje (proste, Pearson) ---
print("\n=== Korelacje z accuracy (Pearson r) ===")
import numpy as np
accs_sorted = sorted(KNOWN_ACCURACY.keys())
acc_vals = np.array([KNOWN_ACCURACY[a] for a in accs_sorted])

def corr_with(metric_name, values_dict):
    vals = np.array([values_dict.get(a, np.nan) for a in accs_sorted])
    mask = ~np.isnan(vals)
    if mask.sum() < 3:
        print(f"  {metric_name}: za malo danych")
        return
    r = np.corrcoef(acc_vals[mask], vals[mask])[0, 1]
    print(f"  {metric_name}: r = {r:.3f}  (n={mask.sum()})")

corr_with("pct_repeat0",     {a: csv_stats.get(a, {}).get("pct_repeat0") for a in accs_sorted})
corr_with("mean_repeat_len", {a: csv_stats.get(a, {}).get("mean_repeat") for a in accs_sorted})
corr_with("mean_cov_ratio",  {a: csv_stats.get(a, {}).get("mean_cov_ratio") for a in accs_sorted})
corr_with("pct_cov_low",     {a: csv_stats.get(a, {}).get("pct_cov_low") for a in accs_sorted})
corr_with("n_junctions",     {a: csv_stats.get(a, {}).get("n") for a in accs_sorted})
corr_with("mean_ctx_len",    {a: csv_stats.get(a, {}).get("mean_ctx_len") for a in accs_sorted})
corr_with("min_dist_train",  min_dist)
