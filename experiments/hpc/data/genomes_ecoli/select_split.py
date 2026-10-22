#!/usr/bin/env python3
"""
Konwertuje wyjscie `mash triangle` na pelna macierz, usuwa near-duplicate
klony, wybiera test_farthest_10 / test_random_10 + odpowiadajace train_90.
"""
import argparse
import random
from pathlib import Path
from collections import defaultdict


def parse_triangle(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    n = int(lines[0].strip())
    names = []
    matrix = [[0.0] * n for _ in range(n)]
    for i, line in enumerate(lines[1:1 + n]):
        parts = line.split("\t")
        name = Path(parts[0]).stem
        names.append(name)
        dists = [float(x) for x in parts[1:]]
        for j, d in enumerate(dists):
            matrix[i][j] = d
            matrix[j][i] = d
    dist = {a: {} for a in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            dist[a][b] = matrix[i][j]
    return names, dist


def dedup_near_identical(names, dist, threshold):
    kept, dropped = [], []
    for name in names:
        if any(dist[name][k] < threshold for k in kept):
            dropped.append(name)
        else:
            kept.append(name)
    return kept, dropped


def min_dist_to_others(name, pool, dist):
    return min(dist[name][o] for o in pool if o != name)


def load_genus_map(path):
    m = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
    return m


def genus_counts(items, genus_map):
    c = defaultdict(int)
    for i in items:
        c[genus_map.get(i, "unknown")] += 1
    return c


def fmt_counts(c):
    return ", ".join(f"{g}={n}" for g, n in sorted(c.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triangle", required=True)
    ap.add_argument("--genus-map", default=None)
    ap.add_argument("--pool-size", type=int, default=100)
    ap.add_argument("--dedup-threshold", type=float, default=0.0005)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names, dist = parse_triangle(args.triangle)
    print(f"Wczytano macierz dla {len(names)} genomow.")

    genus_map = load_genus_map(args.genus_map) if args.genus_map else {}

    kept, dropped = dedup_near_identical(names, dist, args.dedup_threshold)
    print(f"Po dedup (prog={args.dedup_threshold}): {len(kept)}, odrzucono {len(dropped)}.")
    if dropped:
        (outdir / "dropped_near_duplicates.txt").write_text("\n".join(dropped) + "\n")

    if len(kept) < args.pool_size:
        print(f"UWAGA: po dedup zostalo {len(kept)} < --pool-size {args.pool_size}. Uzywam wszystkich.")
        pool = kept
    else:
        random.seed(args.seed)
        pool = random.sample(kept, args.pool_size)

    print(f"Finalna pula: {len(pool)} genomow.")
    if genus_map:
        print(f"Sklad rodzajowy puli: {fmt_counts(genus_counts(pool, genus_map))}")

    min_dists = {n: min_dist_to_others(n, pool, dist) for n in pool}
    farthest_sorted = sorted(pool, key=lambda n: min_dists[n], reverse=True)
    test_farthest = farthest_sorted[:args.n_test]
    train_farthest = [n for n in pool if n not in set(test_farthest)]

    random.seed(args.seed + 1)
    test_random = random.sample(pool, args.n_test)
    train_random = [n for n in pool if n not in set(test_random)]

    def wl(fname, items):
        (outdir / fname).write_text("\n".join(sorted(items)) + "\n")

    wl("test_farthest_10.txt", test_farthest)
    wl("train_90_for_farthest_split.txt", train_farthest)
    wl("test_random_10.txt", test_random)
    wl("train_90_for_random_split.txt", train_random)

    median_nn = sorted(min_dists.values())[len(min_dists) // 2]
    far_min_to_train = [min(dist[t][tr] for tr in train_farthest) for t in test_farthest]

    print("\n--- Weryfikacja izolacji test_farthest_10 ---")
    print(f"Mediana odleglosci do najblizszego sasiada w puli: {median_nn:.5f}")
    print(f"Min. odleglosc test_farthest_10 -> train: {min(far_min_to_train):.5f} "
          f"(srednia: {sum(far_min_to_train)/len(far_min_to_train):.5f})")
    print("Powyzsze powinno byc wyraznie WYZSZE niz mediana.")

    if genus_map:
        print("\n--- Sklad rodzajowy testow ---")
        print(f"test_farthest_10: {fmt_counts(genus_counts(test_farthest, genus_map))}")
        print(f"test_random_10:   {fmt_counts(genus_counts(test_random, genus_map))}")

    print(f"\nPliki zapisane w {outdir}/")


if __name__ == "__main__":
    main()
