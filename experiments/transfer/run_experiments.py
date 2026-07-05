"""Orchestrate the full NxN transfer learning experiment.

Usage (from repo root):
    python experiments/transfer/run_experiments.py [--only-solo] [--only-eval]

Steps:
  1. Solo training: train each organism independently.
  2. Fine-tune: for every pair (src, tgt), fine-tune model_src on data_tgt.
  3. Evaluate: evaluate every model checkpoint on every organism CSV.

Progress is tracked via sentinel files (.done) so the script can be resumed
safely after interruption.

Results land in:
  experiments/transfer/results/solo/<name>/
  experiments/transfer/results/finetune/<src>_to_<tgt>/
  experiments/transfer/results/specialists/<cluster>/   (run manually after NB 04)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

LOG_FILE = Path(__file__).parent / "results" / "run_experiments.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), _file_handler],
)
logger = logging.getLogger(__name__)
logger.info("Log file: %s", LOG_FILE)

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR    = REPO_ROOT / "ml"
RESULTS   = Path(__file__).parent / "results"
CONFIG    = Path(__file__).parent / "config" / "organisms.yaml"


def load_config() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=check)
    return result


def is_done(sentinel: Path) -> bool:
    return sentinel.exists()


def mark_done(sentinel: Path) -> None:
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()


def solo_ckpt_name(name: str) -> str:
    return f"mv1_siamese_cnn_{name}_best.pt"


def finetune_ckpt_name(src: str, tgt: str) -> str:
    return f"mv1_siamese_cnn_{src}_to_{tgt}_best.pt"


def train_solo(name: str, cfg: dict, tr: dict) -> Path:
    """Train a model from scratch on a single organism. Returns checkpoint dir."""
    ckpt_dir = RESULTS / "solo" / name
    sentinel  = ckpt_dir / ".done"

    if is_done(sentinel):
        logger.info("SKIP solo %s (already done)", name)
        return ckpt_dir

    csv_path = REPO_ROOT / cfg["csv"]
    if not csv_path.exists():
        logger.warning("CSV not found for %s: %s - skipping", name, csv_path)
        return ckpt_dir

    run_cmd([
        sys.executable, str(ML_DIR / "main.py"), "train",
        "--data_path",  str(csv_path),
        "--ckpt_dir",   str(ckpt_dir),
        "--run_name",   name,
        "--epochs",     str(tr["epochs"]),
        "--batch_size", str(tr["batch_size"]),
        "--lr",         str(tr["lr_solo"]),
        "--early_stop", str(tr["early_stop"]),
        "--seed",       str(tr["seed"]),
        "--val_split",  str(tr["val_split"]),
        "--embed_dim",  str(tr["embed_dim"]),
        "--hidden_dim", str(tr["hidden_dim"]),
        *(["--augment"] if tr["augment"] else []),
    ])

    mark_done(sentinel)
    logger.info("Solo training done: %s", name)
    return ckpt_dir


def train_finetune(src: str, tgt: str, src_cfg: dict, tgt_cfg: dict, tr: dict) -> Path:
    """Fine-tune model_src on organism_tgt data. Returns checkpoint dir."""
    run_name  = f"{src}_to_{tgt}"
    ckpt_dir  = RESULTS / "finetune" / run_name
    sentinel   = ckpt_dir / ".done"

    if is_done(sentinel):
        logger.info("SKIP finetune %s (already done)", run_name)
        return ckpt_dir

    src_ckpt = RESULTS / "solo" / src / solo_ckpt_name(src)
    if not src_ckpt.exists():
        logger.warning("Solo checkpoint missing for %s - skipping finetune %s", src, run_name)
        return ckpt_dir

    tgt_csv = REPO_ROOT / tgt_cfg["csv"]
    if not tgt_csv.exists():
        logger.warning("CSV not found for %s: %s - skipping finetune %s", tgt, tgt_csv, run_name)
        return ckpt_dir

    run_cmd([
        sys.executable, str(ML_DIR / "main.py"), "train",
        "--data_path",       str(tgt_csv),
        "--ckpt_dir",        str(ckpt_dir),
        "--init_checkpoint", str(src_ckpt),
        "--run_name",        f"{src}_to_{tgt}",
        "--epochs",          str(tr["epochs"]),
        "--batch_size",      str(tr["batch_size"]),
        "--lr",              str(tr["lr_finetune"]),
        "--early_stop",      str(tr["early_stop"]),
        "--seed",            str(tr["seed"]),
        "--val_split",       str(tr["val_split"]),
        "--embed_dim",       str(tr["embed_dim"]),
        "--hidden_dim",      str(tr["hidden_dim"]),
        *(["--augment"] if tr["augment"] else []),
    ])

    mark_done(sentinel)
    logger.info("Fine-tune done: %s -> %s", src, tgt)
    return ckpt_dir


def _find_checkpoint(ckpt_dir: Path) -> Path | None:
    """Find the best checkpoint in a directory (any *_best.pt file)."""
    candidates = list(ckpt_dir.glob("*_best.pt"))
    if not candidates:
        return None
    # Prefer longer, more descriptive names first (solo/finetune names are longer than fallback)
    return max(candidates, key=lambda p: len(p.name))


def evaluate_checkpoint(ckpt_dir: Path, org_name: str, org_cfg: dict) -> Path:
    """Evaluate checkpoint in ckpt_dir on org_name's CSV. Returns output JSON path."""
    ckpt_path = _find_checkpoint(ckpt_dir)
    out_json = ckpt_dir / f"eval_on_{org_name}.json"

    if out_json.exists():
        logger.info("SKIP eval %s on %s (already done)", ckpt_dir.name, org_name)
        return out_json

    if ckpt_path is None:
        logger.warning("No checkpoint found in %s - skipping eval on %s", ckpt_dir, org_name)
        return out_json

    csv_path = REPO_ROOT / org_cfg["csv"]
    if not csv_path.exists():
        logger.warning("CSV not found for %s: %s - skipping eval", org_name, csv_path)
        return out_json

    run_cmd([
        sys.executable, str(ML_DIR / "main.py"), "evaluate",
        "--model_path",  str(ckpt_path),
        "--test_csv",    str(csv_path),
        "--output_json", str(out_json),
    ])

    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-solo",  action="store_true", help="Run solo training only, skip fine-tune")
    parser.add_argument("--only-eval",  action="store_true", help="Run evaluation only (assumes training done)")
    parser.add_argument("--organisms",  nargs="+", default=None,
                        help="Subset of organisms to process (default: all)")
    args = parser.parse_args()

    config = load_config()
    organisms = config["organisms"]
    tr        = config["training"]

    if args.organisms:
        organisms = {k: v for k, v in organisms.items() if k in args.organisms}
        logger.info("Restricted to organisms: %s", list(organisms))

    names = list(organisms.keys())

    # ── 1. Solo training ──────────────────────────────────────────────────────
    if not args.only_eval:
        logger.info("=== Phase 1: Solo training (%d organisms) ===", len(names))
        for name in names:
            train_solo(name, organisms[name], tr)

    # ── 2. Pairwise fine-tuning ───────────────────────────────────────────────
    if not args.only_solo and not args.only_eval:
        n_pairs = len(names) * (len(names) - 1)
        logger.info("=== Phase 2: Pairwise fine-tuning (%d pairs) ===", n_pairs)
        for src in names:
            for tgt in names:
                if src == tgt:
                    continue
                train_finetune(src, tgt, organisms[src], organisms[tgt], tr)

    # ── 3. Evaluation of all checkpoints on all organisms ────────────────────
    logger.info("=== Phase 3: Cross-evaluation ===")

    all_ckpt_dirs: list[Path] = []
    for name in names:
        d = RESULTS / "solo" / name
        if d.exists():
            all_ckpt_dirs.append(d)

    if not args.only_solo:
        for src in names:
            for tgt in names:
                if src == tgt:
                    continue
                d = RESULTS / "finetune" / f"{src}_to_{tgt}"
                if d.exists():
                    all_ckpt_dirs.append(d)

    for ckpt_dir in all_ckpt_dirs:
        for org_name, org_cfg in organisms.items():
            evaluate_checkpoint(ckpt_dir, org_name, org_cfg)

    logger.info("=== All done. Results in %s ===", RESULTS)


if __name__ == "__main__":
    main()
