"""Run provenance: which code and which config produced a training run.

Every training run records the git commit, whether the tracked files had
uncommitted changes, and the full CLI config (REQ031). Collecting this must
never break training: without git (e.g. inside the Docker image, where
``.git/`` is excluded) the commit falls back to the ``GIT_COMMIT``
environment variable, then to ``"unknown"``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UNKNOWN_COMMIT = "unknown"
_GIT_TIMEOUT_S = 10
_DEFAULT_REPO_DIR = Path(__file__).resolve().parent


def _run_git(args: list[str], repo_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command in ``repo_dir`` without raising on a non-zero exit.

    Args:
        args: Arguments after ``git``.
        repo_dir: Working directory for the command.

    Returns:
        The completed process.
    """
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        check=False,
    )


def get_git_info(repo_dir: Path | None = None) -> tuple[str, bool | None]:
    """Return the current commit hash and whether tracked files are modified.

    The commit comes from ``git rev-parse HEAD``. The dirty flag comes from
    ``git diff --quiet HEAD``, which looks only at tracked files, so untracked
    files (venvs, SLURM logs) do not mark the run as dirty.

    When git is missing or ``repo_dir`` is not inside a repository, the
    commit is taken from the ``GIT_COMMIT`` environment variable, or
    ``"unknown"`` when that is unset, and the dirty flag is ``None``.

    Args:
        repo_dir: Directory inside the repository. Defaults to the directory
            of this module, so the result describes the code being run, not
            the caller's working directory.

    Returns:
        Tuple ``(commit, dirty)``. ``dirty`` is ``None`` when unknown.
    """
    repo_dir = repo_dir or _DEFAULT_REPO_DIR
    try:
        head = _run_git(["rev-parse", "HEAD"], repo_dir)
        if head.returncode == 0 and head.stdout.strip():
            diff = _run_git(["diff", "--quiet", "HEAD"], repo_dir)
            dirty = {0: False, 1: True}.get(diff.returncode)
            return head.stdout.strip(), dirty
        logger.warning("git rev-parse HEAD failed: %s", head.stderr.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git not available (%s)", exc)

    commit = os.environ.get("GIT_COMMIT") or UNKNOWN_COMMIT
    logger.warning("Using fallback git commit: %s", commit)
    return commit, None


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Turn parsed CLI arguments into a JSON-serialisable config dict.

    All arguments are kept. ``no_cov`` is added explicitly, because the CLI
    flag ``--no_cov`` is stored by argparse as ``use_cov``.

    Args:
        args: Namespace from ``main.py``'s argument parser.

    Returns:
        Dict of argument name to value, with non-primitive values as strings.
    """
    config: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if value is None or isinstance(value, (bool, int, float, str)):
            config[key] = value
        else:
            config[key] = str(value)
    if "use_cov" in config:
        config["no_cov"] = not config["use_cov"]
    return config


def collect_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Collect git commit, dirty flag and config for one training run.

    Args:
        args: Namespace from ``main.py``'s argument parser.

    Returns:
        Dict with keys ``git_commit``, ``git_dirty`` and ``config``.
    """
    commit, dirty = get_git_info()
    return {"git_commit": commit, "git_dirty": dirty, "config": run_config(args)}
