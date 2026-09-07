"""Pytest configuration for the ml/ test suite.

The project runs its modules with ml/ as the working directory (see CLAUDE.md
key commands), so first-party modules are imported as top-level packages
(``data``, ``models``, ``training``).  Put the ml/ root on ``sys.path`` here so
tests can do the same regardless of where pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))
