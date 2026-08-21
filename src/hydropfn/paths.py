"""Repository paths, resolved from this file rather than the caller's cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
FIGS = ROOT / "figs"
DATA = ROOT / "data"
for _p in (LOGS, FIGS):
    _p.mkdir(exist_ok=True)
