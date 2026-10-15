#!/usr/bin/env python3
"""Create a Copilot BYOK session folder containing starting_point.md and one folder per task.

Usage (from the repository root):
    python byok/prepare_session.py --profile standard --out ~/elb-byok-sessions/qwen-run1
    python byok/prepare_session.py --task python.fix-interval-merge,sql.latest-sensor-reading

Then open ONLY the printed session folder in VS Code and give starting_point.md to Copilot
(agent mode, your BYOK model). Equivalent to `bench byok prepare ...`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # works without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("BENCH_BASE_DIR", str(ROOT))  # configs/, tasks/, artifacts/ of this repo

from engineering_llm_bench.cli.main import app  # noqa: E402

if __name__ == "__main__":
    app(["byok", "prepare", *sys.argv[1:]])
