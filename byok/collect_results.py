#!/usr/bin/env python3
"""Read what the agent produced in a BYOK session, validate it, score it with the hidden
evaluators and store it so the dashboard shows the model.

Usage (from the repository root):
    python byok/collect_results.py ~/elb-byok-sessions/qwen-run1 --model-label "strata/qwen-3.8"
    python byok/collect_results.py SESSION --model-label "strata/qwen-3.8" --dry-run  # check
    python byok/collect_results.py SESSION --model-label "strata/qwen-3.8" --serve     # + dashboard

Options: --provider-label ollama, --local/--remote, --copilot-version X, --session-minutes N,
--only-attempted, --report reports/byok.html, --force. Equivalent to `bench byok collect ...`.
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
    app(["byok", "collect", *sys.argv[1:]])
