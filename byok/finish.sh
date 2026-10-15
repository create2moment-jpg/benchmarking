#!/usr/bin/env bash
# Score the session started by ./byok/start.sh and open the dashboard.
#
#   ./byok/finish.sh                              # model detected from the model server
#   ./byok/finish.sh --model-label "qwen3.8-next" # or name it yourself
#
# Detects which model ran (asks if it cannot tell), validates the agent's work, runs the hidden
# evaluators, moves the code into the model's private output folder, measures the model server
# and opens the dashboard at http://127.0.0.1:8765.
set -euo pipefail
ELB_SETUP_MODE=check
# shellcheck source=byok/_setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/_setup.sh"

"$REPO/.venv/bin/bench" byok finish "$@"
