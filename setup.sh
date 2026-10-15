#!/usr/bin/env bash
# Prepare a benchmark run for a Copilot BYOK model:
#
#   ./setup.sh            # 12 tasks      ./setup.sh smoke   # 4 quick tasks
#
# Installs everything (first run only), creates a new session folder, opens it in VS Code with
# the task commands auto-approved, and copies the prompt to the clipboard.
# Then in VS Code: Copilot Chat -> Agent -> pick your model -> paste -> Enter.
# When the agent is done: ./finish.sh
set -euo pipefail
# shellcheck source=byok/_setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/byok/_setup.sh"

"$REPO/.venv/bin/bench" byok start \
  --profile "${1:-standard}" \
  --python-env "$RUNTIME_VENV" \
  --out "$SESSIONS_DIR/session-$(date +%Y%m%d-%H%M%S)-$RANDOM" \
  --auto-approve
