#!/usr/bin/env bash
# Prepare a benchmark run for a Copilot BYOK model:
#
#   ./setup.sh            # 12 tasks      ./setup.sh smoke   # 4 quick tasks
#
# Installs everything (first run only), creates a new session folder and opens it in VS Code -
# inside a Docker Dev Container when Docker is available (all tools installed, commands
# auto-approved there, only the session folder visible) - and copies the prompt to the clipboard.
# Then in VS Code: Copilot Chat -> Agent -> pick your model -> paste -> Enter.
# When the agent is done: ./finish.sh
set -euo pipefail
# shellcheck source=byok/_setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/byok/_setup.sh"

if [[ "$USE_CONTAINER" == "1" ]]; then
  mode=(--container "$TASKS_IMAGE")
else
  mode=(--python-env "$RUNTIME_VENV" --auto-approve)
fi
"$REPO/.venv/bin/bench" byok start \
  --profile "${1:-standard}" \
  --out "$SESSIONS_DIR/session-$(date +%Y%m%d-%H%M%S)-$RANDOM" \
  "${mode[@]}"
