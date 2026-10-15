#!/usr/bin/env bash
# Start a Copilot BYOK benchmark session with one command.
#
#   ./byok/start.sh              # standard profile (12 tasks)
#   ./byok/start.sh smoke        # 4 quick tasks
#   ./byok/start.sh extended     # all tasks
#
# Sets up the Python environments (first run only), creates a fresh session folder, finds the
# model server Copilot is configured for, copies the prompt to the clipboard and opens the
# session in a new VS Code window. Then: pick your model in Copilot Chat (Agent mode), paste,
# press Enter. When the agent is done, run ./byok/finish.sh.
set -euo pipefail
# shellcheck source=byok/_setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/_setup.sh"

PROFILE="${1:-standard}"
shift || true
"$REPO/.venv/bin/bench" byok start \
  --profile "$PROFILE" \
  --python-env "$RUNTIME_VENV" \
  --out "$SESSIONS_DIR/session-$(date +%Y%m%d-%H%M%S)" \
  "$@"
