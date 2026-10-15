#!/usr/bin/env bash
# Score the session made by ./setup.sh and open the dashboard (http://127.0.0.1:8765).
#   ./finish.sh                          # model detected from the model server
#   ./finish.sh --model-label NAME       # or name it yourself
set -euo pipefail
ELB_SETUP_MODE=check
# shellcheck source=byok/_setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/byok/_setup.sh"

"$REPO/.venv/bin/bench" byok finish "$@"
