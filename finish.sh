#!/usr/bin/env bash
# Score the session made by ./setup.sh and open the dashboard (http://127.0.0.1:8765).
#   ./finish.sh                          # model detected from the model server
#   ./finish.sh --model-label NAME       # or name it yourself
#   ./finish.sh --no-serve               # do not start the dashboard
set -euo pipefail
ELB_SETUP_MODE=check
# shellcheck source=byok/_setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/byok/_setup.sh"

serve=1
for arg in "$@"; do [[ "$arg" == "--no-serve" ]] && serve=0; done
current="$REPO/artifacts/byok_current_session.json"
if [[ "$USE_CONTAINER" == "1" ]] && grep -q '"container_image": "' "$current" 2>/dev/null; then
  # Score in a clean copy of the agent's environment (same image, nothing the agent installed).
  tty=()
  [[ -t 0 && -t 1 ]] && tty=(-t)
  docker run --rm -i "${tty[@]}" --network host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e BENCH_BASE_DIR="$REPO" -e PYTHONPATH="$REPO/src" \
    -v "$REPO:$REPO" -v "$SESSIONS_DIR:$SESSIONS_DIR" -w "$REPO" "$EVAL_IMAGE" \
    python -c 'from engineering_llm_bench.cli.main import main; main()' \
    byok finish --no-serve "$@"
  if [[ "$serve" == "1" ]]; then
    "$REPO/.venv/bin/bench" dashboard
  fi
else
  "$REPO/.venv/bin/bench" byok finish "$@"
fi
