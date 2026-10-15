# shellcheck shell=bash
# Shared, idempotent setup for start.sh / finish.sh (sourced, not run directly):
#   * .venv in the repository      - the benchmark itself + evaluator runtimes
#   * ~/elb-byok-sessions/.runtime-venv - pytest, ruff, numpy, duckdb, torch for the agent's
#     terminal (kept outside the repository so the agent learns nothing about where it is)
# Re-installs only when pyproject.toml or the options change. Set WITH_TORCH=0 to skip PyTorch
# (the PyTorch task is then reported as skipped).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSIONS_DIR="${ELB_SESSIONS_DIR:-$HOME/elb-byok-sessions}"
RUNTIME_VENV="$SESSIONS_DIR/.runtime-venv"
WITH_TORCH="${WITH_TORCH:-1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"  # CPU wheels: small, enough
export BENCH_BASE_DIR="$REPO"

say() { printf '\033[1m[elb]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[elb] %s\033[0m\n' "$*" >&2; }

find_python() {
  local c
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 &&
      "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

PY="$(find_python)" || {
  warn "Python >= 3.11 not found. Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
  exit 2
}

hash_of() { (sha256sum 2>/dev/null || shasum -a 256) | cut -c1-16; }

# ensure_venv DIR STAMP PIP_ARGS...
ensure_venv() {
  local dir="$1" stamp="$2"
  shift 2
  if [[ ! -x "$dir/bin/python" ]]; then
    say "creating virtual environment $dir"
    "$PY" -m venv "$dir" || {
      warn "could not create a venv. Ubuntu/Debian: sudo apt install python3-venv (or python3.12-venv)"
      exit 2
    }
  fi
  if [[ "$(cat "$dir/.elb-stamp" 2>/dev/null || true)" == "$stamp" ]]; then
    return 0
  fi
  if [[ "${ELB_SETUP_MODE:-install}" == "check" && -f "$dir/.elb-stamp" ]]; then
    return 0 # finish.sh: an installed environment is good enough
  fi
  say "installing dependencies into $dir (first run takes a few minutes)"
  "$dir/bin/python" -m pip --version >/dev/null 2>&1 ||
    "$dir/bin/python" -m ensurepip --upgrade --default-pip >/dev/null
  "$dir/bin/python" -m pip install --quiet --upgrade pip
  "$dir/bin/python" -m pip install --quiet "$@"
  if [[ "$WITH_TORCH" == "1" ]]; then
    "$dir/bin/python" -m pip install --quiet torch --index-url "$TORCH_INDEX" ||
      "$dir/bin/python" -m pip install --quiet torch ||
      warn "PyTorch could not be installed: the PyTorch task will be skipped"
  fi
  echo "$stamp" >"$dir/.elb-stamp"
}

STAMP="$(cat "$REPO/pyproject.toml" "$REPO/byok/_setup.sh" | hash_of)-torch$WITH_TORCH"
ensure_venv "$REPO/.venv" "$STAMP" -e "$REPO[dev]"
mkdir -p "$SESSIONS_DIR"
ensure_venv "$RUNTIME_VENV" "$STAMP" pytest ruff "numpy>=1.26" "duckdb>=1.0"

for tool in g++ cmake; do
  command -v "$tool" >/dev/null 2>&1 ||
    warn "$tool not found: C++/CMake tasks will fail or be skipped (sudo apt install g++ cmake)"
done
