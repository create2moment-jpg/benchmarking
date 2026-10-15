# shellcheck shell=bash
# Shared, idempotent setup for setup.sh / finish.sh (sourced, not run directly).
#
# Always: .venv in the repository (the benchmark itself; dashboard).
# With Docker (default when it is available; ELB_MODE=host to opt out):
#   images elb-tasks:<hash> (the agent's Dev Container) and elb-eval:<hash> (scoring), built
#   from docker/ once and rebuilt only when docker/ changes.
# Without Docker: ~/elb-byok-sessions/.runtime-venv with the task runtimes for the agent's
#   terminal, plus the evaluator runtimes in .venv.
# WITH_TORCH=0 skips PyTorch (the PyTorch task is then reported as skipped).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSIONS_DIR="${ELB_SESSIONS_DIR:-$HOME/elb-byok-sessions}"
RUNTIME_VENV="$SESSIONS_DIR/.runtime-venv"
WITH_TORCH="${WITH_TORCH:-1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" # CPU wheels: small, enough
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

hash_of() { (sha256sum 2>/dev/null || shasum -a 256) | cut -c1-12; }

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
  if [[ "$WITH_TORCH" == "1" && "${ELB_NO_TORCH_HERE:-0}" != "1" ]]; then
    "$dir/bin/python" -m pip install --quiet torch --index-url "$TORCH_INDEX" ||
      "$dir/bin/python" -m pip install --quiet torch ||
      warn "PyTorch could not be installed: the PyTorch task will be skipped"
  fi
  echo "$stamp" >"$dir/.elb-stamp"
}

USE_CONTAINER=0
if [[ "${ELB_MODE:-auto}" != "host" ]] && command -v docker >/dev/null 2>&1 &&
  docker info >/dev/null 2>&1; then
  USE_CONTAINER=1
elif [[ "${ELB_MODE:-auto}" == "container" ]]; then
  warn "ELB_MODE=container but Docker is not available (is the daemon running? are you in the docker group?)"
  exit 2
fi

STAMP="$(cat "$REPO/pyproject.toml" "$REPO/byok/_setup.sh" | hash_of)-torch$WITH_TORCH-c$USE_CONTAINER"

if [[ "$USE_CONTAINER" == "1" ]]; then
  # Benchmark only (scoring happens in the eval image).
  ELB_NO_TORCH_HERE=1 ensure_venv "$REPO/.venv" "$STAMP" -e "$REPO"
  IMAGE_TAG="$(cat "$REPO"/docker/* | hash_of)-torch$WITH_TORCH"
  TASKS_IMAGE="elb-tasks:$IMAGE_TAG"
  EVAL_IMAGE="elb-eval:$IMAGE_TAG"
  for target in tasks eval; do
    image="elb-$target:$IMAGE_TAG"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      say "building Docker image $image (first run takes several minutes)"
      docker build --quiet --target "$target" -t "$image" \
        --build-arg WITH_TORCH="$WITH_TORCH" --build-arg TORCH_INDEX="$TORCH_INDEX" \
        "$REPO/docker" >/dev/null
    fi
  done
  if [[ "${ELB_SETUP_MODE:-install}" != "check" ]] && command -v code >/dev/null 2>&1 &&
    ! code --list-extensions 2>/dev/null | grep -qi '^ms-vscode-remote.remote-containers$'; then
    say "installing the VS Code Dev Containers extension"
    code --install-extension ms-vscode-remote.remote-containers >/dev/null ||
      warn "install the 'Dev Containers' extension in VS Code yourself"
  fi
else
  warn "Docker not available: the agent will run on this machine (see instructions/README.md)"
  ensure_venv "$REPO/.venv" "$STAMP" -e "$REPO[dev]"
  mkdir -p "$SESSIONS_DIR"
  ensure_venv "$RUNTIME_VENV" "$STAMP" -r "$REPO/docker/requirements-tasks.txt"
  for tool in g++ cmake; do
    command -v "$tool" >/dev/null 2>&1 ||
      warn "$tool not found: C++/CMake tasks will fail or be skipped (sudo apt install g++ cmake)"
  done
fi
mkdir -p "$SESSIONS_DIR"
