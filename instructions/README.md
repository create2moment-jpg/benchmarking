# Running a model through Copilot BYOK

Needs: VS Code with Copilot and **Docker** (recommended). Without Docker everything still
works, but the agent runs directly on your machine (see the end of this page).

```bash
./setup.sh          # in a terminal, inside this repository   (quick test: ./setup.sh smoke)
```

The first run builds the Docker images (several minutes) and installs the VS Code
**Dev Containers** extension. Then a new VS Code window opens a fresh session folder
**inside a container**:

1. **Copilot Chat → Agent → pick your model.**
2. **Paste (Ctrl+V) → Enter.** The prompt ([command.md](command.md)) is on the clipboard.

When the agent replies `ALL TASKS COMPLETE`:

```bash
./finish.sh         # scores the run (in a clean container) and opens the dashboard
```

For the next model, run `./setup.sh` again.

## Time and server statistics

Every run records how long the agent worked (overall and per task) and, when it can reach the
model server, server-side statistics. They appear on the dashboard's **Overview** (Latest runs
and **Time profile**) and in the terminal after `./finish.sh`.

If the model runs on another machine you reach over SSH, set it up once:

```bash
cp server.example.yaml server.yaml      # then set ssh.target, server.kind and server.url
ssh -o BatchMode=yes user@gpu-server true   # must work without a password prompt
```

`./setup.sh` then logs in during the session (read-only commands) and records: GPU
utilisation / memory / power every few seconds, the loaded model and its GPU share, the number
of LLM requests and how long each took (from the Ollama log, or your `log_command`), token
counters (vLLM / llama.cpp), and any `extra_commands` (e.g. your serving tool's status).
Without `server.yaml` only what the model's HTTP endpoint shows is recorded.

| Statistic | Source |
|---|---|
| Agent working time | first to last LLM request (server log) - else GPU activity - else file changes |
| Setup to finish | `./setup.sh` to `./finish.sh` |
| Time per task | when each task's files changed (estimate) |
| LLM requests, latency p50/p90, model busy time | server log (Ollama; vLLM via `log_command`) or vLLM/llama.cpp "requests running" |
| Tokens in/out, generation tok/s | vLLM / llama.cpp counters; vLLM log throughput |
| GPU utilisation, memory, power | `nvidia-smi` on the server, sampled every `interval_s` |

## What the container gives you

* **Everything installed**: Python 3.12, pytest, ruff, numpy, duckdb, PyTorch (CPU), g++, cmake,
  make. Same versions for every model.
* **The agent may install more** (`pip install`, `sudo apt-get install`) inside the container.
  Scoring runs in a clean copy of the image, so solutions cannot depend on what it installed.
* **No permission prompts**: every command is auto-approved, but only inside the container.
  If the folder is opened on the host instead, nothing is auto-approved.
* **Low risk**: only the session folder is mounted. The agent cannot see this repository
  (hidden tests, other models' code), your home directory or your keys. Limits: 8 GB memory,
  4 CPUs.
* The container shares the host network so Copilot can reach a model served on `localhost`
  (e.g. Ollama). It is not network-isolated.

## Troubleshooting

| What you see | Fix |
|---|---|
| VS Code does not open | Install the `code` command (in VS Code: **Shell Command: Install 'code' command in PATH**), then open the printed folder with **File > Open Folder…**. |
| Window opened, but not in the container (no "Dev Container" in the bottom-left corner) | Click **Reopen in Container** in the pop-up, or press F1 → **Dev Containers: Reopen in Container**. |
| Still asked to **Allow** commands inside the container | Your VS Code version ignores the container's auto-approve setting. Choose the auto-approve option in the approval picker under the chat input; that is low risk inside the container. |
| Copilot cannot reach the model in the container | Use a model served on the host's `localhost` (the container uses the host network), or start Ollama with `OLLAMA_HOST=0.0.0.0`. |
| Agent says there is no `starting_point.md` / `NO SESSION FOLDER OPEN` | The wrong folder is open. Use the window `./setup.sh` opened. |
| Prompt not on the clipboard | Copy it from the terminal output or from [command.md](command.md). |
| Agent stops before finishing | Send: `Continue with the next unfinished task in starting_point.md.` |
| `finish.sh` asks which model ran | Type the name you picked in Copilot, or pass `--model-label NAME`. |
| `setup.sh`: "has not been collected yet" | Run `./finish.sh` first (or delete that session folder to discard it). |
| "server statistics: cannot reach ..." | Check `ssh -o BatchMode=yes <target> true` works; the tasks still run and are scored without statistics. |
| `docker: permission denied` | Add yourself to the `docker` group (`sudo usermod -aG docker $USER`, then log in again). |

## Without Docker

If Docker is not available (or with `ELB_MODE=host ./setup.sh`), `setup.sh` installs the task
runtimes into `~/elb-byok-sessions/.runtime-venv` and opens the session on your machine. It
auto-approves only the commands the tasks need (`python`, `pytest`, `g++`, `cmake`, …); `rm`,
`git`, `pip`, `curl` and similar still ask, and the agent is told not to install anything.
Auto-approved `python` can run any code as your user, so prefer the container.
