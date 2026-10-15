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
| `docker: permission denied` | Add yourself to the `docker` group (`sudo usermod -aG docker $USER`, then log in again). |

## Without Docker

If Docker is not available (or with `ELB_MODE=host ./setup.sh`), `setup.sh` installs the task
runtimes into `~/elb-byok-sessions/.runtime-venv` and opens the session on your machine. It
auto-approves only the commands the tasks need (`python`, `pytest`, `g++`, `cmake`, …); `rm`,
`git`, `pip`, `curl` and similar still ask, and the agent is told not to install anything.
Auto-approved `python` can run any code as your user, so prefer the container.
