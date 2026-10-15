# Instructions for the model under test

`command.md` is the prompt you send to the agent (GitHub Copilot agent mode with your BYOK
model). It is the same for every model, so results stay comparable, and it names no model or
serving stack.

## Run everything, step by step

```bash
# 1. In this repository: create a fresh session folder (outside the repo)
python byok/prepare_session.py --profile standard --out ~/elb-byok-sessions/<model>-run1 \
    --byok-config ~/.config/Code/User/settings.json        # optional: server statistics
bench byok mark-start ~/elb-byok-sessions/<model>-run1     # optional, right before step 4
```

2. In VS Code: **File > Open Folder…** → open `~/elb-byok-sessions/<model>-run1`.
   **Not this repository.** The Explorer must show `starting_point.md`, `AGENTS.md` and
   `tasks/` at the top level. A new window (`code ~/elb-byok-sessions/<model>-run1`) is
   safest.
3. Open Copilot Chat → **Agent** mode → select your BYOK model.
4. Attach `starting_point.md` from the session folder (drag it into the chat or type
   `#file:starting_point.md`), paste the content of [`command.md`](command.md) as the
   message and send it.
5. If it stops before replying `ALL TASKS COMPLETE`, send:
   `Continue with the next unfinished task in starting_point.md, following the same rules.`

```bash
# 6. Back in this repository: validate, score, store, open the dashboard
python byok/collect_results.py ~/elb-byok-sessions/<model>-run1 \
    --model-label "strata/qwen-3.8" --provider-label ollama --local --serve
```

The collector moves the agent's code into the model's private output folder
(`artifacts/model_outputs/<model>/`) and removes the working copy, so collect one model's
session before preparing the next. Details: [byok/README.md](../byok/README.md).

## Troubleshooting

| What you see | Cause and fix |
|---|---|
| "No starting_point.md exists … this workspace is the benchmark harness" | VS Code has this repository open. Prepare a session and open **its** folder (step 2). The repository's `AGENTS.md` tells agents to stop here instead of searching. |
| Agent replies `NO SESSION FOLDER OPEN` | Same cause: the open workspace has no `starting_point.md` at its root. |
| A **Run bash command? Allow / Skip** box for every command | That is VS Code's tool approval, not the benchmark; the prompt cannot change it. You can click **Allow** each time; use the arrow next to **Allow** to allow that command for the session; or change the approval level in the picker under the chat input ("Default Approvals"). Decide this yourself: auto-approved commands run as your user, outside the benchmark's sandbox. Approve only the commands the tasks need (`python`, `pytest`, `ruff`, `g++`, `cmake`, `ctest`, `./<test binary>`), never `rm`, `curl`, `pip` or `git`. |
| "Continue to iterate?" after many steps | VS Code's per-request step limit; click **Continue**, or raise `chat.agent.maxRequests` in your user settings. |
| Agent stops after one task | Send the continue message from step 5. |

`bench byok prepare` prints the same prompt; `src/engineering_llm_bench/byok/command.md` is the
packaged copy (a test keeps both identical).
