# Instructions for the model under test

`command.md` is the prompt you send to the agent (GitHub Copilot agent mode with your BYOK
model) together with the session's `starting_point.md`. It is the same for every model, so
results stay comparable. It names no model or serving stack.

## Run everything, step by step

```bash
# 1. In this repository: create a fresh session folder (outside the repo)
python byok/prepare_session.py --profile standard --out ~/elb-byok-sessions/<model>-run1 \
    --byok-config ~/.config/Code/User/settings.json        # optional: server statistics
bench byok mark-start ~/elb-byok-sessions/<model>-run1     # optional, right before step 3
```

2. In VS Code: **File > Open Folder…** → open `~/elb-byok-sessions/<model>-run1` (only the
   session folder, not this repository).
3. Open Copilot Chat → **Agent** mode → select your BYOK model.
4. Attach `starting_point.md` (drag it into the chat or type `#file:starting_point.md`).
5. Paste the content of [`command.md`](command.md) as the message and send it. Allow terminal
   commands when Copilot asks (pytest, g++, cmake, …).
6. If it stops before replying `ALL TASKS COMPLETE`, send:
   `Continue with the next unfinished task in starting_point.md, following the same rules.`

```bash
# 7. Back in this repository: validate, score, store, open the dashboard
python byok/collect_results.py ~/elb-byok-sessions/<model>-run1 \
    --model-label "strata/qwen-3.8" --provider-label ollama --local --serve
```

The collector moves the agent's code into the model's private output folder
(`artifacts/model_outputs/<model>/`) and removes the working copy, so collect one model's
session before preparing the next. Details: [byok/README.md](../byok/README.md).

`bench byok prepare` prints the same prompt; `src/engineering_llm_bench/byok/command.md` is the
packaged copy (a test keeps both identical).
