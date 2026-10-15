# Provider and harness setup

Models are declared in a models file (`configs/models.example.yaml`; copy it to
`configs/models.yaml`, which the CLI prefers, or pass `--models`). No code changes are needed to
add a model. Check a config with `bench validate-config` and endpoints with `bench doctor`.

## Local models (data stays on this machine)

### Ollama
```bash
ollama pull qwen2.5-coder:14b
ollama serve            # exposes an OpenAI-compatible API on :11434/v1
```
```yaml
- alias: local-qwen
  provider: openai_compatible
  endpoint_type: ollama
  base_url: http://localhost:11434/v1
  model_id: qwen2.5-coder:14b
  serving_runtime: ollama
  quantization: Q4_K_M          # recorded for comparability
  context_limit: 32768          # used to trim old tool observations
  sampling: {temperature: 0.2, top_p: 0.95, seed: 7, max_tokens: 4096}
  timeout_s: 300
  max_concurrency: 1
```
Tip: Ollama's default context is small; set `num_ctx` in a Modelfile or via
`extra_body: {options: {num_ctx: 32768}}`.

### vLLM
```bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000 --api-key "$VLLM_API_KEY"
```
```yaml
- alias: local-vllm
  provider: openai_compatible
  endpoint_type: vllm
  base_url: http://127.0.0.1:8000/v1
  model_id: Qwen/Qwen2.5-Coder-32B-Instruct
  api_key_env: VLLM_API_KEY     # name of the env var, never the key
  serving_runtime: vllm
  quantization: bf16
```
LM Studio, llama.cpp `llama-server` and other servers implementing `/v1/chat/completions` work
the same way (`endpoint_type: lmstudio | llamacpp | other`). Hosts on private networks
(10/8, 172.16/12, 192.168/16, `.local`) count as local.

## Paid / remote models

Remote models need `remote: true` in config **and** `--allow-remote` per run (or
`allow_remote_providers: true`), because task prompts and fixture source code are sent.

### OpenAI-compatible APIs
```yaml
- alias: openai-paid
  provider: openai_compatible
  endpoint_type: openai
  base_url: https://api.openai.com/v1
  model_id: <model id>
  api_key_env: OPENAI_API_KEY
  remote: true
  requests_per_minute: 30
  pricing: {input_per_mtok: <price>, output_per_mtok: <price>, source: "<where from>"}
```
Cost is reported only when you supply `pricing` and the provider reports usage.

### Anthropic (Claude)
Install the official SDK extra: `pip install 'engineering-llm-bench[anthropic]'`.
```yaml
- alias: claude-opus
  provider: anthropic
  endpoint_type: anthropic
  model_id: claude-opus-5-5
  api_key_env: ANTHROPIC_API_KEY
  remote: true
  sampling: {max_tokens: 16000}
  extra_body: {output_config: {effort: high}}
```
Current Claude models reject `temperature`/`top_p`; the adapter does not send sampling values and
uses `extra_body` for provider-specific settings (e.g. effort). A `refusal` stop reason is
recorded as a provider error.

## Execution modes

| Mode | Harness | Use |
|---|---|---|
| A. direct | `--harness direct` | one completion returning whole files; focused coding prompts |
| B. agent | `--harness agent` (built-in loop) or a configured `cli` harness | plan / edit / run tests / revise |
| C. manual | `bench import-run` | workflows that cannot be automated reliably (IDE agents) |

### External agent CLIs (`kind: cli`)
```yaml
harnesses:
  - name: claude-code-cli
    kind: cli
    version: "external"
    argv: ["claude", "-p", "--permission-mode", "acceptEdits"]
    prompt_via: stdin          # stdin | arg ({prompt} placeholder) | file ({prompt_file})
    env_passthrough: [ANTHROPIC_API_KEY]
    remote: true               # network allowed for this harness
```
Run with `bench run --harness claude-code-cli --model <label> --allow-external-harness
--allow-remote`. The `--model` value is a free label stored with results (the CLI chooses its own
model). The benchmark bounds the CLI only by timeout, scrubbed environment, workspace cwd and
(unless `remote`) network denial; usage is recorded as unknown.

## Copilot BYOK sessions (recommended for IDE agents)

`byok/prepare_session.py` writes a session folder with `starting_point.md` for the agent;
`byok/collect_results.py` validates the agent's output, runs the hidden evaluators and stores
the run for the dashboard. See [byok/README.md](../byok/README.md). The lower-level import format
below remains available for other captured workflows.

## Manual import

GitHub Copilot BYOK (and similar IDE agents) has **no public automation API** that this tool can
drive reliably, so it is not automated. Instead, capture evidence and import it; it is scored as
its own model/harness combination (`mode = manual`).

Capture workflow:
1. `bench task validate tasks/<domain>/<task>` (sanity), then copy `tasks/.../fixture` to a
   scratch git repo: `git init && git add -A && git commit -m base`.
2. Run the IDE agent on the prompt from `task.yaml` (`prompt` + `acceptance_criteria`).
3. Save `git diff > attempt.patch`, note elapsed time, token/cost figures **only if the IDE shows
   them**, and optionally export the transcript.
4. Write the import JSON (example: `examples/import/copilot-byok-example.json`):

```json
{
  "schema_version": 1,
  "format": "engineering-llm-bench/import",
  "run": {"model_label": "copilot-byok-<model>", "harness": "copilot-byok-vscode",
          "harness_version": "<extension version>", "provider_label": null, "remote": true},
  "attempts": [{"task_id": "python.fix-interval-merge", "task_version": "1.0.0",
                "patch_file": "attempt.patch", "elapsed_s": 312, "input_tokens": null,
                "output_tokens": null, "cost_usd": null, "summary": "...", "human_notes": "..."}]
}
```
5. `bench import-run --file run.json` applies each patch to a fresh fixture copy and runs the
   task's own evaluators. `--no-evaluate` stores evidence only (attempt marked skipped).
   `--keep-transcripts` stores transcripts (off by default).

Rules: missing data stays `null` (never estimated); `self_reported_passed` is stored but not
trusted; `command_logs` are evidence only and are never executed; patches that touch paths
outside `editable_paths`, use `..`, or create symlinks are rejected; unknown JSON fields are
errors. `bench export-run --run-id X --out run.json` produces the same format (round trip).
