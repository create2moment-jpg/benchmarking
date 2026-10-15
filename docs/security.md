# Security and safe execution

## Threat model

| Actor / input | Trust | Main risks |
|---|---|---|
| Model output (actions, files, summaries) | **untrusted** | arbitrary code via tests/commands, path traversal, exfiltration over the network, resource exhaustion, tampering with evaluator results, prompt-injected instructions in summaries |
| Task files (`task.yaml`, fixtures, evaluator files) | semi-trusted (reviewed in repo) | malicious evaluator commands, path traversal in check specs |
| Imported runs (JSON, patches, command logs, transcripts) | **untrusted** | path traversal / symlinks in patches, executing imported commands, oversized inputs |
| Config files | trusted operator input, validated | credentials committed to config, accidental remote data transfer, unsafe limits |
| Provider responses | untrusted | malformed data, huge payloads |
| Dashboard clients | local user | XSS via stored model output, path traversal in artifact requests |

## Controls

**Data leaving the machine.** Local-first by default. A model is *remote* if `remote: true`;
endpoints on non-local hosts must declare it (config validation refuses otherwise). Runs against
remote providers or remote CLI harnesses require explicit consent (`--allow-remote` or
`allow_remote_providers: true`). Request metadata (task paths etc.) is never sent to providers.
`bench doctor` probes `GET /models` only - no prompts or code. Exports exclude transcripts and
diffs unless `--include-transcripts` / `--include-source` are given.

**Secrets.** Config stores only environment variable *names* (`api_key_env`, `${VAR}` in
headers; literal `Authorization` headers are rejected). Commands run with a scrubbed environment
(no host variables, no proxies, `HOME`/`TMPDIR` in a scratch dir), so API keys are not visible to
model-run code. All persisted logs/traces and all exports pass through `Redactor` (known env
values + secret-shaped patterns). `.env` is gitignored; `.env.example` has placeholders only.

**Command execution** (`tools/runner.py`):
* structured argv, never `shell=True`; executable must be on the global allowlist **and** the
  task's `allowed_commands`; shells, network tools and `rm` cannot be allowlisted;
* executables are resolved by the runner (`python`/`pytest`/`ruff` → the benchmark interpreter);
* arguments with `..` segments or absolute paths outside the workspace are refused;
* `./program` only when the task opts in with `local-binaries`, and only inside the workspace;
* wall-clock timeout (process group killed), `RLIMIT_CPU`, `RLIMIT_AS` (per resource profile),
  `RLIMIT_FSIZE`, no core dumps, output capped (head + tail kept);
* **network**: each command runs in a fresh Linux network namespace with only loopback
  (created in-process with `os.unshare` on Python ≥ 3.12, falling back to `unshare -rn`).
  `isolation.network: deny` fails closed when namespaces are unavailable;
  `deny_if_possible` (default) degrades with `bench doctor` reporting `none`.

**Workspaces.** Each attempt gets a new temp directory outside the repository; the model only
writes inside it (`resolve_inside` rejects traversal and symlink escapes; `editable_paths`
restricts which files may change). Hidden evaluator files are copied in only after the agent
finishes, into a *separate* evaluation copy; a model-created `_hidden/` is deleted first.
Workspaces are deleted after evidence is stored (unless `keep_workspaces`).

**Evaluator integrity.** Evaluators never trust self-reports. JUnit results are cross-checked
with the pytest exit status (a non-zero exit cannot report "all passed"). C++ `CHECK` lines are
scored against `expected_checks` and crashes halve the score. SQL candidates run read-only:
SQLite with an authorizer allowing only SELECT/READ/FUNCTION/RECURSIVE plus a progress-handler
timeout; DuckDB with `enable_external_access=false` and `lock_configuration=true` and an
interrupt timer.

**Imports.** Pydantic validation with size limits; unknown fields are errors; patches are checked
for traversal, symlink/mode changes and non-editable paths before `git apply` (no shell) into a
fresh workspace; imported command logs are stored as evidence and **never executed**.

**Dashboard.** Binds to `127.0.0.1` by default (warning otherwise), GET-only, strict CSP, all
untrusted strings rendered with `textContent`, artifacts served only by registered
(attempt id, name) pairs.

**Storage.** Parameterised SQL; dynamic column names only from an allowlist; WAL +
`synchronous=FULL`; atomic file writes.

## Residual risks and recommendations

* The sandbox bounds accidents and casual misuse; it is **not** a security boundary against a
  determined adversary. Model-written code runs as your user with read access to files your
  user can read (outside the workspace) - argument checks are defence in depth, not confinement.
  For untrusted models run the benchmark inside a container/VM (e.g. `docker run --network none`
  with the repo mounted read-only and `artifacts/` mounted read-write) or under gVisor.
* External CLI harnesses (`kind: cli`) control their own tools; only timeout, environment
  scrubbing (plus explicitly passed-through variables), workspace cwd and (unless `remote`)
  network denial apply. They are disabled unless `--allow-external-harness` is passed.
* Regex redaction cannot catch every secret format; prefer not to give models secrets at all.
* Judge models receive diffs and summaries; using a remote judge sends that data (consent flag).
