"""engineering-llm-bench: reproducible engineering-task benchmark for local and paid LLMs.

The package is organised by responsibility:

* ``config``        - validated YAML configuration (models, harnesses, benchmark settings).
* ``catalog``       - versioned task definitions and the task registry.
* ``providers``     - model adapters (fake, OpenAI-compatible, Anthropic) behind one protocol.
* ``harnesses``     - execution modes: direct completion, built-in agent loop, external CLI.
* ``workspace``     - disposable per-attempt workspaces and safe path handling.
* ``tools``         - sandboxed command runner (allowlist, timeouts, output caps, rlimits).
* ``evaluation``    - independent evaluator checks and transparent rubric scoring.
* ``storage``       - SQLite persistence with migrations and atomic artifact files.
* ``orchestration`` - the run lifecycle (resolve → prepare → execute → verify → score → persist).
* ``importing``     - evidence import for manually captured (e.g. IDE / Copilot BYOK) runs.
* ``reporting``     - statistics, JSON/CSV/HTML exports and secret redaction.
* ``dashboard``     - a small local web dashboard served from persisted data.
"""

__version__ = "0.1.0"

#: Version of the persisted data contracts (DB schema + export format).
DATA_SCHEMA_VERSION = 2
