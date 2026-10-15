"""Validated configuration schemas.

Two files configure a benchmark:

* a *models file* (``configs/models.example.yaml``) declaring model aliases, provider settings,
  local-serving metadata and optional external harnesses;
* a *benchmark file* (``configs/benchmark.example.yaml``) declaring profiles, weights, limits,
  isolation and storage locations.

Every field is validated on load. Unsafe limits, malformed endpoint URLs, invalid weights and
undeclared remote endpoints are rejected with actionable messages. Credentials are never stored
in config: only the *name* of an environment variable that holds them.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engineering_llm_bench.domain import DEFAULT_WEIGHTS, Dimension, Profile, validate_weights

_ALIAS_RE = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")

# Upper bounds considered "safe". Larger values are rejected rather than silently clamped so a typo
# (e.g. 36000 instead of 3600) cannot turn a smoke run into an unbounded one.
MAX_COMMAND_TIMEOUT_S = 3600
MAX_TASK_TIMEOUT_S = 4 * 3600
MAX_RUN_TIMEOUT_S = 48 * 3600
MAX_TURNS = 200
MAX_MEMORY_MB = 64 * 1024
MAX_OUTPUT_BYTES = 50 * 1024 * 1024
MAX_PARALLEL = 32


class StrictModel(BaseModel):
    """Base model: unknown keys are errors (catches typos in YAML)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def is_local_host(host: str) -> bool:
    """Return True when ``host`` refers to the local machine or a private network address.

    Used to stop a config from labelling a public endpoint as local (which would bypass the
    explicit consent required before sending repository contents to remote providers).
    """
    if host in {"localhost", "host.docker.internal"} or host.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private or addr.is_link_local


class SamplingConfig(StrictModel):
    """Sampling parameters recorded with every run for reproducibility."""

    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None
    max_tokens: int = Field(default=4096, ge=16, le=200_000)


class PricingConfig(StrictModel):
    """User-supplied prices. Cost is only reported when usage metadata is also reliable."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    currency: str = "USD"
    source: str = Field(default="user-supplied", description="Where the price came from.")


FakeBehavior = Literal["reference", "noop", "broken", "error", "timeout", "malformed"]


class FakeConfig(StrictModel):
    """Behaviour of the clearly labelled mock provider used for tests and smoke runs."""

    behavior: FakeBehavior = "reference"
    latency_s: float = Field(default=0.0, ge=0.0, le=60.0)


class ModelConfig(StrictModel):
    """A model alias bound to a provider. The alias is the stable label used in history."""

    alias: Annotated[str, Field(pattern=_ALIAS_RE)]
    provider: Literal["fake", "openai_compatible", "anthropic"]
    model_id: str = Field(min_length=1, max_length=256)
    endpoint_type: Literal[
        "ollama", "vllm", "lmstudio", "llamacpp", "openai", "anthropic", "other", "fake"
    ] = "other"
    base_url: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    remote: bool = False
    serving_runtime: str | None = None
    quantization: str | None = None
    context_limit: int | None = Field(default=None, ge=256)
    sampling: SamplingConfig = SamplingConfig()
    timeout_s: float = Field(default=120.0, gt=0, le=MAX_COMMAND_TIMEOUT_S)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_concurrency: int = Field(default=1, ge=1, le=MAX_PARALLEL)
    requests_per_minute: int | None = Field(default=None, ge=1, le=10_000)
    stream: bool = False
    #: Provider-specific request fields passed through verbatim (e.g. Anthropic
    #: ``output_config: {effort: high}``). Recorded in the config fingerprint.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    pricing: PricingConfig | None = None
    fake: FakeConfig | None = None
    notes: str | None = None

    @field_validator("api_key_env")
    @classmethod
    def _env_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_RE.match(value):
            raise ValueError(
                "api_key_env must be the NAME of an environment variable (e.g. OPENAI_API_KEY), "
                "never the key itself"
            )
        return value

    @field_validator("headers")
    @classmethod
    def _headers(cls, value: dict[str, str]) -> dict[str, str]:
        for name, header_value in value.items():
            if not _HEADER_NAME_RE.match(name):
                raise ValueError(f"invalid header name {name!r}")
            if name.lower() in {"authorization", "x-api-key"} and "${" not in header_value:
                raise ValueError(
                    f"header {name!r} must reference an environment variable like ${{MY_KEY}}; "
                    "do not put credentials in config files"
                )
        return value

    @model_validator(mode="after")
    def _provider_requirements(self) -> ModelConfig:
        if self.provider == "fake":
            if self.remote:
                raise ValueError("fake provider cannot be remote")
            return self
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible provider requires base_url")
        if self.base_url is not None:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"malformed endpoint URL {self.base_url!r}: need http(s)://host")
            if parsed.username or parsed.password:
                raise ValueError("endpoint URL must not embed credentials")
            if not self.remote and not is_local_host(parsed.hostname):
                raise ValueError(
                    f"endpoint host {parsed.hostname!r} is not local; set remote: true to "
                    "acknowledge that prompts and source code will be sent to it"
                )
        if self.provider == "anthropic" and not self.remote:
            raise ValueError("anthropic provider is a remote API; set remote: true")
        return self


class HarnessConfig(StrictModel):
    """An execution harness. ``direct`` and ``agent`` are built in; ``cli`` wraps external tools.

    External CLI harnesses (e.g. an agentic coding CLI) manage their own tools, so the benchmark
    can only bound them by timeout, environment scrubbing and workspace isolation. They are
    disabled unless ``--allow-external-harness`` is given.
    """

    name: Annotated[str, Field(pattern=_ALIAS_RE)]
    kind: Literal["direct", "agent", "cli"]
    version: str = "1"
    description: str | None = None
    argv: list[str] = Field(default_factory=list)
    version_argv: list[str] = Field(default_factory=list)
    prompt_via: Literal["arg", "stdin", "file"] = "stdin"
    env_passthrough: list[str] = Field(default_factory=list)
    remote: bool = False

    @model_validator(mode="after")
    def _cli_requirements(self) -> HarnessConfig:
        if self.kind == "cli":
            if not self.argv:
                raise ValueError(f"cli harness {self.name!r} requires argv")
            if self.prompt_via == "arg" and not any("{prompt}" in a for a in self.argv):
                raise ValueError("prompt_via=arg requires a {prompt} placeholder in argv")
        for env in self.env_passthrough:
            if not _ENV_RE.match(env):
                raise ValueError(f"invalid env var name {env!r}")
        return self


BUILTIN_HARNESSES: tuple[HarnessConfig, ...] = (
    HarnessConfig(name="direct", kind="direct", description="Single completion returning files."),
    HarnessConfig(name="agent", kind="agent", description="Built-in plan/edit/test tool loop."),
)


class ModelsFile(StrictModel):
    schema_version: Literal[1] = 1
    models: list[ModelConfig]
    harnesses: list[HarnessConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> ModelsFile:
        aliases = [m.alias for m in self.models]
        dupes = sorted({a for a in aliases if aliases.count(a) > 1})
        if dupes:
            raise ValueError(f"duplicate model aliases: {dupes}")
        names = [h.name for h in self.harnesses]
        builtin = {h.name for h in BUILTIN_HARNESSES}
        clash = sorted({n for n in names if names.count(n) > 1 or n in builtin})
        if clash:
            raise ValueError(f"duplicate or reserved harness names: {clash}")
        return self

    def model(self, alias: str) -> ModelConfig:
        for m in self.models:
            if m.alias == alias:
                return m
        known = ", ".join(sorted(m.alias for m in self.models))
        raise KeyError(f"unknown model alias {alias!r}; known: {known}")

    def all_harnesses(self) -> list[HarnessConfig]:
        return [*BUILTIN_HARNESSES, *self.harnesses]

    def harness(self, name: str) -> HarnessConfig:
        for h in self.all_harnesses():
            if h.name == name:
                return h
        known = ", ".join(sorted(h.name for h in self.all_harnesses()))
        raise KeyError(f"unknown harness {name!r}; known: {known}")


class ProfileConfig(StrictModel):
    """Budget for one profile. Task timeouts are clipped to ``max_task_timeout_s``."""

    description: str = ""
    max_task_timeout_s: float = Field(gt=0, le=MAX_TASK_TIMEOUT_S)
    run_timeout_s: float = Field(gt=0, le=MAX_RUN_TIMEOUT_S)
    max_turns: int = Field(ge=1, le=MAX_TURNS)
    max_total_tokens: int | None = Field(default=None, ge=1000)
    repeats: int = Field(default=1, ge=1, le=20)
    include_tasks: list[str] = Field(default_factory=list)
    exclude_tasks: list[str] = Field(default_factory=list)
    warn_cost: bool = False


class LimitsConfig(StrictModel):
    """Resource limits applied to every command run inside a workspace."""

    command_timeout_s: float = Field(default=120, gt=0, le=MAX_COMMAND_TIMEOUT_S)
    max_output_bytes: int = Field(default=200_000, ge=1000, le=MAX_OUTPUT_BYTES)
    max_file_bytes: int = Field(default=2_000_000, ge=1000, le=512 * 1024 * 1024)
    memory_mb: int = Field(default=4096, ge=128, le=MAX_MEMORY_MB)
    cpu_seconds: int = Field(default=300, ge=1, le=MAX_COMMAND_TIMEOUT_S * 4)
    max_written_file_mb: int = Field(default=128, ge=1, le=16 * 1024)
    max_observation_chars: int = Field(default=12_000, ge=500, le=500_000)


class IsolationConfig(StrictModel):
    """Isolation for commands. ``unshare`` uses Linux user+net namespaces (no network)."""

    network: Literal["deny_if_possible", "deny", "allow"] = "deny_if_possible"
    keep_workspaces: bool = False


class ConcurrencyConfig(StrictModel):
    max_parallel_attempts: int = Field(default=1, ge=1, le=MAX_PARALLEL)
    max_parallel_evaluations: int = Field(default=2, ge=1, le=MAX_PARALLEL)


class BenchmarkConfig(StrictModel):
    schema_version: Literal[1] = 1
    suite_name: str = "engineering-llm-bench"
    suite_version: str = Field(min_length=1)
    tasks_dir: str = "tasks"
    artifacts_dir: str = "artifacts"
    database: str = "artifacts/bench.db"
    allow_remote_providers: bool = False
    default_harness: str = "agent"
    weights: dict[Dimension, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    profiles: dict[Profile, ProfileConfig]
    limits: LimitsConfig = LimitsConfig()
    isolation: IsolationConfig = IsolationConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    command_allowlist: list[str] = Field(
        default_factory=lambda: [
            "python",
            "pytest",
            "ruff",
            "g++",
            "c++",
            "cmake",
            "ctest",
            "make",
            "local-binaries",
        ]
    )
    redact_patterns: list[str] = Field(default_factory=list)

    @field_validator("weights")
    @classmethod
    def _weights(cls, value: dict[Dimension, float]) -> dict[Dimension, float]:
        validate_weights(value)
        return value

    @field_validator("redact_patterns")
    @classmethod
    def _patterns(cls, value: list[str]) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid redact pattern {pattern!r}: {exc}") from exc
        return value

    @field_validator("command_allowlist")
    @classmethod
    def _allowlist(cls, value: list[str]) -> list[str]:
        forbidden = {"sh", "bash", "zsh", "sudo", "su", "curl", "wget", "ssh", "scp", "nc", "rm"}
        bad = sorted(set(value) & forbidden)
        if bad:
            raise ValueError(f"command_allowlist must not contain shells/network/destructive {bad}")
        return value

    @model_validator(mode="after")
    def _profiles(self) -> BenchmarkConfig:
        missing = [p.value for p in Profile if p not in self.profiles]
        if missing:
            raise ValueError(f"profiles missing: {missing}")
        for name, prof in self.profiles.items():
            if prof.max_task_timeout_s > prof.run_timeout_s:
                raise ValueError(f"profile {name}: max_task_timeout_s exceeds run_timeout_s")
        return self
