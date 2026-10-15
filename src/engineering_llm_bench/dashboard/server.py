"""Local dashboard: a small stdlib HTTP server exposing read-only JSON over the results DB.

Design choices: no framework or build step (one HTML/JS/CSS bundle shipped in the package),
GET-only API, loopback bind by default, strict Content-Security-Policy, artifacts served only
by (attempt id, registered artifact name) - never by arbitrary path.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

from engineering_llm_bench.perf.probe import headline
from engineering_llm_bench.redaction import Redactor
from engineering_llm_bench.reporting.export import ReportOptions, build_report
from engineering_llm_bench.storage.artifacts import ArtifactStore
from engineering_llm_bench.storage.db import Store

MAX_ARTIFACT_BYTES = 2_000_000
_STATIC = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
}
_CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:"


class DashboardApi:
    """Pure functions behind the HTTP handler (unit-testable without sockets)."""

    def __init__(self, store: Store, artifacts: ArtifactStore, redactor: Redactor) -> None:
        self.store = store
        self.artifacts = artifacts
        self.redactor = redactor

    def report(self, run_ids: list[str] | None = None) -> dict[str, Any]:
        data = build_report(
            self.store, self.artifacts, self.redactor, ReportOptions(run_ids=run_ids)
        )
        for a in data["attempts"]:  # keep the payload light; details come from /api/attempt
            a.pop("summary", None)
            a.pop("checks", None)
        latest = max((r["finished_at"] or r["created_at"] for r in data["runs"]), default=None)
        data["freshness"] = latest
        return data

    def attempt(self, attempt_id: str) -> dict[str, Any] | None:
        a = self.store.get_attempt(attempt_id)
        if a is None:
            return None
        run = self.store.get_run(a["run_id"]) or {}
        detail = {
            "attempt": a,
            "run": {k: v for k, v in run.items() if not k.endswith("_json")},
            "environment": _loads(run.get("env_json")),
            "model_config": _loads(run.get("model_config_json")),
            "harness_config": _loads(run.get("harness_config_json")),
            "run_config": _loads(run.get("run_config_json")),
            "dimensions": self.store.dimension_scores(attempt_id),
            "checks": self.store.check_results(attempt_id),
            "artifacts": self.store.artifacts(attempt_id),
            "reviews": self.store.reviews(attempt_id),
            "notes": _loads(a.get("notes_json")) or [],
            "performance": _loads(run.get("perf_json")),
            "performance_headline": headline(_loads(run.get("perf_json"))) or None,
        }
        redacted: dict[str, Any] = self.redactor.obj(detail)
        return redacted

    def artifact(self, attempt_id: str, name: str) -> str | None:
        for ref in self.store.artifacts(attempt_id):
            if ref["name"] == name:
                path = self.artifacts.resolve(ref["path"])
                with path.open("rb") as fh:
                    data = fh.read(MAX_ARTIFACT_BYTES + 1)
                text = data[:MAX_ARTIFACT_BYTES].decode("utf-8", errors="replace")
                if len(data) > MAX_ARTIFACT_BYTES:
                    text += "\n...[truncated for display]..."
                return self.redactor.text(text)
        return None


def _loads(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def make_handler(api: DashboardApi) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "elb-dashboard"

        def log_message(self, format: str, *args: Any) -> None:
            return  # keep the terminal quiet; requests carry no secrets but may be noisy

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", _CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path in _STATIC:
                    name, ctype = _STATIC[url.path]
                    body = (
                        resources.files("engineering_llm_bench.dashboard")
                        .joinpath("static", name)
                        .read_bytes()
                    )
                    self._send(200, body, ctype)
                elif url.path == "/api/report":
                    ids = [i for v in query.get("run_ids", []) for i in v.split(",") if i]
                    self._json(api.report(ids or None))
                elif url.path.startswith("/api/attempt/"):
                    detail = api.attempt(url.path.removeprefix("/api/attempt/"))
                    if detail is None:
                        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    else:
                        self._json(detail)
                elif url.path == "/api/artifact":
                    text = api.artifact(query.get("attempt", [""])[0], query.get("name", [""])[0])
                    if text is None:
                        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    else:
                        self._send(200, text.encode(), "text/plain")
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except KeyError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(store: Store, artifacts: ArtifactStore, redactor: Redactor, host: str, port: int) -> None:
    api = DashboardApi(store, artifacts, redactor)
    httpd = ThreadingHTTPServer((host, port), make_handler(api))
    print(f"dashboard: http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
