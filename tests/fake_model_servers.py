"""In-process fakes of an Ollama server and a vLLM-style OpenAI-compatible server, following
their documented HTTP APIs, for testing the performance probe without a real model."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

MODEL = "qwen3.8:latest"


class _Base(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _stream(self, chunks: list[bytes], ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for c in chunks:
            self.wfile.write(f"{len(c):x}\r\n".encode() + c + b"\r\n")
            self.wfile.flush()
            time.sleep(0.005)
        self.wfile.write(b"0\r\n\r\n")


class FakeOllama(_Base):
    generate_calls: ClassVar[int] = 0

    def do_GET(self) -> None:
        if self.path == "/api/version":
            self._json({"version": "0.12.3"})
        elif self.path == "/api/tags":
            self._json(
                {
                    "models": [
                        {
                            "name": MODEL,
                            "size": 5_200_000_000,
                            "details": {
                                "family": "qwen3",
                                "parameter_size": "8.2B",
                                "quantization_level": "Q4_K_M",
                                "format": "gguf",
                            },
                        }
                    ]
                }
            )
        elif self.path == "/api/ps":
            self._json(
                {
                    "models": [
                        {
                            "name": MODEL,
                            "size": 6_000_000_000,
                            "size_vram": 4_500_000_000,
                            "context_length": 8192,
                        }
                    ]
                }
            )
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        body = self._body()
        if self.path == "/api/show":
            self._json(
                {
                    "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M"},
                    "model_info": {
                        "general.architecture": "qwen3",
                        "general.parameter_count": 8_190_000_000,
                        "qwen3.context_length": 40960,
                    },
                    "capabilities": ["completion", "tools"],
                    "template": "x" * 10000,
                }
            )
        elif self.path == "/api/generate":
            FakeOllama.generate_calls += 1
            n = body["options"]["num_predict"]
            chunks = [
                json.dumps({"response": "tok", "done": False}).encode() + b"\n" for _ in range(3)
            ]
            final = {
                "done": True,
                "response": "",
                "total_duration": 2_000_000_000,
                "load_duration": 1_500_000_000 if FakeOllama.generate_calls == 1 else 1_000_000,
                "prompt_eval_count": len(body["prompt"]) // 4,
                "prompt_eval_duration": 100_000_000,
                "eval_count": n,
                "eval_duration": int(n / 40 * 1e9),
            }
            self._stream([*chunks, json.dumps(final).encode() + b"\n"], "application/x-ndjson")
        else:
            self._json({"error": "not found"}, 404)


class FakeVllm(_Base):
    tokens: ClassVar[list[float]] = [1000.0, 500.0]

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json(
                {"data": [{"id": "strata/qwen-3.8", "owned_by": "vllm", "max_model_len": 32768}]}
            )
        elif self.path == "/version":
            self._json({"version": "0.11.0"})
        elif self.path == "/metrics":
            p, g = FakeVllm.tokens
            text = (
                "# HELP vllm:prompt_tokens_total x\n# TYPE vllm:prompt_tokens_total counter\n"
                f'vllm:prompt_tokens_total{{model_name="strata/qwen-3.8"}} {p}\n'
                f'vllm:generation_tokens_total{{model_name="strata/qwen-3.8"}} {g}\n'
                'vllm:num_requests_running{model_name="m"} 0\n'
            )
            self._send(200, text.encode(), "text/plain; version=0.0.4")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        body = self._body()
        if self.path != "/v1/chat/completions":
            self._json({"error": "not found"}, 404)
            return
        n = body["max_tokens"]
        events = [{"choices": [{"delta": {"content": "x"}}]} for _ in range(5)]
        events.append({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": n}})
        chunks = [f"data: {json.dumps(e)}\n\n".encode() for e in events] + [b"data: [DONE]\n\n"]
        FakeVllm.tokens = [FakeVllm.tokens[0] + 50, FakeVllm.tokens[1] + n]
        self._stream(chunks, "text/event-stream")


@contextmanager
def serve(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
