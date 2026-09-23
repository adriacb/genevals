"""Two patterns for a fully custom Executor — no API key, runs offline.

genevals only requires `complete(prompt: str) -> str`; how you get there is
entirely up to you.

  1. BusinessContextExecutor — wraps ANY other Executor, prepending fixed
     business context (a persona, house rules, compliance language) to every
     prompt. Useful when a provider has no first-class "system prompt"
     concept, or you want identical context-injection behavior layered over
     several different providers. (For Anthropic/OpenAI specifically, the
     same effect is available more directly via `system=` — see
     AnthropicExecutor's docstring — this wrapper is for the general case.)

  2. RawHttpExecutor — no Anthropic/OpenAI SDK at all: a stdlib-only HTTP
     POST to a JSON completion endpoint, run in a thread since urllib is
     blocking. This is the pattern for any provider genevals doesn't ship
     an Executor for — a private internal service, a research API, anything
     that just speaks JSON over HTTP.

The demo runs both against a trivial local HTTP server (standing in for
"your own internal service") so it's fully self-contained — no network, no
API key.

Run with: uv run python examples/custom_executor_demo.py
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from genevals import Dataset, Evaluator, Sample, SimpleTarget
from genevals.executors.base import Executor
from genevals.metrics.text import Contains


class BusinessContextExecutor(Executor):
    def __init__(self, inner: Executor, *, business_context: str):
        self._inner = inner
        self._business_context = business_context

    async def complete(self, prompt: str) -> str:
        full_prompt = f"{self._business_context}\n\n{prompt}"
        result = self._inner.complete(full_prompt)
        if inspect.isawaitable(result):
            result = await result
        return result


class RawHttpExecutor(Executor):
    def __init__(self, url: str, *, extra_headers: dict[str, str] | None = None):
        self._url = url
        self._headers = {"Content-Type": "application/json", **(extra_headers or {})}

    async def complete(self, prompt: str) -> str:
        return await asyncio.to_thread(self._post, prompt)

    def _post(self, prompt: str) -> str:
        body = json.dumps({"prompt": prompt}).encode("utf-8")
        request = urllib.request.Request(self._url, data=body, headers=self._headers, method="POST")
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["completion"]


class _LocalEchoHandler(BaseHTTPRequestHandler):
    """Stands in for a real internal completion service, so this example
    needs no network access or API key to run."""

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        completion = f"[echo] {body['prompt']}"
        payload = json.dumps({"completion": completion}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass  # keep the demo's stdout quiet


def main() -> None:
    server = HTTPServer(("127.0.0.1", 0), _LocalEchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        raw = RawHttpExecutor(f"http://127.0.0.1:{server.server_port}")
        business = BusinessContextExecutor(
            raw,
            business_context="You are Acme Corp's support bot. Always mention our 30-day return policy.",
        )

        dataset = Dataset([Sample(id="1", input="Can I return this jacket?", reference="30-day")])
        target = SimpleTarget("business-bot", lambda s: business.complete(s.input))
        report = Evaluator(dataset, [target], [Contains()]).run()

        result = report.results[0]
        print("output:", result.output.text)
        print("contains '30-day':", result.score("contains").passed)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
