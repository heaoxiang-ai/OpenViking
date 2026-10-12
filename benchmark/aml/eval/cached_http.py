"""Durable first-valid caches and shared throttling for read-only/model requests.

Never use this retry helper for Add or other state-changing OV requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import httpx


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def run_lock(directory: Path):
    """Prevent two CLI processes from issuing the same uncached requests."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another runner is using this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class CachedHTTP:
    def __init__(self, client: httpx.AsyncClient, directory: Path, *, attempts: int = 8):
        self.client = client
        self.directory = directory
        self.attempts = attempts
        self.cooldown = 0.0
        self.interval = 0.0
        self.next_start = 0.0
        self.quota_windows: set[int] = set()
        self.gate = asyncio.Lock()
        self.counts: Counter[str] = Counter()
        self.locks: dict[str, asyncio.Lock] = {}

    async def wait_turn(self) -> None:
        while True:
            async with self.gate:
                instant = time.monotonic()
                delay = max(self.cooldown, self.next_start) - instant
                if delay <= 0:
                    self.next_start = instant + self.interval
                    return
            await asyncio.sleep(min(delay, 2))

    async def request(
        self,
        stage: str,
        method: str,
        url: str,
        *,
        payload: dict | None = None,
        params: dict | None = None,
        identity: str = "",
        headers: dict[str, str] | Callable[[], dict[str, str]],
        validate: Callable[[Any], Any],
    ) -> Any:
        # Identity scopes native caches to account/user/snapshot. Credentials
        # and signed HTTP headers are deliberately excluded from artifacts.
        request = {
            "method": method,
            "url": url,
            "payload": payload,
            "params": params,
            "identity": identity,
        }
        key = digest(request)
        directory = self.directory / stage / key
        async with self.locks.setdefault(stage + key, asyncio.Lock()):
            accepted = directory / "accepted.json"
            if accepted.exists():
                saved = json.loads(accepted.read_text(encoding="utf-8"))
                if saved["request"] != request:
                    raise ValueError("cached request does not match")
                self.counts[stage + "_cached"] += 1
                return validate(saved["response"])
            for attempt in range(1, self.attempts + 1):
                await self.wait_turn()
                event: dict[str, Any] = {"attempt": attempt}
                try:
                    response = await self.client.request(
                        method,
                        url,
                        content=json.dumps(payload).encode() if payload is not None else None,
                        params=params,
                        headers={
                            "Content-Type": "application/json",
                            **(headers() if callable(headers) else headers),
                        },
                    )
                    event["status"] = response.status_code
                    self.counts[f"{stage}_{response.status_code}"] += 1
                    if response.status_code == 429:
                        try:
                            delay = max(65.0, float(response.headers.get("retry-after", "65")))
                        except ValueError:
                            delay = 65.0
                        self.cooldown = max(self.cooldown, time.monotonic() + delay)
                        window = int(time.monotonic() // 65)
                        if (
                            re.search(r"qpm|requests? per minute|code-2001", response.text, re.I)
                            and window not in self.quota_windows
                        ):
                            self.interval = max(0.1, self.interval * 1.5)
                            self.quota_windows.add(window)
                    elif 400 <= response.status_code < 500:
                        raise RuntimeError(f"{stage}: permanent HTTP {response.status_code}")
                    response.raise_for_status()
                    body = response.json()
                    result = validate(body)
                    write_json(accepted, {"request": request, "response": body})
                    return result
                except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
                    event["error_type"] = type(exc).__name__
                    if attempt == self.attempts:
                        raise RuntimeError(f"{stage}: technical retries exhausted") from None
                finally:
                    # Do not persist error bodies, Authorization, or API keys.
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event) + "\n")
                await asyncio.sleep(min(3 * attempt, 30))


def completion_text(body: Any, model: str | None = None) -> str:
    if model is not None and body.get("model") != model:
        raise ValueError("unexpected completion model")
    choice = body["choices"][0]
    text = choice["message"]["content"]
    if choice.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip():
        raise ValueError("empty or incomplete completion")
    return text.strip()
