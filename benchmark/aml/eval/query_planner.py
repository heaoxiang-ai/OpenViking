"""Use the pinned RL V7 training template; consume query strings only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

from benchmark.aml.eval.cached_http import CachedHTTP, completion_text

MODEL = "guoxuter/Qwen3.5-2B-Intent-RL-V7"
PROMPT_REVISION = "220433d4a234142638cfcc4ab1ee8dcfc4b4d142"
PROMPT_SHA256 = "7d62e9c860714e0c7df08a6888f60e14da48f6bb15b631bd2edba04bc12132ae"


def planner_payload(question: str, prompt_path: Path) -> dict[str, Any]:
    source = prompt_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != PROMPT_SHA256:
        raise ValueError("planner template differs from the pinned RL V7 training template")
    config = yaml.safe_load(source)
    prompt = Template(config["template"]).render(
        compression_summary="None",
        recent_messages="None",
        current_message=question,
        context_type="",
        target_abstract="",
    )
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config["llm_config"]["temperature"],
        "max_tokens": 1536,
    }


def parse_queries(body: Any, question: str) -> dict[str, Any]:
    text = completion_text(body, MODEL)
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("queries"), list):
        raise ValueError("invalid planner query list")
    if len(parsed["queries"]) > 5:
        raise ValueError("planner exceeds the template maximum of five queries")
    queries = []
    for item in parsed["queries"]:
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            raise ValueError("invalid planner query text")
        query = item["query"].strip()
        if query and query not in queries:
            queries.append(query)
    # context_type, intent, and priority never control routing or ordering.
    return {"queries": queries or [question], "raw_model_output": parsed}


class QueryPlanner:
    def __init__(self, http: CachedHTTP, base_url: str, api_key: str, prompt_path: Path):
        self.http = http
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": "Bearer " + api_key}
        self.prompt_path = prompt_path

    async def rewrite(self, question: str) -> dict[str, Any]:
        return await self.http.request(
            "planner",
            "POST",
            self.url,
            payload=planner_payload(question, self.prompt_path),
            headers=self.headers,
            validate=lambda body: parse_queries(body, question),
        )
