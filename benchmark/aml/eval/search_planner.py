"""Query-only planner -> native Search -> URI union -> rerank -> Top20/owner.

This is benchmark orchestration. It never calls Add or modifies OV storage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import httpx

from benchmark.aml.eval.cached_http import CachedHTTP, digest, run_lock, write_json
from benchmark.aml.eval.evaluate import _read_jsonl
from benchmark.aml.eval.owner_labels import format_search_content
from benchmark.aml.eval.query_planner import PROMPT_SHA256, QueryPlanner
from benchmark.aml.eval.run_public import _bounded_map, _positive_int, _write_jsonl


def aligned_scores(body: Any, count: int) -> list[float]:
    """Provider rows can be score-sorted: id, never position, identifies input."""
    rows = body["result"]["data"]
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("rerank response count mismatch")
    scores = [0.0] * count
    seen = set()
    for row in rows:
        raw_id = row["id"]
        if isinstance(raw_id, bool) or not (
            isinstance(raw_id, int) or isinstance(raw_id, str) and raw_id.isdecimal()
        ):
            raise ValueError("invalid rerank document id")
        index = int(raw_id)
        score = float(row["score"])
        if index not in range(count) or index in seen or not math.isfinite(score):
            raise ValueError("invalid or duplicate rerank document id/score")
        seen.add(index)
        scores[index] = score
    return scores


class VikingDBReranker:
    def __init__(self, http: CachedHTTP, signer: Any):
        self.http = http
        self.signer = signer

    async def rank(self, question: str, documents: list[str]) -> list[float]:
        if not documents or len(documents) > 100:
            raise ValueError("rerank batches must contain 1..100 documents")
        payload = {
            "model_name": self.signer.model_name,
            "model_version": self.signer.model_version,
            "data": [[{"text": text}] for text in documents],
            "query": [{"text": question}],
            "instruction": "Whether the Document answers the Query or matches the content retrieval intent",
        }

        def headers() -> dict[str, str]:
            # Reuse the existing SDK signing helper, but not its positional
            # score parser or silent failure fallback. Re-sign each retry.
            request = self.signer._prepare_request("POST", "/api/vikingdb/rerank", data=payload)
            return request.headers

        return await self.http.request(
            "rerank",
            "POST",
            f"https://{self.signer.host}/api/vikingdb/rerank",
            payload=payload,
            headers=headers,
            validate=lambda body: aligned_scores(body, len(documents)),
        )


class SearchPipeline:
    def __init__(
        self,
        http: CachedHTTP,
        planner: QueryPlanner,
        reranker: VikingDBReranker,
        *,
        base_url: str,
        api_key: str,
        account: str,
        memory_revision: str,
        lane_limit: int = 200,
        topk: int = 20,
    ):
        self.http = http
        self.planner = planner
        self.reranker = reranker
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.account = account
        self.memory_revision = memory_revision
        self.lane_limit = lane_limit
        self.topk = topk

    def headers(self, user: str) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "X-OpenViking-Account": self.account,
            "X-OpenViking-User": user,
        }

    async def native(self, user: str, method: str, path: str, **kwargs: Any) -> Any:
        if (method, path) not in {
            ("POST", "/api/v1/search/search"),
            ("GET", "/api/v1/content/read"),
        }:
            raise ValueError("only read-only native Search/read are allowed")

        def validate(body: Any) -> Any:
            if body["status"] != "ok":
                raise ValueError("native operation did not succeed")
            return body["result"]

        return await self.http.request(
            "native",
            method,
            self.base_url + path,
            headers=self.headers(user),
            identity=digest([self.account, user, self.memory_revision]),
            validate=validate,
            **kwargs,
        )

    async def retrieve(self, row: dict[str, Any]) -> dict[str, Any]:
        user = row["user_id"]
        if (
            not isinstance(user, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", user)
            or user in {".", ".."}
        ):
            raise ValueError("invalid user_id")
        question = row["question"]
        scope = "viking://user/" + user + "/memories/"
        plan = await self.planner.rewrite(question)
        union: dict[str, dict[str, Any]] = {}
        lane_sizes = []
        for query in plan["queries"]:
            result = await self.native(
                user,
                "POST",
                "/api/v1/search/search",
                payload={
                    "query": query,
                    "mode": "list",
                    "context_type": "memory",
                    "target_uri": scope.rstrip("/"),
                    "limit": self.lane_limit,
                    "level": 2,
                    "score_threshold": 0,
                    "read_content": False,
                },
            )
            if result.get("resources") or result.get("skills") or result.get("query_plan"):
                raise ValueError("native Search must not route or plan again")
            hits = result["memories"]
            if len(hits) > self.lane_limit or len({hit["uri"] for hit in hits}) != len(hits):
                raise ValueError("native Search count/unique-URI mismatch")
            lane_sizes.append(len(hits))
            for hit in hits:
                uri = hit["uri"]
                if not uri.startswith(scope) or hit["level"] != 2:
                    raise ValueError("native Search returned another user or level")
                abstract = hit.get("abstract", "")
                if not isinstance(abstract, str) or not abstract.strip():
                    raise ValueError("rerank requires a non-empty memory abstract")
                if uri in union and union[uri]["abstract"] != abstract:
                    raise ValueError("memory changed between Search lanes")
                prior_score = union.get(uri, {}).get("vector_score", float("-inf"))
                union[uri] = {
                    "uri": uri,
                    "abstract": abstract,
                    "vector_score": max(prior_score, float(hit["score"])),
                }
        candidates = list(union.values())
        for start in range(0, len(candidates), 100):
            batch = candidates[start : start + 100]
            scores = await self.reranker.rank(question, [item["abstract"] for item in batch])
            if len(scores) != len(batch) or not all(math.isfinite(s) for s in scores):
                raise ValueError("rerank did not score every candidate")
            for item, score in zip(batch, scores, strict=True):
                item["rerank_score"] = score
        # Stable global sort, after ALL batches; never take each batch's top-k.
        ranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
        selected = [item for item in ranked if item["rerank_score"] > 0][: self.topk]
        contexts = []
        for item in selected:
            content = await self.native(
                user,
                "GET",
                "/api/v1/content/read",
                params={
                    "uri": item["uri"],
                    "raw": "false",
                },
            )
            if not isinstance(content, str) or not content.strip():
                raise ValueError("selected memory has no readable body")
            contexts.append(format_search_content(item["uri"], content))
        return {
            **row,
            "retrieved_context": contexts,
            "retrieved_uris": [item["uri"] for item in selected],
            "retrieval_trace": {
                "planner": plan,
                "lane_sizes": lane_sizes,
                "candidates": candidates,
            },
        }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.questions)
    if any(row.get("unit") != "locomo_refined" for row in rows):
        raise ValueError("this experiment accepts only locomo_refined scoped questions")
    # Native per-lane rerank must be off: final union rerank is the only rerank.
    native_config = json.loads(args.ov_config.read_text(encoding="utf-8"))
    if native_config.get("rerank"):
        raise ValueError("use a dedicated OV config without a rerank block")
    if hashlib.sha256(args.planner_prompt.read_bytes()).hexdigest() != PROMPT_SHA256:
        raise ValueError("planner prompt hash mismatch")
    manifest = {
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "account": args.account,
        "memory_revision": args.memory_revision,
        "ov_base_url": args.ov_base_url,
        "planner_base_url": os.environ["PLANNER_API_BASE"],
        "planner_prompt_sha256": PROMPT_SHA256,
        "lane_limit": args.lane_limit,
        "topk": args.topk,
        "rerank_host": args.rerank_host,
        "rerank_model": "doubao-seed-rerank",
        "rerank_version": "251028",
        "rerank_batch_size": 100,
        "owner_labels": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "search-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("output directory belongs to different search conditions")
    write_json(manifest_path, manifest)
    from openviking.models.rerank.volcengine_rerank import RerankClient

    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30), trust_env=False) as client:
        http = CachedHTTP(client, args.output_dir / "cache")
        planner = QueryPlanner(
            http, os.environ["PLANNER_API_BASE"], os.environ["PLANNER_API_KEY"], args.planner_prompt
        )
        signer = RerankClient(
            ak=os.environ["RERANK_AK"], sk=os.environ["RERANK_SK"], host=args.rerank_host
        )
        pipeline = SearchPipeline(
            http,
            planner,
            VikingDBReranker(http, signer),
            base_url=args.ov_base_url,
            api_key=os.environ["OV_API_KEY"],
            account=args.account,
            memory_revision=args.memory_revision,
            lane_limit=args.lane_limit,
            topk=args.topk,
        )
        write_json(
            args.output_dir / "search-status.json", {"status": "running", "total": len(rows)}
        )
        try:
            results = await _bounded_map(rows, args.concurrency, pipeline.retrieve)
        except Exception:
            write_json(
                args.output_dir / "search-status.json",
                {"status": "failed", "http": dict(http.counts)},
            )
            raise
        temporary = args.output_dir / "retrieval.jsonl.tmp"
        _write_jsonl(temporary, results)
        temporary.replace(args.output_dir / "retrieval.jsonl")
        summary = {
            "status": "completed",
            "search": len(results),
            "new_adds": 0,
            "http": dict(http.counts),
        }
        write_json(args.output_dir / "search-status.json", summary)
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ov-base-url", required=True)
    parser.add_argument("--ov-config", required=True, type=Path)
    parser.add_argument("--account", required=True)
    parser.add_argument(
        "--memory-revision", required=True, help="identifier for an unchanged memory snapshot"
    )
    parser.add_argument("--planner-prompt", required=True, type=Path)
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument("--lane-limit", type=_positive_int, default=200)
    parser.add_argument("--topk", type=_positive_int, default=20)
    parser.add_argument("--rerank-host", default="api-vikingdb.vikingdb.cn-beijing.volces.com")
    args = parser.parse_args(argv)
    try:
        with run_lock(args.output_dir):
            print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))
    except (httpx.HTTPError, OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
