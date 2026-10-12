"""Resumable mini Answer/Judge using the pinned, unchanged public AML prompts."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from benchmark.aml.eval.cached_http import CachedHTTP, completion_text, run_lock, write_json
from benchmark.aml.eval.evaluate import DEFAULT_AML_COMMIT, _read_jsonl, _verify_pipeline
from benchmark.aml.eval.run_public import _positive_int, _write_jsonl

MODEL = "gpt-4o-mini-2024-07-18"
PIPELINE_SHA256 = "31a49fc09e381bc117b59063c7a4cea543a672b23e1264aedb03290239671c1a"
PROMPT_NAMES = {
    "OPEN_ENDED_ANSWER_TEMPLATE",
    "ACCURACY_PROMPT",
    "memory_text",
    "render_answer_prompt",
    "gold_answer",
    "render_accuracy_prompt",
    "parse_judge_label",
}


def load_prompts(repo: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path, info = _verify_pipeline(repo, "data/locomo-refined/pipeline.py", DEFAULT_AML_COMMIT)
    if info["sha256"] != PIPELINE_SHA256:
        raise ValueError("public AML pipeline SHA mismatch")
    # Load only the seven pure prompt/scoring definitions. In particular,
    # never import api_config.py, execute its model client, or change its prompt.
    nodes = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name in PROMPT_NAMES
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in PROMPT_NAMES for target in node.targets
        )
    ]
    if len(nodes) != len(PROMPT_NAMES):
        raise ValueError("public prompt definitions are incomplete")
    namespace: dict[str, Any] = {"json": json, "re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, info


def model_payload(
    prompts: dict[str, Any], stage: str, row: dict, answer: str | None = None
) -> dict:
    if stage == "answer":
        # Gold, evidence, planner traces, URIs, and source annotations are never
        # supplied to the Answer model. Person headers are inside the contents.
        safe = {
            key: row[key]
            for key in (
                "question",
                "retrieved_context",
                "speaker_1_name",
                "speaker_2_name",
            )
            if key in row
        }
        prompt = prompts["render_answer_prompt"](safe)
    elif stage == "judge":
        prompt = prompts["render_accuracy_prompt"](
            {"question": row["question"], "gold_answer": row["gold_answer"]},
            answer,
        )
    else:
        raise ValueError("unknown model stage")
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 16384,
    }


class MiniEvaluator:
    def __init__(
        self,
        http: CachedHTTP,
        prompts: dict[str, Any],
        *,
        endpoints: dict[str, dict[str, Any]],
        answer_concurrency: int = 48,
        judge_concurrency: int = 48,
    ):
        self.http = http
        self.prompts = prompts
        self.endpoints = endpoints
        self.pools = {
            "answer": asyncio.Semaphore(answer_concurrency),
            "judge": asyncio.Semaphore(judge_concurrency),
        }

    async def complete(self, stage: str, row: dict, answer: str | None = None) -> str:
        endpoint = self.endpoints[stage]

        def validate(body: Any) -> str:
            text = completion_text(body, MODEL)
            if stage == "judge":
                self.prompts["parse_judge_label"](text)
            return text

        async with self.pools[stage]:
            return await self.http.request(
                stage,
                "POST",
                endpoint["url"],
                headers=endpoint["headers"],
                params=endpoint.get("params"),
                payload=model_payload(self.prompts, stage, row, answer),
                validate=validate,
            )

    async def answer(self, row: dict) -> str:
        return await self.complete("answer", row)

    async def judge(self, row: dict, answer: str) -> str:
        return await self.complete("judge", row, answer)


def endpoint_from_environment(stage: str) -> dict[str, Any]:
    prefix = stage.upper()
    version = os.getenv(prefix + "_API_VERSION")
    key = os.environ[prefix + "_API_KEY"]
    return {
        "url": os.environ[prefix + "_API_BASE"].rstrip("/") + "/chat/completions",
        "headers": {"api-key": key} if version else {"Authorization": "Bearer " + key},
        "params": {"api-version": version} if version else None,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.retrieval)
    if any(row.get("unit") != "locomo_refined" for row in rows):
        raise ValueError("this evaluator accepts only LoCoMo-Refined")
    prompts, pipeline = load_prompts(args.aml_repo)
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("invalid question")
        if not isinstance(row.get("retrieved_context"), list) or any(
            not isinstance(text, str) for text in row["retrieved_context"]
        ):
            raise ValueError("invalid retrieved_context")
        prompts["gold_answer"](row)
    endpoints = {stage: endpoint_from_environment(stage) for stage in ("answer", "judge")}
    manifest = {
        "retrieval_sha256": hashlib.sha256(args.retrieval.read_bytes()).hexdigest(),
        "pipeline": pipeline,
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 16384,
        "answer_concurrency": args.answer_concurrency,
        "judge_concurrency": args.judge_concurrency,
        "endpoints": {
            stage: {key: value for key, value in endpoint.items() if key != "headers"}
            for stage, endpoint in endpoints.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "eval-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("output directory belongs to different evaluation conditions")
    write_json(manifest_path, manifest)
    status = {"status": "running", "total": len(rows), "answer": 0, "judge": 0}

    def progress() -> None:
        write_json(args.output_dir / "eval-status.json", status)

    progress()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600, connect=30),
        trust_env=False,
        limits=httpx.Limits(max_connections=args.answer_concurrency + args.judge_concurrency),
    ) as client:
        http = CachedHTTP(client, args.output_dir / "model-cache", attempts=12)
        evaluator = MiniEvaluator(
            http,
            prompts,
            endpoints=endpoints,
            answer_concurrency=args.answer_concurrency,
            judge_concurrency=args.judge_concurrency,
        )

        async def one(row: dict) -> dict:
            answer = await evaluator.answer(row)
            status["answer"] += 1
            progress()
            judgment = await evaluator.judge(row, answer)
            label = prompts["parse_judge_label"](judgment)
            result = {
                "id": row["id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "question_type": row.get("question_type", row.get("category")),
                "generated_answer": answer,
                "judge_response": judgment,
                "label": label,
                "is_correct": label == "CORRECT",
            }
            status["judge"] += 1
            progress()
            return result

        tasks = [asyncio.create_task(one(row)) for row in rows]
        try:
            results = list(await asyncio.gather(*tasks))
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            status.update(status="failed", http=dict(http.counts))
            progress()
            raise
        _write_jsonl(
            args.output_dir / "answers.jsonl",
            [{"id": row["id"], "generated_answer": row["generated_answer"]} for row in results],
        )
        _write_jsonl(args.output_dir / "judgments.jsonl", results)
        by_category: dict[str, dict[str, int]] = {}
        for row in results:
            category = by_category.setdefault(str(row["question_type"]), {"total": 0, "correct": 0})
            category["total"] += 1
            category["correct"] += int(row["is_correct"])
        correct = sum(row["is_correct"] for row in results)
        summary = {
            "status": "completed",
            "total": len(results),
            "correct": correct,
            "accuracy": correct / len(results),
            "by_category": by_category,
            "model": MODEL,
            "pipeline": pipeline,
            "http": dict(http.counts),
        }
        write_json(args.output_dir / "summary.json", summary)
        status.update(status="completed", http=dict(http.counts))
        progress()
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--aml-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--answer-concurrency", default=48, type=_positive_int)
    parser.add_argument("--judge-concurrency", default=48, type=_positive_int)
    args = parser.parse_args(argv)
    try:
        with run_lock(args.output_dir):
            print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))
    except (httpx.HTTPError, OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
