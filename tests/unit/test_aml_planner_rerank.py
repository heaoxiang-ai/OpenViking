import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from benchmark.aml.eval.cached_http import CachedHTTP
from benchmark.aml.eval.locomo_mini import MiniEvaluator, load_prompts, model_payload
from benchmark.aml.eval.owner_labels import format_search_content
from benchmark.aml.eval.query_planner import MODEL, parse_queries
from benchmark.aml.eval.search_planner import SearchPipeline, VikingDBReranker, aligned_scores


def completion(text, model=MODEL):
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": text}}]}


class PlannerContractTests(unittest.TestCase):
    def test_routing_and_priority_are_ignored_and_empty_queries_fall_back(self):
        plan = {
            "queries": [
                {"query": "first", "context_type": "resource", "priority": 99},
                {"query": "second", "context_type": "skill", "priority": 1},
                {"query": " first ", "intent": "anything"},
            ]
        }
        self.assertEqual(
            parse_queries(completion(json.dumps(plan)), "original")["queries"], ["first", "second"]
        )
        self.assertEqual(
            parse_queries(completion('{"queries": []}'), "original")["queries"], ["original"]
        )
        with self.assertRaises(ValueError):
            parse_queries(completion('{"queries": ["invalid"]}'), "original")

    def test_score_sorted_rows_are_aligned_to_ids_and_bad_ids_rejected(self):
        self.assertEqual(
            aligned_scores(
                {
                    "result": {
                        "data": [
                            {"id": 1, "score": 0.9},
                            {"id": "0", "score": 0.2},
                        ]
                    }
                },
                2,
            ),
            [0.2, 0.9],
        )
        for bad in (True, 0.5, -1, "missing"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                aligned_scores({"result": {"data": [{"id": bad, "score": 0.8}]}}, 1)
        with self.assertRaises(ValueError):
            aligned_scores({"result": {"data": [{"id": 0, "score": float("nan")}]}}, 1)
        with self.assertRaises(ValueError):
            aligned_scores({"result": {"data": [{"id": 0, "score": 1}, {"id": 0, "score": 1}]}}, 2)

    def test_owner_header_preserves_body_and_distinguishes_event_participants(self):
        content = "ChatLog:\n**user**: [D1:2] Caroline: Hello\n**assistant**: Melanie: Hi"
        self.assertEqual(
            format_search_content("viking://user/u/memories/events/chat.md", content),
            "[记忆归属人: Caroline, Melanie]\n" + content,
        )
        self.assertEqual(
            format_search_content("viking://user/u/memories/preferences/Caroline/diet.md", "text"),
            "[记忆归属人: Caroline]\ntext",
        )

    def test_pinned_real_prompts_exclude_gold_and_evidence_from_answer(self):
        repo = (
            Path(__file__).resolve().parents[2] / "result/aml-public-eval/agent-memory-leaderboard"
        )
        if not repo.exists():
            self.skipTest("pinned public AML checkout is not available")
        prompts, _ = load_prompts(repo)
        row = {
            "question": "Who?",
            "retrieved_context": ["Memory"],
            "gold_answer": "GOLD_CANARY",
            "evidence_dialogue_ids": ["EVIDENCE_CANARY"],
            "retrieval_trace": "TRACE_CANARY",
        }
        answer = model_payload(prompts, "answer", row)
        for canary in ("GOLD_CANARY", "EVIDENCE_CANARY", "TRACE_CANARY"):
            self.assertNotIn(canary, json.dumps(answer))
        self.assertIn("GOLD_CANARY", json.dumps(model_payload(prompts, "judge", row, "response")))


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_union_all_batches_global_top20_only_then_read_and_resume(self):
        seen = []
        scope = "viking://user/u/memories/"

        class Planner:
            async def rewrite(self, _question):
                return {"queries": ["query one", "query two"]}

        class Signer:
            host = "rerank.test"
            model_name = "doubao-seed-rerank"
            model_version = "251028"

            def _prepare_request(self, _method, _path, data):
                class Request:
                    headers = {"X-Test-Signed-Body": json.dumps(data)}

                return Request()

        async def handle(request):
            seen.append(request)
            if request.url.path == "/api/vikingdb/rerank":
                body = json.loads(request.content)
                self.assertEqual(request.content.decode(), request.headers["X-Test-Signed-Body"])
                self.assertEqual(body["query"], [{"text": "original question"}])
                self.assertLessEqual(len(body["data"]), 100)
                scores = [
                    {"id": index, "score": (int(item[0]["text"].split()[-1]) + 1) / 400}
                    for index, item in enumerate(body["data"])
                ]
                return httpx.Response(200, json={"result": {"data": scores[::-1]}})
            self.assertEqual(request.headers["X-OpenViking-User"], "u")
            self.assertEqual(request.headers["X-OpenViking-Account"], "account")
            if request.url.path == "/api/v1/search/search":
                body = json.loads(request.content)
                self.assertEqual(body["mode"], "list")
                self.assertEqual(body["context_type"], "memory")
                self.assertEqual(body["limit"], 200)
                self.assertNotIn("session_id", body)
                numbers = range(200) if body["query"] == "query one" else range(100, 300)
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "result": {
                            "memories": [
                                {
                                    "uri": scope + f"events/{i}.md",
                                    "abstract": f"abstract {i}",
                                    "score": 1.0,
                                    "level": 2,
                                }
                                for i in numbers
                            ]
                        },
                    },
                )
            self.assertEqual(request.url.path, "/api/v1/content/read")
            self.assertTrue(request.url.params["uri"].startswith(scope))
            return httpx.Response(
                200, json={"status": "ok", "result": "ChatLog:\n**user**: Caroline: Hello"}
            )

        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                http = CachedHTTP(client, Path(directory))
                pipeline = SearchPipeline(
                    http,
                    Planner(),
                    VikingDBReranker(http, Signer()),
                    base_url="http://ov.test",
                    api_key="test-only",
                    account="account",
                    memory_revision="fixed",
                )
                row = {
                    "id": "q",
                    "user_id": "u",
                    "question": "original question",
                    "gold_answer": "GOLD_CANARY",
                }
                result = await pipeline.retrieve(row)
                self.assertEqual(len(result["retrieval_trace"]["candidates"]), 300)
                self.assertEqual(
                    result["retrieved_uris"],
                    [scope + f"events/{i}.md" for i in range(299, 279, -1)],
                )
                self.assertEqual(len(result["retrieved_context"]), 20)
                count = len(seen)
                self.assertEqual(await pipeline.retrieve(row), result)
                self.assertEqual(len(seen), count)
                self.assertEqual(
                    sum(request.url.path == "/api/vikingdb/rerank" for request in seen), 3
                )
                self.assertEqual(
                    sum(request.url.path == "/api/v1/content/read" for request in seen), 20
                )
                self.assertNotIn(
                    "GOLD_CANARY", "\n".join(request.content.decode() for request in seen)
                )
                self.assertNotIn(
                    "test-only",
                    "\n".join(path.read_text() for path in Path(directory).rglob("*.json")),
                )

    async def test_another_users_hit_is_rejected_before_rerank_or_read(self):
        class Planner:
            async def rewrite(self, _question):
                return {"queries": ["query"]}

        async def handle(_request):
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "memories": [
                            {
                                "uri": "viking://user/another/memories/file.md",
                                "abstract": "text",
                                "level": 2,
                                "score": 1,
                            },
                        ]
                    },
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                pipeline = SearchPipeline(
                    CachedHTTP(client, Path(directory)),
                    Planner(),
                    None,
                    base_url="http://ov.test",
                    api_key="test",
                    account="a",
                    memory_revision="fixed",
                )
                with self.assertRaisesRegex(ValueError, "another user"):
                    await pipeline.retrieve({"user_id": "u", "question": "question"})

    async def test_valid_wrong_judgment_cached_and_429_retried(self):
        calls = []
        prompts = {
            "render_answer_prompt": lambda row: row["question"] + str(row["retrieved_context"]),
            "render_accuracy_prompt": lambda row, answer: (
                row["question"] + row["gold_answer"] + answer
            ),
            "parse_judge_label": lambda text: json.loads(text)["label"],
        }

        async def handle(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "1"})
            text = "answer" if request.url.path == "/answer" else '{"label": "WRONG"}'
            return httpx.Response(200, json=completion(text, "gpt-4o-mini-2024-07-18"))

        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                http = CachedHTTP(client, Path(directory))

                async def elapsed(_delay):
                    http.cooldown = 0

                evaluator = MiniEvaluator(
                    http,
                    prompts,
                    endpoints={
                        stage: {
                            "url": "http://model.test/" + stage,
                            "headers": {"api-key": "test-only"},
                        }
                        for stage in ("answer", "judge")
                    },
                )
                row = {
                    "question": "question",
                    "retrieved_context": ["memory"],
                    "gold_answer": "GOLD",
                }
                with patch("benchmark.aml.eval.cached_http.asyncio.sleep", elapsed):
                    answer = await evaluator.answer(row)
                judgment = await evaluator.judge(row, answer)
                self.assertEqual(judgment, '{"label": "WRONG"}')
                self.assertEqual(await evaluator.judge(row, answer), judgment)
                self.assertEqual(len(calls), 3)
                self.assertEqual(http.counts["answer_429"], 1)
                self.assertEqual(http.counts["judge_cached"], 1)
                self.assertEqual(evaluator.pools["answer"]._value, 48)
                self.assertEqual(evaluator.pools["judge"]._value, 48)


if __name__ == "__main__":
    unittest.main()
