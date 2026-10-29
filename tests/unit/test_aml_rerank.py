import json

import httpx
import pytest

from benchmark.aml.rerank import RerankError, VikingDBReranker, align_scores


def test_provider_relevance_order_is_mapped_back_to_input_documents():
    assert align_scores(
        {"result": {"data": [{"id": 1, "score": 0.9}, {"id": 0, "score": 0.2}]}}, 2
    ) == [0.2, 0.9]


@pytest.mark.parametrize(
    "items",
    [
        [{"id": 0, "score": 0.8}, {"id": 0, "score": 0.1}],
        [{"id": 0, "score": 0.8}, {"id": 2, "score": 0.1}],
        [{"id": 0, "score": 0.8}],
        [{"id": 0, "score": 0.8}, {"id": 1, "score": float("nan")}],
        [{"id": True, "score": 0.8}, {"id": 0, "score": 0.1}],
    ],
)
def test_invalid_provider_mapping_fails_instead_of_silently_reordering(items):
    with pytest.raises(RerankError):
        align_scores({"result": {"data": items}}, 2)


def make_ranker(tmp_path):
    config = tmp_path / "rerank.json"
    config.write_text(json.dumps({"ak": "test", "sk": "test"}))
    return VikingDBReranker(str(config))


async def test_all_batches_are_scored_before_global_selection(tmp_path, monkeypatch):
    ranker = make_ranker(tmp_path)
    batches = []

    async def batch(query, documents):
        assert query == "question"
        batches.append(list(documents))
        return [int(text) / 1000 for text in documents]

    monkeypatch.setattr(ranker, "_batch", batch)
    hits = [{"uri": str(i), "abstract": str(i)} for i in range(205)]
    ranked = await ranker.rank("question", hits)
    assert list(map(len, batches)) == [100, 100, 5]
    assert [h["uri"] for h in ranked[:20]] == [str(i) for i in range(204, 184, -1)]


async def test_throttle_retries_preserve_id_mapping_and_resign(tmp_path, monkeypatch):
    ranker = make_ranker(tmp_path)
    clock = [0.0]
    attempts = []
    signed = []
    real_client = httpx.AsyncClient

    def handle(request):
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(
            200, json={"result": {"data": [{"id": 1, "score": 0.9}, {"id": 0, "score": 0.2}]}}
        )

    def sign(payload):
        signed.append(clock[0])
        return {}, json.dumps(payload)

    async def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(ranker, "_signed_headers", sign)
    monkeypatch.setattr("benchmark.aml.rerank.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("benchmark.aml.rerank.asyncio.sleep", sleep)
    monkeypatch.setattr(
        "benchmark.aml.rerank.httpx.AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handle)),
    )
    assert await ranker._batch("question", ["A", "B"]) == [0.2, 0.9]
    assert attempts[0] == attempts[1]
    assert len(signed) == 2 and signed[1] >= 65
