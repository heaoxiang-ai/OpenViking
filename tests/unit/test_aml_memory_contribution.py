"""Counterfactual scoring, frozen-context isolation and durable model receipts."""

import argparse
import asyncio
import copy
import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from benchmark.aml.eval import memory_contribution as contribution


@pytest.mark.parametrize(
    "game,expected",
    [
        (lambda mask: bool(mask & 1), [100, 0, 0, 0]),
        (lambda mask: bool(mask & 1 or mask & 4), [50, 0, 50, 0]),
        (lambda mask: bool(mask & 1 and mask & 4), [50, 0, 50, 0]),
        (lambda mask: not bool(mask & 2), [0, -100, 0, 0]),
        (lambda mask: 0.4 + 0.1 * mask.bit_count(), [10, 10, 10, 10]),
    ],
)
def test_exact_shapley_distributes_redundancy_synergy_and_negative_effects(game, expected):
    values = [float(game(mask)) for mask in range(16)]
    effects = contribution.contributions(values)
    assert [effects[kind]["shapley_pp"] for kind in contribution.TYPES] == pytest.approx(expected)
    assert sum(effect["shapley_pp"] for effect in effects.values()) == pytest.approx(
        100 * (values[15] - values[0])
    )
    if values[1] == values[4] == 1 and values[0] == 0:
        assert effects["event"]["removal_loss_pp"] == 0
        assert effects["event"]["standalone_gain_pp"] == 100


def _row(ident="q1", **extra):
    return {
        "id": ident,
        "unit": "longmemeval_s",
        "user_id": "history-" + ident,
        "question": "What was remembered?",
        "gold_answer": "yes",
        "question_type": "single-session",
        "question_date": "2023-10-01",
        "speaker_1_name": "Alice",
        "retrieved_context": [
            "EVENT\nexact formatting",
            "ENTITY",
            "PREFERENCE",
            "PROFILE",
            "OTHER",
        ],
        "retrieved_uris": [
            "viking://user/u/memories/events/2023/event.md",
            "viking://user/u/memories/entities/person/alice.md",
            "viking://user/u/memories/preferences/user/topic.md",
            "viking://user/u/memories/profile.md",
            "viking://user/u/memories/soul.md",
        ],
        **extra,
    }


def test_subsets_preserve_text_order_labels_question_and_gold_without_refill():
    row = _row()
    original = copy.deepcopy(row)
    case = contribution._cases([row], None)[0]
    assert case["types"] == ["event", "entity", "preference", "profile", "other"]
    for mask in range(16):
        subset = contribution.subset_row(case, mask)
        assert subset["question_date"] == row["question_date"]
        assert subset["gold_answer"] == row["gold_answer"]
        assert subset["speaker_1_name"] == "Alice"
        expected = [
            text
            for text, kind in zip(row["retrieved_context"], case["types"], strict=True)
            if kind == "other" or mask & (1 << contribution.TYPES.index(kind))
        ]
        assert subset["retrieved_context"] == expected
        assert len(subset["retrieved_uris"]) == len(expected)
    assert contribution.subset_row(case, 0)["retrieved_context"] == ["OTHER"]
    assert contribution.subset_row(case, 15) == original
    assert row == original


def test_type_provenance_from_metadata_and_declared_types():
    row = _row(retrieved_context=['Event text\n<!-- MEMORY_FIELDS\n{"memory_type":"events"}\n-->'])
    del row["retrieved_uris"]
    assert contribution._memory_types(row) == ["event"]
    row = _row(retrieved_memory_types=["events", "entities", "preferences", "profile", "soul"])
    assert contribution._memory_types(row) == ["event", "entity", "preference", "profile", "other"]


@pytest.mark.parametrize(
    "override,remove,message",
    [
        ({"retrieved_uris": []}, None, "align"),
        ({"retrieved_memory_types": ["entity"] * 5}, None, "conflicting"),
        ({}, "retrieved_uris", "no type provenance"),
        ({"speaker_1_memories": "leaked event"}, None, "all Answer memory"),
        ({"speaker_2_memories": "leaked event"}, None, "all Answer memory"),
        ({}, "user_id", "group-field"),
    ],
)
def test_rejects_missing_or_conflicting_provenance_and_prompt_overrides(override, remove, message):
    row = _row(**override)
    if remove:
        del row[remove]
    with pytest.raises(ValueError, match=message):
        contribution._cases([row], None)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repo = tmp_path / "aml"
    repo.mkdir()
    code = """import json
def render_answer_prompt(row):
    return json.dumps({"question": row["question"], "contexts": row["retrieved_context"]})
def render_accuracy_prompt(row, answer):
    return json.dumps({"gold": row["gold_answer"], "generated": answer})
def parse_judge_label(text):
    label = json.loads(text)["label"]
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError("bad label")
    return label
"""
    for relative in ("data/longmemeval-s/pipeline.py", "data/locomo-refined/pipeline.py"):
        path = repo / relative
        path.parent.mkdir(parents=True)
        path.write_text(code)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "test fixture")
    commit = git("rev-parse", "HEAD")
    retrieval = tmp_path / "retrieval.jsonl"
    public_rows = [_row(), _row("q2", question_type="temporal")]
    retrieval.write_text("".join(json.dumps(row) + "\n" for row in public_rows))
    root = tmp_path / "run"
    root.mkdir()
    args = argparse.Namespace(
        output_dir=root,
        retrieval=retrieval,
        unit=None,
        aml_repo=repo,
        aml_commit=commit,
        group_field=None,
        rounds=2,
        answer_model="test-model",
        judge_model="test-model",
        temperature=0,
        max_tokens=16384,
    )
    manifest = contribution.prepare(args)
    for stage in ("ANSWER", "JUDGE"):
        monkeypatch.setenv(stage + "_API_BASE", "https://model.test/v1")
        monkeypatch.setenv(stage + "_API_KEY", "test-secret-not-written")
    return SimpleNamespace(
        repo=repo, root=root, manifest=manifest, args=args, source_bytes=retrieval.read_bytes()
    )


def _model_handler(calls):
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload["temperature"] == 0
        assert payload["max_tokens"] == 16384
        prompt = json.loads(payload["messages"][0]["content"])
        if "contexts" in prompt:
            assert "gold" not in prompt
            text = (
                "yes"
                if any("EVENT" in text or "ENTITY" in text for text in prompt["contexts"])
                else "no"
            )
        else:
            text = json.dumps(
                {"label": "CORRECT" if prompt["gold"] == prompt["generated"] else "WRONG"}
            )
        return httpx.Response(
            200,
            json={
                "id": f"response-{len(calls)}",
                "choices": [{"finish_reason": "stop", "message": {"content": text}}],
                "usage": {"prompt_tokens": 5},
            },
        )

    return handle


@pytest.mark.asyncio
async def test_end_to_end_all_subsets_rounds_and_resume_keep_first_responses(prepared):
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_model_handler(calls))) as client:
        await contribution.run(prepared.root, prepared.repo, client=client)
        assert len(calls) == 2 * 16 * 2 * 2
        await contribution.run(prepared.root, prepared.repo, client=client)
        assert len(calls) == 2 * 16 * 2 * 2
    summary = contribution.report(prepared.root, bootstrap_samples=25, seed=7)
    assert summary["overall"]["total_gain_pp"] == 100
    assert summary["overall"]["contributions"]["event"]["shapley_pp"] == pytest.approx(50)
    assert summary["overall"]["contributions"]["entity"]["shapley_pp"] == pytest.approx(50)
    assert summary["overall"]["contributions"]["event"]["removal_loss_pp"] == 0
    assert summary["overall"]["contributions"]["profile"]["shapley_pp"] == 0
    assert summary["overall"]["history_groups"] == 2
    assert set(summary["by_category"]) == {"single-session", "temporal"}
    assert summary == contribution.report(prepared.root, bootstrap_samples=25, seed=7)
    assert prepared.args.retrieval.read_bytes() == prepared.source_bytes
    for path in prepared.root.rglob("*.json"):
        assert "test-secret-not-written" not in path.read_text()
    response_ids = [
        json.loads(path.read_text())["response"]["id"]
        for path in prepared.root.glob("round-*/subset-*/*/answer.json")
    ]
    assert len(response_ids) == len(set(response_ids)) == 64
    receipt = prepared.root / "round-01/subset-15/000000/answer.json"
    record = json.loads(receipt.read_text())
    record["text"] = "modified answer"
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="receipt integrity mismatch"):
        contribution.report(prepared.root, bootstrap_samples=10)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,finish,text",
    [("answer", "length", "partial answer"), ("judge", "stop", "invalid judgment")],
)
async def test_successful_but_invalid_response_is_preserved_and_never_resampled(
    prepared, stage, finish, text
):
    calls = []

    def invalid(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": finish, "message": {"content": text}}]},
        )

    path = prepared.root / "invalid" / f"{stage}.json"
    job = contribution._job(prepared.manifest, 0, 0, 0, stage)
    pipeline, _ = contribution._load_pipeline(
        prepared.repo, "longmemeval_s", prepared.args.aml_commit
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(invalid)) as client:
        for _ in range(2):
            with pytest.raises(ValueError, match="no resampling"):
                await contribution._complete(
                    client,
                    path,
                    job,
                    {"model": "test"},
                    endpoint="https://model.test",
                    key="test",
                    pipeline=pipeline,
                    cooldown=contribution._Cooldown(),
                    attempts=3,
                )
    assert len(calls) == 1
    assert len(list(path.parent.glob("*.attempt-*.json"))) == 1
    assert not path.exists()


@pytest.mark.asyncio
async def test_throttle_retries_preserve_failure_evidence_and_respect_attempt_budget(prepared):
    calls = []
    valid = _model_handler(calls)
    attempts = []

    def throttled(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "limited"})
        return valid(request)

    path = prepared.root / "throttled" / "answer.json"
    job = contribution._job(prepared.manifest, 0, 0, 0, "answer")
    pipeline, _ = contribution._load_pipeline(
        prepared.repo, "longmemeval_s", prepared.args.aml_commit
    )
    request = {
        "messages": [{"content": json.dumps({"contexts": []})}],
        "temperature": 0,
        "max_tokens": 16384,
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(throttled)) as client:
        record = await contribution._complete(
            client,
            path,
            job,
            request,
            endpoint="https://model.test",
            key="test",
            pipeline=pipeline,
            cooldown=contribution._Cooldown(),
            attempts=3,
        )
    assert record["text"] == "no"
    assert len(attempts) == 2
    assert json.loads(path.with_name("answer.attempt-001.json").read_text())["http_status"] == 429


@pytest.mark.asyncio
async def test_resume_honors_the_cooldown_saved_before_interruption(prepared):
    calls = []
    valid = _model_handler(calls)
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "60"}, json={"error": "limited"})
        return valid(request)

    class Interrupted(contribution._Cooldown):
        async def wait(self):
            if self.until > contribution.time.monotonic():
                raise asyncio.CancelledError

    class ObserveResume(contribution._Cooldown):
        async def wait(self):
            assert self.until > contribution.time.monotonic() + 30

    path = prepared.root / "interrupted" / "answer.json"
    job = contribution._job(prepared.manifest, 0, 0, 0, "answer")
    pipeline, _ = contribution._load_pipeline(
        prepared.repo, "longmemeval_s", prepared.args.aml_commit
    )
    request = {
        "messages": [{"content": json.dumps({"contexts": []})}],
        "temperature": 0,
        "max_tokens": 16384,
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(asyncio.CancelledError):
            await contribution._complete(
                client,
                path,
                job,
                request,
                endpoint="https://model.test",
                key="test",
                pipeline=pipeline,
                cooldown=Interrupted(),
                attempts=3,
            )
        await contribution._complete(
            client,
            path,
            job,
            request,
            endpoint="https://model.test",
            key="test",
            pipeline=pipeline,
            cooldown=ObserveResume(),
            attempts=3,
        )
    assert len(requests) == 2


def test_report_refuses_incomplete_data_instead_of_scoring_missing_as_wrong(prepared):
    with pytest.raises(ValueError, match="incomplete evaluation"):
        contribution.report(prepared.root, bootstrap_samples=10)
    assert not (prepared.root / "summary.json").exists()


def test_frozen_inputs_and_pipeline_changes_are_rejected(prepared):
    cases = prepared.root / "cases.json"
    original = cases.read_text()
    cases.write_text(original + " ")
    with pytest.raises(ValueError, match="cases SHA"):
        contribution._load_plan(prepared.root)
    cases.write_text(original)
    pipeline = prepared.repo / "data/longmemeval-s/pipeline.py"
    pipeline.write_text(pipeline.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="local modifications"):
        contribution._load_pipeline(prepared.repo, "longmemeval_s", prepared.args.aml_commit)


def test_paired_cluster_bootstrap_is_deterministic_and_keeps_question_weighting():
    records = [
        {"group_id": "shared", "round_values": [[int(bool(mask & 1)) for mask in range(16)]]},
        {"group_id": "shared", "round_values": [[int(bool(mask & 1)) for mask in range(16)]]},
        {"group_id": "other", "round_values": [[0] * 16]},
    ]
    summary = contribution._aggregate(records, 100, 1)
    assert summary["history_groups"] == 2
    assert summary["contributions"]["event"]["shapley_pp"] == pytest.approx(200 / 3)
    assert summary == contribution._aggregate(records, 100, 1)


def test_lock_blocks_overlapping_runs_and_is_released_after_failure(tmp_path):
    with pytest.raises(ValueError, match="locked"), contribution._lock(tmp_path):
        with contribution._lock(tmp_path):
            pass
    assert not (tmp_path / ".running").exists()
