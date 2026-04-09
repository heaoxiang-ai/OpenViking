from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_build_session_messages_maps_haystack_sessions_and_dates():
    module = _load_module(
        "longmemeval_import_to_ov",
        "benchmark/longmemeval/vikingbot/import_to_ov.py",
    )
    item = {
        "question_id": "qid-1",
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 03:24"],
        "haystack_session_ids": ["sess-a", "sess-b"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            [
                {"role": "user", "content": "Degree?"},
            ],
        ],
    }

    sessions = module.build_session_messages(item)

    assert len(sessions) == 2
    assert sessions[0]["meta"]["sample_id"] == "qid-1"
    assert sessions[0]["meta"]["session_key"] == "sess-a"
    assert sessions[0]["meta"]["date_time"] == "2023/05/20 (Sat) 02:21"
    assert sessions[0]["messages"] == [
        {"role": "user", "text": "Hi", "index": 0},
        {"role": "assistant", "text": "Hello", "index": 1},
    ]
    assert sessions[1]["meta"]["session_key"] == "sess-b"


def test_load_longmemeval_qa_extracts_question_answer_and_date(tmp_path: Path):
    module = _load_module(
        "longmemeval_run_eval",
        "benchmark/longmemeval/vikingbot/run_eval.py",
    )
    data = [
        {
            "question_id": "qid-1",
            "question": "What degree did I graduate with?",
            "answer": "Business Administration",
            "question_date": "2023/05/30 (Tue) 23:40",
        },
        {
            "question_id": "qid-2",
            "question": "What tracker did I buy?",
            "answer": "Fitbit Inspire HR",
            "question_date": "2023/02/16 (Thu) 09:10",
        },
    ]
    input_path = tmp_path / "longmemeval.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")

    qa_list = module.load_longmemeval_qa(str(input_path), sample_index=1)

    assert qa_list == [
        {
            "sample_id": "qid-2",
            "question": "What tracker did I buy?",
            "answer": "Fitbit Inspire HR",
            "question_time": "2023-02-16",
            "question_type": "",
            "evidence": [],
        }
    ]


def test_parse_longmemeval_datetime_returns_iso_date():
    module = _load_module(
        "longmemeval_run_eval",
        "benchmark/longmemeval/vikingbot/run_eval.py",
    )

    parsed = module.parse_longmemeval_datetime("2023/05/30 (Tue) 23:40")

    assert parsed.strftime("%Y-%m-%d") == "2023-05-30"


def test_build_full_eval_steps_supports_skip_import():
    module = _load_module(
        "longmemeval_run_full_eval",
        "benchmark/longmemeval/vikingbot/run_full_eval.py",
    )

    steps = module.build_steps(
        python_executable="/usr/bin/python3",
        input_path="/tmp/longmemeval.json",
        output_path="/tmp/result.csv",
        skip_import=True,
    )

    assert [step["name"] for step in steps] == ["eval", "judge", "stats"]
    assert steps[0]["cmd"] == [
        "/usr/bin/python3",
        "benchmark/longmemeval/vikingbot/run_eval.py",
        "/tmp/longmemeval.json",
        "--output",
        "/tmp/result.csv",
        "--threads",
        "20",
    ]


def test_build_sample_agent_id_uses_per_sample_namespace():
    module = _load_module(
        "longmemeval_import_to_ov",
        "benchmark/longmemeval/vikingbot/import_to_ov.py",
    )

    shared = module.build_sample_agent_id("sample-1", "shared")
    per_sample = module.build_sample_agent_id("sample-1", "per-sample")
    per_sample_again = module.build_sample_agent_id("sample-1", "per-sample")
    other_sample = module.build_sample_agent_id("sample-2", "per-sample")

    assert shared == "default"
    assert per_sample.startswith("lm_")
    assert per_sample == per_sample_again
    assert per_sample != other_sample


@pytest.mark.asyncio
async def test_run_import_deferred_submits_before_waiting(monkeypatch):
    module = _load_module(
        "longmemeval_import_to_ov",
        "benchmark/longmemeval/vikingbot/import_to_ov.py",
    )

    item = {
        "question_id": "qid-1",
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 03:24"],
        "haystack_session_ids": ["sess-a", "sess-b"],
        "haystack_sessions": [
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "Bye"}],
        ],
    }

    event_log: list[tuple[str, str, str]] = []
    records: list[dict] = []

    async def fake_submit(messages, openviking_url, semaphore, session_time=None, agent_id="default"):
        session_name = messages[0]["text"]
        event_log.append(("submit", session_name, agent_id))
        return {
            "token_usage": None,
            "task_id": f"task-{session_name}",
            "trace_id": "",
            "agent_id": agent_id,
        }

    async def fake_wait(openviking_url, task_id, semaphore, agent_id="default"):
        event_log.append(("wait", task_id, agent_id))
        return {
            "embedding": 1,
            "vlm": 2,
            "llm_input": 3,
            "llm_output": 4,
            "total": 10,
        }

    monkeypatch.setattr(module, "load_longmemeval_data", lambda path, sample_index=None: [item])
    monkeypatch.setattr(module, "load_ingest_record", lambda: {})
    monkeypatch.setattr(module, "load_success_csv", lambda _: set())
    monkeypatch.setattr(module, "save_ingest_record", lambda record: None)
    monkeypatch.setattr(module, "write_error_record", lambda record, error_path: None)
    monkeypatch.setattr(module, "write_success_record", lambda record, csv_path: records.append(record))
    monkeypatch.setattr(module, "is_already_ingested", lambda *args, **kwargs: False)
    monkeypatch.setattr(module, "mark_ingested", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "submit_viking_ingest", fake_submit)
    monkeypatch.setattr(module, "wait_for_viking_task", fake_wait)

    args = SimpleNamespace(
        input="/tmp/longmemeval.json",
        sample=None,
        sessions=None,
        parallel=2,
        clear_ingest_record=False,
        force_ingest=False,
        success_csv="/tmp/success.csv",
        error_log="/tmp/error.log",
        openviking_url="http://localhost:1933",
        wait_mode="deferred",
        agent_id_mode="per-sample",
    )

    await module.run_import(args)

    assert [event[0] for event in event_log] == ["submit", "submit", "wait", "wait"]
    submit_agent_ids = [event[2] for event in event_log if event[0] == "submit"]
    wait_agent_ids = [event[2] for event in event_log if event[0] == "wait"]
    assert len(set(submit_agent_ids)) == 1
    assert submit_agent_ids == wait_agent_ids
    assert len(records) == 2
