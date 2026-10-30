# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Real backend mapping and extraction decisions with a scripted SDK transport."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from openviking.models.vlm.backends.volcengine_vlm import VolcEngineVLM
from openviking.models.vlm.base import VLMResponse
from openviking.session.memory.extraction_errors import MemoryExtractionError
from tests.session.memory.test_extract_loop_budget_recovery import (
    ScriptedVLM,
    _make_loop,
    _run_loop,
)


def response(content=None, *, finish="stop", reasoning=None, moderation=None):
    return SimpleNamespace(
        id="provider-response-1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, reasoning_content=reasoning, tool_calls=None
                ),
                finish_reason=finish,
                moderation_hit_type=moderation,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=7),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=9),
        ),
    )


def backend(responses):
    calls = []
    scripted = iter(responses)

    async def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return next(scripted)

    vlm = VolcEngineVLM({"provider": "volcengine", "model": "synthetic", "max_tokens": 16384})
    vlm.get_async_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return vlm, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        (response(finish="content_filter"), "content_filter"),
        (response("sdk.commit()", finish="content_filter"), "content_filter"),
        (response("sdk.commit()", moderation="severe_violation"), "content_filter"),
        (response(reasoning="PRIVATE_SYNTHETIC_THOUGHT"), "reasoning_only"),
        (response("<think>PRIVATE_SYNTHETIC_THOUGHT</think>"), "reasoning_only"),
        (response("I cannot help with that request."), "refusal_text"),
        (response("sdk.commit()", finish="length"), "budget_exhausted"),
    ],
)
async def test_non_retryable_replies_stop_before_writes_and_keep_safe_diagnostics(raw, kind):
    expected_calls = 1 if kind in {"content_filter", "refusal_text"} else 3
    vlm, calls = backend([raw] * expected_calls)
    loop = _make_loop(vlm)
    with patch("openviking.session.memory.extract_loop.tracer.set") as trace:
        with pytest.raises(MemoryExtractionError) as error:
            await _run_loop(loop)
    assert error.value.kind == kind
    assert error.value.retryable is False
    assert len(calls) == expected_calls
    loop.finalize_operations.assert_not_awaited()
    failure = error.value.task_failure()
    diagnostics = failure["diagnostics"]
    assert diagnostics["raw_content_length"] == len(raw.choices[0].message.content or "")
    assert diagnostics["response_id"] == "provider-response-1"
    assert diagnostics["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "reasoning_tokens": 9,
        "cached_tokens": 7,
    }
    assert "PRIVATE_SYNTHETIC_THOUGHT" not in json.dumps(failure)
    diagnostic_events = [
        call.args[1]
        for call in trace.call_args_list
        if call.args[0] == "memory.extraction.response_diagnostics"
    ]
    assert len(diagnostic_events) == expected_calls
    assert "PRIVATE_SYNTHETIC_THOUGHT" not in diagnostic_events[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("content", [None, " \n\t"])
async def test_actual_empty_reply_retries_twice_with_identical_input(iterations, recover, content):
    vlm, calls = backend(
        [response(content), response(content), response("sdk.commit()" if recover else content)]
    )
    loop = _make_loop(vlm)
    loop.max_iterations = iterations
    if recover:
        operations, _ = await _run_loop(loop)
        assert not operations.upsert_operations
        loop.finalize_operations.assert_awaited_once()
    else:
        with pytest.raises(MemoryExtractionError) as error:
            await _run_loop(loop)
        assert error.value.kind == "empty_response"
        assert error.value.retryable is False
        assert error.value.diagnostics["response_retries_exhausted"] is True
        assert error.value.diagnostics["response_retry_count"] == 2
        loop.finalize_operations.assert_not_awaited()
    assert len(calls) == 3
    first, second, third = copy.deepcopy(calls)
    # Each transport attempt keeps its own diagnostic ID; model input/budget stay identical.
    assert first["extra_headers"].pop("X-Client-Request-Id") != second["extra_headers"].pop(
        "X-Client-Request-Id"
    )
    third["extra_headers"].pop("X-Client-Request-Id")
    assert first == second == third


@pytest.mark.asyncio
async def test_missing_provider_finish_reason_remains_unknown_in_diagnostics():
    vlm, _ = backend([response(finish=None)] * 3)
    loop = _make_loop(vlm)
    with pytest.raises(MemoryExtractionError) as error:
        await _run_loop(loop)
    assert error.value.diagnostics["finish_reason"] is None
    assert error.value.diagnostics["finish_reason_present"] is False


@pytest.mark.asyncio
async def test_parse_failure_after_repair_is_non_retryable():
    vlm = ScriptedVLM(["missing.update(content='x')"] * 2)
    loop = _make_loop(vlm)
    with pytest.raises(MemoryExtractionError) as error:
        await _run_loop(loop)
    assert len(vlm.calls) == 2
    assert error.value.kind == "parse_error"
    assert error.value.retryable is False
    loop.finalize_operations.assert_not_awaited()


@pytest.mark.asyncio
async def test_response_retry_budget_stays_bounded_after_format_repair():
    vlm = ScriptedVLM(["missing.update(content='x')", *[VLMResponse(content="")] * 3])
    loop = _make_loop(vlm)
    with pytest.raises(MemoryExtractionError) as error:
        await _run_loop(loop)
    assert error.value.kind == "empty_response"
    assert len(vlm.calls) == 4
    assert error.value.diagnostics["response_retry_count"] == 2


@pytest.mark.asyncio
async def test_cleanup_empty_retries_then_stops_without_writes():
    vlm = ScriptedVLM(
        [
            VLMResponse(
                content="",
                response_metadata={
                    "raw_content_has_text": True,
                    "raw_content_length": 12,
                },
            )
        ]
        * 3
    )
    loop = _make_loop(vlm)
    with pytest.raises(MemoryExtractionError) as error:
        await _run_loop(loop)
    assert error.value.kind == "cleanup_empty"
    assert error.value.retryable is False
    assert len(vlm.calls) == 3


@pytest.mark.asyncio
async def test_provider_safety_exception_is_terminal_and_has_no_raw_error_text():
    from openviking.models.model_error import ModelError, ModelErrorKind

    vlm = ScriptedVLM([])
    vlm.get_completion_async = AsyncMock(
        side_effect=ModelError(
            ModelErrorKind.CONTENT_SAFETY,
            "PRIVATE_PROVIDER_ERROR",
            status_code=400,
        )
    )
    loop = _make_loop(vlm)
    with pytest.raises(MemoryExtractionError) as error:
        await _run_loop(loop)
    vlm.get_completion_async.assert_awaited_once()
    assert error.value.kind == "content_filter"
    assert error.value.retryable is False
    assert error.value.diagnostics["provider_status_code"] == 400
    assert "PRIVATE_PROVIDER_ERROR" not in str(error.value)
    assert "PRIVATE_PROVIDER_ERROR" not in json.dumps(error.value.task_failure())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        response(reasoning="PRIVATE_THOUGHT"),
        response("sdk.commit()", finish="length"),
    ],
)
async def test_unusable_response_recovers_on_third_same_input_call(raw):
    vlm, calls = backend([raw, raw, response("sdk.commit()")])
    loop = _make_loop(vlm)
    operations, _ = await _run_loop(loop)
    assert operations is not None
    assert len(calls) == 3
    for call in calls:
        call["extra_headers"].pop("X-Client-Request-Id")
    assert calls[0] == calls[1] == calls[2]
    loop.finalize_operations.assert_awaited_once()


@pytest.mark.asyncio
async def test_response_kind_changes_do_not_reset_shared_retry_budget():
    vlm, calls = backend(
        [
            response(reasoning="PRIVATE_THOUGHT"),
            response("sdk.commit()", finish="length"),
            response(),
        ]
    )
    loop = _make_loop(vlm)
    with pytest.raises(MemoryExtractionError) as failure:
        await _run_loop(loop)
    assert len(calls) == 3
    assert failure.value.kind == "empty_response"
    assert failure.value.diagnostics["response_retry_count"] == 2
    assert failure.value.diagnostics["response_retries_exhausted"] is True
    loop.finalize_operations.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_fault_during_response_recovery_is_not_a_skipped_output():
    from openviking.models.model_error import ModelError, ModelErrorKind

    vlm = ScriptedVLM([])
    error = ModelError(
        ModelErrorKind.AUTHENTICATION_ERROR, "synthetic auth failure", status_code=401
    )
    vlm.get_completion_async = AsyncMock(side_effect=[VLMResponse(content=""), error])
    loop = _make_loop(vlm)
    with pytest.raises(ModelError):
        await _run_loop(loop)
    assert vlm.get_completion_async.await_count == 2
    loop.finalize_operations.assert_not_awaited()
