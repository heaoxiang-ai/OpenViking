# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Input-guard tests for Volcengine sparse-capable embedding modes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking.models.embedder.base import embed_compat
from openviking.models.embedder.volcengine_embedders import (
    VolcengineDenseEmbedder,
    VolcengineHybridEmbedder,
    VolcengineSparseEmbedder,
)

_MULTIMODAL_INPUT = [
    {"type": "text", "text": "diagram of a heat exchanger"},
    {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
    {"type": "video_url", "video_url": {"url": "https://example.test/video.mp4"}},
]


@patch("openviking.models.embedder.volcengine_embedders.volcenginesdkarkruntime.Ark")
def test_sparse_embedder_downgrades_multimodal_input_to_text(_mock_ark):
    embedder = VolcengineSparseEmbedder(model_name="test", api_key="test-key")

    assert embedder.supports_multimodal is False
    assert embedder.prepare_embedding_input(_MULTIMODAL_INPUT) == "diagram of a heat exchanger"


@patch("openviking.models.embedder.volcengine_embedders.volcenginesdkarkruntime.Ark")
def test_hybrid_embedder_preserves_media_for_dense_channel(_mock_ark):
    embedder = VolcengineHybridEmbedder(model_name="test", api_key="test-key")

    assert embedder.supports_multimodal is True
    assert embedder.prepare_embedding_input(_MULTIMODAL_INPUT) == _MULTIMODAL_INPUT


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("content", "expected_requests"),
    [
        (
            "original text",
            [
                {
                    "input": [{"type": "text", "text": "original text"}],
                    "sparse_embedding": {"type": "enabled"},
                }
            ],
        ),
        (
            [{"type": "text", "text": "original text"}],
            [
                {
                    "input": [{"type": "text", "text": "original text"}],
                    "sparse_embedding": {"type": "enabled"},
                }
            ],
        ),
        (
            _MULTIMODAL_INPUT,
            [
                {"input": _MULTIMODAL_INPUT},
                {
                    "input": [{"type": "text", "text": "diagram of a heat exchanger"}],
                    "sparse_embedding": {"type": "enabled"},
                },
            ],
        ),
        (_MULTIMODAL_INPUT[1:], [{"input": _MULTIMODAL_INPUT[1:]}]),
        (
            [{"type": "text", "text": " \n"}, *_MULTIMODAL_INPUT[1:]],
            [{"input": [{"type": "text", "text": " \n"}, *_MULTIMODAL_INPUT[1:]]}],
        ),
    ],
    ids=["text", "text-array", "mixed", "media-only", "blank-text-and-media"],
)
@patch("openviking.models.embedder.volcengine_embedders.volcenginesdkarkruntime.Ark")
async def test_hybrid_routes_dense_media_and_text_sparse_independently(
    mock_ark, asynchronous, content, expected_requests
):
    responses = [
        SimpleNamespace(
            data=SimpleNamespace(
                embedding=[1.0, 0.0], sparse_embedding=[{"index": 7, "value": 0.5}]
            ),
            usage=None,
        ),
        # The text-only dense vector must never replace the multimodal vector.
        SimpleNamespace(
            data=SimpleNamespace(
                embedding=[0.0, 1.0], sparse_embedding=[{"index": 9, "value": 0.8}]
            ),
            usage=None,
        ),
    ]
    client = MagicMock()
    create = AsyncMock(side_effect=responses) if asynchronous else MagicMock(side_effect=responses)
    client.multimodal_embeddings.create = create
    mock_ark.return_value = client
    embedder = VolcengineHybridEmbedder(
        model_name="existing-model", api_key="test-key", dimension=2
    )
    embedder._get_async_client = lambda: client
    if asynchronous:
        result = await embed_compat(embedder, content, is_query=True)
    else:
        result = embedder.embed(embedder.prepare_embedding_input(content), is_query=True)
    assert result.dense_vector == [1.0, 0.0]
    sparse_call = next(
        (i for i, request in enumerate(expected_requests) if "sparse_embedding" in request), None
    )
    assert result.sparse_vector == (
        None if sparse_call is None else {"7": 0.5} if sparse_call == 0 else {"9": 0.8}
    )
    assert [call.kwargs for call in create.call_args_list] == [
        {"model": "existing-model", "extra_headers": embedder.extra_headers, **request}
        for request in expected_requests
    ]


@patch("openviking.models.embedder.volcengine_embedders.volcenginesdkarkruntime.Ark")
def test_hybrid_text_input_mode_keeps_multimodal_guard(mock_ark):
    embedder = VolcengineHybridEmbedder(model_name="test", api_key="test-key", input_type="text")
    assert embedder.supports_multimodal is False
    assert embedder.prepare_embedding_input(_MULTIMODAL_INPUT) == "diagram of a heat exchanger"


@patch("openviking.models.embedder.volcengine_embedders.volcenginesdkarkruntime.Ark")
@patch("openviking.models.embedder.volcengine_embedders.time.perf_counter")
def test_dense_embedder_passes_local_provider_duration_to_metrics(
    mock_perf_counter, mock_ark, monkeypatch
):
    """Embedding call duration must come from this SDK request, not shared embedder state."""
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=3, total_tokens=3),
        data=[SimpleNamespace(embedding=[1.0, 0.0])],
    )
    client = MagicMock()
    client.embeddings.create.return_value = response
    mock_ark.return_value = client
    monkeypatch.setattr(
        "openviking.models.embedder.volcengine_embedders.get_current_telemetry", MagicMock()
    )
    mock_perf_counter.side_effect = [100.0, 100.25]

    embedder = VolcengineDenseEmbedder(
        model_name="test", api_key="test-key", input_type="text", dimension=2
    )
    embedder.update_token_usage = MagicMock()

    embedder.embed("hello")

    assert embedder.update_token_usage.call_args.kwargs["duration_seconds"] == 0.25
