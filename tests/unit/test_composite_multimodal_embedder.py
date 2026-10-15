"""Hybrid channels keep images in dense input and text in sparse input."""

import pytest

from openviking.models.embedder.base import (
    CompositeHybridEmbedder,
    DenseEmbedderBase,
    EmbedResult,
    SparseEmbedderBase,
    embed_compat,
)


class Dense(DenseEmbedderBase):
    def __init__(self, multimodal=True):
        super().__init__("existing-dense")
        self.multimodal = multimodal
        self.seen = []

    @property
    def supports_multimodal(self):
        return self.multimodal

    def embed(self, content, is_query=False):
        self.seen.append((content, is_query))
        return EmbedResult(dense_vector=[1.0, 0.0])

    async def embed_async(self, content, is_query=False):
        return self.embed(content, is_query=is_query)

    def get_dimension(self):
        return 2


class Sparse(SparseEmbedderBase):
    def __init__(self):
        super().__init__("existing-sparse")
        self.seen = []

    def embed(self, content, is_query=False):
        assert isinstance(content, str)
        self.seen.append((content, is_query))
        return EmbedResult(sparse_vector={"word": 0.7})

    async def embed_async(self, content, is_query=False):
        return self.embed(content, is_query=is_query)

    def get_dimension(self):
        return 0


IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("content", "text"),
    [
        ("original\ncode", "original\ncode"),
        (
            [{"type": "text", "text": "first"}, IMAGE, {"type": "text", "text": "second"}, IMAGE],
            "first\nsecond",
        ),
        ([IMAGE], None),
        ([{"type": "text", "text": " "}, IMAGE], None),
    ],
    ids=["text", "mixed-multiple-images", "image-only", "blank-text"],
)
async def test_composite_routes_modalities_to_independent_channels(asynchronous, content, text):
    dense, sparse = Dense(), Sparse()
    embedder = CompositeHybridEmbedder(dense, sparse)
    assert embedder.supports_multimodal is True
    if asynchronous:
        result = await embed_compat(embedder, content, is_query=True)
    else:
        result = embedder.embed(embedder.prepare_embedding_input(content), is_query=True)
    assert dense.seen == [(content, True)]
    assert sparse.seen == ([] if text is None else [(text, True)])
    assert result.dense_vector == [1.0, 0.0]
    assert result.sparse_vector == (None if text is None else {"word": 0.7})
    assert embedder.get_dimension() == 2


def test_text_only_dense_channel_does_not_advertise_image_support():
    embedder = CompositeHybridEmbedder(Dense(multimodal=False), Sparse())
    assert embedder.supports_multimodal is False
    assert (
        embedder.prepare_embedding_input([{"type": "text", "text": "caption"}, IMAGE]) == "caption"
    )
