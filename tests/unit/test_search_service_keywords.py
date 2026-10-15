from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service.search_service import SearchService
from openviking_cli.session.user_id import UserIdentifier


def _ctx():
    return RequestContext(user=UserIdentifier("account", "user"), role=Role.USER)


@pytest.mark.asyncio
async def test_keywords_capability_check_propagates_metadata_failures():
    error = RuntimeError("metadata unavailable")
    check = AsyncMock(side_effect=error)
    vector_store = object()
    fs = SimpleNamespace(
        _get_vector_store=lambda: vector_store,
        _collection_has_fulltext=check,
    )

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        await SearchService(fs).ensure_keywords_search_supported(_ctx())

    check.assert_awaited_once_with(
        vector_store,
        _ctx(),
        supported_modes=("volcengine", "vikingdb"),
        raise_on_error=True,
    )


async def test_multiple_query_images_are_resolved_under_same_request_context():
    read = AsyncMock(return_value=b"png")
    service = SearchService(SimpleNamespace(read_file_bytes=read))
    ctx = _ctx()
    urls = ["data:image/png;base64,YQ==", "viking://resources/photo.png"]
    assert await service._resolve_image_url(urls, ctx) == [urls[0], "data:image/png;base64,cG5n"]
    read.assert_awaited_once_with(urls[1], ctx=ctx)
