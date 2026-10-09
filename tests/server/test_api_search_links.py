# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import pytest

from openviking_cli.retrieve import ContextType, FindResult, MatchedContext


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"include_links": True},
        {"expand_links": True},
        {"include_links": True, "expand_links": True},
    ],
)
async def test_search_forwards_only_enabled_link_options(client, service, monkeypatch, flags):
    captured = {}
    uri = "viking://user/default/memories/events/race.md"
    links = [{"from_uri": uri, "to_uri": "viking://user/default/memories/entities/person.md"}]

    async def fake_search(**kwargs):
        captured.update(kwargs)
        return FindResult(
            memories=[
                MatchedContext(
                    uri=uri,
                    context_type=ContextType.MEMORY,
                    level=2,
                    links=links if flags.get("include_links") else None,
                    backlinks=[] if flags.get("include_links") else None,
                )
            ],
            resources=[],
            skills=[],
        )

    monkeypatch.setattr(service.search, "search", fake_search)
    response = await client.post("/api/v1/search/search", json={"query": "race", **flags})

    assert response.status_code == 200
    for key in ("include_links", "expand_links"):
        assert captured.get(key) == flags.get(key)
    result = response.json()["result"]["memories"][0]
    if flags.get("include_links"):
        assert result["links"] == links
        assert result["backlinks"] == []
    else:
        assert "links" not in result
        assert "backlinks" not in result


@pytest.mark.parametrize("flag", ["include_links", "expand_links"])
@pytest.mark.parametrize(
    "endpoint,extra",
    [
        ("/api/v1/search/search", {"mode": "context"}),
        ("/api/v1/search/find", {}),
    ],
)
async def test_link_options_rejected_on_unsupported_modes(client, endpoint, extra, flag):
    response = await client.post(endpoint, json={"query": "race", flag: True, **extra})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
