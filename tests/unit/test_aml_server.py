"""Competition API contract through the real OpenViking SDK request builder."""

import asyncio
import base64
import json

import httpx
import pytest

from benchmark.aml import server
from benchmark.aml.server import AMLSettings, create_app

IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6lcAAAAASUVORK5CYII="
)
IMAGE_PART = {"type": "image_url", "image_url": {"url": IMAGE_URL}}
IMAGE_SIZE = len(base64.b64decode(IMAGE_URL.partition(",")[2]))


@pytest.fixture
def adapter_client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(tmp_path / "missing.conf"))
    client_type = httpx.AsyncClient

    def build(handler):
        def sdk_client(**kwargs):
            return client_type(**kwargs, transport=httpx.MockTransport(handler))

        monkeypatch.setattr("openviking_sdk.client.httpx.AsyncClient", sdk_client)
        app = create_app(
            settings=AMLSettings(
                openviking_url="http://ov.test",
                openviking_account="competition",
                openviking_api_key="test-ov-key",
                aml_api_key="test-aml-key",
                retry_delay_seconds=0.001,
                task_poll_seconds=0.001,
            )
        )
        return client_type(
            transport=httpx.ASGITransport(app=app),
            base_url="http://adapter.test",
            headers={"Authorization": "Token test-aml-key"},
        )

    return build


@pytest.fixture
def add_client(adapter_client):
    def build(task_status="completed"):
        requests = []

        def handle(request):
            requests.append(request)
            if request.url.path == "/api/v1/sessions/session-1/messages/batch":
                result = {"added": len(json.loads(request.content)["messages"])}
            elif request.url.path == "/api/v1/sessions/session-1/commit":
                assert json.loads(request.content) == {"keep_recent_count": 0}
                result = {"status": "accepted", "task_id": "task-1"}
            else:
                assert request.method == "GET"
                assert request.url.path == "/api/v1/tasks/task-1"
                result = {"status": task_status, "error": "extraction failed"}
            return httpx.Response(200, json={"status": "ok", "result": result})

        return adapter_client(handle), requests

    return build


@pytest.mark.parametrize(
    "content",
    [
        "  Remember my preference for tea.\n",
        "def cache_key(user_id):\n    return f'cache:{user_id}'\n",
    ],
    ids=["textual", "coding"],
)
async def test_add_string_payload_and_commit_are_unchanged(add_client, content):
    client, requests = add_client()
    payload = {
        "request_id": "request-1",
        "user_id": "alice",
        "session_id": "session-1",
        "messages": [
            {"role": "user", "content": content, "timestamp": 0},
            {"role": "assistant", "content": "Acknowledged."},
        ],
    }
    async with client:
        response = await client.post("/add", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "request_id": "request-1",
        "user_id": "alice",
        "session_id": "session-1",
    }
    assert json.loads(requests[0].content) == {
        "messages": [
            {"role": "user", "content": content, "created_at": "1970-01-01T00:00:00.000Z"},
            {"role": "assistant", "content": "Acknowledged."},
        ]
    }
    assert [request.url.path for request in requests] == [
        "/api/v1/sessions/session-1/messages/batch",
        "/api/v1/sessions/session-1/commit",
        "/api/v1/tasks/task-1",
    ]
    assert all(request.headers["X-OpenViking-User"] == "alice" for request in requests)


@pytest.mark.parametrize(
    "parts",
    [
        [IMAGE_PART],
        [
            {"type": "text", "text": "  First image:\n"},
            IMAGE_PART,
            {"type": "text", "text": "The same image again:"},
            IMAGE_PART,
        ],
        [{"type": "text", "text": "Text supplied as a content array."}],
    ],
    ids=["image-only", "ordered-multiple-images", "text-array"],
)
async def test_add_forwards_content_array_as_native_parts(add_client, parts):
    client, requests = add_client()
    async with client:
        response = await client.post(
            "/add",
            json={
                "request_id": "multimodal-1",
                "user_id": "bob",
                "session_id": "session-1",
                "messages": [
                    {"role": "user", "content": "A plain message."},
                    {"role": "assistant", "content": parts, "timestamp": 1704067200000},
                ],
            },
        )
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "request_id": "multimodal-1",
        "user_id": "bob",
        "session_id": "session-1",
    }
    assert json.loads(requests[0].content) == {
        "messages": [
            {"role": "user", "content": "A plain message."},
            {"role": "assistant", "parts": parts, "created_at": "2024-01-01T00:00:00.000Z"},
        ]
    }
    assert len(requests) == 3
    assert all(request.headers["X-OpenViking-User"] == "bob" for request in requests)
    assert all(request.headers["X-OpenViking-Account"] == "competition" for request in requests)
    assert all(request.headers["X-API-Key"] == "test-ov-key" for request in requests)


@pytest.mark.parametrize(
    "content",
    [
        "",
        " \n ",
        [],
        [{"type": "text", "text": " "}],
        [{"type": "image_url", "image_url": {}}],
        [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}}],
        [{"type": "image_url", "image_url": {"url": "data:image/gif;base64,YQ=="}}],
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}],
        [{"type": "tool", "tool_name": "find"}],
    ],
    ids=[
        "empty-string",
        "blank-string",
        "empty-array",
        "blank-text-part",
        "missing-image-url",
        "remote-image",
        "invalid-base64",
        "unsupported-mime",
        "empty-image",
        "unsupported-part",
    ],
)
async def test_add_rejects_invalid_content_before_writing(add_client, content):
    client, requests = add_client()
    async with client:
        response = await client.post(
            "/add",
            json={
                "request_id": "invalid-1",
                "user_id": "alice",
                "session_id": "session-1",
                "messages": [{"role": "user", "content": content}],
            },
        )
    assert response.status_code == 422
    assert requests == []


@pytest.mark.parametrize(
    ("limit", "max_bytes", "image_count", "expected_status"),
    [
        ("MAX_IMAGE_BYTES", IMAGE_SIZE - 1, 1, 422),
        ("MAX_IMAGE_BYTES", IMAGE_SIZE, 1, 200),
        ("MAX_ADD_IMAGE_BYTES", IMAGE_SIZE * 2 - 1, 2, 422),
        ("MAX_ADD_IMAGE_BYTES", IMAGE_SIZE * 2, 2, 200),
    ],
    ids=["image-over-limit", "image-at-limit", "request-over-limit", "request-at-limit"],
)
async def test_add_enforces_image_budget_across_messages(
    add_client, monkeypatch, limit, max_bytes, image_count, expected_status
):
    monkeypatch.setattr(server, limit, max_bytes)
    client, requests = add_client()
    async with client:
        response = await client.post(
            "/add",
            json={
                "request_id": "oversized-1",
                "user_id": "alice",
                "session_id": "session-1",
                "messages": [{"role": "user", "content": [IMAGE_PART]} for _ in range(image_count)],
            },
        )
    assert response.status_code == expected_status
    if expected_status == 422:
        assert requests == []


async def test_add_multimodal_extraction_failure_is_not_reported_as_success(add_client):
    client, requests = add_client(task_status="failed")
    async with client:
        response = await client.post(
            "/add",
            json={
                "request_id": "failed-1",
                "user_id": "alice",
                "session_id": "session-1",
                "messages": [{"role": "user", "content": [IMAGE_PART]}],
            },
        )
    assert response.status_code == 503
    assert "success" not in response.json()
    assert len(requests) == 3


async def test_add_multimodal_requires_existing_authentication(add_client):
    client, requests = add_client()
    async with client:
        response = await client.post(
            "/add",
            headers={"Authorization": "Token wrong-key"},
            json={
                "request_id": "unauthorized-1",
                "user_id": "alice",
                "session_id": "session-1",
                "messages": [{"role": "user", "content": [IMAGE_PART]}],
            },
        )
    assert response.status_code == 401
    assert requests == []


@pytest.mark.parametrize("query", ["Where?", "Which function invalidates src/cache.py?"])
async def test_search_uses_native_find_and_preserves_user_scope_and_aml_response(
    adapter_client, query
):
    seen = []

    async def handle(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/search/find"
        assert request.headers["X-OpenViking-Account"] == "competition"
        assert request.headers["X-API-Key"] == "test-ov-key"
        user = request.headers["X-OpenViking-User"]
        body = json.loads(request.content)
        assert body["target_uri"] == "viking://~/memories"
        assert body["context_type"] == ["memory"]
        assert body["level"] == 2
        assert body["limit"] == 2
        assert "mode" not in body
        assert not body.get("session_id")
        assert not body.get("read_content")
        seen.append((user, body["query"]))
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "memories": [
                        {"uri": f"viking://user/{user}/memories/a", "content": "body"},
                        {"uri": f"viking://user/{user}/memories/b", "abstract": "summary"},
                        {"uri": f"viking://user/{user}/memories/c", "abstract": "extra"},
                    ],
                    "resources": [{"uri": "viking://resources/other", "content": "other"}],
                    "skills": [],
                },
            },
        )

    async with adapter_client(handle) as client:
        responses = await asyncio.gather(
            client.post("/search", json={"user_id": "alice", "query": query, "top_k": 2}),
            client.post(
                "/search",
                json={"user_id": "bob", "query": "Which?", "options": ["A", "B"], "top_k": 2},
            ),
        )
    assert sorted(seen) == [("alice", query), ("bob", "Which?\n\nAnswer options:\n1. A\n2. B")]
    for user, response in zip(("alice", "bob"), responses, strict=True):
        assert response.status_code == 200
        assert response.json() == {
            "data": [
                {"id": f"viking://user/{user}/memories/a", "content": "body"},
                {"id": f"viking://user/{user}/memories/b", "content": "summary"},
            ]
        }


async def test_search_retries_transient_native_find_failure(adapter_client):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadTimeout("temporary failure", request=request)
        return httpx.Response(200, json={"status": "ok", "result": {"memories": []}})

    async with adapter_client(handle) as client:
        response = await client.post(
            "/search", json={"user_id": "alice", "query": "Where?", "top_k": 2}
        )
    assert response.status_code == 200
    assert response.json() == {"data": []}
    assert len(requests) == 2
    assert all(request.url.path == "/api/v1/search/find" for request in requests)
    assert requests[0].content == requests[1].content
    assert requests[0].headers == requests[1].headers


@pytest.mark.parametrize(
    ("query", "text", "images"),
    [
        ([IMAGE_PART], "", IMAGE_URL),
        ([{"type": "text", "text": "Which receipt?"}, IMAGE_PART], "Which receipt?", IMAGE_URL),
        (
            [IMAGE_PART, {"type": "text", "text": "Compare both"}, IMAGE_PART],
            "Compare both",
            [IMAGE_URL, IMAGE_URL],
        ),
        (
            [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}],
            "First\nSecond",
            None,
        ),
    ],
    ids=["image-only", "mixed", "multiple-images", "text-array"],
)
async def test_search_forwards_images_to_native_find(adapter_client, query, text, images):
    seen = []

    def handle(request):
        assert request.url.path == "/api/v1/search/find"
        assert request.headers["X-OpenViking-User"] == "alice"
        body = json.loads(request.content)
        assert body["query"] == text + "\n\nAnswer options:\n1. A\n2. B"
        assert body.get("image_url") == images
        assert body["target_uri"] == "viking://~/memories"
        assert body["context_type"] == ["memory"]
        assert body["level"] == 2
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "memories": [{"uri": "viking://user/alice/memories/a", "content": "Evidence"}]
                },
            },
        )

    async with adapter_client(handle) as client:
        response = await client.post(
            "/search", json={"user_id": "alice", "query": query, "options": ["A", "B"], "top_k": 2}
        )
    assert response.status_code == 200
    assert response.json() == {
        "data": [{"id": "viking://user/alice/memories/a", "content": "Evidence"}]
    }
    assert len(seen) == 1


@pytest.mark.parametrize(
    "query",
    [
        [],
        " ",
        [{"type": "text", "text": " "}],
        [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}}],
    ],
)
async def test_search_rejects_invalid_queries_before_native_find(adapter_client, query):
    def handle(request):
        raise AssertionError("Invalid query must not reach the backend")

    async with adapter_client(handle) as client:
        response = await client.post(
            "/search", json={"user_id": "alice", "query": query, "top_k": 2}
        )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("limit", "maximum", "count", "status"),
    [
        ("MAX_IMAGE_BYTES", IMAGE_SIZE - 1, 1, 422),
        ("MAX_QUERY_IMAGE_BYTES", 2 * IMAGE_SIZE - 1, 2, 422),
        ("MAX_QUERY_IMAGE_BYTES", 2 * IMAGE_SIZE, 2, 200),
    ],
)
async def test_search_validates_per_image_and_aggregate_size(
    adapter_client, monkeypatch, limit, maximum, count, status
):
    monkeypatch.setattr(server, limit, maximum)
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"status": "ok", "result": {"memories": []}})

    async with adapter_client(handle) as client:
        response = await client.post(
            "/search", json={"user_id": "alice", "query": [IMAGE_PART] * count, "top_k": 2}
        )
    assert response.status_code == status
    assert len(seen) == (status == 200)


async def test_image_search_requires_existing_authentication(adapter_client):
    def handle(request):
        raise AssertionError("Unauthorized query must not reach the backend")

    async with adapter_client(handle) as client:
        response = await client.post(
            "/search",
            headers={"Authorization": "Token wrong-key"},
            json={"user_id": "alice", "query": [IMAGE_PART], "top_k": 2},
        )
    assert response.status_code == 401
