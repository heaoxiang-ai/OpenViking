# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Scene recall must retain canonical evidence, scope and lifecycle semantics."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from openviking.models.embedder.base import EmbedResult
from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever, RetrieverMode
from openviking.server.identity import RequestContext, Role
from openviking.session.memory.account_templates import (
    memory_template_data,
    resolve_account_memory_registry,
)
from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.scene_cues import SOURCE_FIELD, source_digest, valid_scene_cue
from openviking.session.memory.utils.content_visibility import visible_content
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.collection_schemas import CollectionSchemas, TextEmbeddingHandler
from openviking.storage.expr import Contains
from openviking.storage.queuefs.embedding_msg import EmbeddingMsg
from openviking.storage.queuefs.process_result import ProcessOutcome
from openviking.storage.scene_cue_index import SceneCueIndex
from openviking.storage.vector_ids import vector_record_id
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking.storage.vikingdb_manager import VikingDBManager
from openviking_cli.retrieve.types import ContextType, TypedQuery
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.memory_config import MemoryConfig, SceneCueConfig
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig

URI = "viking://user/alice/memories/events/2026/09/21/hike.md"
BODY = "# Summary\nA hike.\n# ChatLog\nAlice: I hiked in the rain with Mei."


def context(user="alice", account="test", role=Role.USER):
    return RequestContext(user=UserIdentifier(account_id=account, user_id=user), role=role)


@pytest_asyncio.fixture
async def storage(tmp_path):
    manager = VikingDBManager(VectorDBBackendConfig(path=str(tmp_path), dimension=4))
    await manager.create_collection("context", CollectionSchemas.context_collection("context", 4))
    manager.scene_index = SceneCueIndex(manager, SceneCueConfig(enabled=True))
    auxiliary = manager.scene_index.store
    await auxiliary.create_collection(
        auxiliary.collection_name,
        CollectionSchemas.context_collection(auxiliary.collection_name, 4),
    )
    try:
        yield manager
    finally:
        await manager.close()


async def add_event(storage, uri=URI, ctx=None):
    ctx = ctx or context()
    record = {
        "id": vector_record_id(ctx.account_id, uri, 2),
        "account_id": ctx.account_id,
        "owner_user_id": ctx.user.user_id,
        "uri": uri,
        "context_type": "memory",
        "level": 2,
        "abstract": BODY,
        "vector": [0.1, 1.0, 0.0, 0.0],
        "search_tags": ["trip"],
        "_scene_cue_embedding": ([1.0, 0.1, 0.0, 0.0], None),
    }
    await storage.upsert(record, ctx=ctx)
    return record["id"]


async def recall(storage, ctx=None, **kwargs):
    return await storage.scene_index.search(
        ctx=ctx or context(), query_vector=[1.0, 0.1, 0.0, 0.0], context_type="memory", **kwargs
    )


def test_source_binding_and_hidden_metadata():
    mf = MemoryFile(
        uri=URI,
        content=BODY,
        extra_fields={
            "scene_cue": "Rainy outdoor activity with Mei",
            SOURCE_FIELD: source_digest(BODY),
        },
    )
    raw = MemoryFileUtils.write(mf)
    parsed = MemoryFileUtils.read(raw, uri=URI)
    assert valid_scene_cue(parsed) == "Rainy outdoor activity with Mei"
    assert visible_content(raw, uri=URI).strip() == BODY
    parsed.content += "\nCorrection: Mei was not present."
    assert valid_scene_cue(parsed) == ""


def test_schema_opt_in_does_not_change_primary_templates(monkeypatch):
    config = SimpleNamespace(memory=MemoryConfig())
    monkeypatch.setattr("openviking_cli.utils.config.get_openviking_config", lambda: config)
    baseline = MemoryTypeRegistry().get("events")
    assert "scene_cue" not in {field.name for field in baseline.fields}
    config.memory.scene_cues.enabled = True
    enabled = MemoryTypeRegistry().get("events")
    assert "scene_cue" in {field.name for field in enabled.fields}
    assert enabled.embedding_template == baseline.embedding_template
    assert enabled.content_template == baseline.content_template
    config.memory.scene_cues.enabled = False
    assert MemoryTypeRegistry().get("events") == baseline


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_account_template_cannot_override_scene_switch(monkeypatch, enabled):
    config = SimpleNamespace(memory=MemoryConfig(scene_cues=SceneCueConfig(enabled=not enabled)))
    monkeypatch.setattr("openviking_cli.utils.config.get_openviking_config", lambda: config)
    old = memory_template_data(MemoryTypeRegistry().get("events"))
    config.memory.scene_cues.enabled = enabled
    deployment = MemoryTypeRegistry()
    monkeypatch.setattr(
        "openviking.session.memory.account_templates.read_account_memory_template",
        AsyncMock(side_effect=lambda fs, account, name: old if name == "events" else None),
    )
    resolved = await resolve_account_memory_registry(None, "test", deployment)
    assert ("scene_cue" in {f.name for f in resolved.get("events").fields}) == enabled


@pytest.mark.asyncio
async def test_separate_embeddings_and_canonical_evidence(storage):
    record_id = await add_event(storage)
    primary = (await storage.get_strict([record_id], ctx=context()))[0]
    auxiliary = (await storage.scene_index.store.get_strict([record_id], ctx=context()))[0]
    assert primary["vector"] != auxiliary["vector"]
    assert primary["abstract"] == auxiliary["abstract"] == BODY
    hits = await recall(storage)
    assert [hit["uri"] for hit in hits] == [URI]
    assert hits[0]["abstract"] == BODY


@pytest.mark.asyncio
async def test_account_user_target_tag_and_level_isolation(storage):
    await add_event(storage)
    assert not await recall(storage, ctx=context(user="bob"))
    assert not await recall(storage, ctx=context(account="other"))
    assert not await recall(storage, target_directories=["viking://user/alice/memories/profile"])
    assert not await recall(storage, level=[0, 1])
    assert not await recall(storage, extra_filter=Contains("search_tags", "missing"))
    assert await recall(storage, extra_filter=Contains("search_tags", "trip"))


@pytest.mark.asyncio
async def test_query_rechecks_stale_auxiliary_evidence_and_permissions(storage):
    record_id = await add_event(storage)
    # Simulate interrupted maintenance leaving a stale auxiliary collection.
    record = (await storage.get_strict([record_id], ctx=context()))[0]
    record["abstract"] = "Changed evidence"
    await VikingVectorIndexBackend.upsert(storage, record, ctx=context())
    assert not await recall(storage)
    # Even if an auxiliary backend yields a foreign hit, canonical scope is required.
    foreign = {**record, "abstract": BODY, "_score": 0.99}
    storage.scene_index.store.search_in_tenant = AsyncMock(return_value=[foreign])
    assert not await recall(storage, ctx=context(user="bob"))


@pytest.mark.asyncio
async def test_metadata_refresh_and_edit_invalidation(storage):
    record_id = await add_event(storage)
    await storage.update_search_tags(URI, ["new-tag"], mode="replace", ctx=context())
    assert await recall(storage, extra_filter=Contains("search_tags", "new-tag"))
    record = (await storage.get_strict([record_id], ctx=context()))[0]
    record["abstract"] = "Edited body"
    await storage.upsert(record, ctx=context())
    assert not await storage.scene_index.store.get_strict([record_id], ctx=context())


@pytest.mark.asyncio
async def test_partial_update_and_missing_cue_invalidate_old_view(storage):
    record_id = await add_event(storage)
    await storage.update({"id": record_id, "abstract": "Updated"}, ctx=context())
    assert not await storage.scene_index.store.get_strict([record_id], ctx=context())
    await add_event(storage)
    record = (await storage.get_strict([record_id], ctx=context()))[0]
    record["_scene_cue_embedding"] = None
    await storage.upsert(record, ctx=context())
    assert not await storage.scene_index.store.get_strict([record_id], ctx=context())


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete", "uri", "uris", "user", "account", "clear"])
async def test_deletion_cleans_auxiliary_collection(storage, operation):
    record_id = await add_event(storage)
    root = context(role=Role.ROOT)
    if operation == "delete":
        await storage.delete([record_id], ctx=context())
    elif operation == "uri":
        await storage.remove_by_uri(URI, ctx=context())
    elif operation == "uris":
        await storage.delete_uris(context(), [URI])
    elif operation == "user":
        await storage.delete_user_data("test", "alice", ctx=root)
    elif operation == "account":
        await storage.delete_account_data("test", ctx=root)
    else:
        await storage.clear(ctx=context())
    assert not await storage.scene_index.store.get_strict([record_id], ctx=context())


@pytest.mark.asyncio
async def test_move_and_copy_preserve_auxiliary_mapping(storage):
    await add_event(storage)
    copied = URI.replace("hike.md", "copy.md")
    moved = URI.replace("hike.md", "moved.md")
    await storage.copy_uri_mapping(context(), URI, copied)
    assert {hit["uri"] for hit in await recall(storage)} == {URI, copied}
    await storage.update_uri_mapping(context(), copied, moved)
    assert {hit["uri"] for hit in await recall(storage)} == {URI, moved}


@pytest.mark.asyncio
async def test_quick_fusion_dedup_single_query_embedding_and_recall_switch(storage):
    await add_event(storage)
    embedder = SimpleNamespace(
        prepare_embedding_input=lambda text: text,
        embed_async=AsyncMock(return_value=EmbedResult(dense_vector=[1.0, 0.1, 0.0, 0.0])),
    )
    retriever = HierarchicalRetriever(storage, embedder)
    query = TypedQuery(query="rainy outdoor activity", context_type=ContextType.MEMORY, intent="")
    enabled = await retriever.retrieve(query, context(), limit=10, mode=RetrieverMode.QUICK)
    assert len(enabled.matched_contexts) == 1
    assert enabled.matched_contexts[0].abstract == BODY
    assert embedder.embed_async.call_count == 1
    storage.scene_index.settings.recall_enabled = False
    disabled = await retriever.retrieve(query, context(), limit=10, mode=RetrieverMode.QUICK)
    assert len(disabled.matched_contexts) == 1
    assert enabled.matched_contexts[0].score > disabled.matched_contexts[0].score
    assert await storage.scene_index.store.count(ctx=context()) == 1


@pytest.mark.asyncio
async def test_thinking_reranks_canonical_evidence(storage):
    await add_event(storage)
    retriever = HierarchicalRetriever(storage, None)
    retriever._recursive_search = AsyncMock(return_value=[])
    retriever._rerank_client = object()
    retriever._rerank_scores = AsyncMock(side_effect=lambda q, docs, scores: [0.8] * len(docs))
    result = await retriever.retrieve(
        TypedQuery(query="rainy outdoor activity", context_type=ContextType.MEMORY, intent=""),
        context(),
        mode=RetrieverMode.THINKING,
    )
    assert [hit.uri for hit in result.matched_contexts] == [URI]
    assert retriever._rerank_scores.call_args.args[1] == [BODY]


@pytest.mark.asyncio
@pytest.mark.parametrize("cue_fails", [False, True])
async def test_queue_tracks_auxiliary_embedding_failure(storage, monkeypatch, cue_fails):
    record_id = await add_event(storage)
    results = [EmbedResult(dense_vector=[0.1, 1.0, 0.0, 0.0])]
    results.append(
        ValueError("bad cue input") if cue_fails else EmbedResult(dense_vector=[1.0, 0.1, 0.0, 0.0])
    )
    embedder = SimpleNamespace(
        prepare_embedding_input=lambda text: text,
        embed_async=AsyncMock(side_effect=results),
    )
    config = SimpleNamespace(
        storage=SimpleNamespace(vectordb=SimpleNamespace(name="context")),
        embedding=SimpleNamespace(
            dimension=4,
            get_embedder=lambda: embedder,
            circuit_breaker=SimpleNamespace(
                failure_threshold=5, reset_timeout=1, max_reset_timeout=10
            ),
        ),
    )
    monkeypatch.setattr("openviking_cli.utils.config.get_openviking_config", lambda: config)
    handler = TextEmbeddingHandler(storage)
    record = (await storage.get_strict([record_id], ctx=context()))[0]
    record["_scene_cue"] = "Rainy outdoor activity with Mei"
    msg = EmbeddingMsg(message=BODY, context_data=record)
    result = await handler.on_dequeue({"data": msg.to_json()})
    assert result.outcome == (ProcessOutcome.FAILED if cue_fails else ProcessOutcome.SUCCESS)
    assert [call.args[0] for call in embedder.embed_async.call_args_list] == [
        BODY,
        record["_scene_cue"],
    ]
