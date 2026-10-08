# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Entity association regressions using real local vector storage, no LLM."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from openviking.models.embedder.base import EmbedResult
from openviking.retrieve.entity_linking import index as index_module
from openviking.retrieve.entity_linking._entity_rules import (
    _EntityCandidate,
    _resolve_candidates,
    extract_entities_batch,
)
from openviking.retrieve.entity_linking.index import source_fingerprint
from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever
from openviking.server.identity import RequestContext, Role
from openviking.storage.collection_schemas import CollectionSchemas, TextEmbeddingHandler
from openviking.storage.expr import And, Eq
from openviking.storage.queuefs.embedding_msg import EmbeddingMsg
from openviking.storage.queuefs.process_result import ProcessOutcome
from openviking.storage.record_types import entity_records
from openviking_cli.retrieve.types import ContextType, TypedQuery
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.retrieval_config import EntityLinkingConfig, RetrievalConfig
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig


class Embedder:
    def __init__(self):
        self.inputs = []

    def prepare_embedding_input(self, text):
        return text

    async def embed_async(self, text, is_query=False):
        self.inputs.append((text, is_query))
        vectors = {"Alice": [1, 0, 0, 0], "John": [0, 1, 0, 0], "Pizza": [0, 0, 1, 0]}
        return EmbedResult(dense_vector=vectors.get(text, [1, 0, 0, 0]))


def ctx(user="user1", account="account1"):
    return RequestContext(user=UserIdentifier(account, user), role=Role.USER)


def memory(name, *, user="user1", account="account1", abstract=None, md5="version1"):
    return {
        "id": f"{account}-{user}-{name}",
        "uri": f"viking://user/{user}/memories/events/{name}.md",
        "account_id": account,
        "owner_user_id": user,
        "context_type": "memory",
        "abstract": abstract or name,
        "md5": md5,
        "vector": [1, 0, 0, 0],
        "level": 2,
    }


@pytest_asyncio.fixture
async def storage(tmp_path, vector_backend_factory, monkeypatch, request):
    config = VectorDBBackendConfig(
        backend="local", path=str(tmp_path), dimension=4, sparse_weight=getattr(request, "param", 0)
    )
    store = vector_backend_factory(config)
    assert await store.create_collection(
        "context", CollectionSchemas.context_collection("context", 4)
    )
    monkeypatch.setattr(
        index_module,
        "extract_entities_batch",
        lambda texts: [
            [("PROPER", n) for n in ["Alice", "John", "Pizza"] if n in t] for t in texts
        ],
    )
    yield store
    await store.close()


async def add(store, record, text="Alice"):
    context = ctx(record["owner_user_id"], record["account_id"])
    await store.upsert(record, ctx=context)
    await store.entity_link_index.replace(record, text, Embedder(), context, EntityLinkingConfig())


async def entity_rows(store, context=None, extra=None):
    backend = await store.entity_link_index._backend(context or ctx())
    filt = entity_records() if extra is None else And([entity_records(), extra])
    return await backend.strict_query(filter=filt, limit=1000)


async def search(store, text="Alice", context=None, **kwargs):
    return await store.entity_link_index.search(
        text, Embedder(), context or ctx(), EntityLinkingConfig(), **kwargs
    )


@pytest.mark.asyncio
async def test_entities_share_primary_collection_but_are_not_memories(storage):
    record = memory("favorite-pizza")
    await add(storage, record, "Alice likes Pizza")
    assert await storage.entity_link_index._backend(ctx()) is await storage.get_account_backend(
        "account1"
    )
    assert len(await entity_rows(storage)) == 2
    assert await storage.count(ctx=ctx()) == 1
    assert len(await storage.query(ctx=ctx(), limit=100)) == 1
    assert (await storage.fetch_by_uri(record["uri"], ctx=ctx()))["id"] == record["id"]
    assert (await search(storage))["entities"][0] == {
        "name": "Alice",
        "entity_type": "PROPER",
        "score": pytest.approx(1),
        "memory_uri": record["uri"],
    }


@pytest.mark.asyncio
async def test_entity_lookup_is_independent_and_normal_topk_is_unchanged(storage):
    alice = {**memory("alice"), "vector": [0.8, 0.6, 0, 0]}
    john = {**memory("john"), "vector": [1, 0, 0, 0]}
    await add(storage, alice, "Alice")
    await add(storage, john, "John")
    embedder = Embedder()
    retriever = HierarchicalRetriever(storage, embedder)
    result = await retriever.retrieve(
        TypedQuery(
            "Alice", ContextType.MEMORY, "", target_directories=["viking://user/user1/memories"]
        ),
        ctx(),
        limit=1,
    )
    assert result.matched_contexts[0].uri == john["uri"]
    assert result.matched_contexts[0].score == pytest.approx(1)
    assert embedder.inputs == [("Alice", True)]
    assert (await search(storage))["entities"][0]["memory_uri"] == alice["uri"]


@pytest.mark.asyncio
async def test_entity_scope_and_threshold(storage):
    own = memory("own")
    await add(storage, own, "John")
    await add(storage, memory("foreign", user="user2"), "Alice")
    await add(storage, memory("foreign", account="account2"), "Alice")
    assert (await search(storage))["entities"] == []
    result = await search(storage, "Alice", score_threshold=0)
    assert {x["memory_uri"] for x in result["entities"]} == {own["uri"]}
    assert (await search(storage, "John", target_dirs=[own["uri"]]))["total"] == 1
    assert (await search(storage, "John", target_dirs=["viking://user/user1/memories/other"]))[
        "total"
    ] == 0


@pytest.mark.asyncio
async def test_stale_generation_rejected_then_replaced(storage):
    old = memory("changing", abstract="Alice likes Pizza")
    await add(storage, old, "Alice")
    new = {**old, "md5": "version2", "abstract": "John likes Pizza"}
    await storage.upsert(new, ctx=ctx())
    assert (await search(storage))["total"] == 0
    await storage.entity_link_index.replace(new, "John", Embedder(), ctx(), EntityLinkingConfig())
    assert (await search(storage))["total"] == 0
    assert (await search(storage, "John"))["total"] == 1
    assert len(await entity_rows(storage)) == 1
    assert await storage.get_strict([new["id"]], ctx=ctx())


@pytest.mark.asyncio
async def test_auxiliary_cleanup_never_deletes_primary(storage):
    record = memory("keep")
    await add(storage, record)
    await storage.entity_link_index.remove_uris(ctx(), [record["uri"]])
    assert not await entity_rows(storage)
    assert await storage.get_strict([record["id"]], ctx=ctx())


@pytest.mark.asyncio
async def test_concurrent_shared_entities_keep_all_links(storage):
    records = [memory("first"), memory("second")]
    await asyncio.gather(*(add(storage, r) for r in records))
    assert {r["memory_uri"] for r in (await search(storage))["entities"]} == {
        r["uri"] for r in records
    }


@pytest.mark.asyncio
async def test_deletion_during_entity_write_does_not_resurrect(storage, monkeypatch):
    record = memory("deleted")
    await storage.upsert(record, ctx=ctx())
    backend = await storage.entity_link_index._backend(ctx())
    original = backend.upsert_many

    async def racing_write(rows):
        result = await original(rows)
        await storage.strict_delete([record["id"]], ctx=ctx())
        return result

    monkeypatch.setattr(backend, "upsert_many", racing_write)
    await storage.entity_link_index.replace(
        record, "Alice", Embedder(), ctx(), EntityLinkingConfig()
    )
    assert not await entity_rows(storage)
    assert not await storage.get_strict([record["id"]], ctx=ctx())


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete", "strict_delete", "remove_by_uri", "delete_uris"])
async def test_deletion_removes_only_target_memory_and_its_entities(storage, operation):
    record = memory("deleted")
    other = memory("kept")
    await add(storage, record)
    await add(storage, other)
    if operation == "remove_by_uri":
        assert await storage.remove_by_uri(record["uri"], ctx=ctx()) == 1
    elif operation == "delete_uris":
        await storage.delete_uris(ctx(), [record["uri"]])
    else:
        assert await getattr(storage, operation)([record["id"]], ctx=ctx()) == 1
    assert {r["uri"] for r in await entity_rows(storage)} == {other["uri"]}
    assert await storage.count(ctx=ctx()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("move", [False, True])
async def test_copy_move_keeps_parent_ids_and_entity_ownership(storage, move):
    record = memory("source")
    await add(storage, record)
    destination = "viking://user/user2/memories/events/target.md"
    operation = storage.update_uri_mapping if move else storage.copy_uri_mapping
    await operation(ctx(), record["uri"], destination)
    targets = await storage.query(filter=Eq("uri", destination), ctx=ctx("user2"))
    assert len(targets) == 1
    result = await search(storage, context=ctx("user2"))
    assert result["entities"][0]["memory_uri"] == destination
    links = await entity_rows(storage, extra=Eq("uri", destination))
    assert links[0]["owner_user_id"] == "user2"
    assert links[0]["tags"] == str(targets[0]["id"])
    assert bool(await entity_rows(storage, extra=Eq("uri", record["uri"]))) is not move


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete_user_data", "delete_account_data", "clear"])
async def test_account_user_cleanup_preserves_foreign_records(storage, operation):
    await add(storage, memory("own"))
    await add(storage, memory("other", account="account2"))
    root = RequestContext(user=UserIdentifier("account1", "root"), role=Role.ROOT)
    if operation == "delete_user_data":
        assert await storage.delete_user_data("account1", "user1", ctx=root) == 1
    elif operation == "delete_account_data":
        assert await storage.delete_account_data("account1", ctx=root) == 1
    else:
        assert await storage.clear(ctx=ctx())
    assert not await entity_rows(storage)
    assert len(await entity_rows(storage, context=ctx(account="account2"))) == 1
    assert await storage.count(ctx=ctx(account="account2")) == 1


@pytest.mark.asyncio
async def test_legacy_types_and_scalar_lookup_are_preserved(storage):
    for name, kind in [("absent", None), ("empty", ""), ("file", "file")]:
        record = memory(name)
        if kind is not None:
            record["type"] = kind
        await add(storage, record)
    assert await storage.count(ctx=ctx()) == 3
    assert len(await storage.query(ctx=ctx(), limit=20)) == 3
    assert await storage.filter_in_tenant(ctx=ctx(), extra_filter=Eq("level", 2), limit=20)
    page, _ = await storage.scroll(ctx=ctx(), limit=20)
    assert len(page) == 3
    inventory = await storage.get_incremental_inventory_under_uri(
        "viking://user/user1/memories", ctx=ctx()
    )
    assert len(inventory) == 3


@pytest.mark.asyncio
async def test_empty_query_entities_do_not_embed(storage):
    embedder = Embedder()
    result = await storage.entity_link_index.search(
        "nothing", embedder, ctx(), EntityLinkingConfig()
    )
    assert result == {"query_entities": [], "entities": [], "total": 0}
    assert not embedder.inputs


@pytest.mark.asyncio
async def test_close_does_not_close_shared_backend(storage):
    await add(storage, memory("kept"))
    await storage.entity_link_index.close()
    assert await storage.count(ctx=ctx()) == 1
    with pytest.raises(RuntimeError, match="closing"):
        await storage.entity_link_index._backend(ctx())


@pytest.mark.asyncio
async def test_failed_entity_embeddings_keep_primary(storage):
    record = memory("persisted")
    await storage.upsert(record, ctx=ctx())
    embedder = Embedder()
    embedder.embed_async = AsyncMock(side_effect=RuntimeError("technical failure"))
    with pytest.raises(RuntimeError, match="technical failure"):
        await storage.entity_link_index.replace(
            record, "Alice", embedder, ctx(), EntityLinkingConfig()
        )
    assert await storage.get_strict([record["id"]], ctx=ctx())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["config", "embedding", "timeout"])
async def test_queue_retains_primary_when_entity_indexing_fails(storage, monkeypatch, failure):
    config = EntityLinkingConfig(enabled=True)
    monkeypatch.setattr(index_module, "linking_config", lambda: config)
    if failure == "config":

        def broken_config():
            raise RuntimeError("entity configuration failure")

        monkeypatch.setattr(index_module, "linking_config", broken_config)
    else:
        error = (
            TimeoutError("entity timeout")
            if failure == "timeout"
            else RuntimeError("entity failure")
        )
        monkeypatch.setattr(storage.entity_link_index, "replace", AsyncMock(side_effect=error))
    handler = TextEmbeddingHandler(storage, SimpleNamespace(get_embedder=lambda: Embedder()))
    handler._embedding_provider = SimpleNamespace(bind=lambda account: Embedder())
    record = memory("queued")
    message = EmbeddingMsg(message="Alice", context_data=record)
    result = await handler.on_dequeue({"data": message.to_json()})
    assert result.outcome == ProcessOutcome.SUCCESS
    assert await storage.count(ctx=ctx()) == 1


@pytest.mark.asyncio
async def test_queue_extracts_complete_prose_without_persisting_transient_field(
    storage, monkeypatch
):
    config = EntityLinkingConfig(enabled=True)
    monkeypatch.setattr(index_module, "linking_config", lambda: config)
    handler = TextEmbeddingHandler(storage, SimpleNamespace(get_embedder=lambda: Embedder()))
    handler._embedding_provider = SimpleNamespace(bind=lambda account: Embedder())
    record = {**memory("complete-prose"), "_entity_link_text": "John likes Pizza"}
    message = EmbeddingMsg(message="truncated embedding preview", context_data=record)
    payload = message.to_json()
    result = await handler.on_dequeue({"data": payload})
    assert result.outcome == ProcessOutcome.SUCCESS
    assert {row["abstract"] for row in await entity_rows(storage)} == {"John", "Pizza"}
    persisted = await storage.get_strict([result.value["id"]], ctx=ctx())
    assert len(persisted) == 1
    assert "_entity_link_text" not in persisted[0]
    assert message.context_data["_entity_link_text"] == "John likes Pizza"
    assert message.to_json() == payload


def test_feature_default_is_off():
    assert not RetrievalConfig().entity_linking.enabled


def test_distinct_substring_names_are_not_merged():
    candidates = [
        _EntityCandidate("PROPER", "Sam", "ner", 0, 1, 0.95, 0),
        _EntityCandidate("PROPER", "Samsung", "ner", 3, 4, 0.95, 0),
    ]
    assert _resolve_candidates(candidates) == [("PROPER", "Sam"), ("PROPER", "Samsung")]


def test_source_fingerprint_changes_with_full_file_md5():
    before = memory("same-preview")
    assert source_fingerprint(before) != source_fingerprint({**before, "md5": "version2"})


def test_missing_spacy_returns_aligned_empty_batches(monkeypatch):
    monkeypatch.setattr("openviking.retrieve.entity_linking.nlp.get_nlp_full", lambda: None)
    assert extract_entities_batch(["Alice", "John"]) == [[], []]


def test_real_spacy_extracts_entities_and_identifiers(monkeypatch):
    spacy = pytest.importorskip("spacy")
    pytest.importorskip("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    monkeypatch.setattr("openviking.retrieve.entity_linking.nlp.get_nlp_full", lambda: nlp)
    entities = extract_entities_batch(["Alice works at Microsoft and uses package.module."])
    names = {name for _, name in entities[0]}
    assert "Alice" in names
    assert "Microsoft" in names
    assert "package.module" in names


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", [0, 0.5], indirect=True)
async def test_entity_cosine_on_dense_and_hybrid_main_index(storage):
    await add(storage, memory("alice"), "Alice")
    await add(storage, memory("john"), "John")
    result = await search(storage, score_threshold=0.9)
    assert result["total"] == 1
    assert result["entities"][0]["score"] == pytest.approx(1)


@pytest.mark.asyncio
async def test_current_parent_acl_revocation_and_grant(storage):
    record = {
        **memory("shared"),
        "uri": "viking://resources/shared/note.md",
        "acl_mode": "restricted",
        "acl_direct_grants": ["1:user:user1"],
    }
    await add(storage, record, "Alice")
    # Enable live ACL filtering only after seeding the parent; derived rows carry
    # no grants and must not decide whether the parent is currently visible.
    storage.acl_manager = SimpleNamespace(is_enabled=AsyncMock(return_value=True))
    assert (await search(storage, target_dirs=["viking://resources"]))["total"] == 1
    backend = await storage.get_account_backend("account1")
    await backend.upsert({**record, "acl_direct_grants": []})
    assert (await search(storage, target_dirs=["viking://resources"]))["total"] == 0
    await backend.upsert(record)
    assert (await search(storage, target_dirs=["viking://resources"]))["total"] == 1
