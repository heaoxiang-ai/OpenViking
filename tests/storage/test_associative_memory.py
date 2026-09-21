# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Source provenance, identity, restart safety and typed evidence retrieval."""

import json
import re
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from openviking.message import Message, TextPart
from openviking.models.embedder.base import EmbedResult
from openviking.retrieve.associative_ranking import bm25, budgeted, rrf
from openviking.retrieve.associative_retriever import AssociativeRetriever
from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever
from openviking.server.identity import RequestContext, Role
from openviking.session.associative.builder import AssociativeBuilder, group_messages, scene_batches
from openviking.session.associative.types import Evidence, View, digest
from openviking.session.session import Session
from openviking.storage.associative_index import AssociativeIndex
from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.vikingdb_manager import VikingDBManager
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.associative_memory_config import AssociativeMemoryConfig
from openviking_cli.utils.config.memory_config import MemoryConfig
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig

ROOT = "viking://user/alice/memories/associative"
SOURCE = "viking://user/alice/sessions/test/history/archive_001/source.json"
STAMP = "2026-09-21T00:00:00Z"


def test_partial_commit_recovery_keeps_associative_coverage():
    # Associative extraction finished but its long-term sibling failed. Persisting
    # and reloading the archive must not schedule the finished messages again.
    saved = json.loads(
        json.dumps(Session._serialize_completed_memory_steps({"associative": {"m1", "m2"}}))
    )
    recovered = {"long_term": {"m1"}}
    Session._merge_completed_memory_steps(recovered, saved)
    pending = {
        step: [mid for mid in ("m1", "m2", "m3") if mid not in recovered.get(step, set())]
        for step in ("long_term", "associative")
    }
    assert pending == {"long_term": ["m2", "m3"], "associative": ["m3"]}


def context(user="alice", account="test"):
    return RequestContext(user=UserIdentifier(account_id=account, user_id=user), role=Role.USER)


class Files:
    def __init__(self):
        self.data = {SOURCE: "Original conversation"}

    async def read_file(self, uri, ctx):
        if uri not in self.data:
            raise FileNotFoundError(uri)
        return self.data[uri]

    async def write_file(self, uri, content, ctx):
        self.data[uri] = content


class Embedder:
    def __init__(self):
        self.calls = 0

    def prepare_embedding_input(self, text):
        return text

    async def embed_async(self, text, is_query=False):
        self.calls += 1
        return EmbedResult(dense_vector=[1.0, 0.0, 0.0, 0.0])


@pytest_asyncio.fixture
async def store(tmp_path):
    manager = VikingDBManager(VectorDBBackendConfig(path=str(tmp_path), dimension=4))
    await manager.create_collection("context", CollectionSchemas.context_collection("context", 4))
    index = AssociativeIndex(manager, AssociativeMemoryConfig(enabled=True, final_profile_k=0))
    manager.associative_index = index
    await index.store.create_collection(
        index.store.collection_name,
        CollectionSchemas.context_collection(index.store.collection_name, 4),
    )
    try:
        yield manager, index, Files(), Embedder()
    finally:
        await manager.close()


def node(kind, text, **kwargs):
    return Evidence(
        f"{ROOT}/{kind}s/test.md",
        kind,
        text,
        text,
        SOURCE,
        digest("Original conversation"),
        STAMP,
        **kwargs,
    )


async def publish(store, *nodes):
    manager, index, fs, embed = store
    for entry in nodes:
        await index.publish(entry, fs=fs, ctx=context(), embedder=embed, cache=SOURCE + "/cache")


def test_config_and_evidence_budgets():
    assert not MemoryConfig().associative.enabled
    with pytest.raises(ValueError, match="not both"):
        MemoryConfig(associative={"enabled": True}, scene_cues={"enabled": True})
    groups = {
        kind: [{"kind": kind, "rank": n} for n in range(30)]
        for kind in ("scene", "item", "profile")
    }
    result = budgeted(groups, {"scene": 5, "item": 15, "profile": 1}, 21)
    assert Counter(row["kind"] for row in result) == {"scene": 5, "item": 15, "profile": 1}
    result = budgeted(groups, {"scene": 5, "item": 15, "profile": 1}, 5)
    assert len(result) == 5
    assert {row["kind"] for row in result} >= {"scene", "item"}
    fused = rrf([{"single_high_score": 100, "agreement": 0.1}, {"agreement": 0.1}])
    assert fused["agreement"] > fused["single_high_score"]
    assert (
        bm25(
            "watercolor knee",
            {"right": "watercolor after knee injury", "wrong": "running marathon"},
        )["right"]
        > 0
    )


def test_peer_turns_never_cross_targets():
    messages = [
        Message("a", "user", [TextPart("Alice self")]),
        Message("b", "assistant", [TextPart("Self reply")]),
        Message("c", "user", [TextPart("Bob secret")], peer_id="bob"),
        Message("d", "assistant", [TextPart("Bob reply")]),
        Message("e", "user", [TextPart("Carol secret")], peer_id="carol"),
        Message("f", "assistant", [TextPart("Carol reply")]),
    ]
    grouped = group_messages(messages, context(), True, {"bob"}, True)
    assert [m["id"] for m in grouped["__self"]] == ["a", "b"]
    assert [m["id"] for m in grouped["bob"]] == ["c", "d"]
    assert "carol" not in grouped
    assert group_messages(messages, context(), False, set(), False) == {}

    prefix = Message("prefix", "assistant", [TextPart("Earlier speaker's fact")])
    self_only = group_messages([prefix, *messages[:2]], context(), True, set(), True)
    assert [m["id"] for m in self_only["__self"]] == ["prefix", "a", "b"]
    assert group_messages([prefix], context(), True, set(), True)["__self"][0]["id"] == "prefix"
    mixed = group_messages([prefix, *messages], context(), True, {"bob"}, True)
    assert "prefix" not in {m["id"] for turns in mixed.values() for m in turns}


@pytest.mark.asyncio
async def test_views_are_separate_hidden_scoped_and_idempotent(store):
    manager, index, fs, embed = store
    scene = node(
        "scene",
        "Evan paints while his knee heals.",
        views=[
            View("dialogue", "raw", "Evan: Painting keeps me busy."),
            View("scene", "scene", "creative recovery activity"),
            View("horizon", "GOAL_ARC", "future low-impact recovery pastime", 0.8),
        ],
    )
    await publish(store, scene)
    calls = embed.calls
    await publish(store, scene)
    assert embed.calls == calls
    assert "future low-impact" not in fs.data[scene.uri]
    corpus = await index.corpus(context(), [ROOT])
    assert list(corpus) == [scene.uri]
    views = await index.vectors(
        context(), [1, 0, 0, 0], ["evidence", "dialogue", "scene", "horizon"], [ROOT]
    )
    assert len(views) == 4 and len({row["id"] for row in views}) == 4
    assert not await index.corpus(context("bob"), [ROOT])
    assert not await index.corpus(context(account="other"), [ROOT])
    assert (
        await index.evidence(corpus[scene.uri], fs=fs, ctx=context(), source_cache={}) == scene.text
    )
    fs.data[scene.uri] += " changed"
    assert await index.evidence(corpus[scene.uri], fs=fs, ctx=context(), source_cache={}) is None
    fs.data[scene.uri] = scene.text
    fs.data[SOURCE] += " changed"
    assert await index.evidence(corpus[scene.uri], fs=fs, ctx=context(), source_cache={}) is None


@pytest.mark.asyncio
async def test_copy_move_delete_preserve_all_views(store):
    manager, index, fs, embed = store
    scene = node(
        "scene",
        "Grounded scene",
        views=[View("scene", "object", "one cue"), View("scene", "emotion", "another cue")],
    )
    await publish(store, scene)
    target = scene.uri.replace("test.md", "copied.md")
    fs.data[target] = fs.data[scene.uri]
    await manager.copy_uri_mapping(context(), scene.uri, target)
    assert len(await index.vectors(context(), [1, 0, 0, 0], ["evidence", "scene"], [ROOT])) == 6
    moved = target.replace("copied.md", "moved.md")
    fs.data[moved] = fs.data.pop(target)
    await manager.update_uri_mapping(context(), target, moved)
    corpus = await index.corpus(context(), [ROOT])
    assert set(corpus) == {scene.uri, moved}
    await manager.delete_uris(context(), [scene.uri, moved])
    assert not await index.corpus(context(), [ROOT])
    assert not await index.vectors(context(), [1, 0, 0, 0], ["scene"], [ROOT])


@pytest.mark.asyncio
async def test_retrieval_follows_graph_reranks_only_evidence_and_checks_dependencies(store):
    manager, index, fs, embed = store
    topic = node("topic", "Evan knee recovery")
    scene = node(
        "scene",
        "Evan paints during recovery.",
        topic_uris=[topic.uri],
        views=[
            View("dialogue", "raw", "I paint"),
            View("horizon", "GOAL_ARC", "TRIGGER_ONLY", 0.8),
        ],
    )
    item = node(
        "item",
        "Evan uses watercolor to stay busy while his knee heals.",
        scene_uris=[scene.uri],
        topic_uris=[topic.uri],
        dependencies={scene.uri: digest(scene.text)},
        views=[View("entity", "one", "TRIGGER_ONLY", 0.9, "recovery hobby")],
    )
    await publish(store, topic, scene, item)
    seen = []

    async def rerank(query, docs, fallback):
        seen.extend(docs)
        assert all("TRIGGER_ONLY" not in doc for doc in docs)
        return [0.8] * len(docs)

    retriever = AssociativeRetriever(index, fs, rerank)
    result = await retriever.retrieve(
        "recovery hobby", [1, 0, 0, 0], context(), targets=[ROOT], extra_filter=None, limit=20
    )
    assert {row["uri"] for row in result} == {scene.uri, item.uri}
    assert seen and all(row["abstract"] == fs.data[row["uri"]] for row in result)
    fs.data.pop(scene.uri)
    result = await retriever.retrieve(
        "recovery hobby", [1, 0, 0, 0], context(), targets=[ROOT], extra_filter=None, limit=20
    )
    assert not result


@pytest.mark.asyncio
async def test_valid_model_response_reused_after_resume():
    builder = object.__new__(AssociativeBuilder)
    import asyncio

    builder.index = SimpleNamespace(model_slots=asyncio.Semaphore(1))
    builder.fs, builder.ctx = Files(), context()
    builder.settings = AssociativeMemoryConfig()
    builder.vlm = SimpleNamespace(
        model="seed", get_completion_async=AsyncMock(return_value='{"ok": true}')
    )

    def validate(value):
        assert value["ok"] is True

    await builder.ask("source only", SOURCE, validate)
    await builder.ask("source only", SOURCE, validate)
    builder.vlm.get_completion_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_full_build_has_all_four_trigger_families_and_resumes_without_model_resampling(store):
    manager, index, fs, embed = store
    requests = []

    async def complete(*, prompt, **kwargs):
        requests.append(prompt)
        if "scene memory generation specialist" in prompt:
            output = {
                "title": "Evan knee recovery",
                "summary": "Evan paints while recovering.",
                "content": "Evan paints watercolors while recovering from a knee injury.",
            }
        elif "cognitive-memory triggers" in prompt:
            output = {
                "scene_attributes": {
                    k: f"grounded {k}" for k in ["scene", "object", "event", "emotion"]
                },
                "horizon_channels": {
                    k: {
                        "sent": "creative recovery pastime" if k == "GOAL_ARC" else None,
                        "confidence": 0.8 if k == "GOAL_ARC" else 0,
                    }
                    for k in [
                        "PERSONAL_ARC",
                        "AVOIDANCE_HABIT",
                        "GOAL_ARC",
                        "BELIEF_IN_ACTION",
                        "LEGACY_ANCHOR",
                    ]
                },
            }
        elif "topic extraction specialist" in prompt:
            output = {
                "title": "Evan recovery painting",
                "keywords": ["Evan", "recovery", "watercolors"],
            }
        elif "determining whether a memory scene belongs" in prompt:
            output = {"results": [{"topic_id": "topic_1", "match": True}]}
        elif "updating topics while maintaining" in prompt:
            output = {
                "title": "Evan recovery painting",
                "keywords": ["Evan", "recovery", "watercolors", "continued practice"],
            }
        elif "pulling out searchable memory items" in prompt:
            output = {
                "items": [
                    {
                        "content": "Evan paints watercolor during knee recovery.",
                        "scene_ids": re.findall(r"Scene ID: (scene_\d+)", prompt),
                        "temporal": "2026-09-21",
                        "keywords": ["watercolor"],
                        "query_patterns": ["Which recovery hobby suits Evan?"],
                    }
                ]
            }
        elif "one semantic step above a memory item" in prompt:
            output = {
                "triggers": [
                    {
                        "concept": "recovery pastime",
                        "bridge": "knee injury -> low impact creative hobby",
                        "confidence": 0.9,
                    }
                ]
            }
        else:
            raise AssertionError(prompt[:160])
        return json.dumps(output)

    vlm = SimpleNamespace(model="seed", get_completion_async=complete)
    config = SimpleNamespace(
        memory=MemoryConfig(associative={"enabled": True}),
        vlm=SimpleNamespace(model="seed", get_vlm_instance=lambda: vlm),
        embedding=SimpleNamespace(
            get_embedder=lambda: embed, model_dump=lambda **kw: {"fixture": True}
        ),
    )
    messages = [
        Message(
            "u",
            "user",
            [TextPart("Evan: I paint watercolors while my knee heals.")],
            created_at=STAMP,
        ),
        Message("a", "assistant", [TextPart("That sounds relaxing.")], created_at=STAMP),
    ]
    builder = AssociativeBuilder(index=index, fs=fs, ctx=context(), config=config)
    archive = SOURCE.rsplit("/", 1)[0]
    first = await builder.build(
        messages, archive, allow_self=True, allowed_peers=set(), peer_enabled=False
    )
    before_calls, before_embeddings = len(requests), embed.calls
    second = await builder.build(
        messages, archive, allow_self=True, allowed_peers=set(), peer_enabled=False
    )
    assert [c.uri for c in first] == [c.uri for c in second]
    assert len(requests) == before_calls and embed.calls == before_embeddings
    corpus = await index.corpus(context(), [ROOT])
    assert Counter(row["name"] for row in corpus.values()) == {"topic": 1, "scene": 1, "item": 1}
    fact = next(row for row in corpus.values() if row["name"] == "item")
    assert fact["metadata"]["lexical_text"].count("Which recovery hobby suits Evan?") == 2
    assert fact["abstract"].count("Which recovery hobby suits Evan?") == 1
    assert "Which recovery hobby suits Evan?" not in fs.data[fact["uri"]]
    assert "Which recovery hobby suits Evan?" not in fact["metadata"]["rerank_text"]
    views = await index.vectors(
        context(),
        [1, 0, 0, 0],
        ["evidence", "dialogue", "scene", "horizon", "entity", "bridge", "joint"],
        [ROOT],
    )
    assert {row["type"] for row in views} == {
        "evidence",
        "dialogue",
        "scene",
        "horizon",
        "entity",
        "bridge",
        "joint",
    }
    assert all("creative recovery pastime" not in fs.data[row["uri"]] for row in corpus.values())

    # A later archive updates the same topic, and supplements facts across BOTH scenes.
    await builder.build(
        messages, archive + "-later", allow_self=True, allowed_peers=set(), peer_enabled=False
    )
    corpus = await index.corpus(context(), [ROOT])
    topics = [row for row in corpus.values() if row["name"] == "topic"]
    scenes = [row for row in corpus.values() if row["name"] == "scene"]
    assert len(topics) == len(scenes) == 2
    assert len({row["metadata"]["topics"][0] for row in topics + scenes}) == 1
    connected = next(row for row in corpus.values() if len(row["metadata"]["scenes"]) == 2)
    assert await index.evidence(connected, fs=fs, ctx=context(), source_cache={})
    old_source = next(
        row["metadata"]["source_uri"]
        for row in scenes
        if "-later/" not in row["metadata"]["source_uri"]
    )
    fs.data[old_source] += " changed"
    assert await index.evidence(connected, fs=fs, ctx=context(), source_cache={}) is None


def test_fact_batches_cover_every_scene_with_overlap():
    scenes = [node("scene", "word " * 12) for _ in range(5)]
    for i, scene in enumerate(scenes):
        scene.uri = f"{ROOT}/scenes/{i}.md"
    from openviking.utils.token_estimation import estimate_text_tokens

    size = estimate_text_tokens(scenes[0].text)
    batches = list(scene_batches(scenes, size * 2))
    assert {s.uri for batch in batches for s in batch} == {s.uri for s in scenes}
    assert all(len(batch) == 2 for batch in batches)
    assert all(a[-1].uri == b[0].uri for a, b in zip(batches, batches[1:], strict=False))
    with pytest.raises(ValueError, match="input budget"):
        list(scene_batches(scenes, size - 1))


@pytest.mark.asyncio
async def test_transfer_respects_file_manifest_and_replaces_old_views(store):
    manager, index, fs, embed = store
    scene = node("scene", "Grounded", views=[View("scene", "a", "cue")])
    omitted = node("item", "Not copied")
    destination = Evidence.from_dict(scene.to_dict())
    destination.uri = scene.uri.replace("/associative/", "/copy/")
    destination.views.append(View("scene", "old", "stale cue"))
    await publish(store, scene, omitted, destination)
    target = ROOT.replace("/associative", "/copy")
    await manager.copy_uri_mapping(context(), ROOT, target, recursive=True, source_uris=[scene.uri])
    rows = await index.vectors(context(), [1, 0, 0, 0], ["evidence", "scene"], [target])
    assert len(rows) == 2
    assert {row["uri"] for row in rows} == {destination.uri}


@pytest.mark.asyncio
async def test_required_reranker_does_not_silently_evaluate_fallback():
    retriever = HierarchicalRetriever.__new__(HierarchicalRetriever)
    retriever.rerank_max_input_tokens = 0

    def fail(*args):
        raise RuntimeError("provider unavailable")

    retriever._rerank_client = SimpleNamespace(rerank_batch=fail)
    with pytest.raises(RuntimeError, match="reranking failed"):
        await retriever._rerank_scores("query", ["evidence"], [0.03], strict=True)
    assert await retriever._rerank_scores("query", ["evidence"], [0.03]) == [0.03]
