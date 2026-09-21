# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Archive-driven T-Mem build stages on OV models, storage and identity boundaries.

No evaluation query, gold answer, or existing extraction schema enters this pipeline.
Accepted model output and the materialization manifest are durable before indexing.
"""

import json
from collections import defaultdict
from types import SimpleNamespace

from openviking.models.embedder.base import embed_compat
from openviking.models.vlm.llm import parse_json_from_response
from openviking.retrieve.associative_ranking import bm25, cosine, rrf
from openviking.server.error_mapping import is_not_found_error
from openviking.session.associative.tmem_prompts.memory_item_prompts import ITEM_EXTRACTION_PROMPT
from openviking.session.associative.tmem_prompts.scene_prompts import (
    CONV_BOUNDARY_DETECTION_PROMPT,
    DEFAULT_CUSTOM_INSTRUCTIONS,
    SCENE_GENERATION_PROMPT,
)
from openviking.session.associative.tmem_prompts.topic_prompts import (
    TOPIC_EXTRACTION_PROMPT,
    TOPIC_MATCH_PROMPT,
    TOPIC_UPDATE_PROMPT,
)
from openviking.session.associative.tmem_prompts.trigger_prompts import (
    ENTITY_BRIDGE_TRIGGER_PROMPT,
    HORIZON_TRIGGER_KEYS,
    SCENE_TRIGGER_KEYS,
    build_prompt,
)
from openviking.session.associative.types import Evidence, View, digest
from openviking.session.memory.memory_isolation_handler import (
    MemoryIsolationHandler,
    peer_user_space,
)
from openviking.utils.token_estimation import estimate_text_tokens, truncate_text_to_token_budget
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


def group_messages(messages, ctx, allow_self, allowed_peers, peer_enabled):
    """A peer never receives another peer's user turn or its assistant continuation."""
    isolation = MemoryIsolationHandler(
        ctx,
        SimpleNamespace(messages=messages),
        allow_self=allow_self,
        allowed_peer_ids=allowed_peers,
        peer_memory_enabled=peer_enabled,
    )
    groups = defaultdict(list)
    # A self-only archive can begin with an assistant turn (including an entirely
    # assistant-only batch). Preserve it. In mixed/peer scope an unowned prefix
    # remains ambiguous and must not be broadcast into another person's memory.
    target = (
        "__self"
        if allow_self
        and not allowed_peers
        and not any(getattr(m, "peer_id", None) for m in messages)
        else None
    )
    for message in messages:
        if message.role == "user":
            target = isolation._message_target_id(message)
        elif message.role == "assistant" and getattr(message, "peer_id", None):
            target = isolation._message_target_id(
                SimpleNamespace(role="user", peer_id=message.peer_id)
            )
        if not target or message.role not in {"user", "assistant"}:
            continue
        content = "\n".join(
            part.text for part in message.parts if getattr(part, "type", None) == "text"
        )
        if content.strip():
            groups[target].append(
                {
                    "id": message.id,
                    "role": message.role,
                    "timestamp": message.created_at,
                    "content": content,
                }
            )
    return dict(groups)


def dialogue(messages):
    return "\n".join(f"[{m['timestamp']}] {m['role']}: {m['content']}" for m in messages)


def required_text(data, name):
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Associative model output missing {name}")
    return value.strip()


def scene_batches(scenes, token_budget):
    """Bound fact extraction without dropping a scene; overlap for connected facts."""
    batch, tokens = [], 0
    for scene in scenes:
        size = estimate_text_tokens(scene.text)
        if size > token_budget:
            raise ValueError(f"Scene exceeds associative fact input budget: {scene.uri}")
        if batch and tokens + size > token_budget:
            yield batch
            overlap = batch[-1]
            overlap_tokens = estimate_text_tokens(overlap.text)
            batch = [overlap] if overlap_tokens + size <= token_budget else []
            tokens = overlap_tokens if batch else 0
        batch.append(scene)
        tokens += size
    if batch:
        yield batch


class AssociativeBuilder:
    def __init__(self, *, index, fs, ctx, config):
        self.index, self.fs, self.ctx, self.config = index, fs, ctx, config
        self.settings = config.memory.associative
        self.vlm = config.vlm.get_vlm_instance()
        self.embedder = config.embedding.get_embedder()

    async def read_json(self, uri):
        try:
            return json.loads(await self.fs.read_file(uri, ctx=self.ctx))
        except Exception as exc:
            if is_not_found_error(exc):
                return None
            raise

    async def write_json(self, uri, value):
        await self.fs.write_file(uri, json.dumps(value, ensure_ascii=False), ctx=self.ctx)

    async def ask(self, prompt, cache, validate):
        # Include model identity/output settings, never credentials, in durable cache keys.
        key = digest([getattr(self.vlm, "model", ""), prompt, self.settings.max_output_tokens])
        uri = f"{cache}/model-{key}.json"
        cached = await self.read_json(uri)
        if cached is not None:
            validate(cached)
            return cached
        for attempt in range(3):
            async with self.index.model_slots:
                response = await self.vlm.get_completion_async(
                    prompt=prompt,
                    thinking=False,
                    max_tokens=self.settings.max_output_tokens,
                )
            try:
                data = parse_json_from_response(response)
                if not isinstance(data, dict):
                    raise ValueError("Associative extraction requires a JSON object")
                validate(data)
            except (ValueError, KeyError, TypeError):
                # Invalid structured output is technical failure, not an accepted sample.
                if attempt == 2:
                    raise
                continue
            await self.write_json(uri, data)
            return data
        raise AssertionError("unreachable")

    async def split(self, messages, cache):
        groups, current = [], []
        for message in messages:
            if estimate_text_tokens(dialogue([message])) > self.settings.max_scene_tokens:
                raise ValueError(f"Message exceeds associative scene input budget: {message['id']}")
            split = len(current) >= self.settings.max_scene_messages
            split = split or (
                bool(current)
                and estimate_text_tokens(dialogue(current + [message]))
                > self.settings.max_scene_tokens
            )
            if not split and len(current) >= 2:
                prompt = CONV_BOUNDARY_DETECTION_PROMPT.format(
                    conversation_history=dialogue(current),
                    new_messages=dialogue([message]),
                    time_gap_info=f"Last timestamp: {current[-1]['timestamp']}; next: {message['timestamp']}",
                )

                def valid(result):
                    if type(result.get("should_end")) is not bool:
                        raise ValueError("Missing boolean scene boundary")

                result = await self.ask(prompt, cache, valid)
                split = result["should_end"]
            if split:
                groups.append(current)
                current = []
            current.append(message)
        if current:
            groups.append(current)
        return groups

    async def scene(self, messages, root, source, source_sha, cache, tags):
        prompt = SCENE_GENERATION_PROMPT.format(
            conversation_start_time=messages[0]["timestamp"],
            conversation=dialogue(messages),
            custom_instructions=DEFAULT_CUSTOM_INSTRUCTIONS,
        )

        def valid(result):
            for name in ("title", "summary", "content"):
                required_text(result, name)

        result = await self.ask(prompt, cache, valid)
        scene_id = digest([source, [m["id"] for m in messages]])[:24]
        node = Evidence(
            uri=f"{root}/scenes/{scene_id}.md",
            kind="scene",
            text=f"# {result['title']}\n\n{result['content']}\n\n## Source dialogue\n{dialogue(messages)}\n",
            embedding_text=result["content"],
            source_uri=source,
            source_sha256=source_sha,
            timestamp=messages[0]["timestamp"],
            message_ids=[m["id"] for m in messages],
            search_tags=tags,
            views=[View("dialogue", "original", dialogue(messages))],
        )
        node.embedding_text = " ".join([result["title"], result["summary"], result["content"]])
        node.lexical_text = " ".join(
            [result["title"]] * 3 + [result["summary"]] * 2 + [result["content"]]
        )
        node.rerank_text = result["content"]
        if len(messages) <= self.settings.trigger_max_messages:

            def valid_triggers(result):
                for key in SCENE_TRIGGER_KEYS:
                    required_text(result["scene_attributes"], key)
                for key in HORIZON_TRIGGER_KEYS:
                    channel = result["horizon_channels"][key]
                    if channel.get("sent") is not None and not isinstance(channel["sent"], str):
                        raise ValueError("Invalid Horizon sentence")
                    confidence = channel.get("confidence")
                    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                        raise ValueError("Invalid Horizon confidence")

            cues = await self.ask(build_prompt("A", dialogue(messages)), cache, valid_triggers)
            node.views.extend(
                View("scene", key, cues["scene_attributes"][key]) for key in SCENE_TRIGGER_KEYS
            )
            for key in HORIZON_TRIGGER_KEYS:
                channel = cues["horizon_channels"][key]
                if channel["sent"] and channel["sent"].strip():
                    node.views.append(View("horizon", key, channel["sent"], channel["confidence"]))
        return node, result

    async def topics(self, scene, description, existing, root, cache):
        # Keep immutable versions, with one stable topic root for all scene links.
        # New keywords from any version remain searchable, without destructive updates.
        topics = {}
        for row in existing.values():
            if row["name"] != "topic":
                continue
            root_uri = next(iter(row["metadata"].get("topics", [])), row["uri"])
            revision = (row["metadata"].get("topic_revision", 0), row["uri"])
            if root_uri not in topics or revision > (
                topics[root_uri]["metadata"].get("topic_revision", 0),
                topics[root_uri]["uri"],
            ):
                topics[root_uri] = row
        ranked = bm25(description["summary"], {uri: row["abstract"] for uri, row in topics.items()})
        if topics:
            embedding = await embed_compat(self.embedder, description["summary"], is_query=True)
            vectors = await self.index.vectors(
                self.ctx,
                embedding.dense_vector,
                ["evidence"],
                [root],
                uris={uri for uri, row in existing.items() if row["name"] == "topic"},
            )
            dense_scores = {}
            for row in vectors:
                topic_root = next(iter(row["metadata"].get("topics", [])), row["uri"])
                dense_scores[topic_root] = max(
                    dense_scores.get(topic_root, -1), cosine(embedding.dense_vector, row["vector"])
                )
            ranked = rrf([ranked, dense_scores])
        shortlist = sorted(topics, key=lambda uri: (-ranked.get(uri, 0), uri))[
            : self.settings.topic_top_k
        ]
        matches = []
        if shortlist:
            labels = {f"topic_{i}": uri for i, uri in enumerate(shortlist, 1)}
            prompt = TOPIC_MATCH_PROMPT.format(
                scene_subject=description["title"],
                scene_summary=description["summary"],
                num_topics=len(labels),
                topics_text=json.dumps(
                    {
                        key: truncate_text_to_token_budget(
                            topics[uri]["abstract"],
                            self.settings.max_scene_tokens * 3 // len(labels),
                        )
                        for key, uri in labels.items()
                    }
                ),
            )

            def valid(result):
                if not isinstance(result.get("results"), list):
                    raise ValueError("Missing topic matches")
                for match in result["results"]:
                    if match.get("topic_id") not in labels or type(match.get("match")) is not bool:
                        raise ValueError("Unknown topic or invalid match")

            result = await self.ask(prompt, cache, valid)
            matches = [labels[row["topic_id"]] for row in result["results"] if row["match"]]

        def valid_new(result):
            required_text(result, "title")
            if not isinstance(result.get("keywords"), list) or not all(
                isinstance(k, str) for k in result["keywords"]
            ):
                raise ValueError("Invalid topic keywords")

        if matches:
            versions = []
            for topic_uri in matches:
                result = await self.ask(
                    TOPIC_UPDATE_PROMPT.format(
                        existing_topic=truncate_text_to_token_budget(
                            topics[topic_uri]["abstract"], self.settings.max_scene_tokens * 3
                        ),
                        new_scene=description["content"],
                    ),
                    cache,
                    valid_new,
                )
                text = f"# {result['title']}\n\n" + ", ".join(result["keywords"])
                versions.append(
                    Evidence(
                        f"{root}/topics/{digest([topic_uri, scene.uri, text])[:24]}.md",
                        "topic",
                        text,
                        text,
                        scene.source_uri,
                        scene.source_sha256,
                        scene.timestamp,
                        scene_uris=[scene.uri],
                        topic_uris=[topic_uri],
                        search_tags=scene.search_tags,
                        topic_revision=topics[topic_uri]["metadata"].get("topic_revision", 0) + 1,
                    )
                )
                versions[-1].embedding_text = " ".join([result["title"]] * 2 + result["keywords"])
                versions[-1].rerank_text = versions[-1].embedding_text
            return matches, versions

        result = await self.ask(
            TOPIC_EXTRACTION_PROMPT.format(scenes=description["content"]), cache, valid_new
        )
        # An immutable per-scene topic root avoids cross-session lost updates. Later scenes
        # attach to the existing root through their own topic_uris, without rewriting it.
        uri = f"{root}/topics/{digest([scene.uri, result['title']])[:24]}.md"
        text = f"# {result['title']}\n\n" + ", ".join(result["keywords"])
        return [uri], [
            Evidence(
                uri,
                "topic",
                text,
                " ".join([result["title"]] * 2 + result["keywords"]),
                scene.source_uri,
                scene.source_sha256,
                scene.timestamp,
                scene_uris=[scene.uri],
                topic_uris=[uri],
                search_tags=scene.search_tags,
                rerank_text=" ".join([result["title"]] * 2 + result["keywords"]),
            )
        ]

    async def items(self, topic_uri, topic_title, scenes, root, source, source_sha, cache, tags):
        labels = {f"scene_{i}": scene for i, scene in enumerate(scenes, 1)}
        prompt = ITEM_EXTRACTION_PROMPT.format(
            topic_id=topic_uri,
            topic_title=topic_title,
            scenes_content="\n\n".join(
                f"Scene ID: {key}\n{scene.text}" for key, scene in labels.items()
            ),
            reference_time=scenes[-1].timestamp,
        )

        def valid(result):
            if not isinstance(result.get("items"), list):
                raise ValueError("Missing atomic items")
            for item in result["items"]:
                required_text(item, "content")
                for field in ("keywords", "query_patterns"):
                    values = item.get(field) or []
                    if not isinstance(values, list) or not all(
                        isinstance(value, str) for value in values
                    ):
                        raise ValueError(f"Invalid fact {field}")
                links = item.get("scene_ids")
                if (
                    not isinstance(links, list)
                    or not links
                    or any(link not in labels for link in links)
                ):
                    raise ValueError("Item references a nonexistent scene")

        result = await self.ask(prompt, cache, valid)
        nodes = []
        for item in result["items"]:
            scene_uris = list(dict.fromkeys(labels[key].uri for key in item["scene_ids"]))
            text = item["content"]
            if item.get("temporal"):
                text += f"\nTime: {item['temporal']}"
            if item.get("spatial"):
                text += f"\nLocation: {item['spatial']}"
            uri = f"{root}/items/{digest([topic_uri, text, scene_uris])[:24]}.md"
            node = Evidence(
                uri,
                "item",
                text,
                text,
                source,
                source_sha,
                scenes[-1].timestamp,
                scene_uris=scene_uris,
                topic_uris=[topic_uri],
                search_tags=tags,
            )
            keywords, patterns = (
                " ".join(item.get("keywords") or []),
                " ".join(item.get("query_patterns") or []),
            )
            qualifiers = [str(item.get(key) or "") for key in ("temporal", "spatial")]
            node.embedding_text = " ".join([item["content"], patterns, keywords, *qualifiers])
            node.lexical_text = " ".join(
                [item["content"]] * 3 + [patterns] * 2 + [keywords, *qualifiers]
            )
            node.rerank_text = " ".join([item["content"], keywords])
            node.dependencies = {
                labels[key].uri: digest(labels[key].text) for key in item["scene_ids"]
            }
            node.dependencies.update(
                {labels[key].source_uri: labels[key].source_sha256 for key in item["scene_ids"]}
            )
            trigger_prompt = ENTITY_BRIDGE_TRIGGER_PROMPT.format(
                trigger_count=self.settings.triggers_per_item,
                item_content=text,
                item_temporal=item.get("temporal", ""),
            )

            def valid_trigger(result):
                triggers = result.get("triggers")
                if (
                    not isinstance(triggers, list)
                    or len(triggers) > self.settings.triggers_per_item * 2
                ):
                    raise ValueError("Invalid item triggers")
                for trigger in triggers:
                    required_text(trigger, "concept")
                    required_text(trigger, "bridge")
                    confidence = trigger.get("confidence")
                    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                        raise ValueError("Invalid trigger confidence")

            triggers = await self.ask(trigger_prompt, cache, valid_trigger)
            for number, trigger in enumerate(triggers["triggers"]):
                if trigger["confidence"] < self.settings.trigger_confidence:
                    continue
                concept, bridge = trigger["concept"], trigger["bridge"]
                for family, cue in (
                    ("entity", concept),
                    ("bridge", bridge),
                    ("joint", concept + "\n" + bridge),
                ):
                    node.views.append(
                        View(family, str(number), cue, trigger["confidence"], concept)
                    )
            nodes.append(node)
        return nodes

    async def build(
        self, messages, archive_uri, *, allow_self, allowed_peers, peer_enabled, tags=None
    ):
        groups = group_messages(messages, self.ctx, allow_self, allowed_peers, peer_enabled)
        contexts = []
        for target, selected in groups.items():
            user_space = peer_user_space(self.ctx.user.user_id, target)
            logger.info("Associative build: user_space=%s messages=%d", user_space, len(selected))
            root = f"viking://user/{user_space}/memories/associative"
            # Cache input per identity; source snapshots contain only this target's turns.
            fingerprint = digest(
                [
                    selected,
                    self.settings.model_dump(),
                    self.config.vlm.model,
                    self.config.embedding.model_dump(mode="json"),
                ]
            )
            cache = f"{archive_uri}/associative/{digest(root)[:16]}/{fingerprint}"
            source = f"{cache}/source.json"
            snapshot = json.dumps(
                {"archive_uri": archive_uri, "messages": selected}, ensure_ascii=False
            )
            old = await self.read_json(source)
            if old is None:
                await self.fs.write_file(source, snapshot, ctx=self.ctx)
            else:
                assert old == json.loads(snapshot), "Associative source changed"
            source_sha = digest(snapshot)
            manifest_uri = f"{cache}/manifest.json"
            manifest = await self.read_json(manifest_uri)
            if manifest is None:
                corpus_uri = f"{cache}/prior-corpus.json"
                existing = await self.read_json(corpus_uri)
                if existing is None:
                    existing = await self.index.corpus(self.ctx, [root])
                    await self.write_json(corpus_uri, existing)
                nodes, topic_scenes = [], defaultdict(list)
                chunks = await self.split(selected, cache)
                logger.info("Associative build: user_space=%s scenes=%d", user_space, len(chunks))
                for chunk in chunks:
                    scene, description = await self.scene(
                        chunk, root, source, source_sha, cache, tags or []
                    )
                    topic_uris, created = await self.topics(
                        scene, description, existing, root, cache
                    )
                    scene.topic_uris = topic_uris
                    nodes.extend(created)
                    nodes.append(scene)
                    for topic in created:
                        existing[topic.uri] = {
                            "uri": topic.uri,
                            "name": "topic",
                            "abstract": topic.embedding_text,
                            "metadata": topic.metadata(),
                        }
                    for topic_uri in topic_uris:
                        topic_scenes[topic_uri].append(scene)
                for topic_uri, related in topic_scenes.items():
                    # Include earlier scenes of the SAME topic for supplementary connected
                    # facts. Preserve each fact's exact source-scene hashes for invalidation.
                    related_before = []
                    source_cache = {}
                    token_count = sum(estimate_text_tokens(scene.text) for scene in related)
                    for row in sorted(
                        existing.values(), key=lambda r: r.get("created_at", ""), reverse=True
                    ):
                        if row["name"] != "scene" or topic_uri not in row["metadata"]["topics"]:
                            continue
                        if len(related_before) >= 15:
                            break
                        text = await self.index.evidence(
                            row, fs=self.fs, ctx=self.ctx, source_cache=source_cache
                        )
                        if (
                            not text
                            or token_count + estimate_text_tokens(text)
                            > self.settings.max_scene_tokens * 3
                        ):
                            continue
                        token_count += estimate_text_tokens(text)
                        metadata = row["metadata"]
                        related_before.append(
                            Evidence(
                                row["uri"],
                                "scene",
                                text,
                                row["abstract"],
                                metadata["source_uri"],
                                metadata["source_sha256"],
                                row["created_at"],
                                topic_uris=metadata["topics"],
                            )
                        )
                    for batch in scene_batches(
                        related_before + related, self.settings.max_scene_tokens * 3
                    ):
                        nodes.extend(
                            await self.items(
                                topic_uri,
                                existing[topic_uri]["abstract"],
                                batch,
                                root,
                                source,
                                source_sha,
                                cache,
                                tags or [],
                            )
                        )
                nodes = list({node.uri: node for node in nodes}.values())
                manifest = {
                    "source_sha256": source_sha,
                    "nodes": [node.to_dict() for node in nodes],
                }
                await self.write_json(manifest_uri, manifest)
            if manifest["source_sha256"] != source_sha:
                raise ValueError("Associative materialization source changed")
            logger.info(
                "Associative materialization: user_space=%s nodes=%d",
                user_space,
                len(manifest["nodes"]),
            )
            for raw in manifest["nodes"]:
                node = Evidence.from_dict(raw)
                await self.index.publish(
                    node, fs=self.fs, embedder=self.embedder, ctx=self.ctx, cache=cache
                )
                if node.kind != "topic":
                    contexts.append(
                        SimpleNamespace(uri=node.uri, category=f"associative_{node.kind}")
                    )
            await self.write_json(f"{cache}/complete.json", {"manifest_sha256": digest(manifest)})
        return contexts
