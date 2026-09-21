# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Topic -> scene -> item retrieval with four trigger families and evidence reranking."""

import asyncio
from difflib import SequenceMatcher

from openviking.retrieve.associative_ranking import bm25, budgeted, cosine, rrf
from openviking.storage.expr import Contains
from openviking.telemetry import get_current_telemetry


class AssociativeRetriever:
    def __init__(self, index, fs, reranker):
        self.index, self.fs, self.reranker = index, fs, reranker
        self.settings = index.settings

    async def retrieve(
        self, query, vector, ctx, *, targets, extra_filter, limit, threshold=0, score_gte=False
    ):
        corpus = await self.index.corpus(ctx, targets, extra_filter)
        if not corpus:
            return None  # Existing namespaces without this evidence layer retain ordinary OV retrieval.
        source_cache, validated = {}, {}
        telemetry = get_current_telemetry()

        async def body(uri):
            if uri not in validated:
                validated[uri] = await self.index.evidence(
                    corpus[uri], fs=self.fs, ctx=ctx, source_cache=source_cache
                )
            return validated[uri]

        async def dense(uris):
            rows = await self.index.vectors(ctx, vector, ["evidence"], targets, extra_filter, uris)
            return {row["uri"]: cosine(vector, row["vector"]) for row in rows if row["uri"] in uris}

        async def rank(uris, top_k, *, distinct_topics=False):
            if not uris:
                return []
            scores = rrf(
                [
                    bm25(
                        query,
                        {
                            uri: corpus[uri]["metadata"].get("lexical_text")
                            or corpus[uri]["abstract"]
                            for uri in uris
                        },
                    ),
                    await dense(uris),
                ]
            )
            candidate_ids = sorted(scores, key=lambda uri: (-scores[uri], uri))[
                : self.settings.initial_candidates
            ]
            documents = await asyncio.gather(*(body(uri) for uri in candidate_ids))
            candidates = [
                (uri, text) for uri, text in zip(candidate_ids, documents, strict=True) if text
            ]
            if not candidates:
                return []
            fallback = [scores[uri] for uri, _ in candidates]
            # Triggers never enter the reranker. Its input is independently grounded evidence.
            ranked_scores = await self.reranker(
                query,
                [corpus[uri]["metadata"].get("rerank_text") or text for uri, text in candidates],
                fallback,
            )
            ranked = [
                (uri, score) for (uri, _), score in zip(candidates, ranked_scores, strict=True)
            ]
            ranked = sorted(ranked, key=lambda entry: (-entry[1], entry[0]))
            if distinct_topics:
                roots, unique = set(), []
                for uri, score in ranked:
                    root = next(iter(corpus[uri]["metadata"].get("topics", [])), uri)
                    if root not in roots:
                        roots.add(root)
                        unique.append((uri, score))
                ranked = unique
            return ranked[:top_k]

        topic_ids = {uri for uri, row in corpus.items() if row["name"] == "topic"}
        scene_ids = {uri for uri, row in corpus.items() if row["name"] == "scene"}
        item_ids = {uri for uri, row in corpus.items() if row["name"] == "item"}
        topics = await rank(topic_ids, self.settings.topic_top_k, distinct_topics=True)
        topic_roots = (
            set().union(
                *({topic, *corpus[topic]["metadata"].get("topics", [])} for topic, _ in topics)
            )
            if topics
            else set()
        )
        topic_pool = {
            uri for uri in scene_ids if set(corpus[uri]["metadata"]["topics"]) & topic_roots
        }

        async def family_scores(families):
            rows = await self.index.vectors(ctx, vector, families, targets, extra_filter)
            scores = {}
            for row in rows:
                uri = row["uri"]
                if uri in scene_ids:
                    scores[uri] = max(scores.get(uri, -1), cosine(vector, row["vector"]))
            return scores

        # Stage5: max within each view family; rank fusion ACROSS raw dialogue,
        # descriptive scenes and associative horizons. No raw-score max fusion.
        channels = await asyncio.gather(
            family_scores(["dialogue"]),
            family_scores(["scene"]),
            family_scores(["horizon"]),
        )
        associative_scores = rrf(list(channels), k=30)
        extra_scenes = set(
            sorted(associative_scores, key=lambda uri: (-associative_scores[uri], uri))[
                : self.settings.trigger_top_k
            ]
        )
        scenes = await rank(topic_pool | extra_scenes, self.settings.scene_top_k)
        selected_scenes = {uri for uri, _ in scenes}
        connected_items = {
            uri for uri in item_ids if set(corpus[uri]["metadata"]["scenes"]) & selected_scenes
        }

        # Entity/Bridge: concept, bridge and joint embedding views; group duplicate
        # concepts before applying the trigger budget so one concept may recall multiple facts.
        trigger_rows = await self.index.vectors(
            ctx, vector, ["entity", "bridge", "joint"], targets, extra_filter
        )
        triggers = {}
        for row in trigger_rows:
            if row["uri"] not in item_ids:
                continue
            view = row["metadata"]["view"]
            score = cosine(vector, row["vector"])
            if (
                view["confidence"] < self.settings.trigger_confidence
                or score < self.settings.trigger_cosine_gate
            ):
                continue
            concept = " ".join(view["concept"].casefold().split())
            key = next(
                (old for old in triggers if SequenceMatcher(None, old, concept).ratio() >= 0.9),
                concept,
            )
            trigger = triggers.setdefault(key, {"score": score, "items": set()})
            trigger["score"] = max(trigger["score"], score)
            trigger["items"].add(row["uri"])
        fired = sorted(triggers.values(), key=lambda trigger: -trigger["score"])[
            : self.settings.trigger_top_k
        ]
        extra_items = set().union(*(trigger["items"] for trigger in fired)) if fired else set()
        items = await rank(connected_items | extra_items, self.settings.item_top_k)

        def records(ranked):
            return [
                {
                    **corpus[uri],
                    "abstract": validated[uri],
                    "_score": score,
                    "_final_score": score,
                    "category": f"associative_{corpus[uri]['name']}",
                }
                for uri, score in ranked
                if (score >= threshold if score_gte else score > threshold)
            ]

        profiles = []
        if self.settings.final_profile_k:
            profile_filter = self.index.primary._merge_filters(
                extra_filter, Contains("uri", "/memories/profile.md")
            )
            profiles = await self.index.primary.filter_in_tenant(
                ctx=ctx,
                context_type="memory",
                target_directories=targets,
                extra_filter=profile_filter,
                level=[2],
                limit=self.settings.final_profile_k,
            )
            profiles = [
                {**row, "_score": 1.0, "_final_score": 1.0}
                for row in profiles
                if row["uri"].endswith("/memories/profile.md")
            ]
        result = budgeted(
            {"scene": records(scenes), "item": records(items), "profile": profiles},
            {
                "scene": self.settings.final_scene_k,
                "item": self.settings.final_item_k,
                "profile": self.settings.final_profile_k,
            },
            limit,
        )
        telemetry.set(
            "search.associative",
            {
                "topic_candidates": len(topic_ids),
                "topic_selected": len(topics),
                "scene_topic_pool": len(topic_pool),
                "scene_trigger_candidates": len(extra_scenes),
                "item_connected_pool": len(connected_items),
                "item_trigger_candidates": len(extra_items),
                "fired_item_triggers": len(fired),
                "result_count": len(result),
            },
        )
        return result
