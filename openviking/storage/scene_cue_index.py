# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Auxiliary Event vectors in the same vector service, never an evidence store."""

from typing import Any

from openviking.server.identity import RequestContext
from openviking.storage.expr import In
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend


class SceneCueIndex:
    def __init__(self, primary, settings):
        self.primary = primary
        self.settings = settings
        config = primary._config.model_copy(
            update={"name": primary.collection_name + "_scene_cues"}
        )
        self.store = VikingVectorIndexBackend(config)
        # Use the authoritative account ACL settings/materialization logic.
        self.store.acl_manager = primary.acl_manager

    @property
    def recall_enabled(self) -> bool:
        return self.settings.enabled and self.settings.recall_enabled

    @staticmethod
    def is_event(record: dict) -> bool:
        return (
            record.get("context_type") == "memory"
            and record.get("level", 2) == 2
            and "/memories/events/" in record.get("uri", "")
        )

    async def sync(self, record: dict, ctx: RequestContext, *, refresh=False, embedding=None):
        """Refresh metadata, replace a cue vector, or invalidate a stale view.

        Canonical writes come first. Query-time validation makes partial failures safe:
        a dangling/stale auxiliary record cannot substitute for the current evidence.
        """
        if not self.is_event(record):
            return
        record_id = record["id"]
        old = await self.store.get_strict([record_id], ctx=ctx)
        if refresh:
            if not embedding:
                await self.store.delete([record_id], ctx=ctx)
                return
            vector, sparse = embedding
        elif old and old[0].get("abstract") == record.get("abstract"):
            vector = old[0].get("vector")
            sparse = old[0].get("sparse_vector")
        else:
            if old:
                await self.store.delete([record_id], ctx=ctx)
            return
        payload = {**record, "vector": vector, "sparse_vector": sparse or {}}
        if not await self.store.upsert(payload, ctx=ctx):
            raise RuntimeError(f"Failed to write scene cue vector for {record['uri']}")

    async def search(self, *, ctx: RequestContext, **kwargs: Any) -> list[dict]:
        """Use identical scope/ACL filters, then revalidate against canonical records.

        Revalidation covers revoked permissions, changed tags, edits and deletes even
        if an auxiliary write failed, or this feature was disabled during maintenance.
        Only canonical evidence is returned; cue text never becomes an answer source.
        """
        if not self.recall_enabled:
            return []
        if kwargs.get("context_type") not in (None, "memory"):
            return []
        levels = kwargs.get("level")
        if levels is not None and 2 not in levels:
            return []
        hits = await self.store.search_in_tenant(ctx=ctx, **kwargs)
        if not hits:
            return []
        filters = self.primary._merge_filters(
            kwargs.get("extra_filter"), In("uri", [hit["uri"] for hit in hits])
        )
        current = await self.primary.filter_in_tenant(
            ctx=ctx,
            context_type="memory",
            target_directories=kwargs.get("target_directories"),
            extra_filter=filters,
            level=[2],
            limit=len(hits),
        )
        by_uri = {record["uri"]: record for record in current}
        return [
            {**by_uri[hit["uri"]], "_score": hit["_score"]}
            for hit in hits
            if hit["uri"] in by_uri and hit.get("abstract") == by_uri[hit["uri"]].get("abstract")
        ]
