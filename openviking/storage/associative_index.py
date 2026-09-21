# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tenant-scoped multi-view index for independently persisted associative evidence."""

import asyncio
import json

from openviking.models.embedder.base import embed_compat
from openviking.server.error_mapping import is_not_found_error
from openviking.session.associative.types import Evidence, View, digest
from openviking.storage.expr import And, Eq, In
from openviking.storage.vector_migration import rewrite_transfer_uri, uri_in_transfer_scope
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking.utils.time_utils import get_current_timestamp
from openviking_cli.exceptions import PermissionDeniedError

FIELDS = [
    "id",
    "uri",
    "type",
    "name",
    "description",
    "abstract",
    "search_tags",
    "created_at",
    "updated_at",
    "level",
    "context_type",
]


class AssociativeBackend(VikingVectorIndexBackend):
    """Use OV's ACL, entry manifests and rollback for all filesystem transfers."""

    def _rewrite_transfer_record(self, record, **kwargs):
        payload = super()._rewrite_transfer_record(record, **kwargs)
        metadata = json.loads(record["description"])
        source, target = kwargs["source_uri"], kwargs["target_uri"]

        def remap(uri):
            return (
                rewrite_transfer_uri(uri, source, target)
                if uri_in_transfer_scope(uri, source, recursive=True)
                else uri
            )

        metadata["scenes"] = [remap(uri) for uri in metadata.get("scenes", [])]
        metadata["topics"] = [remap(uri) for uri in metadata.get("topics", [])]
        metadata["dependencies"] = {
            remap(uri): value for uri, value in metadata.get("dependencies", {}).items()
        }
        # Provenance remains tied to the immutable original archive.
        view = metadata["view"]
        payload["id"] = AssociativeIndex.record_id(
            kwargs["ctx"].account_id, payload["uri"], view["family"], view["key"]
        )
        payload["description"] = json.dumps(metadata, ensure_ascii=False)
        return payload


class AssociativeIndex:
    def __init__(self, primary, settings):
        self.primary = primary
        self.settings = settings
        self.store = AssociativeBackend(
            primary._config.model_copy(update={"name": primary.collection_name + "_associative"})
        )
        self.store.acl_manager = primary.acl_manager
        self.model_slots = asyncio.Semaphore(settings.max_concurrent)
        self.embedding_slots = asyncio.Semaphore(settings.max_concurrent)

    @staticmethod
    def record_id(account: str, uri: str, family: str, key: str) -> str:
        return digest([account, uri, family, key])[:32]

    async def scope(self, ctx, targets=None, extra_filter=None):
        return self.store._build_scope_filter(
            ctx=ctx,
            context_type="memory",
            target_directories=targets,
            extra_filter=extra_filter,
            level=[2],
            acl_enabled=await self.store._acl_enabled(ctx),
        )

    async def corpus(self, ctx, targets=None, extra_filter=None) -> dict[str, dict]:
        scope = await self.scope(ctx, targets, extra_filter)
        result = {}
        cursor = None
        seen_cursors = set()
        while True:
            page, next_cursor = await self.store.scroll(
                ctx=ctx,
                filter=And([scope, Eq("type", "evidence")]),
                output_fields=FIELDS,
                limit=256,
                cursor=cursor,
            )
            for row in page:
                row["metadata"] = json.loads(row["description"])
                result[row["uri"]] = row
            if next_cursor is None:
                break
            if next_cursor in seen_cursors:
                raise RuntimeError("Associative corpus cursor repeated")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return result

    async def vectors(self, ctx, vector, families, targets=None, extra_filter=None, uris=None):
        scope = await self.scope(ctx, targets, extra_filter)
        filters = [scope, In("type", families)]
        if uris is not None:
            if not uris:
                return []
            filters.append(In("uri", list(uris)))
        rows = await self.store.search(
            ctx=ctx,
            query_vector=vector,
            filter=And(filters),
            output_fields=FIELDS + ["vector"],
            limit=self.settings.initial_candidates,
        )
        for row in rows:
            row["metadata"] = json.loads(row["description"])
        return rows

    async def publish(self, node: Evidence, *, fs, embedder, ctx, cache):
        """Idempotently write evidence and all views; accepted embeddings survive retries."""
        await fs.write_file(node.uri, node.text, ctx=ctx)
        views = [View("evidence", "primary", node.embedding_text), *node.views]
        expected = [
            self.record_id(ctx.account_id, node.uri, view.family, view.key) for view in views
        ]

        async def publish_view(view):
            record_id = self.record_id(ctx.account_id, node.uri, view.family, view.key)
            payload_key = digest([node.metadata(), view.__dict__])
            cache_uri = f"{cache}/embedding-{payload_key}.json"
            try:
                embedded = json.loads(await fs.read_file(cache_uri, ctx=ctx))
            except Exception as exc:
                if not is_not_found_error(exc):
                    raise
                async with self.embedding_slots:
                    result = await embed_compat(embedder, view.text, is_query=False)
                embedded = {
                    "vector": result.dense_vector,
                    "sparse_vector": result.sparse_vector or {},
                }
                await fs.write_file(cache_uri, json.dumps(embedded), ctx=ctx)
            metadata = {**node.metadata(), "view": view.__dict__}
            payload = {
                "id": record_id,
                "uri": node.uri,
                "type": view.family,
                "name": node.kind,
                "description": json.dumps(metadata, ensure_ascii=False),
                "abstract": node.embedding_text,
                "context_type": "memory",
                "level": 2,
                "account_id": ctx.account_id,
                "owner_user_id": ctx.user.user_id,
                "search_tags": node.search_tags,
                "created_at": node.timestamp,
                "updated_at": get_current_timestamp(),
                "active_count": 0,
                **embedded,
            }
            if not await self.store.upsert(payload, ctx=ctx):
                raise RuntimeError("Associative index write failed")

        # Settle all in-flight views before propagating a failure to the native task.
        results = await asyncio.gather(
            *(publish_view(view) for view in views), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        # Changed/deleted views must not survive a resumed materialization.
        old, _ = await self.store._read_uri_transfer_entries(
            ctx, [node.uri], include_full_records=False
        )
        obsolete = [row["id"] for row in old if row["id"] not in expected]
        if obsolete:
            await self.store.delete(obsolete, ctx=ctx)

    async def evidence(self, row, *, fs, ctx, source_cache):
        """Recheck current filesystem ACL, evidence identity, and immutable source archive."""
        try:
            text = await fs.read_file(row["uri"], ctx=ctx)
            metadata = row["metadata"]
            if digest(text) != metadata["body_sha256"]:
                return None
            source = metadata["source_uri"]
            if source not in source_cache:
                source_cache[source] = digest(await fs.read_file(source, ctx=ctx))
            if source_cache[source] != metadata["source_sha256"]:
                return None
            for uri, expected in metadata.get("dependencies", {}).items():
                if uri not in source_cache:
                    source_cache[uri] = digest(await fs.read_file(uri, ctx=ctx))
                if source_cache[uri] != expected:
                    return None
            return text
        except Exception as exc:
            if is_not_found_error(exc) or isinstance(exc, (PermissionError, PermissionDeniedError)):
                return None
            raise

    async def transfer(self, ctx, source_uri, target_uri, recursive=False, *, move=False, **kwargs):
        method = self.store.update_uri_mapping if move else self.store.copy_uri_mapping
        return await method(ctx, source_uri, target_uri, recursive, **kwargs)
