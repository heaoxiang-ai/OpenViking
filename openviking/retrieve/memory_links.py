# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Request-scoped native memory links with one-hop candidate expansion."""

import asyncio
import math
from typing import Any, Dict, List, Optional

from openviking.core.uri_validation import validate_request_viking_uri
from openviking.server.identity import RequestContext
from openviking.session.memory.dataclass import StoredLink
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.expr import And, FilterExpr, In, RawDSL
from openviking.storage.vikingdb_manager import VikingDBManagerProxy
from openviking_cli.exceptions import InvalidURIError
from openviking_cli.retrieve.types import MatchedContext
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryLinks:
    """Read link metadata and resolve targets through the ordinary search scope."""

    LOOKUP_BATCH_SIZE = 200

    def __init__(
        self,
        fs: Any,
        proxy: VikingDBManagerProxy,
        ctx: RequestContext,
        target_directories: List[str],
        scope_dsl: Optional[FilterExpr | Dict[str, Any]],
    ):
        self.fs = fs
        self.proxy = proxy
        self.ctx = ctx
        self.target_directories = target_directories
        self.scope_dsl = scope_dsl
        self._semaphore = asyncio.Semaphore(10)
        self._metadata: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._readable: set[str] = set()

    async def _read(self, uri: str) -> Dict[str, List[Dict[str, Any]]]:
        if uri in self._metadata:
            return self._metadata[uri]
        metadata: Dict[str, List[Dict[str, Any]]] = {"links": [], "backlinks": []}
        try:
            async with self._semaphore:
                raw = await self.fs.read_file(uri, ctx=self.ctx)
            memory = MemoryFileUtils.read(raw, uri=uri)
            self._readable.add(uri)
            for key in metadata:
                for value in getattr(memory, key):
                    try:
                        link = StoredLink.model_validate(value)
                        source = validate_request_viking_uri(link.from_uri, self.ctx)
                        target = validate_request_viking_uri(link.to_uri, self.ctx)
                    except (TypeError, ValueError, InvalidURIError):
                        continue
                    if source == target:
                        continue
                    if (key == "links" and source != uri) or (key == "backlinks" and target != uri):
                        continue
                    link.weight = (
                        min(1.0, max(0.0, link.weight)) if math.isfinite(link.weight) else 0.5
                    )
                    metadata[key].append(
                        {**link.model_dump(), "from_uri": source, "to_uri": target}
                    )
        except Exception as exc:
            # Stale/deleted/unreadable files must not break ordinary search.
            logger.debug("Cannot read memory links for %s: %s", uri, type(exc).__name__)
        self._metadata[uri] = metadata
        return metadata

    @staticmethod
    def _other(uri: str, link: Dict[str, Any]) -> str:
        return link["to_uri"] if link["from_uri"] == uri else link["from_uri"]

    async def _resolve(self, uris: List[str]) -> Dict[str, Dict[str, Any]]:
        if not uris:
            return {}
        # Reuse exactly the tenant/ACL/target/filter machinery used by vector recall.
        # An untrusted URI in a memory file is never itself an access grant.
        records = {}
        for start in range(0, len(uris), self.LOOKUP_BATCH_SIZE):
            batch = uris[start : start + self.LOOKUP_BATCH_SIZE]
            conditions: List[FilterExpr] = [In("uri", batch)]
            if self.scope_dsl:
                conditions.append(
                    RawDSL(self.scope_dsl) if isinstance(self.scope_dsl, dict) else self.scope_dsl
                )
            offset = 0
            while True:
                rows = await self.proxy.filter_in_tenant(
                    context_type="memory",
                    target_directories=self.target_directories,
                    extra_filter=And(conditions),
                    level=[2],
                    limit=len(batch),
                    offset=offset,
                )
                records.update({row["uri"]: row for row in rows if row.get("uri") in batch})
                # Legacy duplicate index rows must not crowd out other targets.
                if len(rows) < len(batch) or all(uri in records for uri in batch):
                    break
                offset += len(rows)
        await asyncio.gather(*(self._read(uri) for uri in records))
        return {uri: row for uri, row in records.items() if uri in self._readable}

    async def _load(self, uris: List[str]):
        metadata = await asyncio.gather(*(self._read(uri) for uri in uris))
        targets = list(
            dict.fromkeys(
                self._other(uri, link)
                for uri, fields in zip(uris, metadata, strict=True)
                for links in fields.values()
                for link in links
            )
        )
        records = await self._resolve(targets)
        return metadata, records

    async def expand(self, candidates: List[Dict[str, Any]]):
        """Add every eligible L2 memory referenced by the initial seeds, once."""
        seeds = [
            row["uri"]
            for row in candidates
            if row.get("context_type") == "memory" and row.get("level") == 2
        ]
        metadata, records = await self._load(seeds)
        seen = {row["uri"] for row in candidates}
        weights: Dict[str, float] = {}
        for uri, fields in zip(seeds, metadata, strict=True):
            for links in fields.values():
                for link in links:
                    target = self._other(uri, link)
                    if (
                        target in records
                        and target not in seen
                        and str(records[target].get("abstract", "")).strip()
                    ):
                        weights[target] = max(weights.get(target, 0.0), link["weight"])
        additions = []
        for uri in sorted(weights, key=lambda target: (-weights[target], target)):
            additions.append({**records[uri], "_score": 0.0, "_link_expanded": True})
        return candidates + additions

    async def attach(self, matches: List[MatchedContext]) -> None:
        """Expose only links whose opposite endpoint passes the current query scope."""
        memories = [
            match for match in matches if match.context_type.value == "memory" and match.level == 2
        ]
        metadata, records = await self._load([match.uri for match in memories])
        for match, fields in zip(memories, metadata, strict=True):
            for key, links in fields.items():
                setattr(
                    match,
                    key,
                    [link for link in links if self._other(match.uri, link) in records],
                )
