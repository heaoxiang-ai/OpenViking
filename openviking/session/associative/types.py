# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Serializable evidence and retrieval-only views; generated views never enter evidence."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal


def digest(value) -> str:
    text = (
        value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    )
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class View:
    family: str
    key: str
    text: str
    confidence: float = 1.0
    concept: str = ""


@dataclass
class Evidence:
    uri: str
    kind: Literal["topic", "scene", "item"]
    text: str
    embedding_text: str
    source_uri: str
    source_sha256: str
    timestamp: str
    scene_uris: list[str] = field(default_factory=list)
    topic_uris: list[str] = field(default_factory=list)
    message_ids: list[str] = field(default_factory=list)
    views: list[View] = field(default_factory=list)
    search_tags: list[str] = field(default_factory=list)
    dependencies: dict[str, str] = field(default_factory=dict)
    lexical_text: str = ""
    rerank_text: str = ""
    topic_revision: int = 0

    def metadata(self) -> dict:
        return {
            "version": 1,
            "kind": self.kind,
            "body_sha256": digest(self.text),
            "source_uri": self.source_uri,
            "source_sha256": self.source_sha256,
            "scenes": self.scene_uris,
            "topics": self.topic_uris,
            "message_ids": self.message_ids,
            "dependencies": self.dependencies,
            "lexical_text": self.lexical_text or self.embedding_text,
            "rerank_text": self.rerank_text or self.text,
            "topic_revision": self.topic_revision,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Evidence":
        value = dict(value)
        value["views"] = [View(**view) for view in value.get("views", [])]
        return cls(**value)
