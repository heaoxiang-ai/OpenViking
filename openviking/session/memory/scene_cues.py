# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Source-bound scene descriptors, kept outside Event evidence and its primary embedding."""

import hashlib

from openviking.session.memory.dataclass import MemoryField, MemoryFile, MemoryTypeSchema
from openviking.session.memory.merge_op.base import FieldType, MergeOp

SCENE_FIELD = "scene_cue"
SOURCE_FIELD = "scene_cue_source_sha256"
MAX_CUE_CHARS = 2000


def scene_field() -> MemoryField:
    return MemoryField(
        name=SCENE_FIELD,
        field_type=FieldType.STRING,
        merge_op=MergeOp.IMMUTABLE,
        description=(
            "A concise descriptive retrieval cue (at most 2000 characters) grounded ONLY in "
            "this Event's selected message ranges. Describe the situation, participants, "
            "explicit time/place, stated intent and concrete facts, including relevant details "
            "in the ChatLog that a short event title might omit. Preserve who said/did what; "
            "assistant suggestions are not user actions or preferences. Use the source language. "
            "Do not invent facts, future scenarios, hypothetical questions or answers. "
            "This is a retrieval aid, not evidence; the original Summary/ChatLog is authoritative."
        ),
    )


def source_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def configure_scene_field(schema: MemoryTypeSchema, enabled: bool) -> None:
    """Keep this deployment-owned field consistent across old account templates."""
    schema.fields = [field for field in schema.fields if field.name != SCENE_FIELD]
    if enabled:
        schema.fields.append(scene_field())


def valid_scene_cue(memory: MemoryFile) -> str:
    """Reject stale descriptors after a manual body edit or a legacy metadata-only copy."""
    cue = memory.extra_fields.get(SCENE_FIELD)
    if not isinstance(cue, str) or not cue.strip() or len(cue) > MAX_CUE_CHARS:
        return ""
    if memory.extra_fields.get(SOURCE_FIELD) != source_digest(memory.content or ""):
        return ""
    return cue.strip()
