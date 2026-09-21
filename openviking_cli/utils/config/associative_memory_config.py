# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One opt-in switch for the complete associative evidence pipeline."""

from pydantic import BaseModel, Field


class AssociativeMemoryConfig(BaseModel):
    enabled: bool = False
    max_concurrent: int = Field(default=32, ge=1, le=256)
    max_scene_messages: int = Field(default=20, ge=2, le=100)
    max_scene_tokens: int = Field(default=6000, ge=512)
    trigger_max_messages: int = Field(default=10, ge=1)
    triggers_per_item: int = Field(default=5, ge=1, le=10)
    trigger_confidence: float = Field(default=0.7, ge=0, le=1)
    trigger_cosine_gate: float = Field(default=0.85, ge=-1, le=1)
    trigger_top_k: int = Field(default=10, ge=1)
    initial_candidates: int = Field(default=250, ge=20)
    topic_top_k: int = Field(default=15, ge=1)
    scene_top_k: int = Field(default=24, ge=1)
    item_top_k: int = Field(default=40, ge=1)
    final_scene_k: int = Field(default=5, ge=1)
    final_item_k: int = Field(default=15, ge=1)
    final_profile_k: int = Field(default=1, ge=0, le=5)
    max_output_tokens: int = Field(default=16384, ge=1024)
    rerank_required: bool = True
