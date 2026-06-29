# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace

from openviking.storage.queuefs import semantic_processor as semantic_processor_module
from openviking.storage.queuefs.semantic_processor import SemanticProcessor


def _patch_semantic_limits(monkeypatch, *, abstract_max_chars=256, overview_max_chars=4000):
    config = SimpleNamespace(
        semantic=SimpleNamespace(
            abstract_max_chars=abstract_max_chars,
            overview_max_chars=overview_max_chars,
        )
    )
    monkeypatch.setattr(semantic_processor_module, "get_openviking_config", lambda: config)


def test_structured_l0_l1_output_uses_model_abstract_and_cleans_overview(monkeypatch):
    _patch_semantic_limits(monkeypatch)
    processor = SemanticProcessor()
    generated = """<!-- ABSTRACT -->
This is a complete L0 sentence generated specifically for retrieval.

<!-- OVERVIEW -->
# README

This is the L1 overview body.

## Quick Navigation

- Read README.md
"""

    abstract = processor._extract_abstract_from_overview(generated)
    overview, abstract = processor._enforce_size_limits(generated, abstract)

    assert abstract == "This is a complete L0 sentence generated specifically for retrieval."
    assert overview.startswith("# README")
    assert "<!-- ABSTRACT -->" not in overview
    assert "<!-- OVERVIEW -->" not in overview


def test_abstract_truncation_prefers_complete_sentence(monkeypatch):
    _patch_semantic_limits(monkeypatch, abstract_max_chars=80)
    processor = SemanticProcessor()
    abstract = (
        "This is a complete sentence. "
        "This second sentence contains onboarding material that would be cut."
    )

    overview, abstract = processor._enforce_size_limits("# README\n\nBody", abstract)

    assert overview == "# README\n\nBody"
    assert abstract == "This is a complete sentence."

