# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reserved derived-record discriminator in the unified context collection."""

from openviking.storage.expr import Eq, RawDSL

ENTITY_LINK_TYPE = "entity_link"


def entity_records():
    return Eq("type", ENTITY_LINK_TYPE)


def context_records():
    # Negative membership keeps legacy records with empty/missing type visible.
    return RawDSL({"op": "must_not", "field": "type", "conds": [ENTITY_LINK_TYPE]})
