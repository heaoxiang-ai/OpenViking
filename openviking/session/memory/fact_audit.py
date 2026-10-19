# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Model-declared session diagnostics, separate from extraction policy.

The records describe the model's decisions; they are not proof of semantic
coverage and never authorize writes, change routing, or trigger another call.
"""

from __future__ import annotations

import io
import json
import tokenize
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from openviking.session.memory.utils.json_parser import JsonUtils

# Leading underscores cannot be dynamic Pydantic operation field names, so this
# diagnostic key cannot collide with an enabled custom memory type.
AUDIT_KEY = "_memory_audit"
AUDIT_COMMENT_PREFIX = "# memory_audit:"


class FactDecision(BaseModel):
    """A compact, model-declared disposition, separate from memory operations."""

    model_config = ConfigDict(extra="forbid")

    fact: StrictStr = Field(min_length=1)
    source_indexes: list[Annotated[StrictInt, Field(ge=0)]] = Field(default_factory=list)
    source_memory: StrictStr = ""
    decision: Literal["create", "update", "already_present", "omit", "remove"]
    destinations: list[StrictStr] = Field(default_factory=list)
    reason: StrictStr = ""

    @model_validator(mode="after")
    def validate_disposition(self) -> FactDecision:
        if not self.fact.strip():
            raise ValueError("fact must be non-blank")
        if not self.source_indexes and not self.source_memory.strip():
            raise ValueError("a source message index or read-memory reference is required")
        if any(not destination.strip() for destination in self.destinations):
            raise ValueError("destinations must be non-blank references")
        if self.decision in {"create", "update", "already_present"} and not self.destinations:
            raise ValueError("saved or already-present facts require a destination")
        if self.decision in {"omit", "remove"} and not self.reason.strip():
            raise ValueError("omitted or removed facts require a reason")
        if self.decision in {"already_present", "remove"} and not self.source_memory.strip():
            raise ValueError("already-present or removed facts require a read-memory reference")
        return self


def render_fact_audit_rules(output_format: str) -> str:
    """Render a protocol-compatible metadata envelope, without a new SDK method."""
    envelope = (
        'Emit each record as a standalone Python comment: # memory_audit: {"fact": ..., ...}. '
        "Place comments outside all string literals, before the corresponding SDK calls. "
        "Comments are diagnostic metadata, not SDK methods. A no-change program may have "
        "audit comments followed by sdk.commit()."
        if output_format == "python"
        else f'Add a top-level "{AUDIT_KEY}" array to the operations JSON. The array is '
        "diagnostic metadata, not a memory type or write operation. Keep it outside all "
        "memory fields. A no-change response may include audit records and empty operations."
    )
    return (
        "\n\n## Fact Decision Audit\n"
        f"{envelope}\n"
        "Use concise records with fact (a short factual statement), source_indexes (the "
        "Conversation History indexes, not dataset labels), source_memory (an existing read "
        "object binding or page_id when relevant), decision (create/update/already_present/"
        "omit/remove), destinations (object variables in Python or page_ids in JSON), and "
        "reason (required for omit/remove). References are strings. For an omitted fact "
        "destinations may be empty; for a moved fact name the surviving destination. Group "
        "pure small talk rather than copying whole conversations. Account for eligible new "
        "facts and each substantive old fact removed or weakened by an update/delete; "
        "unchanged old facts do not need individual records. For a removal, cite an explicit "
        "correction, retraction, deletion request, or duplicate/move destination. Keep concise "
        "decisions outside memory content; do not output private reasoning. This audit is only a "
        "model report: do not claim that it independently verifies semantic completeness "
        "or successful persistence.\n"
    )


def collect_fact_audit(content: str, output_format: str) -> dict[str, Any]:
    """Collect diagnostics without making otherwise-valid writes fail or retry.

    Only real Python COMMENT tokens are inspected. A marker inside quoted
    conversation/memory text can never become an audit record.
    """
    records = []
    errors = []
    raw_records: list[Any] = []
    try:
        if output_format == "python":
            for token in tokenize.generate_tokens(io.StringIO(content).readline):
                if token.type == tokenize.COMMENT and token.string.startswith(AUDIT_COMMENT_PREFIX):
                    try:
                        raw_records.append(json.loads(token.string[len(AUDIT_COMMENT_PREFIX) :]))
                    except (TypeError, ValueError):
                        errors.append("invalid_comment_json")
        elif output_format == "json":
            parsed = JsonUtils.loads(content)
            if isinstance(parsed, dict) and AUDIT_KEY in parsed:
                raw = parsed[AUDIT_KEY]
                if isinstance(raw, list):
                    raw_records = raw
                else:
                    errors.append("audit_must_be_an_array")
    except (ValueError, tokenize.TokenError, IndentationError, SyntaxError):
        errors.append("unreadable_audit_envelope")
    for index, raw_record in enumerate(raw_records):
        try:
            records.append(FactDecision.model_validate(raw_record).model_dump())
        except (TypeError, ValueError):
            # Avoid duplicating unvalidated model text in errors; raw I/O is already traced.
            errors.append(f"invalid_record:{index}")
    return {
        "status": "invalid" if errors else "reported" if records else "missing",
        "model_declared": True,
        "records": records,
        "errors": errors,
    }
