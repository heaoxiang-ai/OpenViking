# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Structured extraction failures carried through asynchronous Commit tasks."""

from copy import deepcopy
from typing import Any


class MemoryExtractionError(RuntimeError):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        retryable: bool = False,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.diagnostics = deepcopy(diagnostics or {})

    def task_failure(self) -> dict[str, Any]:
        """No program, prompt, reasoning text, or provider error text is persisted here."""
        return {
            "source": "memory_extraction",
            "kind": self.kind,
            "retryable": self.retryable,
            "diagnostics": deepcopy(self.diagnostics),
        }
