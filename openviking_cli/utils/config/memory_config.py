# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class MemoryConfig(BaseModel):
    """Memory configuration for OpenViking."""

    version: str = Field(
        default="v1",
        description="Memory implementation version: 'v1' (legacy) or 'v2' (new templating system)",
    )
    agent_scope_mode: str = Field(
        default="user+agent",
        description=(
            "Agent memory namespace mode: 'user+agent' keeps agent memory isolated by "
            "(user_id, agent_id), while 'agent' shares agent memory across users of the same agent."
        ),
    )

    custom_templates_dir: str = Field(
        default="",
        description="Custom memory templates directory. If set, templates from this directory will be loaded in addition to built-in templates",
    )
    v2_lock_retry_interval_seconds: float = Field(
        default=0.2,
        ge=0.0,
        description=(
            "Retry interval (seconds) when SessionCompressorV2 fails to acquire memory subtree "
            "locks. Set to 0 for immediate retries."
        ),
    )
    v2_lock_max_retries: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum retries for SessionCompressorV2 memory lock acquisition. "
            "0 means unlimited retries."
        ),
    )

    llm_tools: List[str] = Field(
        default_factory=lambda: ["read", "search"],
        description=(
            "Read-only tools exposed to memory extraction LLM. "
            "Supported values: 'read', 'search', 'ls'."
        ),
    )

    model_config = {"extra": "forbid"}

    @field_validator("agent_scope_mode")
    @classmethod
    def validate_agent_scope_mode(cls, value: str) -> str:
        if value not in {"user+agent", "agent"}:
            raise ValueError("memory.agent_scope_mode must be 'user+agent' or 'agent'")
        return value

    @field_validator("llm_tools")
    @classmethod
    def validate_llm_tools(cls, value: List[str]) -> List[str]:
        allowed = {"read", "search", "ls"}
        unique_tools = []
        seen = set()

        for tool in value:
            if tool not in allowed:
                raise ValueError(
                    "memory.llm_tools only supports: 'read', 'search', 'ls'"
                )
            if tool not in seen:
                unique_tools.append(tool)
                seen.add(tool)

        if "read" not in seen:
            raise ValueError("memory.llm_tools must include 'read'")

        return unique_tools

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MemoryConfig":
        """Create configuration from dictionary."""
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
