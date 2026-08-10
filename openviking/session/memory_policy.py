# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Memory extraction policy for session commits."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

from openviking_cli.exceptions import InvalidArgumentError

_POLICY_KEYS = {"self", "peer", "memory_types", "working_memory"}
_TARGET_KEYS = {"enabled", "memory_types"}
_WORKING_MEMORY_KEYS = {"enabled"}
_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"", "0", "false", "no", "off"}

AGENT_MEMORY_TYPES = {"cases", "trajectories", "experiences"}
_EXPERIENCE_DEPENDENCIES = AGENT_MEMORY_TYPES


def _memory_policy_shape_error(key: str) -> str:
    if key == "working_memory":
        return "memory_policy.working_memory must be an object"
    return "memory_policy target must be an object"


def _memory_policy_keys_error(key: str) -> str:
    if key == "working_memory":
        return "memory_policy.working_memory supports only: enabled"
    return "memory_policy target supports only: enabled, memory_types"


def _parse_enabled(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value

    warnings.warn(
        f"memory_policy.{key}.enabled should be a boolean; legacy coercion is deprecated",
        FutureWarning,
        stacklevel=3,
    )
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False

    return bool(value)


def _parse_memory_types(data: Any, *, field: str) -> Optional[set[str]]:
    if data is None:
        return None
    if not isinstance(data, list):
        raise InvalidArgumentError(f"{field} must be a list")
    memory_types = set()
    for item in data:
        if not isinstance(item, str) or not item:
            raise InvalidArgumentError(f"{field} must contain non-empty strings")
        memory_types.add(item)
    return memory_types


def _parse_target(
    data: Any,
    *,
    default_enabled: bool,
    key: str,
) -> tuple[bool, Optional[set[str]], bool]:
    if data is None:
        return default_enabled, None, False
    if not isinstance(data, dict):
        raise InvalidArgumentError(_memory_policy_shape_error(key))
    allowed_keys = _WORKING_MEMORY_KEYS if key == "working_memory" else _TARGET_KEYS
    extra_keys = set(data) - allowed_keys
    if extra_keys:
        raise InvalidArgumentError(_memory_policy_keys_error(key))
    enabled = _parse_enabled(data["enabled"], key=key) if "enabled" in data else default_enabled
    if key == "working_memory":
        return enabled, None, False
    has_memory_types = "memory_types" in data
    memory_types = _parse_memory_types(
        data.get("memory_types"),
        field=f"memory_policy.{key}.memory_types",
    )
    return enabled, memory_types, has_memory_types


@dataclass
class MemoryPolicy:
    """Effective memory policy for one commit.

    ``self`` and ``peer`` have independent type filters. The legacy top-level
    ``memory_types`` input is still accepted and normalized into this shape.
    """

    self_enabled: bool = True
    peer_enabled: bool = True
    self_memory_types: Optional[set[str]] = None
    peer_memory_types: Optional[set[str]] = None
    working_memory_enabled: bool = True

    @property
    def memory_types(self) -> Optional[set[str]]:
        """Compatibility view used by callers that only need the union."""
        if self.self_memory_types is None or self.peer_memory_types is None:
            return None
        return set(self.self_memory_types) | set(self.peer_memory_types)

    @classmethod
    def default(cls) -> "MemoryPolicy":
        return cls()

    @classmethod
    def from_dict(cls, data: Any) -> "MemoryPolicy":
        if data is None:
            return cls.default()
        if isinstance(data, MemoryPolicy):
            return data
        if not isinstance(data, dict):
            raise InvalidArgumentError("memory_policy must be an object")
        extra_keys = set(data) - _POLICY_KEYS
        if extra_keys:
            raise InvalidArgumentError(
                "memory_policy supports only: " + ", ".join(sorted(_POLICY_KEYS))
            )

        self_enabled, self_types, self_types_explicit = _parse_target(
            data.get("self"), default_enabled=True, key="self"
        )
        peer_enabled, peer_types, peer_types_explicit = _parse_target(
            data.get("peer"), default_enabled=True, key="peer"
        )
        working_enabled, _, _ = _parse_target(
            data.get("working_memory"), default_enabled=True, key="working_memory"
        )

        legacy_types = _parse_memory_types(
            data.get("memory_types"), field="memory_policy.memory_types"
        )
        if not self_types_explicit:
            self_types = None if legacy_types is None else set(legacy_types)
        if not peer_types_explicit:
            peer_types = None if legacy_types is None else set(legacy_types) - AGENT_MEMORY_TYPES

        if peer_types and peer_types & AGENT_MEMORY_TYPES:
            raise InvalidArgumentError(
                "memory_policy.peer.memory_types does not support agent memory types: "
                + ", ".join(sorted(peer_types & AGENT_MEMORY_TYPES))
            )

        return cls(
            self_enabled=self_enabled,
            peer_enabled=peer_enabled,
            self_memory_types=self_types,
            peer_memory_types=peer_types,
            working_memory_enabled=working_enabled,
        )

    def validate_memory_types(self, known_memory_types: set[str]) -> None:
        configured = set()
        if self.self_memory_types is not None:
            configured.update(self.self_memory_types)
        if self.peer_memory_types is not None:
            configured.update(self.peer_memory_types)
        unknown = configured - known_memory_types
        if unknown:
            raise InvalidArgumentError(
                "Unknown memory_policy memory types: " + ", ".join(sorted(unknown))
            )

    def resolve(
        self,
        known_memory_types: set[str],
        *,
        agent_evolution_enabled: bool,
    ) -> "MemoryPolicy":
        """Expand defaults and apply the account-scoped Agent switch."""
        self_types = (
            set(known_memory_types)
            if self.self_memory_types is None
            else set(self.self_memory_types)
        )
        if "experiences" in self_types:
            self_types.update(_EXPERIENCE_DEPENDENCIES)
        peer_types = (
            set(known_memory_types) - AGENT_MEMORY_TYPES
            if self.peer_memory_types is None
            else set(self.peer_memory_types)
        )
        if not agent_evolution_enabled:
            self_types -= AGENT_MEMORY_TYPES
        return MemoryPolicy(
            self_enabled=self.self_enabled,
            peer_enabled=self.peer_enabled,
            self_memory_types=self_types,
            peer_memory_types=peer_types,
            working_memory_enabled=self.working_memory_enabled,
        )

    def to_dict(self) -> dict[str, Any]:
        self_target: dict[str, Any] = {"enabled": self.self_enabled}
        peer_target: dict[str, Any] = {"enabled": self.peer_enabled}
        if self.self_memory_types is not None:
            self_target["memory_types"] = sorted(self.self_memory_types)
        if self.peer_memory_types is not None:
            peer_target["memory_types"] = sorted(self.peer_memory_types)
        data: dict[str, Any] = {
            "self": self_target,
            "peer": peer_target,
        }
        if not self.working_memory_enabled:
            data["working_memory"] = {"enabled": False}
        return data
