# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
import warnings
from pathlib import Path

import pytest

from openviking.session.memory.dataclass import MemoryField, MemoryTypeSchema
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.merge_op.base import FieldType
from openviking.session.memory_policy import MemoryPolicy
from openviking.session.session import _effective_memory_types, _effective_self_memory_types
from openviking_cli.exceptions import InvalidArgumentError


def test_memory_policy_defaults_to_self_and_peer():
    policy = MemoryPolicy.from_dict(None)

    assert policy.self_enabled is True
    assert policy.peer_enabled is True
    assert policy.memory_types is None
    assert policy.working_memory_enabled is True


def test_memory_policy_can_disable_peer_memory():
    policy = MemoryPolicy.from_dict({"peer": {"enabled": False}})

    assert policy.self_enabled is True
    assert policy.peer_enabled is False


def test_effective_memory_types_respect_disabled_targets():
    policy = MemoryPolicy.from_dict(
        {
            "self": {"enabled": False, "memory_types": ["experiences"]},
            "peer": {"enabled": False, "memory_types": ["profile"]},
        }
    )

    assert _effective_self_memory_types(policy) == set()
    assert _effective_memory_types(policy) == set()


def test_memory_policy_uses_top_level_memory_types():
    policy = MemoryPolicy.from_dict(
        {
            "self": {"enabled": False},
            "peer": {"enabled": True},
            "memory_types": ["profile", "events"],
        }
    )

    assert policy.self_enabled is False
    assert policy.peer_enabled is True
    assert policy.memory_types == {"profile", "events"}
    assert policy.to_dict() == {
        "self": {"enabled": False, "memory_types": ["events", "profile"]},
        "peer": {"enabled": True, "memory_types": ["events", "profile"]},
    }
    assert policy.to_legacy_dict() == {
        "self": {"enabled": False},
        "peer": {"enabled": True},
        "memory_types": ["events", "profile"],
    }


def test_memory_policy_legacy_fallback_does_not_expand_asymmetric_user_types():
    policy = MemoryPolicy.from_dict(
        {
            "self": {"enabled": True, "memory_types": ["profile", "experiences"]},
            "peer": {"enabled": True, "memory_types": ["events"]},
        }
    ).resolve(
        {"cases", "events", "experiences", "profile", "trajectories"},
        agent_evolution_enabled=True,
    )

    assert policy.to_legacy_dict() == {
        "self": {"enabled": True},
        "peer": {"enabled": True},
        "memory_types": ["cases", "experiences", "trajectories"],
    }


def test_memory_policy_can_disable_working_memory():
    policy = MemoryPolicy.from_dict({"working_memory": {"enabled": False}})

    assert policy.self_enabled is True
    assert policy.peer_enabled is True
    assert policy.working_memory_enabled is False
    assert policy.to_dict() == {
        "self": {"enabled": True},
        "peer": {"enabled": True},
        "working_memory": {"enabled": False},
    }


def test_memory_policy_rejects_invalid_working_memory_shape():
    with pytest.raises(InvalidArgumentError, match="working_memory must be an object"):
        MemoryPolicy.from_dict({"working_memory": False})

    with pytest.raises(InvalidArgumentError, match="working_memory supports only: enabled"):
        MemoryPolicy.from_dict({"working_memory": {"enabled": True, "mode": "summary"}})


def test_memory_policy_legacy_enabled_compatibility_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "memory_policy_compat.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))

    for case in cases:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            policy = MemoryPolicy.from_dict({"peer": {"enabled": case["value"]}})

        assert policy.peer_enabled is case["expected"]
        deprecations = [item for item in caught if item.category is FutureWarning]
        assert bool(deprecations) is case["deprecated"]


def test_memory_policy_legacy_enabled_warning_is_user_visible():
    with pytest.warns(FutureWarning, match="legacy coercion is deprecated"):
        MemoryPolicy.from_dict({"peer": {"enabled": "false"}})


def test_memory_policy_rejects_invalid_memory_types():
    policy = MemoryPolicy.from_dict({"memory_types": ["profile", "missing"]})

    with pytest.raises(InvalidArgumentError, match="missing"):
        policy.validate_memory_types({"profile"})

    policy = MemoryPolicy.from_dict({"self": {"enabled": True, "memory_types": ["experiences"]}})
    assert policy.self_memory_types == {"experiences"}
    assert policy.resolve(
        {"cases", "events", "experiences", "profile", "trajectories"},
        agent_evolution_enabled=True,
    ).self_memory_types == {"cases", "experiences", "trajectories"}


def test_memory_policy_supports_independent_self_and_peer_types():
    policy = MemoryPolicy.from_dict(
        {
            "self": {"enabled": True, "memory_types": ["profile", "experiences"]},
            "peer": {"enabled": True, "memory_types": ["events"]},
        }
    )

    assert policy.self_memory_types == {"experiences", "profile"}
    assert policy.peer_memory_types == {"events"}


def test_memory_policy_rejects_agent_types_for_peer_memory():
    with pytest.raises(InvalidArgumentError, match="does not support agent memory types"):
        MemoryPolicy.from_dict({"peer": {"enabled": True, "memory_types": ["experiences"]}})


def test_memory_policy_legacy_agent_dependencies():
    evolution_policy = MemoryPolicy.from_dict({"memory_types": ["profile", "experiences"]})
    assert evolution_policy.memory_types == {
        "profile",
        "cases",
        "trajectories",
        "experiences",
    }

    incomplete_policy = MemoryPolicy.from_dict(
        {"memory_types": ["profile", "cases", "trajectories"]}
    )
    assert incomplete_policy.memory_types == {"profile"}


async def test_initialize_memory_files_respects_memory_type_filter(monkeypatch):
    class FakeVikingFS:
        def __init__(self):
            self.written_uris = []

        async def read_file(self, uri, ctx=None):
            raise FileNotFoundError(uri)

        async def write_file(self, uri, content, ctx=None):
            del content, ctx
            self.written_uris.append(uri)

    def schema(memory_type: str, filename: str) -> MemoryTypeSchema:
        return MemoryTypeSchema(
            memory_type=memory_type,
            directory="viking://user/{{ user_space }}/memories",
            filename_template=filename,
            content_template="{{ content }}",
            fields=[
                MemoryField(
                    name="content",
                    field_type=FieldType.STRING,
                    init_value=memory_type,
                )
            ],
        )

    fake_fs = FakeVikingFS()
    monkeypatch.setattr("openviking.storage.viking_fs.get_viking_fs", lambda: fake_fs)

    registry = MemoryTypeRegistry(load_schemas=False)
    registry.register(schema("identity", "identity.md"))
    registry.register(schema("profile", "profile.md"))

    ctx = type("Ctx", (), {"user": type("User", (), {"user_id": "alice"})()})()
    await registry.initialize_memory_files(ctx, allowed_memory_types={"profile"})

    assert fake_fs.written_uris == ["viking://user/alice/memories/profile.md"]
