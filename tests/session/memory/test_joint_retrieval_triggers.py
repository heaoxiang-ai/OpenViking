# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression cases for joint extraction, grounding and memory update lifecycle."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.extraction_output_protocol import (
    ExtractionOutputContext,
    create_extraction_output_protocol,
)
from openviking.session.memory.memory_type_registry import get_default_registry
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.page_id_map import PageIdMap
from openviking.session.memory.retrieval_triggers import (
    TRIGGER_FIELD,
    attach_extracted,
    valid_cached,
)
from openviking.session.memory.schema_model_generator import SchemaModelGenerator
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.memory_trigger_index import MemoryTriggerIndex
from openviking_cli.utils.config.memory_trigger_config import MemoryTriggerConfig

SETTINGS = MemoryTriggerConfig(enabled=True)
PREF_URI = "viking://user/conversation/memories/preferences/joanna/food.md"
EVENT_URI = "viking://user/conversation/memories/events/2023/09/15/jam.md"


def cue(
    *,
    subject="joanna",
    text="What dietary restrictions apply when cooking?",
    anchor="Cannot have dairy products",
    family="bridge",
):
    return {
        "family": family,
        "subject": subject,
        "text": text,
        "anchor": anchor,
        "confidence": 0.95,
    }


def preference(owner="joanna", body="Cannot have dairy products"):
    return MemoryFile(
        uri=PREF_URI,
        memory_type="preferences",
        content=body,
        extra_fields={"user": owner, "topic": "food"},
    )


def attach(memory, views):
    return attach_extracted(memory, views, settings=SETTINGS)


def test_same_conversation_preferences_embed_distinct_owners():
    outputs = []
    for owner in ("joanna", "nate"):
        outputs.append(attach(preference(owner), [cue(subject=owner)])[0]["text"])
    assert outputs == [
        "joanna: What dietary restrictions apply when cooking?",
        "nate: What dietary restrictions apply when cooking?",
    ]
    assert attach(preference(), [cue(subject="nate")]) == []


def test_anchor_does_not_validate_invented_date_even_at_high_confidence():
    body = "# 2023-09-15\nDave and his band rehearsed in Boston on 2023-09-14."
    memory = MemoryFile(uri=EVENT_URI, content=body)
    anchor = "Dave and his band rehearsed in Boston on 2023-09-14."
    views = [
        cue(subject="Dave", anchor=anchor, text=f"What happened at Dave's rehearsal on {date}?")
        for date in ("2023-09-13", "2023-09-15", "2023-09-14")
    ]
    accepted = attach(memory, views)
    assert len(accepted) == 1
    assert "2023-09-14" in accepted[0]["text"]


def test_short_event_quote_can_keep_its_subject_from_the_same_source_line():
    line = "On 2023-09-14, Dave said his band rehearsed; they did not record the session."
    memory = MemoryFile(uri=EVENT_URI, content="# Summary\n" + line)
    accepted = attach(
        memory,
        [
            cue(
                subject="Dave",
                anchor="they did not record the session",
                text="Why was Dave's rehearsal on 2023-09-14 not recorded?",
            )
        ],
    )
    assert len(accepted) == 1
    assert accepted[0]["anchor"] == line
    assert valid_cached(memory, SETTINGS) == accepted


@pytest.mark.parametrize("separator", ["\n", " "])
def test_event_quote_cannot_borrow_subject_from_an_unrelated_sentence(separator):
    body = (
        "Dave attended a concert."
        + separator
        + "Calvin said his band rehearsed and did not record it."
    )
    memory = MemoryFile(uri=EVENT_URI, content=body)
    assert (
        attach(
            memory,
            [
                cue(
                    subject="Dave",
                    anchor="did not record it",
                    text="Why was Dave's rehearsal not recorded?",
                )
            ],
        )
        == []
    )


def test_event_quote_retains_named_speaker_for_first_person_sentence():
    body = "**assistant**: [D19:3] Dave: Hey Calvin! I wish we had recorded the jam."
    memory = MemoryFile(uri=EVENT_URI, content=body)
    accepted = attach(
        memory,
        [
            cue(
                subject="Dave",
                anchor="I wish we had recorded the jam",
                text="Did Dave record his band rehearsal?",
            )
        ],
    )
    assert accepted[0]["anchor"] == body


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "bad",
        {"views": []},
        [None],
        [cue(family=[])],
        [cue(subject="someone")],
        [cue(anchor="invented")],
        [cue(text="Can Joanna drink 3 glasses of milk?")],
    ],
)
def test_invalid_optional_output_preserves_the_memory(bad):
    memory = preference()
    assert attach(memory, bad) == []
    assert memory.content == "Cannot have dairy products"


def test_social_only_anchors_and_redundant_views_are_dropped():
    memory = preference()
    duplicate = cue(text="What dietary restrictions apply when cooking!!!")
    same_intent = cue(text="Which dishes should Joanna avoid when planning dinner?")
    assert len(attach(memory, [cue(), duplicate, same_intent])) == 1
    bye = "John: Thanks, gonna go, sorry. Cheers! Bye!"
    memory = MemoryFile(uri=EVENT_URI, content=bye)
    assert (
        attach(memory, [cue(subject="John", anchor=bye, text="Looking for casual chat goodbyes")])
        == []
    )


@pytest.mark.parametrize("family", ["concept", "entity"])
def test_removed_cue_families_are_not_generated_or_reused(family):
    memory = preference()
    accepted = attach(memory, [cue(), cue(family=family)])
    assert len(accepted) == 1
    assert accepted[0]["family"] == "bridge"
    # Keep other valid families in an old mixed cache, rather than invalidating
    # the entire cache or passing the removed family on to the embedding queue.
    memory.extra_fields[TRIGGER_FIELD]["views"].append(cue(family=family))
    assert len(valid_cached(memory, SETTINGS)) == 1
    attach(memory, None)
    assert [v["family"] for v in memory.extra_fields[TRIGGER_FIELD]["views"]] == ["bridge"]


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("events", ["bridge", "horizon", "scene"]),
        ("entities", ["bridge"]),
        ("preferences", ["bridge"]),
    ],
)
def test_output_schema_only_exposes_remaining_families(kind, expected):
    schema = get_default_registry().get(kind)
    model = SchemaModelGenerator([schema], trigger_settings=SETTINGS).create_flat_data_model(schema)
    field = model.model_json_schema()["properties"][TRIGGER_FIELD]
    assert field["items"]["properties"]["family"]["enum"] == expected
    assert "concept:" not in field["description"]


def test_chinese_subject_is_supported_without_spaces():
    body = "小王不能吃虾。"
    memory = MemoryFile(uri=EVENT_URI, content=body)
    accepted = attach(
        memory, [cue(subject="小王", anchor=body, text="给小王安排聚餐时要避开什么食物？")]
    )
    assert len(accepted) == 1
    assert not accepted[0]["text"].startswith("小王:")


def test_missing_output_keeps_only_current_body_and_identity_cache():
    memory = preference()
    accepted = attach(memory, [cue()])
    assert attach(memory, None) == accepted
    memory.extra_fields["user"] = "nate"
    assert valid_cached(memory, SETTINGS) is None
    assert attach(memory, None) == []
    assert TRIGGER_FIELD not in memory.extra_fields
    memory = preference()
    attach(memory, [cue()])
    memory.content = "Can now have dairy products"
    assert attach(memory, None) == []
    assert TRIGGER_FIELD not in memory.extra_fields


def test_explicit_empty_list_replaces_old_cues_and_payload_cannot_forge_cache():
    memory = preference()
    attach(memory, [cue()])
    forged_state = deepcopy(memory.extra_fields[TRIGGER_FIELD])
    assert attach(memory, []) == []
    assert valid_cached(memory, SETTINGS) == []
    assert attach(memory, forged_state) == []
    assert TRIGGER_FIELD not in memory.extra_fields


def protocol_context(enabled=True, existing=False):
    schema = get_default_registry().get("preferences")
    config = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
    with patch("openviking_cli.utils.config.get_openviking_config", return_value=config):
        model = SchemaModelGenerator(
            [schema], trigger_settings=MemoryTriggerConfig(enabled=enabled)
        ).create_structured_operations_model()
    page_ids, files = PageIdMap(), {}
    if existing:
        memory = preference()
        attach(memory, [cue()])
        page_ids.get_page_id(memory.uri)
        files[memory.uri] = memory
    return ExtractionOutputContext(
        operations_model=model,
        schemas=(schema,),
        link_enabled=False,
        page_id_map=page_ids,
        read_file_contents=files,
    )


@pytest.mark.parametrize("mode", ["json", "python"])
@pytest.mark.parametrize("existing", [False, True])
def test_both_protocols_preserve_joint_cues_on_create_and_update(mode, existing):
    context = protocol_context(existing=existing)
    protocol = create_extraction_output_protocol(mode)
    contract = protocol.render_contract(context)
    assert "SAME output" in contract
    assert "final merged memory" in contract
    if mode == "python":
        signature = next(
            line for line in contract.splitlines() if "sdk.create_preferences(" in line
        )
        assert signature.index("content:") < signature.index("retrieval_triggers:")
    if mode == "json":
        output = json.dumps(
            {
                "preferences": [
                    {
                        "page_id": 1 if existing else 100,
                        "user": "joanna",
                        "topic": "food",
                        "content": "Cannot have dairy products",
                        TRIGGER_FIELD: [cue()],
                    }
                ]
            }
        )
    elif existing:
        bindings = protocol.render_new_bindings(context, source="test")
        assert "source_sha256" not in bindings
        output = f"preferences_1.update(retrieval_triggers={[cue()]!r})\nsdk.commit()"
    else:
        output = (
            f"sdk.create_preferences(user='joanna', topic='food', "
            f"content='Cannot have dairy products', retrieval_triggers={[cue()]!r})\n"
            "sdk.commit()"
        )
    operations, error = protocol.parse(output, context)
    assert error is None
    assert operations.preferences[0].model_dump()[TRIGGER_FIELD] == [cue()]


@pytest.mark.parametrize("mode", ["json", "python"])
def test_disabled_trigger_feature_does_not_change_prompt(mode):
    contract = create_extraction_output_protocol(mode).render_contract(protocol_context(False))
    assert TRIGGER_FIELD not in contract


@pytest.mark.parametrize("mode", ["json", "python"])
def test_malformed_optional_cues_do_not_drop_a_parsed_memory(mode):
    context = protocol_context()
    protocol = create_extraction_output_protocol(mode)
    if mode == "python":
        output = (
            "sdk.create_preferences(user='joanna', topic='food', "
            "content='Cannot have dairy products', retrieval_triggers={'bad': 1})\n"
            "sdk.commit()"
        )
    else:
        output = json.dumps(
            {
                "preferences": [
                    {
                        "page_id": 100,
                        "user": "joanna",
                        "topic": "food",
                        "content": "Cannot have dairy products",
                        TRIGGER_FIELD: {"bad": 1},
                    }
                ]
            }
        )
    operations, error = protocol.parse(output, context)
    assert error is None
    assert len(operations.preferences) == 1
    assert operations.preferences[0].content == "Cannot have dairy products"


@pytest.mark.asyncio
async def test_extract_loop_generates_memory_and_cues_in_one_response():
    schema = get_default_registry().get("preferences")
    provider = MagicMock()
    provider.get_memory_schemas.return_value = [schema]
    provider.get_output_language.return_value = "en"
    provider.get_tools.return_value = []
    provider.get_extract_context.return_value = SimpleNamespace(page_id_map=PageIdMap())
    provider.read_file_contents = {}
    provider.instruction.return_value = "Extract memory operations."
    provider.prefetch = AsyncMock(return_value=[])
    output = (
        f"sdk.create_preferences(user='joanna', topic='food', "
        f"content='Cannot have dairy products', retrieval_triggers={[cue()]!r})\n"
        "sdk.commit()"
    )
    vlm = SimpleNamespace(model="test", get_completion_async=AsyncMock(return_value=output))
    loop = ExtractLoop(vlm=vlm, viking_fs=MagicMock(), context_provider=provider, max_iterations=1)
    empty = ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
    loop.resolve_operations = AsyncMock(return_value=(empty, []))
    loop._check_unread_existing_files = AsyncMock(return_value={})
    config = SimpleNamespace(memory=SimpleNamespace(link_enabled=False, triggers=SETTINGS))
    with (
        patch("openviking.session.memory.extract_loop.get_openviking_config", return_value=config),
        patch("openviking_cli.utils.config.get_openviking_config", return_value=config),
    ):
        await loop.run()
    vlm.get_completion_async.assert_awaited_once()
    assert "SAME output" in vlm.get_completion_async.call_args.kwargs["messages"][0]["content"]
    parsed = loop.resolve_operations.call_args.args[0]
    assert parsed.preferences[0].model_dump()[TRIGGER_FIELD] == [cue()]


@pytest.mark.asyncio
async def test_updater_checks_final_body_without_a_trigger_model_call():
    index = object.__new__(MemoryTriggerIndex)
    index.settings = SETTINGS
    # No model_slots or VLM: any attempt to use the previous generator fails.
    updater = MemoryUpdater(
        registry=get_default_registry(), vikingdb=SimpleNamespace(trigger_index=index)
    )
    storage = {}

    async def read(uri, **kwargs):
        return storage.get(uri, "")

    async def write(uri, raw, **kwargs):
        storage[uri] = raw

    updater._get_viking_fs = lambda: SimpleNamespace(read_file=read, write_file=write)
    operation = ResolvedOperation(
        memory_type="preferences",
        uris=[PREF_URI],
        memory_fields={
            "user": "joanna",
            "topic": "food",
            "content": "Cannot have dairy products",
            TRIGGER_FIELD: [cue()],
        },
    )
    await updater._apply_upsert(operation, MagicMock())
    memory = MemoryFileUtils.read(storage[PREF_URI], uri=PREF_URI)
    assert memory.content.strip() == "Cannot have dairy products"
    assert len(valid_cached(memory, SETTINGS)) == 1
    operation.memory_fields["content"] = "Now tolerates dairy products"
    operation.memory_fields.pop(TRIGGER_FIELD)
    await updater._apply_upsert(operation, MagicMock())
    memory = MemoryFileUtils.read(storage[PREF_URI], uri=PREF_URI)
    assert TRIGGER_FIELD not in memory.extra_fields
    assert "Now tolerates" in memory.content
