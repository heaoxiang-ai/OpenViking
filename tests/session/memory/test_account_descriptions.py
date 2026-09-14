# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Account descriptions are text; only exact deployment copies retain Jinja."""

import pytest

from openviking.session.memory.account_templates import (
    EDITABLE_MEMORY_TEMPLATE_FIELDS,
    _complete_template,
    _parse_template,
    _validate_template,
    memory_template_data,
)
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.schema_model_generator import (
    SchemaModelGenerator,
    SchemaPromptGenerator,
)


@pytest.mark.parametrize("memory_type", EDITABLE_MEMORY_TEMPLATE_FIELDS)
@pytest.mark.parametrize(
    "text",
    [
        "{{ language }}",
        "{% if language == 'en' %}English{% else %}中文{% endif %}",
        "{{ cycler.__init__.__globals__.__builtins__.len('harmless') }}",
        "{% for n in range(3) %}x{% endfor %}",
        "{% include 'not_a_file' %}",
        "literal {{ and {% unclosed",
    ],
)
def test_account_descriptions_are_literal_in_all_schema_prompts(memory_type, text):
    defaults = memory_template_data(MemoryTypeRegistry().get(memory_type))
    editable = EDITABLE_MEMORY_TEMPLATE_FIELDS[memory_type]
    supplied = {
        "description": "TYPE: " + text,
        "fields": [{"name": name, "description": name + ": " + text} for name in editable],
    }
    data = _complete_template(defaults, supplied, memory_type)
    schema = _validate_template(data, memory_type, deployment_defaults=defaults)
    # Registry snapshots must preserve private provenance without exposing it in YAML/API data.
    schema = schema.model_copy(deep=True)
    assert schema._account_description
    assert "_account_description" not in memory_template_data(schema)
    assert schema.description == supplied["description"]

    generator = SchemaModelGenerator([schema], template_context={"language": "en"})
    operations = generator.create_structured_operations_model()
    model = generator.create_flat_data_model(schema)
    prompts = SchemaPromptGenerator([schema], template_context={"language": "en"})
    type_prompt = prompts.generate_type_descriptions()
    field_prompt = prompts.generate_field_descriptions(memory_type)
    assert supplied["description"] in operations.model_fields[memory_type].description
    assert supplied["description"] in type_prompt
    for field in schema.fields:
        if field.name in editable:
            assert field._account_description
            assert "_account_description" not in field.model_dump()
            assert field.description in model.model_fields[field.name].description
            assert field.description in type_prompt
            assert field.description in field_prompt


@pytest.mark.parametrize("request_kind", ["empty", "roundtrip", "type_only", "field_only"])
def test_unchanged_deployment_descriptions_keep_existing_language_rendering(request_kind):
    deployment = MemoryTypeRegistry().get("profile")
    deployment.description = "TYPE {{ language.upper() }}"
    deployment.fields[0].description = "FIELD {{ language.upper() }}"
    defaults = memory_template_data(deployment)
    supplied = {
        "empty": {},
        "roundtrip": defaults,
        "type_only": {"description": "CUSTOM {{ language }}"},
        "field_only": {"fields": [{"name": "content", "description": "CUSTOM {{ language }}"}]},
    }[request_kind]
    data = _complete_template(defaults, supplied, "profile")
    schema = _validate_template(data, "profile", deployment_defaults=defaults)
    generator = SchemaModelGenerator([schema], template_context={"language": "en"})
    operations = generator.create_structured_operations_model()
    field_description = generator.create_flat_data_model(schema).model_fields["content"].description
    type_description = operations.model_fields["profile"].description
    assert (
        "CUSTOM {{ language }}" if request_kind == "type_only" else "TYPE EN"
    ) in type_description
    assert (
        "CUSTOM {{ language }}" if request_kind == "field_only" else "FIELD EN"
    ) in field_description
    assert not deployment._account_description
    assert not deployment.fields[0]._account_description


def test_persisted_descriptions_do_not_trust_client_flags_or_require_jinja_syntax():
    import yaml

    deployment = MemoryTypeRegistry().get("profile")
    defaults = memory_template_data(deployment)
    data = memory_template_data(deployment)
    data["description"] = "{{ cycler.__init__.__globals__.__builtins__.len('harmless') }}"
    data["_account_description"] = False
    data["fields"][0]["description"] = "literal {{ unclosed"
    data["fields"][0]["_account_description"] = False
    raw = yaml.safe_dump(data).encode()
    parsed = _parse_template(raw, "profile")
    schema = _validate_template(parsed, "profile", deployment_defaults=defaults)
    assert schema._account_description
    assert schema.fields[0]._account_description
    prompt = SchemaPromptGenerator([schema]).generate_type_descriptions()
    assert data["description"] in prompt
    assert data["fields"][0]["description"] in prompt


def test_deployment_change_recomputes_description_trust_without_changing_old_snapshot():
    deployment = MemoryTypeRegistry().get("profile")
    deployment.description = "TYPE {{ language.upper() }}"
    deployment.fields[0].description = "FIELD {{ language.upper() }}"
    stored = memory_template_data(deployment)
    old = _validate_template(stored, "profile", deployment_defaults=stored)
    deployment.description = "NEW TYPE {{ language }}"
    deployment.fields[0].description = "NEW FIELD {{ language }}"
    new = _validate_template(
        stored, "profile", deployment_defaults=memory_template_data(deployment)
    )

    assert not old._account_description
    assert not old.fields[0]._account_description
    assert new._account_description
    assert new.fields[0]._account_description
    old_prompt = SchemaPromptGenerator([old], {"language": "en"}).generate_type_descriptions()
    new_prompt = SchemaPromptGenerator([new], {"language": "en"}).generate_type_descriptions()
    assert "TYPE EN" in old_prompt and "FIELD EN" in old_prompt
    assert stored["description"] in new_prompt
    assert stored["fields"][0]["description"] in new_prompt
