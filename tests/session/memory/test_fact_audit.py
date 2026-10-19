# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Fact diagnostics stay separate from writes and cannot inject executable code."""

import json

import pytest

from openviking.session.memory.fact_audit import AUDIT_KEY, collect_fact_audit


def _decision(**overrides):
    return {
        "fact": "Bought a helmet for 120 yuan",
        "source_indexes": [12],
        "decision": "create",
        "destinations": ["purchase_event"],
        **overrides,
    }


def test_python_collects_only_real_comment_tokens_including_fenced_output():
    record = _decision()
    content = (
        "```python\n"
        'sdk.set_profile(content="""\n'
        '# memory_audit: {"this": "is quoted memory content"}\n'
        '""")\n'
        f"# memory_audit: {json.dumps(record)}\n"
        "sdk.commit()\n```"
    )

    audit = collect_fact_audit(content, "python")

    assert audit["status"] == "reported"
    assert audit["model_declared"] is True
    assert len(audit["records"]) == 1
    assert audit["records"][0]["fact"] == record["fact"]


@pytest.mark.parametrize("output_format", ["python", "json"])
def test_omission_and_old_fact_removal_keep_source_and_reason(output_format):
    records = [
        _decision(decision="omit", destinations=[], reason="hypothetical purchase, not confirmed"),
        _decision(
            fact="The old price was 110 yuan",
            decision="remove",
            source_memory="entities_1",
            destinations=["entities_1"],
            reason="user corrected the recorded amount in message 12",
        ),
    ]
    content = (
        "\n".join(f"# memory_audit: {json.dumps(record)}" for record in records) + "\nsdk.commit()"
        if output_format == "python"
        else json.dumps({"events": [], AUDIT_KEY: records})
    )

    audit = collect_fact_audit(content, output_format)

    assert audit["status"] == "reported"
    assert [record["decision"] for record in audit["records"]] == ["omit", "remove"]
    assert audit["records"][1]["source_memory"] == "entities_1"


@pytest.mark.parametrize(
    "record",
    [
        _decision(decision="omit", destinations=[]),
        _decision(decision="remove", reason="length target"),
        _decision(decision="already_present"),
        _decision(destinations=[]),
        _decision(source_indexes=[-1]),
        _decision(source_indexes=[True]),
        _decision(source_indexes=["12"]),
        _decision(source_indexes=[]),
        _decision(fact=" "),
        _decision(destinations=[" "]),
        _decision(extra_write="malicious"),
    ],
)
def test_invalid_audit_records_are_diagnostic_errors(record):
    audit = collect_fact_audit(f"# memory_audit: {json.dumps(record)}\nsdk.commit()", "python")

    assert audit["status"] == "invalid"
    assert audit["records"] == []
    assert audit["errors"] == ["invalid_record:0"]


def test_missing_and_malformed_audits_are_not_silently_reported_as_complete():
    assert collect_fact_audit("sdk.commit()", "python")["status"] == "missing"
    assert collect_fact_audit('{"events": []}', "json")["status"] == "missing"
    assert collect_fact_audit('{"_memory_audit": {}}', "json")["status"] == "invalid"
    audit = collect_fact_audit("# memory_audit: __import__('os').abort()\nsdk.commit()", "python")
    assert audit["status"] == "invalid"
    assert audit["errors"] == ["invalid_comment_json"]


def test_json_does_not_treat_audit_text_inside_memory_as_metadata():
    response = json.dumps({"profile": [{"content": '# memory_audit: {"bad": true}'}]})
    assert collect_fact_audit(response, "json")["status"] == "missing"
