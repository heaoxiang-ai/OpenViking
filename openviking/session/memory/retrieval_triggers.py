# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Jointly extracted retrieval cues, validated against the final memory before indexing.

These checks establish provenance, not general semantic entailment. No additional
LLM is called here; invalid optional cues never prevent writing ordinary memory.
"""

import hashlib
import json
import math
import re

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

TRIGGER_FIELD = "retrieval_triggers"
MEMORY_TYPES = {"events", "entities", "preferences"}
FAMILIES = {"bridge", "scene", "horizon"}
# `entity` was the old name of the removed concept cue, not a native Entity.
RETIRED_FAMILIES = {"entity", "concept"}
JOINT_MAX_VIEWS = 3
PROMPT = """Retrieval-only cues for THIS memory, generated in the SAME output
as its ordinary fields. Return a list of 0 to MAX_VIEWS objects, each with
family, subject, text, anchor, confidence. Do not create separate memories.
Write the native content/summary FIRST, then retrieval_triggers as the LAST
argument of the SDK call (or last field of the JSON object). For each anchor,
copy a contiguous phrase from the content/summary you JUST wrote. Do not quote
the input chat instead if you have paraphrased it in the memory body. For event
cues, prefer the new summary and include the named subject in that quote.
Example: {"family": "bridge", "subject": "Joanna", "text": "What dietary restrictions should be considered when cooking for Joanna?", "anchor": "Cannot have dairy products", "confidence": 0.9}.
For a substantive memory (a concrete preference, constraint, possession, creation,
or experience), normally provide 1 to MAX_VIEWS grounded search cues. A useful
question or use situation for an EXISTING fact is sufficient: cues need different
retrieval wording, NOT additional facts. The fact already appearing in the native
body is expected and is NOT a reason to return [].
Use [] when there is no grounded useful cue, especially greetings, thanks,
farewells, or content without a reusable fact. Do not fill a quota or paraphrase
the same fact once per family.
Never just repeat the summary, title, entity card, or a generic topic label.
bridge: a question or hypothetical use situation in which the supported fact helps.
For events only, scene: a distinctive situation retaining subject and concrete
object/action; horizon: a possible later retrieval situation, expressed as a
question or hypothetical, never a claim that a future action actually happened.
Every text must be self-contained and name its subject. For preferences, subject
is the existing `user` (preference owner); for entities, it is the existing `name`.
For events, quote one explicit subject from the grounding anchor. Preserve other
participants and their roles. Do not turn another person's suggestion into an
action, preference, plan, or accomplishment of the subject. Preserve negation,
uncertainty, plans versus completed actions, and preference changes.
anchor must be ONE verbatim continuous span of the FINAL memory body after edits,
not a URI, metadata, stitched quotes, or an old version. Include the specific
fact and its qualifiers. For events, quote the summary you produce or the actual
chat text in the selected ranges. Never expand ranges just to ground a cue.
Copy dates, times, counts and quantities literally from that anchor, or omit them.
Never calculate a new date from 'yesterday'/'last night'. Never introduce names,
actions, outcomes or causal claims unsupported by the memory. Use its language.
Each text <= 500 chars. Confidence is only a filter, not proof of grounding.
On updates, replace the entire cue list for the final merged memory; use [] to
remove cues. Ordinary memory remains the only answer evidence.
"""


def extraction_description(settings):
    return PROMPT.replace("MAX_VIEWS", str(min(JOINT_MAX_VIEWS, settings.max_triggers)))


def source_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def memory_type_for_uri(uri: str) -> str | None:
    for kind in MEMORY_TYPES:
        if f"/memories/{kind}/" in uri and uri.endswith(".md"):
            if not uri.rsplit("/", 1)[-1].startswith("."):
                return kind
    return None


def _normalized(text):
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def _mentions(text, subject):
    # Word boundaries for Latin identifiers, substring matching for names in
    # scripts without spaces (e.g. 小王不能吃虾).
    return (
        re.search(
            r"(?<![a-z0-9])" + re.escape(_normalized(subject)) + r"(?![a-z0-9])",
            _normalized(text),
        )
        is not None
    )


def _owner(memory, kind):
    value = memory.extra_fields.get({"preferences": "user", "entities": "name"}.get(kind))
    return value.strip() if isinstance(value, str) else ""


def _subject_anchor(body, anchor, subject):
    """Keep a named subject with its quote, using its sentence or speaker turn.

    A model often quotes a predicate ("his band rehearsed") without repeating
    the name earlier in the summary/chat turn. Expand to that exact contiguous
    sentence rather than rejecting it or borrowing a name from elsewhere in the file.
    This is a provenance check, not proof of semantic role assignment.
    """
    if _mentions(anchor, subject):
        return anchor
    start = 0
    while (pos := body.find(anchor, start)) >= 0:
        left = body.rfind("\n", 0, pos) + 1
        right = body.find("\n", pos + len(anchor))
        right = right if right >= 0 else len(body)
        line = body[left:right]
        # A named chat speaker can govern a later first-person sentence in the
        # same turn. Otherwise don't borrow a name from another sentence.
        speaker = re.match(r"\*\*(?:user|assistant)\*\*:\s*(?:\[[^\]]+\]\s*)?([^:\n]+):", line)
        if speaker and _normalized(speaker[1]) == _normalized(subject):
            return line
        stops = list(re.finditer(r"[.!?。！？](?:\s+|$)", body[left:pos]))
        start_sentence = left + stops[-1].end() if stops else left
        stop = re.search(r"[.!?。！？](?:\s+|$)", body[pos + len(anchor) : right])
        end_sentence = pos + len(anchor) + stop.end() if stop else right
        context = body[start_sentence:end_sentence].strip()
        if _mentions(context, subject):
            return context
        start = pos + 1
    raise ValueError("Retrieval trigger subject is not grounded")


def identity_hash(memory):
    return source_hash(
        json.dumps(
            {key: memory.extra_fields.get(key) for key in ("user", "name")},
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def validate_views(value, body, kind, settings):
    """Structural validation shared with legacy caches; not an entailment check."""
    if not isinstance(value, list) or len(value) > settings.max_triggers:
        raise ValueError("Invalid retrieval trigger list")
    allowed = FAMILIES if kind == "events" else {"bridge"}
    accepted, seen = [], set()
    for view in value:
        if isinstance(view, dict) and isinstance(view.get("family"), str):
            if view["family"] in RETIRED_FAMILIES:
                continue
        if (
            not isinstance(view, dict)
            or not isinstance(view.get("family"), str)
            or view["family"] not in allowed
        ):
            raise ValueError("Invalid retrieval trigger family")
        text, anchor, confidence = view.get("text"), view.get("anchor"), view.get("confidence")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500 or not _normalized(text):
            raise ValueError("Invalid retrieval trigger text")
        if not isinstance(anchor, str) or not anchor.strip() or anchor not in body:
            raise ValueError("Retrieval trigger has no exact source anchor")
        if (
            type(confidence) not in (int, float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Invalid retrieval trigger confidence")
        key = _normalized(text)
        if confidence >= settings.min_confidence and key not in seen:
            accepted.append(
                {
                    "family": view["family"],
                    "text": text.strip(),
                    "anchor": anchor,
                    "confidence": confidence,
                    **({"subject": view["subject"]} if "subject" in view else {}),
                }
            )
            seen.add(key)
    return accepted


# Literal numeric spans include ISO dates/times, fractions, and decimal counts.
# Requiring them in the *anchor* prevents borrowing a date from another event in
# the same body. Written-out dates and semantic role errors still need prompting.
_NUMBERS = re.compile(r"\d+(?:[-./:,]\d+)*")
_SOCIAL_ONLY = re.compile(
    r"(?:(?:thanks(?: so much| a lot)?|thank you|bye|goodbye|cheers|later|take care|"
    r"see you(?: later| soon)?|ttyl|gotta go|gonna go|sorry|hi|hello|"
    r"再见|谢谢|回头见|拜拜|你好)\s*)+"
)


def _grounded_view(view, memory, settings):
    kind = memory_type_for_uri(memory.uri or "")
    views = validate_views([view], memory.content, kind, settings)
    if not views:
        return None
    result = views[0]
    subject = result.get("subject")
    if not isinstance(subject, str) or not _normalized(subject) or len(subject) > 120:
        raise ValueError("Missing explicit retrieval trigger subject")
    subject = subject.strip()
    owner = _owner(memory, kind)
    if owner:
        if _normalized(subject) != _normalized(owner):
            raise ValueError("Retrieval trigger subject differs from memory owner")
        subject = owner
    elif kind in {"preferences", "entities"}:
        raise ValueError("Retrieval trigger subject is not grounded")
    else:
        result["anchor"] = _subject_anchor(memory.content, result["anchor"], subject)
    if _normalized(subject) in {
        "he",
        "she",
        "they",
        "someone",
        "user",
        "a person",
        "他",
        "她",
        "用户",
    }:
        raise ValueError("Retrieval trigger subject is ambiguous")
    text, anchor = result["text"], result["anchor"]
    if not _mentions(text, subject):
        text = f"{subject}: {text}"
    if len(text) > 500:
        raise ValueError("Retrieval trigger with subject exceeds text budget")
    if not set(_NUMBERS.findall(text)) <= set(_NUMBERS.findall(anchor + " " + subject)):
        raise ValueError("Retrieval trigger adds an unsupported date or quantity")
    social_anchor = re.sub(r"^\s*" + re.escape(subject) + r"\s*[:：]\s*", "", anchor, flags=re.I)
    if _SOCIAL_ONLY.fullmatch(_normalized(social_anchor)):
        raise ValueError("Retrieval trigger only anchors a greeting or farewell")
    return {**result, "text": text, "subject": subject}


def valid_cached(memory, settings):
    state = memory.extra_fields.get(TRIGGER_FIELD)
    if not isinstance(state, dict) or state.get("version") not in (1, 2):
        return None
    if state.get("source_sha256") != source_hash(memory.content):
        return None
    kind = memory_type_for_uri(memory.uri or "")
    if not kind:
        return None
    try:
        views = validate_views(state.get("views"), memory.content, kind, settings)
        if state["version"] == 2:
            if state.get("identity_sha256") != identity_hash(memory):
                return None
            if any(_grounded_view(view, memory, settings) != view for view in views):
                return None
        return views
    except ValueError:
        return None


def attach_extracted(memory, proposed, *, settings):
    """Validate joint output after rendering/merge, without any model request.

    Missing output can keep an unchanged v2 cache. Changed evidence, explicit [], or
    invalid output clears old cues; callers continue to index the native memory.
    """
    if proposed is None:
        state = memory.extra_fields.get(TRIGGER_FIELD, {})
        cached = valid_cached(memory, settings)
        if isinstance(state, dict) and state.get("version") == 2 and cached is not None:
            state["views"] = cached
            return cached
        memory.extra_fields.pop(TRIGGER_FIELD, None)
        return []
    memory.extra_fields.pop(TRIGGER_FIELD, None)
    if not isinstance(proposed, list):
        logger.warning("Ignoring malformed optional retrieval triggers for %s", memory.uri)
        return []
    views, seen = [], set()
    for item in proposed[: settings.max_triggers]:
        try:
            view = _grounded_view(item, memory, settings)
        except ValueError as exc:
            logger.warning("Discarding retrieval trigger for %s: %s", memory.uri, exc)
            continue
        if view is None:
            continue
        key = _normalized(view["text"])
        # One cue per family and evidence span prevents repeated rephrasings.
        intent = (view["family"], _normalized(view["anchor"]))
        if key in seen or intent in seen:
            continue
        seen.update((key, intent))
        views.append(view)
        if len(views) >= min(JOINT_MAX_VIEWS, settings.max_triggers):
            break
    memory.extra_fields[TRIGGER_FIELD] = {
        "version": 2,
        "generation": "joint_extraction",
        "source_sha256": source_hash(memory.content),
        "identity_sha256": identity_hash(memory),
        "prompt_sha256": source_hash(extraction_description(settings)),
        "views": views,
    }
    return views
