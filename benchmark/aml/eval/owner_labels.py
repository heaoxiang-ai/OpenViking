"""Render the same per-memory person headers as the frozen Top20 experiment."""

import re


def _person_label(value: str) -> str:
    value = value.strip().replace("_", " ")
    if (
        not value
        or value.startswith(".")
        or len(value) > 80
        or not any(char.isalnum() for char in value)
        or any(not (char.isalnum() or char in " -.'") for char in value)
        or value.casefold()
        in {"user", "assistant", "system", "tool", "profile", "user profile", "summary", "overview"}
        or value.casefold().startswith("conv-")
    ):
        return ""
    return value


def format_search_content(uri: str, content: str) -> str:
    relative_path = uri.partition("/memories/")[2] if uri.startswith("viking://") else ""
    parts = relative_path.split("/")
    people: list[str] = []
    if len(parts) >= 3 and parts[0] == "preferences":
        people = [_person_label(parts[1])]
    elif len(parts) >= 3 and parts[:2] in (["entities", "person"], ["entities", "people"]):
        people = [_person_label(parts[2].removesuffix(".md"))]
    elif relative_path == "profile.md":
        people = [_person_label(name) for name in re.findall(r"^# ([^\r\n]+)$", content, re.M)]
    elif parts[0] == "events" and "ChatLog:" in content:
        chat_log = content.split("ChatLog:", 1)[1]
        for match in re.finditer(r"^\*\*([^*\r\n]+)\*\*:[ \t]*([^\r\n]*)", chat_log, re.M):
            speaker = re.match(r"(?:\[D\d+:\d+\]\s*)?([^:\r\n]+):", match[2])
            people.append(_person_label(speaker[1] if speaker else match[1]))
        # Event speakers are participants; the header does not assign every
        # fact in a conversation to every participant.
    people = list(dict.fromkeys(person for person in people if person))
    label = ", ".join(people) if people else "未明确"
    return f"[记忆归属人: {label}]\n{content}"
