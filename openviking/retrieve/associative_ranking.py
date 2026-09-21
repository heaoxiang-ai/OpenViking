# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""BM25, rank fusion and independent evidence budgets for associative retrieval."""

import math
import re
from collections import Counter


def terms(text: str) -> list[str]:
    # Latin words and individual CJK characters; no network tokenizer dependency.
    return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.casefold())


def bm25(query: str, documents: dict[str, str]) -> dict[str, float]:
    if not documents:
        return {}
    counts = {key: Counter(terms(text)) for key, text in documents.items()}
    lengths = {key: sum(values.values()) for key, values in counts.items()}
    average = sum(lengths.values()) / len(counts) or 1
    query_terms = set(terms(query))
    frequency = Counter(
        term for counts_ in counts.values() for term in counts_ if term in query_terms
    )
    result = {}
    for key, doc in counts.items():
        score = 0.0
        for term in query_terms:
            n = doc.get(term, 0)
            if n:
                idf = math.log(1 + (len(counts) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * n * 2.5 / (n + 1.5 * (0.25 + 0.75 * lengths[key] / average))
        if score > 0:
            result[key] = score
    return result


def rrf(channels: list[dict[str, float]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for channel in channels:
        valid = {uri: score for uri, score in channel.items() if math.isfinite(score)}
        for rank, uri in enumerate(sorted(valid, key=lambda u: (-valid[u], u)), 1):
            scores[uri] = scores.get(uri, 0.0) + 1 / (k + rank)
    return scores


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    norm = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    value = sum(x * y for x, y in zip(left, right, strict=True)) / norm if norm else 0.0
    return value if math.isfinite(value) else 0.0


def budgeted(groups: dict[str, list[dict]], budgets: dict[str, int], limit: int) -> list[dict]:
    """Allocate a small requested limit proportionally without collapsing to one type."""
    available = {kind: min(budgets[kind], len(rows)) for kind, rows in groups.items()}
    total = sum(available.values())
    if not total or limit <= 0:
        return []
    if total <= limit:
        quotas = available
    else:
        exact = {kind: n * limit / total for kind, n in available.items()}
        quotas = {kind: int(n) for kind, n in exact.items()}
        for kind in sorted(exact, key=lambda key: (-(exact[key] - quotas[key]), key))[
            : limit - sum(quotas.values())
        ]:
            quotas[kind] += 1
    return [row for kind, rows in groups.items() for row in rows[: quotas[kind]]]
