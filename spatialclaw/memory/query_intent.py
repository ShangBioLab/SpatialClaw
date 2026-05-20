"""Low-overhead heuristics for memory-related query intent detection.

These helpers keep phrase lists and regexes out of bot/core.py while
preserving the same lightweight, substring-based matching strategy.
"""

from __future__ import annotations

import re

_MEMORY_RECALL_CUES = tuple(
    cue.lower()
    for cue in (
        "remember",
        "memory",
        "recall",
        "what do you remember",
        "记得",
        "记住",
        "当前记忆",
        "现在记住",
        "回忆",
        "刚才",
        "刚刚",
        "最近",
        "上次",
    )
)

_MEMORY_RECENCY_OR_OUTPUT_CUES = tuple(
    cue.lower()
    for cue in (
        "刚才",
        "刚刚",
        "最近",
        "上次",
        "latest",
        "recent",
        "current",
        "当前",
        "结果目录",
        "输出目录",
        "output path",
        "output directory",
        "结果路径",
    )
)

_WORKFLOW_CONTINUATION_CUES = tuple(
    cue.lower()
    for cue in (
        "继续之前",
        "继续上次",
        "延续之前",
        "沿用上次",
        "恢复刚才",
        "恢复之前",
        "previous workflow",
        "continue the previous",
        "continue previous",
        "resume the previous",
        "resume previous",
        "same workflow",
        "continue earlier workflow",
    )
)

_WORKFLOW_CONTINUATION_PAIRS = (
    ("继续", "workflow"),
    ("继续", "分析"),
    ("沿用", "workflow"),
    ("沿用", "分析"),
    ("恢复", "workflow"),
    ("恢复", "分析"),
    ("continue", "workflow"),
    ("continue", "analysis"),
    ("resume", "workflow"),
    ("resume", "analysis"),
)

_CLUSTER_PATTERN = re.compile(r"\bcluster[\s_:-]*(\d+)\b")
_DOMAIN_PATTERN = re.compile(r"\b(?:spatial[\s_-]*)?domain[\s_:-]*(\d+)\b")
_CELL_TYPE_PATTERN = re.compile(r"\bcell[\s_-]*type[\s_:-]*(\d+)\b")


def _normalized_lower_text(user_text: str) -> str:
    return str(user_text or "").lower()


def _contains_any_phrase(lower_text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in lower_text for cue in cues)


def looks_like_memory_recall_query(user_text: str) -> bool:
    """Return whether the user is asking about stored memory."""
    return _contains_any_phrase(_normalized_lower_text(user_text), _MEMORY_RECALL_CUES)


def looks_like_workflow_continuation_query(user_text: str) -> bool:
    """Return whether the user wants to continue or resume a prior workflow."""
    lower = _normalized_lower_text(user_text)
    if _contains_any_phrase(lower, _WORKFLOW_CONTINUATION_CUES):
        return True
    return any(left in lower and right in lower for left, right in _WORKFLOW_CONTINUATION_PAIRS)


def looks_like_recent_or_output_query(user_text: str) -> bool:
    """Return whether the user is asking for recent workflow state or output info."""
    return _contains_any_phrase(
        _normalized_lower_text(user_text),
        _MEMORY_RECENCY_OR_OUTPUT_CUES,
    )


def extract_memory_focus_terms(user_text: str) -> list[str]:
    """Extract normalized entity anchors such as cluster IDs from a query."""
    lower = _normalized_lower_text(user_text)
    terms: list[str] = []

    for match in _CLUSTER_PATTERN.finditer(lower):
        terms.append(f"cluster_{match.group(1)}")

    for match in _DOMAIN_PATTERN.finditer(lower):
        idx = match.group(1)
        terms.append(f"domain_{idx}")
        terms.append(f"cluster_{idx}")

    for match in _CELL_TYPE_PATTERN.finditer(lower):
        terms.append(f"cell_type_{match.group(1)}")

    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered
