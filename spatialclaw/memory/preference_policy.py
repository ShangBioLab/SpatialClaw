"""Preference memory policy helpers.

This module keeps preference-specific rules out of bot/core.py so the
prompt orchestration layer does not need to hardcode preference keywords.

Design goals:
- Preferences should be concise user tendencies, not factual notes.
- Preferences should be updateable through stable keys.
- Workflow facts (datasets, outputs, detailed results) should live in
  dataset / analysis / insight memories instead of preference memory.
"""

from __future__ import annotations

from typing import Any

PREFERENCE_MAX_KEY_LENGTH = 64
PREFERENCE_MAX_VALUE_LENGTH = 120
PREFERENCE_MAX_VALUE_TOKENS = 16

_PREFERENCE_ALLOWED_STEMS = {
    "citation",
    "dpi",
    "explanation",
    "figure",
    "format",
    "language",
    "markdown",
    "method",
    "model",
    "palette",
    "provider",
    "resolution",
    "response",
    "style",
    "theme",
    "timezone",
    "tone",
    "verbosity",
    "workflow",
}

_WORKFLOW_STATE_STEMS = {
    "dataset",
    "method",
    "output",
    "parameter",
    "resolution",
    "workflow",
}

_FACT_LIKE_VALUE_MARKERS = (
    "\n",
    "\r",
    "http://",
    "https://",
    "```",
    ".h5ad",
    ".h5mu",
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".png",
    ".pdf",
    "/",
    "\\",
)


def normalize_preference_key(key: str) -> str:
    """Normalize a preference key into a stable update-friendly token."""
    text = str(key or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_preference_value(value: Any) -> str:
    """Convert a scalar preference value into a compact single-line string."""
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        raise ValueError(
            "Preference values must be short scalar settings, not structured objects."
        )

    return " ".join(text.strip().split())


def _preference_key_stems(key: str) -> set[str]:
    return {part for part in normalize_preference_key(key).split("_") if part}


def is_allowed_preference_key(key: str) -> bool:
    """Return whether a key looks like a tendency/default setting."""
    normalized = normalize_preference_key(key)
    if not normalized or len(normalized) > PREFERENCE_MAX_KEY_LENGTH:
        return False

    stems = _preference_key_stems(normalized)
    if any(stem in _PREFERENCE_ALLOWED_STEMS for stem in stems):
        return True
    return normalized.startswith(("current_", "default_", "preferred_"))


def is_workflow_state_preference_key(key: str) -> bool:
    """Return whether a preference key can affect workflow-state arbitration."""
    stems = _preference_key_stems(key)
    return any(stem in _WORKFLOW_STATE_STEMS for stem in stems)


def validate_preference_payload(key: str, value: Any) -> tuple[str, str]:
    """Validate and normalize a preference payload.

    Returns:
        (normalized_key, normalized_value)
    """
    normalized_key = normalize_preference_key(key)
    if not normalized_key:
        raise ValueError("Preference key cannot be empty.")
    if len(normalized_key) > PREFERENCE_MAX_KEY_LENGTH:
        raise ValueError(
            f"Preference key is too long (max {PREFERENCE_MAX_KEY_LENGTH} characters)."
        )
    if not is_allowed_preference_key(normalized_key):
        raise ValueError(
            "Preference keys must describe concise user tendencies or defaults, "
            "such as language, format, style, method, workflow, or resolution."
        )

    normalized_value = normalize_preference_value(value)
    if not normalized_value:
        raise ValueError("Preference value cannot be empty.")
    if len(normalized_value) > PREFERENCE_MAX_VALUE_LENGTH:
        raise ValueError(
            "Preference values must stay short and concise; "
            f"max length is {PREFERENCE_MAX_VALUE_LENGTH} characters."
        )
    if len(normalized_value.split()) > PREFERENCE_MAX_VALUE_TOKENS:
        raise ValueError(
            "Preference values must be short tendencies, not long factual notes."
        )

    lowered_value = normalized_value.lower()
    if any(marker in lowered_value for marker in _FACT_LIKE_VALUE_MARKERS):
        raise ValueError(
            "Preference values cannot contain file paths, URLs, multiline notes, "
            "or result-document style content."
        )

    return normalized_key, normalized_value


def preference_to_prompt_entry(key: str, value: Any) -> tuple[str, str] | None:
    """Return a prompt-safe preference entry, or None if it should be hidden."""
    try:
        prompt_key, prompt_value = validate_preference_payload(key, value)
    except ValueError:
        return None
    return prompt_key, prompt_value
