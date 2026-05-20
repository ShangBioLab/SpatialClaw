"""Promotion heuristics for episodic and semantic memory storage.

This module keeps promotion policy separate from the storage engine so
the graph core can remain unchanged while the bot evolves its memory
admission logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

LAYER_EPISODIC = "episodic"
LAYER_SEMANTIC = "semantic"
VALID_LAYERS = (LAYER_EPISODIC, LAYER_SEMANTIC, "all")


@dataclass(slots=True)
class PromotionDecision:
    """Result of evaluating whether a memory should enter semantic storage."""

    store_episodic: bool = True
    promote_to_semantic: bool = False
    importance: float = 0.0
    confidence: float = 0.0
    evidence_uri: Optional[str] = None
    reasons: list[str] = field(default_factory=list)


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 3)))


def evaluate_promotion(memory: Any) -> PromotionDecision:
    """Return a conservative promotion decision for a memory object.

    The rules intentionally prefer precision over recall: semantic memory
    pollution is more harmful than leaving a useful fact in episodic memory.
    """

    memory_type = getattr(memory, "memory_type", "")

    if memory_type == "preference":
        return PromotionDecision(
            store_episodic=False,
            promote_to_semantic=True,
            importance=0.95,
            confidence=0.95,
            evidence_uri=None,
            reasons=["User preference is stable and directly reusable."],
        )

    if memory_type == "project_context":
        return PromotionDecision(
            store_episodic=False,
            promote_to_semantic=True,
            importance=0.9,
            confidence=0.9,
            evidence_uri=None,
            reasons=["Project context should persist across future sessions."],
        )

    if memory_type == "dataset":
        importance = 0.35
        confidence = 0.45
        reasons: list[str] = ["Dataset metadata is useful for episodic continuity."]

        if getattr(memory, "file_exists", True):
            confidence += 0.2
            reasons.append("Dataset path is available.")
        if getattr(memory, "platform", None):
            importance += 0.15
            confidence += 0.15
            reasons.append("Platform metadata makes the dataset reusable.")
        if getattr(memory, "n_obs", None) is not None or getattr(memory, "n_vars", None) is not None:
            importance += 0.15
            confidence += 0.1
            reasons.append("Shape metadata confirms this is a stable dataset record.")

        promote = confidence >= 0.75 and importance >= 0.55
        if promote:
            reasons.append("Dataset qualifies for semantic reuse.")

        return PromotionDecision(
            store_episodic=True,
            promote_to_semantic=promote,
            importance=_clamp_score(importance),
            confidence=_clamp_score(confidence),
            evidence_uri=getattr(memory, "file_path", None),
            reasons=reasons,
        )

    if memory_type == "analysis":
        importance = 0.4
        confidence = 0.4
        reasons = ["Analysis execution is initially treated as episodic workflow state."]

        if getattr(memory, "status", "") == "completed":
            importance += 0.2
            confidence += 0.25
            reasons.append("Completed analyses are more trustworthy than failed runs.")
        if getattr(memory, "source_dataset_id", ""):
            importance += 0.1
            confidence += 0.1
            reasons.append("Dataset lineage is attached.")
        if getattr(memory, "output_path", ""):
            importance += 0.15
            confidence += 0.1
            reasons.append("Output path provides recoverable evidence.")

        promote = (
            getattr(memory, "status", "") == "completed"
            and bool(getattr(memory, "output_path", "") or getattr(memory, "source_dataset_id", ""))
        )
        if promote:
            reasons.append("Analysis qualifies for semantic recall.")

        return PromotionDecision(
            store_episodic=True,
            promote_to_semantic=promote,
            importance=_clamp_score(importance),
            confidence=_clamp_score(confidence),
            evidence_uri=getattr(memory, "output_path", None),
            reasons=reasons,
        )

    if memory_type == "insight":
        importance = 0.35
        confidence = 0.45
        reasons = ["Biological insight starts as tentative unless corroborated."]

        if getattr(memory, "confidence", "") == "user_confirmed":
            importance += 0.35
            confidence += 0.35
            reasons.append("User confirmation strongly supports promotion.")
        if getattr(memory, "evidence", ""):
            importance += 0.2
            confidence += 0.15
            reasons.append("Explicit evidence supports the biological label.")

        promote = (
            getattr(memory, "confidence", "") == "user_confirmed"
            or bool(getattr(memory, "evidence", ""))
        )
        if promote:
            reasons.append("Insight qualifies for semantic memory.")

        evidence_uri = getattr(memory, "source_analysis_id", None) or None
        return PromotionDecision(
            store_episodic=True,
            promote_to_semantic=promote,
            importance=_clamp_score(importance),
            confidence=_clamp_score(confidence),
            evidence_uri=evidence_uri,
            reasons=reasons,
        )

    return PromotionDecision(
        store_episodic=True,
        promote_to_semantic=False,
        importance=0.25,
        confidence=0.25,
        reasons=["Unknown memory type defaults to episodic storage only."],
    )
