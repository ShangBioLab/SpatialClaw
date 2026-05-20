"""Typed memory records used by SpatialClaw's layered memory store."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .preference_policy import validate_preference_payload


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Session(BaseModel):
    """User session persisted across CLI, TUI, and bot runs."""

    session_id: str
    user_id: str
    platform: Literal["telegram", "feishu", "cli", "tui"]
    created_at: datetime = Field(default_factory=utcnow)
    last_activity: datetime = Field(default_factory=utcnow)
    preferences: dict[str, Any] = Field(default_factory=dict)
    active: bool = True


class BaseMemory(BaseModel):
    """Base record shared by all typed memory categories."""

    memory_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    memory_type: str
    created_at: datetime = Field(default_factory=utcnow)
    source_session_id: str | None = None
    memory_layer: Literal["episodic", "semantic"] | None = None
    importance: float | None = None
    confidence_score: float | None = None
    evidence_uri: str | None = None
    promoted_at: datetime | None = None
    promotion_reasons: list[str] = Field(default_factory=list)


class DatasetMemory(BaseMemory):
    """Physical spatial dataset metadata."""

    memory_type: Literal["dataset"] = "dataset"
    file_path: str
    platform: str | None = None
    n_obs: int | None = None
    n_vars: int | None = None
    preprocessing_state: Literal["raw", "qc", "normalized", "clustered"] = "raw"
    file_exists: bool = True

    @field_validator("file_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if value.startswith("/"):
            raise ValueError("Absolute paths not allowed")
        return value


class AnalysisMemory(BaseMemory):
    """Analysis execution record with lineage and task trajectory fields."""

    memory_type: Literal["analysis"] = "analysis"
    source_dataset_id: str
    parent_analysis_id: str | None = None
    skill: str
    method: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    output_path: str | None = None
    status: Literal["completed", "failed", "interrupted"] = "completed"
    executed_at: datetime = Field(default_factory=utcnow)
    duration_seconds: float = 0.0
    task_main: str = ""
    task_trajectory: str = ""
    key_steps: str = ""
    fail_reason: str = ""
    cluster_id: int | None = None


class PreferenceMemory(BaseMemory):
    """Concise user preference or habit."""

    memory_type: Literal["preference"] = "preference"
    domain: str
    key: str
    value: Any
    is_strict: bool = False
    updated_at: datetime = Field(default_factory=utcnow)

    @field_validator("key")
    @classmethod
    def validate_preference_key(cls, value: str) -> str:
        normalized_key, _ = validate_preference_payload(value, "placeholder")
        return normalized_key

    @field_validator("value")
    @classmethod
    def validate_preference_value(cls, value: Any, info) -> str:
        key = info.data.get("key", "")
        _, normalized_value = validate_preference_payload(key, value)
        return normalized_value


class InsightMemory(BaseMemory):
    """Biological interpretation or reusable task rule."""

    memory_type: Literal["insight"] = "insight"
    source_analysis_id: str
    entity_type: str
    entity_id: str
    biological_label: str
    evidence: str = ""
    confidence: Literal["user_confirmed", "ai_predicted"] = "ai_predicted"
    rule: str = ""
    score: float = 2.0
    positive_task_uris: list[str] = Field(default_factory=list)
    negative_task_uris: list[str] = Field(default_factory=list)


class ProjectContextMemory(BaseMemory):
    """Global scientific project context."""

    memory_type: Literal["project_context"] = "project_context"
    project_goal: str = ""
    species: str | None = None
    tissue_type: str | None = None
    disease_model: str | None = None


MEMORY_TYPE_CLASSES: dict[str, type[BaseMemory]] = {
    "dataset": DatasetMemory,
    "analysis": AnalysisMemory,
    "preference": PreferenceMemory,
    "insight": InsightMemory,
    "project_context": ProjectContextMemory,
}

MEMORY_TYPE_DOMAINS: dict[str, str] = {
    "dataset": "dataset",
    "analysis": "analysis",
    "preference": "preference",
    "insight": "insight",
    "project_context": "project",
}
