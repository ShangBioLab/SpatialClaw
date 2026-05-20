"""Typed episodic / semantic memory store backed by graph storage."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from .memory_client import MemoryClient
from .preference_policy import (
    is_workflow_state_preference_key,
    preference_to_prompt_entry,
)
from .promotion import (
    LAYER_EPISODIC,
    LAYER_SEMANTIC,
    VALID_LAYERS,
    evaluate_promotion,
)
from .types import (
    AnalysisMemory,
    BaseMemory,
    DatasetMemory,
    InsightMemory,
    MEMORY_TYPE_CLASSES,
    MEMORY_TYPE_DOMAINS,
    PreferenceMemory,
    ProjectContextMemory,
    Session,
    utcnow,
)

logger = logging.getLogger(__name__)


_TYPE_TO_DOMAIN = MEMORY_TYPE_DOMAINS

_LAYER_TO_DOMAIN = {
    LAYER_EPISODIC: "episodic",
    LAYER_SEMANTIC: "semantic",
}

_TYPE_CLASSES = MEMORY_TYPE_CLASSES


def _memory_to_uri_path(memory: BaseMemory) -> str:
    """Convert a memory object to a URI path within its domain."""
    if isinstance(memory, DatasetMemory):
        return memory.file_path.replace("/", "_")
    elif isinstance(memory, AnalysisMemory):
        return f"{memory.skill}/{memory.memory_id}"
    elif isinstance(memory, PreferenceMemory):
        return f"{memory.domain}/{memory.key}"
    elif isinstance(memory, InsightMemory):
        return f"{memory.entity_type}/{memory.entity_id}"
    elif isinstance(memory, ProjectContextMemory):
        return "current"
    return memory.memory_id


def filter_context_preferences_for_continuation(memory_context: str) -> str:
    """Remove workflow-state preferences from prompt context during continuation."""
    if not memory_context.strip():
        return memory_context

    sections: list[tuple[str | None, list[str]]] = []
    current_header: str | None = None
    current_lines: list[str] = []

    for line in memory_context.splitlines():
        if line.startswith("**") and line.endswith("**:"):
            if current_header is not None or current_lines:
                sections.append((current_header, current_lines))
            current_header = line
            current_lines = []
            continue
        current_lines.append(line)

    if current_header is not None or current_lines:
        sections.append((current_header, current_lines))

    filtered_sections: list[str] = []
    for header, lines in sections:
        if header == "**User Preferences**:":
            kept_lines: list[str] = []
            for line in lines:
                stripped = line.strip()
                if not stripped.startswith("- "):
                    if stripped:
                        kept_lines.append(line)
                    continue
                key_part = stripped[2:].split(":", 1)[0].strip()
                if is_workflow_state_preference_key(key_part):
                    continue
                kept_lines.append(line)

            if kept_lines:
                filtered_sections.append(header)
                filtered_sections.extend(kept_lines)
            continue

        if header is not None:
            filtered_sections.append(header)
        filtered_sections.extend(lines)

    return "\n".join(filtered_sections).strip()


def _normalize_session_path(session_id: str) -> str:
    """Normalize a session ID into a stable path fragment."""
    return (session_id or "default").strip().replace("/", "_")


def _session_root_path(session_id: str) -> str:
    return _normalize_session_path(session_id)


def _memory_root_path(session_id: str, memory_type: str) -> str:
    return f"{_session_root_path(session_id)}/{memory_type}"


def _memory_uri(session_id: str, layer: str, memory: BaseMemory) -> str:
    domain = _LAYER_TO_DOMAIN[layer]
    base = _memory_root_path(session_id, memory.memory_type)
    return f"{domain}://{base}/{_memory_to_uri_path(memory)}"


def _session_id_from_path(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return None
    return parts[0]


def _infer_memory_type(path: str) -> Optional[str]:
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 2 and parts[1] in _TYPE_CLASSES:
        return parts[1]
    if parts and parts[0] in _TYPE_CLASSES:
        return parts[0]
    return None


def _layer_domains(layer: str) -> tuple[str, ...]:
    if layer == "all":
        return (_LAYER_TO_DOMAIN[LAYER_SEMANTIC], _LAYER_TO_DOMAIN[LAYER_EPISODIC])

    domain = _LAYER_TO_DOMAIN.get(layer)
    return (domain,) if domain else ()


def _memory_sort_key(memory: BaseMemory) -> tuple[float, int]:
    created_at = memory.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    layer_rank = 1 if memory.memory_layer == LAYER_SEMANTIC else 0
    return (created_at.timestamp(), layer_rank)


def _with_layer_metadata(
    memory: BaseMemory,
    *,
    session_id: str,
    layer: str,
    importance: float,
    confidence: float,
    evidence_uri: Optional[str],
    promotion_reasons: list[str],
    promoted_at: Optional[datetime],
) -> BaseMemory:
    """Stamp one layer-specific view of a memory object."""
    return memory.model_copy(
        update={
            "source_session_id": session_id,
            "memory_layer": layer,
            "importance": importance,
            "confidence_score": confidence,
            "evidence_uri": evidence_uri,
            "promoted_at": promoted_at if layer == LAYER_SEMANTIC else None,
            "promotion_reasons": list(promotion_reasons),
        }
    )


def _memory_to_content(memory: BaseMemory) -> str:
    """Serialize a memory to JSON content for graph storage."""
    return memory.model_dump_json()


def _content_to_memory(content: str, memory_type: str) -> Optional[BaseMemory]:
    """Deserialize graph memory content to a Pydantic model."""
    cls = _TYPE_CLASSES.get(memory_type)
    if not cls:
        return None
    try:
        return cls.model_validate_json(content)
    except Exception:
        return None


class LayeredMemoryStore:
    """Typed store for episodic and semantic Memory records."""

    def __init__(self, database_url: Optional[str] = None):
        self._client = MemoryClient(database_url)

    async def initialize(self) -> None:
        """Initialize storage backend."""
        await self._client.initialize()

    async def close(self) -> None:
        """Close backend resources."""
        await self._client.close()

    async def create_session(self, user_id: str, platform: str, chat_id: str = "", session_id: str = None) -> Session:
        """Create a new session in the graph memory."""
        session_id = session_id or uuid.uuid4().hex[:16]
        session = Session(
            session_id=session_id,
            user_id=user_id,
            platform=platform,
        )

        await self._client.remember(
            uri=f"session://{session_id}",
            content=session.model_dump_json(),
            disclosure=f"Session for user {user_id} on {platform}",
        )

        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID."""
        mem = await self._client.recall(f"session://{session_id}")
        if not mem or not mem.get("content"):
            return None
        try:
            return Session.model_validate_json(mem["content"])
        except Exception as e:
            logger.warning("Failed to parse session %s: %s", session_id, e)
            return None

    async def update_session(self, session_id: str, updates: dict) -> None:
        """Update session fields."""
        session = await self.get_session(session_id)
        if not session:
            return

        data = session.model_dump()
        data.update(updates)
        data["last_activity"] = utcnow().isoformat()

        updated = Session.model_validate(data)
        await self._client.remember(
            uri=f"session://{session_id}",
            content=updated.model_dump_json(),
        )

    async def save_memory(self, session_id: str, memory: BaseMemory) -> str:
        """Save a memory, return memory_id."""
        await self._store_memory_layers(session_id, memory)

        # Sync preference to the session object for frontend visibility
        if memory.memory_type == "preference" and hasattr(memory, "key") and hasattr(memory, "value"):
            session = await self.get_session(session_id)
            if session:
                updates = session.preferences.copy()
                updates[memory.key] = memory.value
                await self.update_session(session_id, {"preferences": updates})

        return memory.memory_id

    def build_memory_uri(
        self,
        session_id: str,
        memory: BaseMemory,
        *,
        layer: str | None = None,
    ) -> str:
        """Build the graph URI for one typed memory in one concrete layer."""
        layer_name = layer or memory.memory_layer
        if layer_name not in (LAYER_EPISODIC, LAYER_SEMANTIC):
            raise ValueError(f"Layer must be {LAYER_EPISODIC!r} or {LAYER_SEMANTIC!r}")
        return _memory_uri(session_id, layer_name, memory)

    async def get_memory_uris(
        self,
        session_id: str,
        memory: BaseMemory,
        *,
        layer: str = "all",
    ) -> list[str]:
        """Return existing graph URIs for a typed memory across requested layers."""
        if layer not in VALID_LAYERS:
            return []

        layer_names = (
            (LAYER_SEMANTIC, LAYER_EPISODIC)
            if layer == "all"
            else (layer,)
        )
        uris: list[str] = []
        for layer_name in layer_names:
            uri = _memory_uri(session_id, layer_name, memory)
            stored = await self._client.recall(uri)
            if not stored or not stored.get("content"):
                continue
            stored_obj = _content_to_memory(stored["content"], memory.memory_type)
            if stored_obj and stored_obj.memory_id == memory.memory_id:
                uris.append(uri)
        return uris

    async def _store_memory_layers(self, session_id: str, memory: BaseMemory) -> None:
        """Persist a memory to its eligible layers."""
        decision = evaluate_promotion(memory)
        promoted_at = memory.promoted_at or (utcnow() if decision.promote_to_semantic else None)
        layered_memories: list[tuple[str, BaseMemory]] = []

        if decision.store_episodic:
            layered_memories.append(
                (
                    LAYER_EPISODIC,
                    _with_layer_metadata(
                        memory,
                        session_id=session_id,
                        layer=LAYER_EPISODIC,
                        importance=decision.importance,
                        confidence=decision.confidence,
                        evidence_uri=decision.evidence_uri,
                        promotion_reasons=decision.reasons,
                        promoted_at=promoted_at,
                    ),
                )
            )

        if decision.promote_to_semantic:
            layered_memories.append(
                (
                    LAYER_SEMANTIC,
                    _with_layer_metadata(
                        memory,
                        session_id=session_id,
                        layer=LAYER_SEMANTIC,
                        importance=decision.importance,
                        confidence=decision.confidence,
                        evidence_uri=decision.evidence_uri,
                        promotion_reasons=decision.reasons,
                        promoted_at=promoted_at,
                    ),
                )
            )

        if not layered_memories:
            layered_memories.append(
                (
                    LAYER_EPISODIC,
                    _with_layer_metadata(
                        memory,
                        session_id=session_id,
                        layer=LAYER_EPISODIC,
                        importance=decision.importance,
                        confidence=decision.confidence,
                        evidence_uri=decision.evidence_uri,
                        promotion_reasons=decision.reasons,
                        promoted_at=promoted_at,
                    ),
                )
            )

        for layer_name, layered_memory in layered_memories:
            await self._client.remember(
                uri=_memory_uri(session_id, layer_name, layered_memory),
                content=_memory_to_content(layered_memory),
                disclosure=f"{layer_name.title()} memory from session {session_id}",
            )

    async def get_memories(
        self,
        session_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
        layer: str = "all",
    ) -> list[BaseMemory]:
        """Retrieve memories, optionally filtered by type."""
        if layer not in VALID_LAYERS:
            return []

        if memory_type:
            if memory_type not in _TYPE_TO_DOMAIN:
                return []

        collected: list[BaseMemory] = []
        memory_types = [memory_type] if memory_type else list(_TYPE_TO_DOMAIN.keys())

        async def _collect(uri: str, expected_type: Optional[str]):
            """Recursively collect leaf memories from a domain tree."""
            children = await self._client.list_children(uri)
            for child in children:
                child_uri = f"{child['domain']}://{child['path']}"
                mem = await self._client.recall(child_uri)
                if mem and mem.get("content"):
                    inferred_type = expected_type or _infer_memory_type(child.get("path", ""))
                    obj = _content_to_memory(mem["content"], inferred_type or "")
                    if obj:
                        collected.append(obj)
                    else:
                        # Not a valid leaf — recurse into this container node
                        await _collect(child_uri, inferred_type)

        for domain in _layer_domains(layer):
            for mtype in memory_types:
                root_uri = f"{domain}://{_memory_root_path(session_id, mtype)}"
                await _collect(root_uri, mtype)

        results: list[BaseMemory] = []
        seen_memory_ids: set[str] = set()
        for memory in sorted(collected, key=_memory_sort_key, reverse=True):
            if memory.memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(memory.memory_id)
            results.append(memory)
            if len(results) >= limit:
                break

        return results

    async def update_memory(self, memory_id: str, updates: dict) -> None:
        """Update memory fields — search and update in place."""
        results = await self._client.search(memory_id, limit=50)
        existing_uris: dict[str, str] = {}
        base_obj: BaseMemory | None = None
        session_id: str | None = None

        for r in results:
            mem = await self._client.recall(r["uri"])
            if mem and mem.get("content"):
                try:
                    data = json.loads(mem["content"])
                    if data.get("memory_id") == memory_id:
                        data.update(updates)
                        inferred_type = data.get("memory_type") or _infer_memory_type(r.get("path", ""))
                        obj = _content_to_memory(json.dumps(data), inferred_type or "")
                        if not obj:
                            continue

                        if base_obj is None or obj.memory_layer == LAYER_SEMANTIC:
                            base_obj = obj
                        session_id = session_id or obj.source_session_id or _session_id_from_path(r.get("path", ""))
                        if obj.memory_layer:
                            existing_uris[obj.memory_layer] = r["uri"]
                except json.JSONDecodeError:
                    continue

        if base_obj and session_id:
            base_data = base_obj.model_dump()
            base_data.update(
                {
                    "source_session_id": session_id,
                    "memory_layer": None,
                    "importance": None,
                    "confidence_score": None,
                    "evidence_uri": None,
                    "promoted_at": None,
                    "promotion_reasons": [],
                }
            )
            base_data.update(updates)

            memory_cls = _TYPE_CLASSES[base_obj.memory_type]
            updated_obj = memory_cls.model_validate(base_data)
            decision = evaluate_promotion(updated_obj)
            promoted_at = updated_obj.promoted_at or (utcnow() if decision.promote_to_semantic else None)

            desired_layers: list[str] = []
            if decision.store_episodic:
                desired_layers.append(LAYER_EPISODIC)
            if decision.promote_to_semantic:
                desired_layers.append(LAYER_SEMANTIC)
            if not desired_layers:
                desired_layers.append(LAYER_EPISODIC)

            desired_uris: set[str] = set()
            for layer_name in desired_layers:
                layered_memory = _with_layer_metadata(
                    updated_obj,
                    session_id=session_id,
                    layer=layer_name,
                    importance=decision.importance,
                    confidence=decision.confidence,
                    evidence_uri=decision.evidence_uri,
                    promotion_reasons=decision.reasons,
                    promoted_at=promoted_at,
                )
                uri = _memory_uri(session_id, layer_name, layered_memory)
                desired_uris.add(uri)
                await self._client.remember(
                    uri=uri,
                    content=_memory_to_content(layered_memory),
                    disclosure=f"{layer_name.title()} memory from session {session_id}",
                )

            for existing_uri in existing_uris.values():
                if existing_uri in desired_uris:
                    continue
                try:
                    await self._client.forget(existing_uri)
                except ValueError:
                    pass

            if updated_obj.memory_type == "preference" and hasattr(updated_obj, "key") and hasattr(updated_obj, "value"):
                session = await self.get_session(session_id)
                if session:
                    preferences = session.preferences.copy()
                    preferences[updated_obj.key] = updated_obj.value
                    await self.update_session(session_id, {"preferences": preferences})

    async def delete_session(self, session_id: str) -> None:
        """Delete session."""
        await self._delete_uri_tree(f"episodic://{_session_root_path(session_id)}")
        await self._delete_uri_tree(f"semantic://{_session_root_path(session_id)}")
        try:
            await self._client.forget(f"session://{session_id}")
        except ValueError:
            pass

    async def search_memories(
        self,
        session_id: str,
        query: str,
        memory_type: Optional[str] = None,
        layer: str = "all",
    ) -> list[BaseMemory]:
        """Search memories by content."""
        if layer not in VALID_LAYERS:
            return []
        if memory_type and memory_type not in _TYPE_TO_DOMAIN:
            return []

        memories: list[BaseMemory] = []
        seen_memory_ids: set[str] = set()
        session_prefix = f"{_session_root_path(session_id)}/"

        for domain in _layer_domains(layer):
            results = await self._client.search(query, limit=50, domain=domain)
            for r in results:
                path = r.get("path", "")
                if not path.startswith(session_prefix):
                    continue

                inferred_type = _infer_memory_type(path)
                if memory_type and inferred_type != memory_type:
                    continue

                mem = await self._client.recall(r["uri"])
                if not mem or not mem.get("content"):
                    continue

                obj = _content_to_memory(mem["content"], inferred_type or "")
                if not obj or obj.memory_id in seen_memory_ids:
                    continue

                seen_memory_ids.add(obj.memory_id)
                memories.append(obj)
                if len(memories) >= 20:
                    return memories

        return memories

    async def load_context(self, session_id: str) -> str:
        """Load layered context for LLM prompt injection."""
        project_ctx = await self.get_memories(
            session_id, "project_context", limit=1, layer=LAYER_SEMANTIC
        )
        datasets = await self.get_memories(
            session_id, "dataset", limit=1, layer=LAYER_EPISODIC
        )
        analyses = await self.get_memories(
            session_id, "analysis", limit=3, layer="all"
        )
        prefs = await self.get_memories(
            session_id, "preference", limit=5, layer=LAYER_SEMANTIC
        )
        insights = await self.get_memories(
            session_id, "insight", limit=3, layer=LAYER_SEMANTIC
        )

        parts = []

        if project_ctx:
            pc = project_ctx[0]
            ctx_parts = []
            if pc.project_goal:
                ctx_parts.append(f"Goal: {pc.project_goal}")
            if pc.species:
                ctx_parts.append(f"Species: {pc.species}")
            if pc.tissue_type:
                ctx_parts.append(f"Tissue: {pc.tissue_type}")
            if pc.disease_model:
                ctx_parts.append(f"Disease: {pc.disease_model}")
            if ctx_parts:
                parts.append("**Project Context**: " + " | ".join(ctx_parts))

        if datasets:
            ds = datasets[0]
            parts.append(
                f"**Current Dataset**: {ds.file_path} "
                f"({ds.platform or 'unknown'}, {ds.n_obs or '?'} obs, {ds.preprocessing_state})"
            )

        if analyses:
            parts.append("**Recent Analyses**:")
            for i, analysis in enumerate(analyses[:3], 1):
                skill_name = analysis.skill or "unknown_skill"
                method_name = analysis.method or "unknown_method"
                line = f"{i}. {skill_name} ({method_name}) - {analysis.status}"
                if analysis.output_path:
                    line += f" | output: {analysis.output_path}"
                parts.append(line)

        if prefs:
            pref_lines: list[str] = []
            for pref in prefs:
                prompt_entry = preference_to_prompt_entry(pref.key, pref.value)
                if prompt_entry is None:
                    continue
                prompt_key, prompt_value = prompt_entry
                pref_lines.append(f"- {prompt_key}: {prompt_value}")

            if pref_lines:
                parts.append("**User Preferences**:")
                parts.extend(pref_lines)

        if insights:
            parts.append("**Known Insights**:")
            for insight in insights:
                confidence = "confirmed" if insight.confidence == "user_confirmed" else "predicted"
                line = (
                    f"- {insight.entity_type} {insight.entity_id}: "
                    f"{insight.biological_label} ({confidence})"
                )
                if getattr(insight, "evidence", ""):
                    line += f" | evidence: {insight.evidence}"
                parts.append(line)

        return "\n\n".join(parts) if parts else ""

    async def _delete_uri_tree(self, uri: str) -> None:
        """Recursively delete a URI subtree from leaves to root."""
        mem = await self._client.recall(uri)
        if not mem:
            return

        children = await self._client.list_children(uri)
        for child in children:
            await self._delete_uri_tree(f"{child['domain']}://{child['path']}")

        try:
            await self._client.forget(uri)
        except ValueError:
            pass

    # ------------------------------------------------------------------
