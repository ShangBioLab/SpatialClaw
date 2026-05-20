"""Memory-augmented reasoning operator for SpatialClaw.

This module is the copied-in local MAR execution layer for ReMemR1-style
memory reasoning. It keeps the operator code inside ``spatialclaw/agents``
while still delegating model inference to an external OpenAI-compatible API
or dedicated reasoning endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests
from openai import AsyncOpenAI

NO_MEMORY = "No previous memory"
NO_RECALLED_MEMORY = "No memory was recalled."

TEMPLATE = """You are presented with a problem, a retrieved memory chunk that may contain useful evidence, a recalled memory, and a previous working memory.

You should generate response in the following format:
- Output your reasoning in <thinking>...</thinking>.
- Update the working memory in exactly one <update>...</update> block.
- If the current memory is insufficient and you need to revisit older working memory, also output exactly one <recall>...</recall> block.

<problem>
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

<memory_chunk>
{chunk}
</memory_chunk>

Updated memory:
"""

TEMPLATE_FINAL = """You are assisting another agent. Based on the accumulated working memory, produce a concise reasoning context for a downstream tool-calling assistant.

Return exactly one <summary>...</summary> block containing:
- the key facts that matter for the current query
- the relevant prior analyses or dataset context
- any recommended next analytical direction

<problem>
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

Final reasoning context:
"""


def _get_env(primary_key: str, legacy_key: str, default: str = "") -> str:
    value = os.getenv(primary_key, "").strip()
    if value:
        return value
    legacy_value = os.getenv(legacy_key, "").strip()
    if legacy_value:
        return legacy_value
    return default


@dataclass
class MemoryCandidate:
    candidate_id: str
    memory_type: str
    text: str
    source: str = "memory"
    rank_score: float = 0.0


@dataclass
class RecallEntry:
    label: str
    text: str
    score_bias: float = 0.0


class MemoryAugmentedReasoningOperator:
    """ReMemR1-style pre-reasoning operator over SpatialClaw memory."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_candidates: int = 16,
        max_chunks: int = 8,
        max_memory_tokens: int = 768,
        max_summary_tokens: int = 1024,
        max_chunk_chars: int = 1600,
        invoke_mode: str = "auto",
        min_query_chars: int = 24,
        reason_endpoint: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"
        self.temperature = temperature
        self.top_p = top_p
        self.max_candidates = max_candidates
        self.max_chunks = max_chunks
        self.max_memory_tokens = max_memory_tokens
        self.max_summary_tokens = max_summary_tokens
        self.max_chunk_chars = max_chunk_chars
        self.invoke_mode = invoke_mode
        self.min_query_chars = min_query_chars
        self.reason_endpoint = reason_endpoint.strip()
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @classmethod
    def from_env(cls) -> "MemoryAugmentedReasoningOperator | None":
        base_url = _get_env("SPATIALCLAW_MAR_BASE_URL", "SPATIALCLAW_REMEM_BASE_URL")
        model = _get_env("SPATIALCLAW_MAR_MODEL", "SPATIALCLAW_REMEM_MODEL")
        if not base_url or not model:
            return None
        return cls(
            base_url=base_url,
            model=model,
            api_key=_get_env("SPATIALCLAW_MAR_API_KEY", "SPATIALCLAW_REMEM_API_KEY"),
            temperature=float(_get_env("SPATIALCLAW_MAR_TEMPERATURE", "SPATIALCLAW_REMEM_TEMPERATURE", "0.0")),
            top_p=float(_get_env("SPATIALCLAW_MAR_TOP_P", "SPATIALCLAW_REMEM_TOP_P", "1.0")),
            max_candidates=int(_get_env("SPATIALCLAW_MAR_MAX_CANDIDATES", "SPATIALCLAW_REMEM_MAX_CANDIDATES", "16")),
            max_chunks=int(_get_env("SPATIALCLAW_MAR_MAX_CHUNKS", "SPATIALCLAW_REMEM_MAX_CHUNKS", "8")),
            max_memory_tokens=int(_get_env("SPATIALCLAW_MAR_MAX_MEMORY_TOKENS", "SPATIALCLAW_REMEM_MAX_MEMORY_TOKENS", "768")),
            max_summary_tokens=int(_get_env("SPATIALCLAW_MAR_MAX_SUMMARY_TOKENS", "SPATIALCLAW_REMEM_MAX_SUMMARY_TOKENS", "1024")),
            max_chunk_chars=int(_get_env("SPATIALCLAW_MAR_MAX_CHUNK_CHARS", "SPATIALCLAW_REMEM_MAX_CHUNK_CHARS", "1600")),
            invoke_mode=_get_env("SPATIALCLAW_MAR_INVOKE_MODE", "SPATIALCLAW_REMEM_INVOKE_MODE", "auto").lower(),
            min_query_chars=int(_get_env("SPATIALCLAW_MAR_MIN_QUERY_CHARS", "SPATIALCLAW_REMEM_MIN_QUERY_CHARS", "24")),
            reason_endpoint=_get_env("SPATIALCLAW_MAR_REASON_ENDPOINT", "SPATIALCLAW_REMEM_REASON_ENDPOINT"),
        )

    def should_invoke(self, user_query: str) -> bool:
        query = (user_query or "").strip()
        if not query:
            return False
        if query.startswith("/"):
            return False

        mode = self.invoke_mode.lower()
        if mode == "always":
            return True
        if mode == "never":
            return False

        if len(query) < self.min_query_chars:
            return False

        lowered = query.lower()
        cues = (
            "remember",
            "previous",
            "earlier",
            "last time",
            "before",
            "again",
            "continue",
            "resume",
            "compared with",
            "what did we",
            "which dataset",
            "which method",
            "based on",
            "according to",
            "history",
            "memory",
            "之前",
            "上次",
            "继续",
            "回顾",
            "根据之前",
            "还记得",
            "哪个数据集",
            "哪个方法",
        )
        return any(cue in lowered for cue in cues)

    async def prepare_context(
        self,
        *,
        session_id: str,
        user_query: str,
        memory_store: Any,
    ) -> dict[str, Any]:
        if not user_query.strip():
            return {"summary": "", "retrieved_ids": [], "callbacks": [], "candidate_count": 0}

        candidates = await self._collect_candidates(
            session_id=session_id,
            user_query=user_query,
            memory_store=memory_store,
        )
        if not candidates:
            return {"summary": "", "retrieved_ids": [], "callbacks": [], "candidate_count": 0}

        if self.reason_endpoint:
            return await self._call_reason_endpoint(
                session_id=session_id,
                user_query=user_query,
                candidates=candidates[: self.max_chunks],
            )

        return await self._reason_over_candidates(
            user_query=user_query,
            candidates=candidates[: self.max_chunks],
        )

    async def _collect_candidates(
        self,
        *,
        session_id: str,
        user_query: str,
        memory_store: Any,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        seen: set[str] = set()

        def add_memory(memory: Any, memory_type: str, source: str) -> None:
            candidate_id = str(getattr(memory, "memory_id", "")) or f"{memory_type}:{len(candidates)}"
            if candidate_id in seen:
                return
            text = self._memory_to_text(memory, memory_type)
            if not text:
                return
            seen.add(candidate_id)
            candidates.append(
                MemoryCandidate(
                    candidate_id=candidate_id,
                    memory_type=memory_type,
                    text=text,
                    source=source,
                    rank_score=self._score_candidate(
                        user_query=user_query,
                        memory_type=memory_type,
                        text=text,
                        source=source,
                    ),
                )
            )

        if hasattr(memory_store, "search_memories"):
            try:
                for mem in await memory_store.search_memories(session_id, user_query):
                    add_memory(mem, getattr(mem, "memory_type", "memory"), "search")
            except Exception:
                pass

        recent_specs = [
            ("project_context", 1),
            ("dataset", 2),
            ("analysis", 4),
            ("insight", 4),
            ("preference", 4),
        ]
        for memory_type, limit in recent_specs:
            try:
                memories = await memory_store.get_memories(session_id, memory_type, limit=limit)
            except Exception:
                memories = []
            for mem in memories:
                add_memory(mem, memory_type, f"recent:{memory_type}")

        return self._rank_candidates(user_query, candidates)

    def _memory_to_text(self, memory: Any, memory_type: str) -> str:
        if memory_type == "dataset":
            return self._format_memory_block(
                "dataset",
                [
                    ("file_path", getattr(memory, "file_path", "")),
                    ("platform", getattr(memory, "platform", "")),
                    ("n_obs", getattr(memory, "n_obs", "")),
                    ("n_vars", getattr(memory, "n_vars", "")),
                    ("preprocessing_state", getattr(memory, "preprocessing_state", "")),
                    ("created_at", getattr(memory, "created_at", None)),
                ],
            )
        if memory_type == "analysis":
            return self._format_memory_block(
                "analysis",
                [
                    ("source_dataset_id", getattr(memory, "source_dataset_id", "")),
                    ("parent_analysis_id", getattr(memory, "parent_analysis_id", "")),
                    ("skill", getattr(memory, "skill", "")),
                    ("method", getattr(memory, "method", "")),
                    ("status", getattr(memory, "status", "")),
                    ("output_path", getattr(memory, "output_path", "")),
                    ("parameters", getattr(memory, "parameters", {})),
                    ("executed_at", getattr(memory, "executed_at", None)),
                ],
            )
        if memory_type == "preference":
            return self._format_memory_block(
                "preference",
                [
                    ("domain", getattr(memory, "domain", "")),
                    ("key", getattr(memory, "key", "")),
                    ("value", getattr(memory, "value", "")),
                    ("updated_at", getattr(memory, "updated_at", None)),
                ],
            )
        if memory_type == "insight":
            return self._format_memory_block(
                "insight",
                [
                    ("source_analysis_id", getattr(memory, "source_analysis_id", "")),
                    ("entity_type", getattr(memory, "entity_type", "")),
                    ("entity_id", getattr(memory, "entity_id", "")),
                    ("biological_label", getattr(memory, "biological_label", "")),
                    ("evidence", getattr(memory, "evidence", "")),
                    ("confidence", getattr(memory, "confidence", "")),
                    ("created_at", getattr(memory, "created_at", None)),
                ],
            )
        if memory_type == "project_context":
            return self._format_memory_block(
                "project_context",
                [
                    ("project_goal", getattr(memory, "project_goal", "")),
                    ("species", getattr(memory, "species", "")),
                    ("tissue_type", getattr(memory, "tissue_type", "")),
                    ("disease_model", getattr(memory, "disease_model", "")),
                    ("created_at", getattr(memory, "created_at", None)),
                ],
            )
        if hasattr(memory, "model_dump_json"):
            return self._truncate(memory.model_dump_json())
        return self._truncate(str(memory))

    def _format_memory_block(self, memory_type: str, fields: list[tuple[str, Any]]) -> str:
        lines = [f"[MEMORY_TYPE={memory_type}]"]
        for key, value in fields:
            rendered = self._stringify_value(value)
            if rendered:
                lines.append(f"{key}: {rendered}")
        return self._truncate("\n".join(lines))

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value if item not in (None, ""))
        return str(value).strip()

    def _truncate(self, text: str) -> str:
        text = (text or "").strip()
        if len(text) <= self.max_chunk_chars:
            return text
        head = self.max_chunk_chars // 2
        tail = self.max_chunk_chars - head - len("\n...(truncated)...\n")
        return text[:head] + "\n...(truncated)...\n" + text[-tail:]

    def _preferred_memory_types(self, user_query: str) -> list[str]:
        lowered = (user_query or "").lower()
        order = ["analysis", "dataset", "project_context", "insight", "preference"]

        cue_map = {
            "dataset": ("dataset", "sample", "file", "h5ad", "visium"),
            "analysis": ("analysis", "method", "pipeline", "preprocess", "cluster", "run"),
            "preference": ("prefer", "language", "format", "english", "chinese", "中文"),
            "insight": ("cell type", "marker", "biological", "insight", "annotation"),
            "project_context": ("goal", "species", "tissue", "disease", "project"),
        }

        preferred = None
        for memory_type, cues in cue_map.items():
            if any(cue in lowered for cue in cues):
                preferred = memory_type
                break

        if preferred and preferred in order:
            order.remove(preferred)
            order.insert(0, preferred)
        return order

    def _score_candidate(
        self,
        *,
        user_query: str,
        memory_type: str,
        text: str,
        source: str,
    ) -> float:
        query_tokens = self._tokenize(user_query)
        text_tokens = self._tokenize(text)
        union = query_tokens | text_tokens
        overlap = len(query_tokens & text_tokens)
        lexical_score = overlap / len(union) if union else 0.0

        preferred_order = self._preferred_memory_types(user_query)
        type_bonus = 0.0
        if memory_type in preferred_order:
            type_bonus = max(0.0, 0.75 - (preferred_order.index(memory_type) * 0.22))

        source_bonus = 0.1 if source == "search" else 0.0
        return lexical_score + type_bonus + source_bonus

    def _rank_candidates(self, user_query: str, candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        preferred_order = self._preferred_memory_types(user_query)
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.rank_score,
                preferred_order.index(candidate.memory_type) if candidate.memory_type in preferred_order else len(preferred_order),
                candidate.candidate_id,
            ),
        )

        diversified: list[MemoryCandidate] = []
        seen_types: set[str] = set()
        for candidate in ordered:
            if candidate.memory_type in seen_types:
                continue
            diversified.append(candidate)
            seen_types.add(candidate.memory_type)
            if len(diversified) >= self.max_candidates:
                break

        for candidate in ordered:
            if candidate in diversified:
                continue
            diversified.append(candidate)
            if len(diversified) >= self.max_candidates:
                break
        return diversified

    async def _reason_over_candidates(
        self,
        *,
        user_query: str,
        candidates: list[MemoryCandidate],
    ) -> dict[str, Any]:
        working_memory = NO_MEMORY
        recalled_memory = NO_RECALLED_MEMORY
        history_entries: list[RecallEntry] = []
        retrieved_ids: list[str] = []
        callbacks: list[str] = []
        step_trace: list[dict[str, Any]] = []

        for step, candidate in enumerate(candidates):
            prompt = TEMPLATE.format(
                prompt=user_query,
                recalled_memory=recalled_memory,
                memory=working_memory,
                chunk=candidate.text,
            )
            response = await self._chat(prompt, max_tokens=self.max_memory_tokens)
            updated_memory = self._parse_update_memory(response) or working_memory
            recall_query = self._parse_recall_query(response)

            working_memory = updated_memory
            retrieved_ids.append(candidate.candidate_id)

            recalled_from = None
            if recall_query:
                callbacks.append(recall_query)
                recalled_entry = self._retrieve_from_history(recall_query, history_entries)
                recalled_memory = recalled_entry.text if recalled_entry else NO_RECALLED_MEMORY
                recalled_from = recalled_entry.label if recalled_entry else None
            else:
                recalled_memory = NO_RECALLED_MEMORY

            step_trace.append(
                {
                    "step": step,
                    "candidate_id": candidate.candidate_id,
                    "candidate_type": candidate.memory_type,
                    "candidate_source": candidate.source,
                    "rank_score": round(candidate.rank_score, 4),
                    "recall_query": recall_query,
                    "recalled_from": recalled_from,
                }
            )

            history_entries.append(
                RecallEntry(
                    label=f"chunk:{candidate.candidate_id}",
                    text=candidate.text,
                    score_bias=0.1,
                )
            )
            history_entries.append(
                RecallEntry(
                    label=f"working_memory:{candidate.candidate_id}",
                    text=updated_memory,
                    score_bias=0.2,
                )
            )

        final_prompt = TEMPLATE_FINAL.format(
            prompt=user_query,
            recalled_memory=recalled_memory,
            memory=working_memory,
        )
        final_response = await self._chat(final_prompt, max_tokens=self.max_summary_tokens)
        summary = self._parse_summary(final_response) or final_response.strip()

        return {
            "summary": summary,
            "retrieved_ids": retrieved_ids,
            "callbacks": callbacks,
            "candidate_count": len(candidates),
            "mode": "local_loop",
            "used_candidate_types": [c.memory_type for c in candidates],
            "candidate_briefs": [
                {
                    "id": c.candidate_id,
                    "type": c.memory_type,
                    "source": c.source,
                    "rank_score": round(c.rank_score, 4),
                }
                for c in candidates
            ],
            "trace": {
                "steps": len(step_trace),
                "step_trace": step_trace,
            },
        }

    async def _call_reason_endpoint(
        self,
        *,
        session_id: str,
        user_query: str,
        candidates: list[MemoryCandidate],
    ) -> dict[str, Any]:
        payload = {
            "session_id": session_id,
            "query": user_query,
            "model": self.model,
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "type": candidate.memory_type,
                    "text": candidate.text,
                }
                for candidate in candidates
            ],
            "config": {
                "max_chunks": self.max_chunks,
                "max_memory_tokens": self.max_memory_tokens,
                "max_summary_tokens": self.max_summary_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def _post() -> dict[str, Any]:
            response = requests.post(
                self.reason_endpoint,
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()

        data = await asyncio.to_thread(_post)

        summary = str(data.get("summary", "") or "").strip()
        return {
            "summary": summary,
            "retrieved_ids": data.get("used_memory_ids") or [c.candidate_id for c in candidates],
            "callbacks": data.get("callbacks", []),
            "candidate_count": len(candidates),
            "mode": "remote_endpoint",
            "used_candidate_types": [c.memory_type for c in candidates],
            "candidate_briefs": [
                {
                    "id": c.candidate_id,
                    "type": c.memory_type,
                    "source": c.source,
                    "rank_score": round(c.rank_score, 4),
                }
                for c in candidates
            ],
            "trace": data.get("trace", {}),
            "raw": data,
        }

    async def _chat(self, prompt: str, *, max_tokens: int) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def _retrieve_from_history(self, query: str, history_memories: list[RecallEntry]) -> RecallEntry | None:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return None
        best_score = 0.0
        best_memory = None
        for memory in history_memories:
            memory_tokens = self._tokenize(memory.text)
            union = query_tokens | memory_tokens
            if not union:
                continue
            score = (len(query_tokens & memory_tokens) / len(union)) + memory.score_bias
            if score > best_score:
                best_score = score
                best_memory = memory
        return best_memory

    def _tokenize(self, text: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
        return {token for token in cleaned.split() if token}

    def _parse_recall_query(self, text: str) -> str | None:
        match = re.search(r"<recall>(.+?)</recall>", text, flags=re.DOTALL)
        return match.group(1).strip() if match else None

    def _parse_update_memory(self, text: str) -> str | None:
        match = re.search(r"<update>(.+?)</update>", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        cleaned = re.sub(r"<recall>.*?</recall>", "", text, flags=re.DOTALL).strip()
        return cleaned or None

    def _parse_summary(self, text: str) -> str | None:
        match = re.search(r"<summary>(.+?)</summary>", text, flags=re.DOTALL)
        return match.group(1).strip() if match else None


MAROperator = MemoryAugmentedReasoningOperator
ReMemReasoningOperator = MemoryAugmentedReasoningOperator

__all__ = [
    "MAROperator",
    "MemoryAugmentedReasoningOperator",
    "MemoryCandidate",
    "NO_MEMORY",
    "NO_RECALLED_MEMORY",
    "RecallEntry",
    "ReMemReasoningOperator",
    "TEMPLATE",
    "TEMPLATE_FINAL",
]
