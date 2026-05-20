"""Post-task insight extraction for SpatialClaw layered memory."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from typing import Callable

from .layered_store import AnalysisMemory, BaseMemory, InsightMemory, LayeredMemoryStore

logger = logging.getLogger(__name__)


SYSTEM_COMPARE = (
    "You are a helpful assistant that improves reusable rules by comparing "
    "successful and failed spatial analysis task trajectories."
)

USER_COMPARE = """
Here are two task trajectories:

Task 1 (Successful):
{task1}
{task1_trajectory}

Task 2 (Failed):
{task2}
{task2_trajectory}
Fail reason: {fail_reason}

Existing rules:
{existing_rules}

Based on the comparison, update the rules. Use the following operations:
- ADD: <new rule text ending with period>
- EDIT <rule_number>: <updated rule text ending with period>
- REMOVE <rule_number>: <rule text>
- AGREE <rule_number>: <rule text>
"""

SYSTEM_SUCCESS = (
    "You are a helpful assistant that derives concise reusable rules from "
    "successful spatial analysis task trajectories."
)

USER_SUCCESS = """
Here are successful task trajectories:
{success_history}

Existing rules:
{existing_rules}

Based on these successes, update the rules using ADD / EDIT / REMOVE / AGREE operations.
"""

SYSTEM_MERGE = "You are a helpful assistant that merges and deduplicates a list of rules."

USER_MERGE = """
Merge the following rules into at most {limited_number} concise, non-overlapping rules:
{current_rules}
"""


class InsightExtractor:
    """Extract, retrieve, and refine reusable insight rules."""

    START_THRESHOLD = 5
    ROUNDS_PER_REFINEMENT = 5
    INSIGHT_SAMPLE_COUNT = 5
    MAX_RULE_COUNT = 10

    def __init__(
        self,
        store: LayeredMemoryStore,
        llm_callable: Callable[[str, str], str],
    ):
        self._store = store
        self._llm = llm_callable

    async def on_task_complete(
        self,
        session_id: str,
        analysis: AnalysisMemory,
        memory_size: int,
    ) -> None:
        """Run periodic insight refinement after enough task memories exist."""
        if (
            memory_size >= self.START_THRESHOLD
            and memory_size % self.ROUNDS_PER_REFINEMENT == 0
        ):
            await self.refine_insights(session_id, self.INSIGHT_SAMPLE_COUNT)

    async def refine_insights(self, session_id: str, num_points: int) -> None:
        """Sample recent analyses and update reusable insight rules."""
        all_analyses: list[AnalysisMemory] = await self._store.get_memories(
            session_id,
            memory_type="analysis",
            limit=200,
        )
        if not all_analyses:
            return

        for _ in range(num_points):
            anchor = random.choice(all_analyses)
            successful = [item for item in all_analyses if item.status == "completed"][:3]
            failed = [item for item in all_analyses if item.status == "failed"][:1]

            if anchor.status == "completed" and anchor not in successful:
                successful.append(anchor)
            elif anchor.status != "completed" and anchor not in failed:
                failed.append(anchor)

            related_uris = await self._analysis_uris(session_id, successful + failed)
            related_insights = await self._find_related_insights(
                session_id,
                related_uris,
                threshold=max(1, len(related_uris) // 2),
            )

            await self._refine_insight_rules(
                session_id,
                successful,
                failed,
                related_insights,
            )

        await self._prune_insights(session_id)

    async def _refine_insight_rules(
        self,
        session_id: str,
        successful: list[AnalysisMemory],
        failed: list[AnalysisMemory],
        related_insights: list[InsightMemory],
    ) -> None:
        """Run LLM rule refinement over successful and failed trajectories."""
        rule_text = self._format_rules(related_insights)

        for i, failed_analysis in enumerate(failed):
            if i >= len(successful):
                break
            successful_analysis = successful[i]
            user_prompt = USER_COMPARE.format(
                task1=successful_analysis.skill,
                task1_trajectory=successful_analysis.task_trajectory,
                task2=failed_analysis.skill,
                task2_trajectory=failed_analysis.task_trajectory,
                fail_reason=failed_analysis.fail_reason,
                existing_rules=rule_text,
            )
            response = self._llm(SYSTEM_COMPARE, user_prompt)
            operations = self._parse_rules(response)
            task_uris = await self._analysis_uris(
                session_id,
                [successful_analysis, failed_analysis],
            )
            await self._update_rules(
                session_id,
                task_uris,
                operations,
                related_insights,
            )

        if successful:
            history = "\n".join(
                f"task{i}:\n{task.skill}\n{task.key_steps}"
                for i, task in enumerate(successful)
            )
            user_prompt = USER_SUCCESS.format(
                success_history=history,
                existing_rules=rule_text,
            )
            response = self._llm(SYSTEM_SUCCESS, user_prompt)
            operations = self._parse_rules(response)
            task_uris = await self._analysis_uris(session_id, successful)
            await self._update_rules(
                session_id,
                task_uris,
                operations,
                related_insights,
            )

    async def apply_feedback(
        self,
        session_id: str,
        insight_uris: list[str],
        reward: float,
    ) -> None:
        """Reinforce or penalize insight scores after a task outcome."""
        for uri in insight_uris:
            stored = await self._store._client.recall(uri)
            if not stored or not stored.get("content"):
                continue
            try:
                data = json.loads(stored["content"])
                memory_id = data.get("memory_id")
                if not memory_id:
                    continue
                await self._store.update_memory(
                    memory_id,
                    {"score": data.get("score", 2.0) + reward},
                )
            except Exception as exc:
                logger.debug("Insight feedback skipped for %s: %s", uri, exc)

        await self._prune_insights(session_id)

    async def query_insights(
        self,
        session_id: str,
        task_main: str,
        top_k: int = 10,
    ) -> list[InsightMemory]:
        """Retrieve reusable insight rules related to a task description."""
        _, _, insights = await self.retrieve_for_task(
            session_id=session_id,
            task_main=task_main,
            successful_topk=0,
            failed_topk=0,
            insight_topk=top_k,
        )
        return insights

    async def merge_insights(self, session_id: str) -> None:
        """Merge overlapping rule insights within coarse task clusters."""
        all_analyses: list[AnalysisMemory] = await self._store.get_memories(
            session_id,
            memory_type="analysis",
            limit=500,
        )
        all_insights: list[InsightMemory] = await self._store.get_memories(
            session_id,
            memory_type="insight",
            limit=500,
        )

        cluster_task_uris: dict[int, list[str]] = {}
        for analysis in all_analyses:
            cluster_id = analysis.cluster_id if analysis.cluster_id is not None else 0
            cluster_task_uris.setdefault(cluster_id, [])
            cluster_task_uris[cluster_id].extend(
                await self._memory_uris(session_id, analysis)
            )

        for cluster_id, task_uris in cluster_task_uris.items():
            task_uri_set = set(task_uris)
            related = [
                insight
                for insight in all_insights
                if insight.rule and task_uri_set.intersection(insight.positive_task_uris)
            ]
            rules = [insight.rule for insight in related]
            if len(rules) < 2:
                continue

            merged_rules = self._merge_rules(rules)
            for index, rule in enumerate(merged_rules):
                insight = InsightMemory(
                    source_analysis_id=task_uris[0] if task_uris else "merged",
                    entity_type="rule_cluster",
                    entity_id=f"{cluster_id}_{index}",
                    biological_label=rule[:80],
                    evidence="Merged from related task-rule memories.",
                    rule=rule,
                    score=2.0,
                    positive_task_uris=task_uris,
                )
                await self._store.save_memory(session_id, insight)
                await self._link_insight_to_tasks(session_id, insight, task_uris)

    async def retrieve_for_task(
        self,
        session_id: str,
        task_main: str,
        successful_topk: int = 2,
        failed_topk: int = 1,
        insight_topk: int = 3,
        hop: int = 1,
    ) -> tuple[list[AnalysisMemory], list[AnalysisMemory], list[InsightMemory]]:
        """Retrieve related successful tasks, failed tasks, and insight rules."""
        all_analyses: list[AnalysisMemory] = await self._store.get_memories(
            session_id,
            memory_type="analysis",
            limit=100,
        )

        scored = sorted(
            [
                (analysis, self._jaccard(task_main, analysis.task_main))
                for analysis in all_analyses
                if analysis.task_main
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        seed_analyses = [analysis for analysis, score in scored if score > 0][:3]

        expanded_uris: set[str] = set()
        for seed in seed_analyses:
            seed_uri = await self._primary_uri(session_id, seed)
            if not seed_uri:
                continue
            expanded_uris.add(seed_uri)
            try:
                neighbors = await self._store._client._graph.get_task_neighbors(
                    seed_uri,
                    hop=hop,
                    similarity_threshold=0.1,
                )
                expanded_uris.update(neighbors)
            except Exception as exc:
                logger.debug("Task-neighbor retrieval skipped for %s: %s", seed_uri, exc)

        uri_to_analysis = await self._analysis_uri_map(session_id, all_analyses)
        expanded_analyses = [
            uri_to_analysis[uri]
            for uri in expanded_uris
            if uri in uri_to_analysis
        ]
        if not expanded_analyses:
            expanded_analyses = seed_analyses

        expanded_scored = sorted(
            [
                (analysis, self._jaccard(task_main, analysis.task_main))
                for analysis in expanded_analyses
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        successful = [
            analysis
            for analysis, _ in expanded_scored
            if analysis.status == "completed"
        ][:successful_topk]
        failed = [
            analysis
            for analysis, _ in expanded_scored
            if analysis.status == "failed"
        ][:failed_topk]

        related_uris = set()
        for analysis in expanded_analyses:
            related_uris.update(await self._memory_uris(session_id, analysis))

        all_insights: list[InsightMemory] = await self._store.get_memories(
            session_id,
            memory_type="insight",
            limit=200,
        )
        insight_scored = []
        for insight in all_insights:
            if not insight.rule:
                continue
            overlap = len(related_uris.intersection(insight.positive_task_uris))
            if overlap:
                insight_scored.append((insight, overlap))
        insight_scored.sort(key=lambda item: item[1], reverse=True)
        top_insights = [insight for insight, _ in insight_scored[:insight_topk]]

        return successful, failed, top_insights

    async def _find_related_insights(
        self,
        session_id: str,
        task_uris: list[str],
        threshold: float = 1.0,
    ) -> list[InsightMemory]:
        """Find insight rules linked to enough of the supplied task URIs."""
        if not task_uris:
            return []

        all_insights: list[InsightMemory] = await self._store.get_memories(
            session_id,
            memory_type="insight",
            limit=200,
        )
        task_uri_set = set(task_uris)
        scored = []
        for insight in all_insights:
            score = len(task_uri_set.intersection(insight.positive_task_uris))
            if score >= threshold:
                scored.append((insight, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [insight for insight, _ in scored]

    def _format_rules(self, insights: list[InsightMemory]) -> str:
        rules = [insight.rule for insight in insights if insight.rule]
        if not rules:
            return "1. (no existing rules)"
        return "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, 1))

    def _merge_rules(self, rules: list[str]) -> list[str]:
        if not rules:
            return []

        merged: list[str] = []
        for index in range(0, len(rules), 10):
            batch = rules[index : index + 10]
            user = USER_MERGE.format(
                current_rules="\n".join(batch),
                limited_number=max(1, len(batch) // 3),
            )
            response = self._llm(SYSTEM_MERGE, user)
            parsed = self._parse_numbered_list(response)
            merged.extend(parsed or batch[:1])

        return merged[: self.MAX_RULE_COUNT]

    @staticmethod
    def _parse_numbered_list(text: str) -> list[str]:
        pattern = r"\d+\.\s+(.*?)(?=\n\d+\.|\Z)"
        items = re.findall(pattern, text.strip(), flags=re.DOTALL)
        return [item.strip() for item in items if item.strip()]

    @staticmethod
    def _parse_rules(llm_text: str) -> list[tuple[str, str]]:
        pattern = re.compile(
            r"^\s*(ADD|EDIT|REMOVE|AGREE)(?:\s+(\d+))?\s*:\s*(.+?)\s*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        operations: list[tuple[str, str]] = []
        for op_type, op_index, text in pattern.findall(llm_text or ""):
            cleaned_text = text.strip()
            if not cleaned_text or not cleaned_text.endswith("."):
                continue
            operation = op_type.upper()
            if op_index:
                operation = f"{operation} {op_index}"
            operations.append((operation, cleaned_text))
        return operations

    async def _update_rules(
        self,
        session_id: str,
        relative_task_uris: list[str],
        operations: list[tuple[str, str]],
        current_insights: list[InsightMemory],
    ) -> None:
        """Apply parsed rule operations to typed insight memories."""
        for operation, rule_text in operations:
            op_parts = operation.split()
            op_type = op_parts[0]
            rule_idx = int(op_parts[1]) - 1 if len(op_parts) > 1 else None

            if op_type == "ADD":
                digest = hashlib.sha1(rule_text.encode("utf-8")).hexdigest()[:12]
                insight = InsightMemory(
                    source_analysis_id=relative_task_uris[0]
                    if relative_task_uris
                    else "auto",
                    entity_type="rule",
                    entity_id=digest,
                    biological_label=rule_text[:80],
                    evidence="Derived from related spatial analysis task trajectories.",
                    rule=rule_text,
                    score=2.0,
                    positive_task_uris=relative_task_uris,
                )
                await self._store.save_memory(session_id, insight)
                await self._link_insight_to_tasks(session_id, insight, relative_task_uris)
                continue

            if rule_idx is None or not 0 <= rule_idx < len(current_insights):
                continue

            insight = current_insights[rule_idx]
            if op_type in {"AGREE", "EDIT"}:
                updates = {
                    "score": insight.score + 1,
                    "positive_task_uris": sorted(
                        set(insight.positive_task_uris).union(relative_task_uris)
                    ),
                }
                if op_type == "EDIT":
                    updates["rule"] = rule_text
                    updates["biological_label"] = rule_text[:80]
                await self._store.update_memory(insight.memory_id, updates)
            elif op_type == "REMOVE":
                await self._store.update_memory(
                    insight.memory_id,
                    {"score": insight.score - 1},
                )

        await self._prune_insights(session_id)

    async def _prune_insights(self, session_id: str) -> None:
        """Delete insight memories whose score has fallen to zero or below."""
        all_insights: list[InsightMemory] = await self._store.get_memories(
            session_id,
            memory_type="insight",
            limit=500,
        )
        for insight in all_insights:
            if insight.score > 0:
                continue
            for uri in await self._memory_uris(session_id, insight):
                try:
                    await self._store._client.forget(uri)
                except Exception as exc:
                    logger.debug("Insight pruning skipped for %s: %s", uri, exc)

    async def _link_insight_to_tasks(
        self,
        session_id: str,
        insight: InsightMemory,
        task_uris: list[str],
    ) -> None:
        insight_uri = await self._primary_uri(session_id, insight)
        if not insight_uri:
            return
        for task_uri in task_uris:
            try:
                await self._store._client._graph.add_insight_task_edge(
                    insight_uri=insight_uri,
                    task_uri=task_uri,
                )
            except Exception as exc:
                logger.debug("Insight-task edge skipped for %s: %s", task_uri, exc)

    async def _analysis_uris(
        self,
        session_id: str,
        analyses: list[AnalysisMemory],
    ) -> list[str]:
        uris: list[str] = []
        for analysis in analyses:
            uris.extend(await self._memory_uris(session_id, analysis))
        return uris

    async def _analysis_uri_map(
        self,
        session_id: str,
        analyses: list[AnalysisMemory],
    ) -> dict[str, AnalysisMemory]:
        uri_map: dict[str, AnalysisMemory] = {}
        for analysis in analyses:
            for uri in await self._memory_uris(session_id, analysis):
                uri_map[uri] = analysis
        return uri_map

    async def _primary_uri(
        self,
        session_id: str,
        memory: BaseMemory,
    ) -> str | None:
        uris = await self._memory_uris(session_id, memory)
        return uris[0] if uris else None

    async def _memory_uris(
        self,
        session_id: str,
        memory: BaseMemory,
    ) -> list[str]:
        return await self._store.get_memory_uris(session_id, memory)

    @staticmethod
    def _jaccard(left: str, right: str) -> float:
        left_tokens = set((left or "").lower().split())
        right_tokens = set((right or "").lower().split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
