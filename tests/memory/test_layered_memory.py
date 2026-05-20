"""Tests for minimal layered memory behavior in LayeredMemoryStore."""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from spatialclaw.memory.layered_store import (
    AnalysisMemory,
    LayeredMemoryStore,
    DatasetMemory,
    InsightMemory,
    PreferenceMemory,
    ProjectContextMemory,
)


@pytest_asyncio.fixture
async def store():
    """Create a LayeredMemoryStore backed by a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    db_url = f"sqlite+aiosqlite:///{db_path}"
    s = LayeredMemoryStore(db_url)
    await s.initialize()

    yield s

    await s.close()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_dataset_stays_episodic_when_metadata_is_weak(store):
    session = await store.create_session("user1", "cli")

    memory = DatasetMemory(file_path="data/raw.h5ad")
    await store.save_memory(session.session_id, memory)

    episodic = await store.get_memories(
        session.session_id,
        "dataset",
        layer="episodic",
    )
    semantic = await store.get_memories(
        session.session_id,
        "dataset",
        layer="semantic",
    )
    all_layers = await store.get_memories(session.session_id, "dataset")

    assert len(episodic) == 1
    assert episodic[0].memory_id == memory.memory_id
    assert episodic[0].memory_layer == "episodic"
    assert episodic[0].source_session_id == session.session_id
    assert semantic == []
    assert len(all_layers) == 1


@pytest.mark.asyncio
async def test_typed_memory_uris_resolve_existing_layers(store):
    session = await store.create_session("user1b", "cli")

    analysis = AnalysisMemory(
        source_dataset_id="dataset-1",
        skill="spatial-domain-identification",
        method="leiden",
        output_path="output/domain-run",
        status="completed",
    )
    await store.save_memory(session.session_id, analysis)

    all_uris = await store.get_memory_uris(session.session_id, analysis)
    short_uris = await store.get_memory_uris(
        session.session_id,
        analysis,
        layer="episodic",
    )
    long_uris = await store.get_memory_uris(
        session.session_id,
        analysis,
        layer="semantic",
    )

    assert len(all_uris) == 2
    assert short_uris == [
        f"episodic://{session.session_id}/analysis/{analysis.skill}/{analysis.memory_id}"
    ]
    assert long_uris == [
        f"semantic://{session.session_id}/analysis/{analysis.skill}/{analysis.memory_id}"
    ]


@pytest.mark.asyncio
async def test_completed_analysis_is_promoted_to_semantic(store):
    session = await store.create_session("user2", "cli")

    memory = AnalysisMemory(
        source_dataset_id="dataset-1",
        skill="spatial-domain-identification",
        method="leiden",
        output_path="output/run-1",
        status="completed",
    )
    await store.save_memory(session.session_id, memory)

    episodic = await store.get_memories(
        session.session_id,
        "analysis",
        layer="episodic",
    )
    semantic = await store.get_memories(
        session.session_id,
        "analysis",
        layer="semantic",
    )
    all_layers = await store.get_memories(session.session_id, "analysis")

    assert len(episodic) == 1
    assert len(semantic) == 1
    assert len(all_layers) == 1
    assert episodic[0].memory_id == memory.memory_id
    assert semantic[0].memory_id == memory.memory_id
    assert semantic[0].memory_layer == "semantic"
    assert semantic[0].promoted_at is not None
    assert semantic[0].importance is not None


@pytest.mark.asyncio
async def test_insight_can_transition_from_episodic_to_semantic_after_update(store):
    session = await store.create_session("user2b", "cli")

    memory = InsightMemory(
        source_analysis_id="analysis-1",
        entity_type="cluster",
        entity_id="5",
        biological_label="tentative label",
        confidence="ai_predicted",
    )
    await store.save_memory(session.session_id, memory)

    semantic_before = await store.get_memories(
        session.session_id,
        "insight",
        layer="semantic",
    )
    assert semantic_before == []

    await store.update_memory(
        memory.memory_id,
        {
            "confidence": "user_confirmed",
            "evidence": "validated by marker genes",
            "biological_label": "validated niche",
        },
    )

    episodic_after = await store.get_memories(
        session.session_id,
        "insight",
        layer="episodic",
    )
    semantic_after = await store.get_memories(
        session.session_id,
        "insight",
        layer="semantic",
    )

    assert len(episodic_after) == 1
    assert len(semantic_after) == 1
    assert episodic_after[0].memory_id == memory.memory_id
    assert semantic_after[0].memory_id == memory.memory_id
    assert episodic_after[0].memory_layer == "episodic"
    assert episodic_after[0].importance == 0.9
    assert episodic_after[0].confidence_score == 0.95
    assert semantic_after[0].memory_layer == "semantic"
    assert semantic_after[0].confidence == "user_confirmed"
    assert semantic_after[0].evidence == "validated by marker genes"
    assert semantic_after[0].biological_label == "validated niche"
    assert semantic_after[0].importance == 0.9
    assert semantic_after[0].confidence_score == 0.95


@pytest.mark.asyncio
async def test_insight_downgrade_removes_stale_semantic_layer(store):
    session = await store.create_session("user2c", "cli")

    memory = InsightMemory(
        source_analysis_id="analysis-2",
        entity_type="cluster",
        entity_id="7",
        biological_label="confirmed label",
        confidence="user_confirmed",
    )
    await store.save_memory(session.session_id, memory)

    semantic_before = await store.get_memories(
        session.session_id,
        "insight",
        layer="semantic",
    )
    assert len(semantic_before) == 1

    await store.update_memory(
        memory.memory_id,
        {
            "confidence": "ai_predicted",
            "biological_label": "tentative revision",
            "evidence": "",
        },
    )

    episodic_after = await store.get_memories(
        session.session_id,
        "insight",
        layer="episodic",
    )
    semantic_after = await store.get_memories(
        session.session_id,
        "insight",
        layer="semantic",
    )

    assert len(episodic_after) == 1
    assert semantic_after == []
    assert episodic_after[0].memory_id == memory.memory_id
    assert episodic_after[0].confidence == "ai_predicted"
    assert episodic_after[0].biological_label == "tentative revision"
    assert episodic_after[0].importance == 0.35
    assert episodic_after[0].confidence_score == 0.45


@pytest.mark.asyncio
async def test_preference_is_semantic_only_and_updates_session(store):
    session = await store.create_session("user3", "cli")

    memory = PreferenceMemory(
        domain="spatial",
        key="cluster_method",
        value="leiden",
    )
    await store.save_memory(session.session_id, memory)

    episodic = await store.get_memories(
        session.session_id,
        "preference",
        layer="episodic",
    )
    semantic = await store.get_memories(
        session.session_id,
        "preference",
        layer="semantic",
    )
    loaded_session = await store.get_session(session.session_id)

    assert episodic == []
    assert len(semantic) == 1
    assert semantic[0].memory_layer == "semantic"
    assert loaded_session is not None
    assert loaded_session.preferences["cluster_method"] == "leiden"


@pytest.mark.asyncio
async def test_load_context_prefers_semantic_but_keeps_current_episodic(store):
    session = await store.create_session("user4", "cli")

    dataset = DatasetMemory(
        file_path="data/visium.h5ad",
        platform="Visium",
        n_obs=5000,
    )
    await store.save_memory(session.session_id, dataset)

    analysis = AnalysisMemory(
        source_dataset_id=dataset.memory_id,
        skill="spatial-preprocessing",
        method="scanpy",
        output_path="output/preprocess",
        status="completed",
    )
    await store.save_memory(session.session_id, analysis)

    preference = PreferenceMemory(
        domain="spatial",
        key="cluster_method",
        value="leiden",
    )
    await store.save_memory(session.session_id, preference)

    project = ProjectContextMemory(
        project_goal="Map the tumor microenvironment",
        species="human",
        tissue_type="breast",
    )
    await store.save_memory(session.session_id, project)

    insight = InsightMemory(
        source_analysis_id=analysis.memory_id,
        entity_type="cluster",
        entity_id="3",
        biological_label="immune niche",
        evidence="marker genes support immune infiltration",
        confidence="user_confirmed",
    )
    await store.save_memory(session.session_id, insight)

    context = await store.load_context(session.session_id)

    assert "**Project Context**" in context
    assert "Map the tumor microenvironment" in context
    assert "**Current Dataset**" in context
    assert "data/visium.h5ad" in context
    assert "**Recent Analyses**" in context
    assert "spatial-preprocessing (scanpy) - completed" in context
    assert "output/preprocess" in context
    assert "**User Preferences**" in context
    assert "cluster_method: leiden" in context
    assert "**Known Insights**" in context
    assert "cluster 3: immune niche (confirmed)" in context
    assert "evidence: marker genes support immune infiltration" in context


@pytest.mark.asyncio
async def test_load_context_includes_insight_evidence(store):
    session = await store.create_session("user4a", "cli")
    await store.save_memory(
        session.session_id,
        InsightMemory(
            source_analysis_id="",
            entity_type="analysis_output",
            entity_id="deepst_result_files",
            biological_label=(
                "spatial_domain_identification.png; "
                "umap_spatial_comparison.png; "
                "domain_sizes.png; "
                "workflow_summary.png"
            ),
            confidence="user_confirmed",
            evidence="2026-05-18 real GPU DeepST output files",
        ),
    )

    context = await store.load_context(session.session_id)

    assert "analysis_output deepst_result_files" in context
    assert "spatial_domain_identification.png" in context
    assert "umap_spatial_comparison.png" in context
    assert "domain_sizes.png" in context
    assert "workflow_summary.png" in context
    assert "evidence: 2026-05-18 real GPU DeepST output files" in context


@pytest.mark.asyncio
async def test_load_context_only_renders_prompt_safe_preferences(store):
    session = await store.create_session("user4b", "cli")

    await store.save_memory(
        session.session_id,
        PreferenceMemory(
            domain="global",
            key="response_style",
            value="concise",
        ),
    )

    context = await store.load_context(session.session_id)

    assert "**User Preferences**" in context
    assert "response_style: concise" in context
    assert "output/report.md" not in context


@pytest.mark.asyncio
async def test_session_isolation_is_enforced_by_layered_paths(store):
    session_a = await store.create_session("userA", "cli")
    session_b = await store.create_session("userB", "cli")

    await store.save_memory(
        session_a.session_id,
        DatasetMemory(file_path="data/a.h5ad", platform="Visium"),
    )
    await store.save_memory(
        session_b.session_id,
        DatasetMemory(file_path="data/b.h5ad", platform="MERFISH"),
    )

    memories_a = await store.get_memories(session_a.session_id, "dataset")
    memories_b = await store.get_memories(session_b.session_id, "dataset")

    assert [memory.file_path for memory in memories_a] == ["data/a.h5ad"]
    assert [memory.file_path for memory in memories_b] == ["data/b.h5ad"]


@pytest.mark.asyncio
async def test_delete_session_removes_layered_memories(store):
    session = await store.create_session("user5", "cli")

    await store.save_memory(
        session.session_id,
        DatasetMemory(file_path="data/remove_me.h5ad", platform="Visium"),
    )
    await store.save_memory(
        session.session_id,
        PreferenceMemory(domain="spatial", key="method", value="leiden"),
    )

    await store.delete_session(session.session_id)

    assert await store.get_session(session.session_id) is None
    assert await store.get_memories(session.session_id, "dataset") == []
    assert await store.get_memories(session.session_id, "preference") == []
    assert await store.search_memories(
        session.session_id,
        "leiden",
        memory_type="preference",
        layer="semantic",
    ) == []


def test_default_database_url_uses_project_config_dir(monkeypatch, tmp_path):
    import spatialclaw.memory.database as database

    project_root = tmp_path / "project"
    other_cwd = tmp_path / "other"
    project_root.mkdir()
    other_cwd.mkdir()
    monkeypatch.setattr(database, "_PROJECT_ROOT", project_root)
    monkeypatch.chdir(other_cwd)

    expected = project_root / ".config" / "spatialclaw" / "memory.db"
    assert database._get_default_db_url() == f"sqlite+aiosqlite:///{expected}"


def test_configured_database_url_overrides_project_default(monkeypatch, tmp_path):
    import spatialclaw.memory.database as database

    configured = f"sqlite+aiosqlite:///{tmp_path / 'custom.db'}"
    monkeypatch.setenv("SPATIALCLAW_MEMORY_DB_URL", configured)

    assert database._get_database_url() == configured


@pytest.mark.asyncio
async def test_default_legacy_memory_db_is_archived_and_recreated(monkeypatch, tmp_path):
    import spatialclaw.memory.database as database

    monkeypatch.setattr(database, "_PROJECT_ROOT", tmp_path)
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    db_path = tmp_path / ".config" / "spatialclaw" / "memory.db"
    db_path.parent.mkdir(parents=True)

    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY,
            node_uuid TEXT,
            content TEXT NOT NULL,
            is_retired BOOLEAN DEFAULT 0,
            superseded_by_id INTEGER,
            created_at DATETIME
        )
        """
    )
    connection.execute("CREATE TABLE paths (domain TEXT, path TEXT)")
    connection.execute("INSERT INTO paths(domain, path) VALUES ('short_term', 'legacy')")
    connection.commit()
    connection.close()

    db = database.DatabaseManager()
    await db.init_db()
    await db.close()

    backups = list((tmp_path / ".run" / "backups").glob("memory-legacy-*.db"))
    assert len(backups) == 1

    rebuilt = sqlite3.connect(db_path)
    columns = {row[1] for row in rebuilt.execute("PRAGMA table_info(memories)")}
    rebuilt.close()
    assert {"deprecated", "migrated_to"} <= columns


@pytest.mark.asyncio
async def test_insight_extractor_uses_layered_task_uris(store):
    from spatialclaw.memory.insight_extractor import InsightExtractor

    session = await store.create_session("user6", "cli")
    success = AnalysisMemory(
        source_dataset_id="dataset-6",
        skill="spatial-domain-identification",
        method="leiden",
        output_path="output/success",
        status="completed",
        task_main="leiden domain clustering",
        task_trajectory="QC completed and Leiden found stable domains.",
        key_steps="QC, normalize, Leiden clustering",
        cluster_id=2,
    )
    failure = AnalysisMemory(
        source_dataset_id="dataset-6",
        skill="spatial-domain-identification",
        method="leiden",
        output_path="output/failure",
        status="failed",
        task_main="leiden domain clustering failed",
        task_trajectory="Run failed before neighbor graph construction.",
        fail_reason="Missing spatial neighbors.",
        cluster_id=2,
    )
    await store.save_memory(session.session_id, success)
    await store.save_memory(session.session_id, failure)

    def fake_llm(system: str, user: str) -> str:
        return "ADD: Build spatial neighbors before Leiden domain clustering."

    extractor = InsightExtractor(store=store, llm_callable=fake_llm)
    await extractor.refine_insights(session.session_id, num_points=1)

    insights = await store.get_memories(session.session_id, "insight")
    assert any("spatial neighbors" in insight.rule for insight in insights)

    rule = next(insight for insight in insights if "spatial neighbors" in insight.rule)
    assert any(uri.startswith("semantic://") for uri in rule.positive_task_uris)
    assert any(uri.startswith("episodic://") for uri in rule.positive_task_uris)

    successful, failed, retrieved_insights = await extractor.retrieve_for_task(
        session.session_id,
        "leiden domain clustering",
        successful_topk=1,
        failed_topk=1,
        insight_topk=3,
    )

    assert [item.memory_id for item in successful] == [success.memory_id]
    assert [item.memory_id for item in failed] == [failure.memory_id]
    assert any(rule.memory_id == item.memory_id for item in retrieved_insights)
