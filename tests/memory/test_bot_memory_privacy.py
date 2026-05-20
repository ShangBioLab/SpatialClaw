from pathlib import Path

import bot.core as bot_core


def test_memory_safe_path_keeps_project_relative(monkeypatch, tmp_path):
    project_root = tmp_path / "SpatialClaw"
    project_root.mkdir()
    dataset = project_root / "data" / "brain.h5ad"
    dataset.parent.mkdir()
    dataset.touch()
    monkeypatch.setattr(bot_core, "PROJECT_ROOT", project_root)

    assert bot_core._memory_safe_path(dataset) == "data/brain.h5ad"


def test_memory_safe_path_redacts_external_absolute(monkeypatch, tmp_path):
    project_root = tmp_path / "SpatialClaw"
    external = tmp_path / "private" / "patient_a" / "brain.h5ad"
    project_root.mkdir()
    external.parent.mkdir(parents=True)
    external.touch()
    monkeypatch.setattr(bot_core, "PROJECT_ROOT", project_root)

    assert bot_core._memory_safe_path(external) == "brain.h5ad"
    assert str(external.parent) not in bot_core._memory_safe_path(external)


def test_memory_safe_path_preserves_relative_display(monkeypatch, tmp_path):
    monkeypatch.setattr(bot_core, "PROJECT_ROOT", tmp_path)

    assert bot_core._memory_safe_path(Path("output/run-1")) == "output/run-1"
