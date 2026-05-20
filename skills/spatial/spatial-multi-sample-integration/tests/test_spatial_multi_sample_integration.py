"""Tests for spatial-multi-sample-integration skill."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SKILL_SCRIPT = Path(__file__).resolve().parent.parent / "spatial_multi_sample_integration.py"


def test_demo_mode(tmp_path):
    """Demo mode should run and produce standard outputs."""
    output_dir = tmp_path / "multi_int_out"
    result = subprocess.run(
        [sys.executable, str(SKILL_SCRIPT), "--demo", "--output", str(output_dir)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(SKILL_SCRIPT.parent),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (output_dir / "report.md").exists()
    assert (output_dir / "result.json").exists()
    assert (output_dir / "processed.h5ad").exists()
    assert (output_dir / "tables" / "integration_metrics.csv").exists()


def test_result_json_keys(tmp_path):
    """result.json should include integration summary keys."""
    output_dir = tmp_path / "multi_int_out_json"
    result = subprocess.run(
        [sys.executable, str(SKILL_SCRIPT), "--demo", "--output", str(output_dir)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(SKILL_SCRIPT.parent),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    data = json.loads((output_dir / "result.json").read_text())
    assert data["skill"] == "spatial-multi-sample-integration"
    assert data["summary"]["n_batches"] >= 2
    assert "batch_mixing_before" in data["summary"]
    assert "batch_mixing_after" in data["summary"]