"""Tests for the spatial-visualization skill."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SKILL_SCRIPT = Path(__file__).resolve().parent.parent / "spatial_visualization.py"


def test_demo_mode(tmp_path):
    """spatial-visualization --demo should run and create outputs."""
    output_dir = tmp_path / "viz_out"
    result = subprocess.run(
        [sys.executable, str(SKILL_SCRIPT), "--demo", "--output", str(output_dir)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(SKILL_SCRIPT.parent),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (output_dir / "report.md").exists()
    assert (output_dir / "result.json").exists()
    assert (output_dir / "figures").exists()
    assert any(output_dir.joinpath("figures").glob("*.png"))