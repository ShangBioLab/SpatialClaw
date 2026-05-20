"""Tests for the spatial-deconvolution skill."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_SCRIPT = Path(__file__).resolve().parent.parent / "spatial_deconvolution.py"


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "deconvolution_out"


def test_demo_mode_is_not_supported(tmp_output):
    """spatial-deconvolution does not expose a demo mode without reference data."""
    result = subprocess.run(
        [sys.executable, str(SKILL_SCRIPT), "--demo", "--output", str(tmp_output)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(SKILL_SCRIPT.parent),
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --demo" in (result.stderr + result.stdout)


@pytest.mark.skip(reason="Demo mode requires real reference data")
def test_demo_report_content(tmp_output):
    pass

@pytest.mark.skip(reason="Demo mode requires real reference data")
def test_demo_result_json(tmp_output):
    pass
