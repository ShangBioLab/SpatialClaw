"""Shared filesystem locations for the SpatialClaw repository."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

ROOT_CLI = PROJECT_ROOT / "spatialclaw.py"
SKILLS_DIR = PROJECT_ROOT / "skills"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SESSIONS_DIR = PROJECT_ROOT / "sessions"
SOUL_MD = PROJECT_ROOT / "SOUL.md"
