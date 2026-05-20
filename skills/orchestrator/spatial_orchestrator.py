#!/usr/bin/env python3
"""Top-level spatial orchestrator wrapper.

The canonical implementation lives under
``skills/spatial/spatial-orchestrator/spatial_orchestrator.py``.  This module
restores the top-level ``skills/orchestrator/`` entry without duplicating the
routing tables or reintroducing removed non-spatial orchestration code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_SPATIAL_ORCHESTRATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "spatial"
    / "spatial-orchestrator"
    / "spatial_orchestrator.py"
)


def _load_spatial_orchestrator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_spatialclaw_spatial_orchestrator",
        _SPATIAL_ORCHESTRATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spatial orchestrator from {_SPATIAL_ORCHESTRATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_impl = _load_spatial_orchestrator()

KEYWORD_MAP = _impl.KEYWORD_MAP
EXTENSION_MAP = _impl.EXTENSION_MAP
NAMED_PIPELINES = _impl.NAMED_PIPELINES
SKILL_DESCRIPTIONS = _impl.SKILL_DESCRIPTIONS

route_query = _impl.route_query
route_file = _impl.route_file
list_skills = _impl.list_skills
run_pipeline = _impl.run_pipeline
write_routing_report = _impl.write_routing_report
write_pipeline_report = _impl.write_pipeline_report
main = _impl.main


if __name__ == "__main__":
    main()
