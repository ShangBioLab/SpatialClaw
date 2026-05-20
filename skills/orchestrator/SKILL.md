---
name: spatial-orchestrator-top-level
description: >-
  Top-level spatial orchestration entry that delegates to the canonical
  spatial-orchestrator skill without restoring removed non-spatial skill trees.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, routing, orchestration]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    os: [macos, linux]
    trigger_keywords:
      - orchestrate
      - route
      - pipeline
---

# Top-Level Spatial Orchestrator

This directory is the restored top-level orchestration entry for the current
spatial-only SpatialClaw architecture. It delegates all routing and pipeline
execution to `skills/spatial/spatial-orchestrator/`.

It intentionally does not contain or advertise deleted non-spatial omics skills.

## CLI Reference

```bash
python skills/orchestrator/spatial_orchestrator.py --list-skills
python skills/orchestrator/spatial_orchestrator.py --demo --output <dir>
python skills/orchestrator/spatial_orchestrator.py \
  --query "find spatially variable genes" \
  --output <dir>
```

The canonical registered skill remains:

```bash
python spatialclaw.py run spatial-orchestrator --demo --output <dir>
```
