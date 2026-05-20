---
name: spatial-orchestrator
description: >-
  Route natural language requests and file inputs to registered SpatialClaw
  skills, and run named spatial analysis pipelines.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, routing, orchestration, pipeline]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "🧭"
    os: [macos, linux]
    trigger_keywords:
      - route
      - orchestrate
      - pipeline
      - recommend skill
---

# Spatial Orchestrator

Spatial Orchestrator routes user intent to the current registered spatial skill
set. It never advertises unregistered implementation modules as runnable skills.

## CLI Reference

```bash
python spatialclaw.py run spatial-orchestrator \
  --query "find spatially variable genes" \
  --output <dir>

python spatialclaw.py run spatial-orchestrator \
  --pipeline standard \
  --input <preprocessed.h5ad> \
  --output <dir>

python skills/spatial/spatial-orchestrator/spatial_orchestrator.py --list-skills
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Optional spatial `.h5ad` input |
| `--output` | required | Output directory |
| `--demo` | off | Run demo routing/pipeline |
| `--query` | - | Natural language spatial analysis request |
| `--pipeline` | - | Named pipeline to run |
| `--list-skills` | off | Print registered spatial skills |
| `--timeout` | `600` | Per-step timeout in seconds |

## Named Pipelines

- `standard`: preprocessing, domain identification, DE, SVG detection, statistics
- `full`: standard pipeline plus cell communication and enrichment
- `integration`: multi-sample integration followed by domains and DE
- `spatial_only`: preprocessing, SVG detection, statistics
- `cancer`: preprocessing, CNV, DE, enrichment

## Safety

- Routes only to canonical skill names returned by `python spatialclaw.py list`.
- Executes locally through `spatialclaw.py run`.
- Writes `report.md`, `result.json`, and reproducibility commands.
