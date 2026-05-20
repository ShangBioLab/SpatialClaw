---
name: spatial-targets
description: >-
  Python API skill for exploratory spatial biomarker discovery, target
  prioritization, drug annotation, and research-use evidence summaries.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, biomarkers, targets, drug-matching, translational]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_targets.py
    library_module: skills.spatial._lib.targets
    trigger_keywords:
      - spatial biomarker
      - target discovery
      - drug annotation
      - research target prioritization
---

# Spatial Targets

Spatial Targets summarizes spatial analysis results into exploratory biomarkers, candidate research targets, and drug annotation summaries.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing.

## Research-Use Boundary

Outputs are exploratory and research-use only. They do not recommend therapies, determine clinical actionability, diagnose disease, or provide clinical decision support. In-code drug annotations are static research references, not treatment guidance.

## Python API

```python
from skills.spatial._lib.targets import run_targets

result = run_targets(adata, method="biomarker_discovery", domain_key="spatial_domain")
```

## Capabilities

- Spatial biomarker discovery from domains or niches.
- Research target prioritization from expression and spatial specificity.
- Drug annotation against curated in-code target-drug mappings.
- Research-use evidence summary generation.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_targets_smoke`.
