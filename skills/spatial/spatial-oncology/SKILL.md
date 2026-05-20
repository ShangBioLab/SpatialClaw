---
name: spatial-oncology
description: >-
  Python API skill for oncology-focused spatial downstream analysis including
  tumor ecosystem mapping and exploratory response-associated pattern summaries.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, oncology, tumor, cancer, research-use]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_oncology.py
    library_module: skills.spatial._lib.oncology
    trigger_keywords:
      - tumor ecosystem
      - tumor edge
      - response-associated patterns
      - exploratory outcome association
      - cancer spatial analysis
---

# Spatial Oncology

Spatial Oncology maps cancer-related spatial ecosystems and summarizes tumor, immune, stromal, and outcome-associated research patterns.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing.

## Research-Use Boundary

Outputs are exploratory and research-use only. They do not diagnose disease, predict individual patient outcomes, recommend treatment, or provide clinical decision support. Placeholder or fallback outputs must not be interpreted as validated clinical predictions.

## Python API

```python
from skills.spatial._lib.oncology import run_oncology

result = run_oncology(adata, method="tumor_ecosystem", cell_type_key="cell_type")
```

## Capabilities

- Tumor core, edge, and interface mapping.
- Supervised niche discovery by condition.
- Exploratory cohort outcome association helpers.
- Response-associated spatial pattern summaries.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_oncology_smoke`.
