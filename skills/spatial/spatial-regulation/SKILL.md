---
name: spatial-regulation
description: >-
  Python API skill for spatial transcription-factor activity and regulatory
  network inference.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, regulation, transcription-factor, grn, tf-activity]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_regulation.py
    library_module: skills.spatial._lib.regulation
    trigger_keywords:
      - regulatory network
      - transcription factor
      - TF activity
      - SpaGRN
      - SCENIC
---

# Spatial Regulation

Spatial Regulation infers transcription-factor activity and regulatory programs in spatial context.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing.

## Python API

```python
from skills.spatial._lib.regulation import run_regulation

result = run_regulation(adata, method="tf_activity", species="human")
```

## Capabilities

- TF activity scoring from expression matrices.
- Optional SpaGRN and SCENIC-style regulatory workflows.
- Optional paired RNA/ATAC regulatory inference helpers.
- Spatial regulatory pattern summaries.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_regulation_smoke`.
