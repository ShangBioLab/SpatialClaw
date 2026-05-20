---
name: spatial-histology
description: >-
  Python API skill for histology image analysis linked to spatial
  transcriptomics workflows, including tissue segmentation and nuclei/cell
  detection.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, histology, image-analysis, segmentation, pathology]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_histology.py
    library_module: skills.spatial._lib.histology
    trigger_keywords:
      - histology
      - tissue segmentation
      - nucleus detection
      - cell segmentation
      - pathology image
---

# Spatial Histology

Spatial Histology provides downstream image analysis for spatial transcriptomics and spatial pathology datasets. It wraps reusable functions from `skills.spatial._lib.histology`.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing.

## Python API

```python
from skills.spatial._lib.histology import run_histology

result = run_histology(image, task="tissue_segmentation", method="otsu")
```

## Capabilities

- Tissue segmentation with Otsu or watershed methods.
- Nucleus detection with optional Cellpose or StarDist backends.
- Cell segmentation with optional Cellpose-compatible backends.
- Patch-level histology analysis for downstream spatial workflows.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_histology_smoke`.
