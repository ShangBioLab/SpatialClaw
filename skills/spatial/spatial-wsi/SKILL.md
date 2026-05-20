---
name: spatial-wsi
description: >-
  Python API skill for whole-slide image loading, tiling, feature extraction,
  and optional multiple-instance learning inference.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, wsi, whole-slide-image, pathology, mil]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_wsi.py
    library_module: skills.spatial._lib.wsi
    trigger_keywords:
      - whole slide image
      - WSI
      - tiling
      - patch extraction
      - MIL
---

# Spatial WSI

Spatial WSI handles whole-slide image ingestion and patch-level processing for spatial pathology workflows.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing. Optional MIL outputs are research-use signals only and are not diagnostic predictions.

## Python API

```python
from skills.spatial._lib.wsi import run_wsi_pipeline

result = run_wsi_pipeline("slide.svs", patch_size=256)
```

## Capabilities

- WSI metadata loading through optional OpenSlide or TIFF backends.
- Tissue region detection.
- Patch extraction and feature extraction.
- Optional MIL inference on patch features.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_wsi_smoke`.
