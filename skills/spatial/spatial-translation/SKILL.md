---
name: spatial-translation
description: >-
  Python API skill for cross-modal spatial translation, including
  histology-to-expression prediction and super-resolution helpers.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, cross-modal, histology, expression-prediction, super-resolution]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_translation.py
    library_module: skills.spatial._lib.translation
    trigger_keywords:
      - histology to expression
      - cross-modal translation
      - super resolution
      - image to transcriptomics
---

# Spatial Translation

Spatial Translation supports cross-modal prediction and virtual spatial modality generation.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing. Placeholder or synthetic predictions are for method prototyping and must not be interpreted as validated biological measurements.

## Python API

```python
from skills.spatial._lib.translation import run_translation

result = run_translation(adata, images=None, method="hist2st")
```

## Capabilities

- Histology-to-expression prediction helpers.
- Histology-to-protein prediction helpers.
- Super-resolution reconstruction helpers.
- Cross-slice interpolation helpers.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_translation_smoke`.
