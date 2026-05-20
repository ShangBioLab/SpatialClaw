---
name: spatial-morphology
description: >-
  Python API skill for morphology feature extraction and image-expression
  feature fusion in spatial analysis workflows.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, morphology, histology, image-features, fusion]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_morphology.py
    library_module: skills.spatial._lib.morphology
    trigger_keywords:
      - morphology
      - image features
      - histology features
      - image omics fusion
---

# Spatial Morphology

Spatial Morphology extracts handcrafted or deep image features and links them to spatial expression matrices.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing.

## Python API

```python
from skills.spatial._lib.morphology import run_morphology

result = run_morphology(adata, images=None, method="deep_features")
```

## Capabilities

- Handcrafted image features for color, texture, shape, and intensity.
- Deep feature extraction when optional torch/vision dependencies are available.
- Image-expression feature fusion for downstream spatial modeling.
- IHC/IF quantification helpers.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_morphology_smoke`.
