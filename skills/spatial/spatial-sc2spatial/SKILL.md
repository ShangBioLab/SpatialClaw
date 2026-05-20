---
name: spatial-sc2spatial
description: >-
  Python API skill for mapping reference single-cell annotations and
  expression programs onto spatial transcriptomics data.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, reference-mapping, label-transfer, tangram, imputation]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_sc2spatial.py
    library_module: skills.spatial._lib.sc2spatial
    trigger_keywords:
      - label transfer
      - reference mapping
      - Tangram
      - SpaGE
      - CellTrek
---

# Spatial Reference Mapping

Spatial Reference Mapping transfers information from reference single-cell data to spatial transcriptomics datasets.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing.

## Python API

```python
from skills.spatial._lib.sc2spatial import run_sc2spatial_mapping

result = run_sc2spatial_mapping(adata_sp, adata_ref, method="label_transfer")
```

## Capabilities

- KNN label transfer from reference cell annotations.
- Optional Tangram-style mapping.
- Optional SpaGE/gimVI expression imputation and embedding helpers.
- Optional CellTrek-style cell localization.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_sc2spatial_smoke`.
