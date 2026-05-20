---
name: spatial-niches
description: >-
  Python API skill for spatial niche and tissue microenvironment discovery,
  characterization, and cross-condition comparison.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, niches, microenvironment, neighborhood, cell-types]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_niches.py
    library_module: skills.spatial._lib.niches
    trigger_keywords:
      - spatial niche
      - microenvironment
      - neighborhood composition
      - CellCharter
      - NicheCompass
---

# Spatial Niches

Spatial Niches identifies local tissue microenvironments from spatial coordinates, expression, and cell-type annotations.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing.

## Python API

```python
from skills.spatial._lib.niches import run_niche_identification

result = run_niche_identification(adata, method="leiden_niche", cell_type_key="cell_type")
```

## Capabilities

- Leiden-based spatial niche discovery.
- Optional CellCharter, scNiche, NicheCompass, and NiCo integrations.
- Niche characterization by cell-type composition and marker programs.
- Cross-condition niche comparison.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_niches_smoke`.
