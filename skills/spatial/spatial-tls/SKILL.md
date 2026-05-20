---
name: spatial-tls
description: >-
  Python API skill for tertiary lymphoid structure detection, maturity typing,
  ecosystem analysis, and exploratory outcome association.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, tls, immunology, tumor-microenvironment, lymphoid-structure]
metadata:
  SPATIALCLAW:
    domain: spatial
    interface: python-api
    entrypoint: spatial_tls.py
    library_module: skills.spatial._lib.tls
    trigger_keywords:
      - tertiary lymphoid structure
      - TLS
      - immune niche
      - B cell
      - T cell
---

# Spatial TLS

Spatial TLS detects and characterizes tertiary lymphoid structures in spatial omics data.

## Interface

Python API only. This skill is intentionally not registered for `spatialclaw run` or other CLI execution routing.

## Research-Use Boundary

Outputs are exploratory and research-use only. TLS labels, maturity scores, and outcome associations are cohort-level analytical summaries, not diagnostic findings, prognostic predictions, or treatment recommendations.

## Python API

```python
from skills.spatial._lib.tls import run_tls

result = run_tls(adata, method="tls_detection", cell_type_key="cell_type")
```

## Capabilities

- TLS region detection from B/T-cell co-localization.
- TLS maturity classification.
- TLS ecosystem composition summaries.
- Optional exploratory association with outcome metadata.

## Validation

Covered by `tests/spatial/test_library_only_skills.py::test_spatial_tls_smoke`.
