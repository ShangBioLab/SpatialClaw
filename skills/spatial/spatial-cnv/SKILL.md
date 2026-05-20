---
name: spatial-cnv
description: >-
  Copy number variation inference from spatial transcriptomics expression data
  using Python-native inferCNVpy.
version: 0.3.0
author: SPATIALCLAW Team
license: MIT
tags: [spatial, CNV, copy number, inferCNV, cancer]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "🧫"
    os: [macos, linux]
    install:
      - kind: pip
        package: scanpy
        bins: []
    trigger_keywords:
      - copy number variation
      - CNV
      - inferCNV
      - chromosomal aberration
      - cancer clone
---

# Spatial CNV

Spatial CNV infers large-scale chromosomal gains and losses from spatial transcriptomics expression patterns.

## Core Capability

1. **inferCNVpy**: Expression-based CNV inference using `adata.X` as log-normalized expression.
2. **Spatial CNV mapping**: Overlay CNV scores on spatial coordinates.
3. **Reproducible reporting**: Write `report.md`, `result.json`, processed AnnData, and figures.

## Input

| Format | Extension | Required Fields |
|--------|-----------|-----------------|
| AnnData | `.h5ad` | `X` log-normalized expression, gene positions in `var`; `obsm["spatial"]` for spatial plots |

## CLI

```bash
python skills/spatial/spatial-cnv/spatial_cnv.py \
  --input <preprocessed.h5ad> \
  --method infercnvpy \
  --reference-key cell_type \
  --reference-cat Normal \
  --output <dir>

python skills/spatial/spatial-cnv/spatial_cnv.py --demo --output <dir>
spatialclaw run spatial-cnv --input <file.h5ad> --output <dir>
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Preprocessed spatial `.h5ad` |
| `--output` | required | Output directory |
| `--demo` | off | Run built-in demo |
| `--method` | `infercnvpy` | CNV backend |
| `--reference-key` | `cell_type` | `.obs` column identifying reference categories |
| `--reference-cat` | - | One or more reference categories |
| `--window-size` | `100` | Genomic smoothing window size |
| `--step` | `10` | Genomic smoothing step |

## Method

inferCNVpy subtracts reference expression in log-space and smooths expression along ordered genomic windows. Genes must have chromosome/start/end annotations in `adata.var`.

## Output

```
output_directory/
├── report.md
├── result.json
├── processed.h5ad
└── figures/
    ├── cnv_heatmap.png
    └── cnv_spatial.png
```

## Dependencies

Required Python packages:

- `scanpy`

Optional Python package:

- `infercnvpy`

## Safety

- Local-first processing.
- Reports include SPATIALCLAW disclaimers.
- Parameters and outputs are recorded for reproducibility.

## Citations

- [inferCNVpy](https://github.com/icbi-lab/infercnvpy) — Python inferCNV for single-cell and spatial data.
- [Tirosh et al. 2016](https://doi.org/10.1126/science.aad0501) — expression-based CNV inference in tumors.
