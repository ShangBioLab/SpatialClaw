---
name: spatial-multi-sample-integration
description: >-
  Multi-sample spatial transcriptomics integration and batch correction with
  automatic method selection (STAligner, Harmony, BBKNN, Scanorama) and robust fallback.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, integration, batch-correction, staligner, harmony, bbknn, scanorama]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "🧬"
    os: [macos, linux]
    install:
      - kind: pip
        package: scanpy
        bins: []
    trigger_keywords:
      - multi sample integration
      - batch correction
      - integrate samples
      - staligner
      - harmony
      - bbknn
      - scanorama
---

# 🧬 Spatial Multi-sample Integration

You are **Spatial Multi-sample Integration**, a specialised SPATIALCLAW agent for integrating multiple spatial samples and reducing batch effects while preserving biological structure.

## Why This Exists

- **Without it**: Multi-sample embeddings are dominated by technical variation and sample-specific drift.
- **With it**: A unified command performs integration, recomputes neighbourhoods/UMAP, and reports batch-mixing quality.
- **Why SPATIALCLAW**: Consistent outputs (report, result JSON, processed h5ad, figures, reproducibility bundle) that match the rest of the spatial skills.

## Core Capabilities

1. **Auto method routing**: Select the first available method among STAligner, Harmony, BBKNN, and Scanorama.
2. **Explicit method mode**: Force `staligner`, `harmony`, `bbknn`, or `scanorama` for reproducible runs.
3. **Robust fallback**: If optional integration dependencies are missing, falls back to PCA baseline and still produces outputs.
4. **Quality metrics**: Computes neighbour-graph batch mixing entropy before and after integration.
5. **Visual diagnostics**: Writes UMAP plots colored by batch and cluster.

## Input Formats

| Format | Extension | Required Fields | Example |
|--------|-----------|-----------------|---------|
| AnnData | `.h5ad` | `X`, `obs[batch_key]`, preferably `obsm["X_pca"]` | `merged_samples.h5ad` |
| Demo | n/a | `--demo` flag | Built from `spatial-preprocessing --demo` |

## CLI Reference

```bash
# Auto-select available integration method (recommended)
python skills/spatial/spatial-multi-sample-integration/spatial_multi_sample_integration.py \
  --input <merged.h5ad> --output <dir> --batch-key sample --method auto

# Force a method
python skills/spatial/spatial-multi-sample-integration/spatial_multi_sample_integration.py \
  --input <merged.h5ad> --output <dir> --batch-key sample --method staligner

# Force another method
python skills/spatial/spatial-multi-sample-integration/spatial_multi_sample_integration.py \
  --input <merged.h5ad> --output <dir> --batch-key sample --method harmony

# Demo
python skills/spatial/spatial-multi-sample-integration/spatial_multi_sample_integration.py \
  --demo --output /tmp/spatial_multi_int_demo
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | (required unless `--demo`) | Multi-sample AnnData file |
| `--output` | `results/spatial_multi_sample_integration` | Output report directory |
| `--demo` | off | Run the built-in multi-sample integration demo |
| `--batch-key` | `batch` | Column in `adata.obs` identifying batches |
| `--method` | `auto` | `auto`, `staligner`, `harmony`, `bbknn`, or `scanorama` |

## Outputs

```
output_dir/
├── report.md
├── result.json
├── processed.h5ad
├── figures/
│   ├── umap_by_batch.png
│   ├── umap_by_cluster.png
│   └── batch_mixing.png
├── tables/
│   └── integration_metrics.csv
└── reproducibility/
    ├── commands.sh
    └── environment.txt
```
