---
name: spatial-deconvolution
description: >-
  Cell-type deconvolution for spatial transcriptomics by mapping a
  single-cell RNA-seq reference onto spatial data.
version: 0.3.0
author: SPATIALCLAW
license: MIT
tags: [spatial, deconvolution, cell-type, tangram, stereoscope, graphst]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      python: ">=3.10"
    entrypoint: spatial_deconvolution.py
    inputs:
      - h5ad
    outputs:
      - processed.h5ad
      - result.json
      - report.md
---

# Spatial Transcriptomics Deconvolution

Cell-type deconvolution for spatial transcriptomics: infer the proportion of each cell type per spot by mapping a single-cell RNA-seq reference onto the spatial data.

Supported methods: **Tangram** (default), **Stereoscope**, **GraphST-guided**.

---

## Why This Exists

- **The Problem:** Spatial platforms (Visium, Slide-seq…) capture multiple cells per spot, losing single-cell resolution.
- **The Solution:** Align a high-resolution scRNA-seq reference to spatial data to estimate cell-type proportions per spot.
- **Why SPATIALCLAW:** Wraps three complementary algorithms with a unified `--method` interface, handles gene-ID matching, and stores results under both method-specific and unified obsm keys.

---

## Methods Overview

| Method | Algorithm | Best For | Key Dependency |
|---|---|---|---|
| `tangram` | Deep-learning cell→space mapping (clusters mode) | General purpose, GPU-friendly | `tangram-sc` |
| `stereoscope` | Negative-Binomial probabilistic model | Count-data accuracy, multiple cell types | `scvi-tools` |
| `graphst` | GraphST spatial embedding + NNLS | Noisy/sparse ST data, captures spatial context | `GraphST` |

---

## Output Keys

All methods write to **two** obsm slots:

| Key | Content |
|---|---|
| `<method>_ct_pred` | Method-specific proportions (e.g., `tangram_ct_pred`) |
| `deconvolution_ct_pred` | **Unified key** — always present, same data |

The unified `deconvolution_ct_pred` key lets downstream skills (visualisation, enrichment) work regardless of which method was used.

---

## Workflow

1. **Load** spatial (.h5ad) and scRNA reference (.h5ad).
2. **Intersect** genes shared by both datasets.
3. **Deconvolve** using the chosen method.
4. **Output** updated spatial .h5ad with proportion matrices.

---

## CLI Reference

```bash
# Tangram (default)
python spatialclaw.py run spatial-deconvolution \
  --input  <st_data.h5ad> \
  --reference <sc_data.h5ad> \
  --cell-type-key <column>

# Stereoscope
python spatialclaw.py run spatial-deconvolution \
  --input  <st_data.h5ad> \
  --reference <sc_data.h5ad> \
  --cell-type-key <column> \
  --method stereoscope \
  --n-epochs 10000

# GraphST-guided
python spatialclaw.py run spatial-deconvolution \
  --input  <st_data.h5ad> \
  --reference <sc_data.h5ad> \
  --cell-type-key <column> \
  --method graphst
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--input` | — | Spatial transcriptomics .h5ad |
| `--reference` | — | scRNA reference .h5ad |
| `--cell-type-key` | — | Column in reference `.obs` with cell type labels |
| `--method` | `tangram` | `tangram` \| `stereoscope` \| `graphst` |
| `--n-epochs` | method default* | Training epochs override |
| `--output` | — | Output directory |
| `--no-gpu` | false | Force CPU |

\* Method epoch defaults: tangram=1000, stereoscope=200, graphst=1000

Demo mode is intentionally not registered for this skill because real
deconvolution requires a user-provided scRNA-seq reference and cell-type label.

---

## Tool Call Reference (AI agent mode)

```json
{
  "skill": "spatial-deconvolution",
  "file_path": "<path_to_spatial.h5ad>",
  "output_dir": "<output_directory>",
  "extra_args": [
    "--reference", "<path_to_sc_reference.h5ad>",
    "--cell-type-key", "<cell_type_column>",
    "--method", "stereoscope"
  ]
}
```

### Required extra_args
- `--reference <path>` — scRNA reference .h5ad.
- `--cell-type-key <column>` — cell type column in the reference `.obs`; ask the user when the column is unknown.

### Method selection
- User says "Tangram" → `--method tangram` (default, can omit)
- User says "Stereoscope" → `--method stereoscope`
- User says "GraphST" → `--method graphst`

### Bot tool parameters
- Use `file_path` for the spatial `.h5ad` input.
- Use `output_dir` for the result directory when the user asks for a specific destination.
- Use `extra_args` for `--reference`, `--cell-type-key`, `--method`, `--n-epochs`, and `--no-gpu`.
- Stereoscope and GraphST are deep learning methods that can take 10-60 minutes; tell the user the expected runtime before launching them.

---

## Dependencies

```bash
# Tangram
pip install tangram-sc

# Stereoscope
pip install scvi-tools

# GraphST
pip install GraphST
```

All methods additionally require: `scanpy`, `anndata`, `scipy`, `numpy`.
