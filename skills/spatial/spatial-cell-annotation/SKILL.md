---
name: spatial-cell-annotation
description: >-
  Cell type annotation for spatial transcriptomics data using marker-based
  scoring, Tangram mapping, scANVI transfer, or CellAssign probabilistic models.
version: 0.2.0
author: SPATIALCLAW
license: MIT
tags: [spatial, annotation, cell-type, tangram, scanvi, cellassign, marker-genes]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "🏷️"
    os: [macos, linux]
    install:
      - kind: pip
        package: scanpy
        bins: []
    trigger_keywords:
      - cell type annotation
      - annotate cell types
      - Tangram
      - scANVI
      - CellAssign
      - marker genes
---

# 🏷️ Spatial Annotate

You are **Spatial Annotate**, a specialised SPATIALCLAW agent for cell type annotation. Your role is to assign biologically meaningful cell type labels to spatial transcriptomics spots/cells using multiple methods with varying accuracy-complexity tradeoffs.

## Why This Exists

- **Without it**: Manual literature search for markers, inconsistent annotation across projects
- **With it**: One command annotates all spots with cell types, produces spatial maps and reports
- **Why SPATIALCLAW**: Unified interface across 4 methods — from zero-reference marker scoring to deep learning transfer

## Workflow

1. **Calculate**: Prepare modalities and normalize batch representations.
2. **Execute**: Run chosen annotation mechanism across spatial structures.
3. **Assess**: Quantify annotation probabilities versus bio-preservation.
4. **Generate**: Save annotated matrices and compute UMAP/spatial graphs.
5. **Report**: Synthesize report with annotation metadata.

## Core Capabilities

1. **Marker-based** (default, fast): No reference needed — scores cluster markers against built-in cell type signatures. Uses `adata.X` (log-normalized)
2. **Tangram**: Maps single-cell reference to spatial data via deep learning. Uses `adata.X` (log-normalized) for both reference and spatial
3. **scANVI**: Semi-supervised variational inference for label transfer. Uses `adata.layers["counts"]` (raw counts, NB model)
4. **CellAssign**: Probabilistic assignment using predefined marker gene panels. Uses `adata.layers["counts"]` (raw counts, NB model)

## Input Formats

| Format | Extension | Required Fields | Example |
|--------|-----------|-----------------|---------|
| AnnData (preprocessed) | `.h5ad` | `X` (normalised), `layers["counts"]` (raw), `obsm["spatial"]`, clusters | `preprocessed.h5ad` |
| Reference (for tangram/scanvi) | `.h5ad` | `X`, `obs["cell_type"]` | `reference_sc.h5ad` |

### Input Matrix Convention

Different annotation methods have different statistical assumptions about the input expression data:

| Method | Input Matrix | Rationale |
|--------|-------------|-----------|
| `marker_based` | `adata.X` (log-normalized) | Marker scoring and Wilcoxon test operate on continuous expression where gene magnitudes are comparable |
| `tangram` | `adata.X` (log-normalized) | Both scRNA-seq and spatial must be on the same normalized scale for the mapping optimization |
| `scanvi` | `adata.layers["counts"]` (raw) | VAE generative model assumes negative-binomial / ZINB count likelihood |
| `cellassign` | `adata.layers["counts"]` (raw) | Probabilistic model assumes negative-binomial count likelihood; size factors computed from raw counts |

**Core principle**: Whether a method uses counts or normalized data depends on whether it has a count-based probabilistic model (NB/ZINB/GLM) internally.

**Data layout requirement**: Preprocessing must store raw counts before normalization:

```python
adata.layers["counts"] = adata.X.copy()   # before normalize_total + log1p
adata.X = lognorm_expr                     # after normalize_total + log1p
```

If `layers["counts"]` is missing, count-based methods (scanvi, cellassign) will fall back to `adata.raw` (if available) or `adata.X` with a warning.

## CLI Reference

```bash
# Marker-based (default, no reference needed)
python skills/spatial/spatial-cell-annotation/spatial_cell_annotation.py \
  --input <preprocessed.h5ad> --output <dir>

# Tangram transfer
python skills/spatial/spatial-cell-annotation/spatial_cell_annotation.py \
  --input <file> --method tangram --reference <sc_ref.h5ad> --output <dir>

# scANVI transfer
python skills/spatial/spatial-cell-annotation/spatial_cell_annotation.py \
  --input <file> --method scanvi --reference <sc_ref.h5ad> --output <dir>

# Demo
python skills/spatial/spatial-cell-annotation/spatial_cell_annotation.py --demo --output /tmp/annotate_demo

# Via CLI (using 'spatialclaw run' or 'python spatialclaw.py run')
spatialclaw run spatial-cell-annotation --input <file> --output <dir>
spatialclaw run spatial-cell-annotation --demo
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Preprocessed spatial `.h5ad` |
| `--output` | required | Output directory |
| `--demo` | off | Run built-in synthetic demo |
| `--method` | `marker_based` | `marker_based`, `tangram`, `scanvi`, or `cellassign` |
| `--reference` | - | scRNA reference `.h5ad` for reference-based methods |
| `--cell-type-key` | `cell_type` | Reference `.obs` column with cell type labels |
| `--cluster-key` | `leiden` | Spatial `.obs` cluster column for marker-based summaries |
| `--species` | `human` | Species used for built-in marker sets: `human` or `mouse` |
| `--batch-key` | - | Optional batch column for scANVI-style workflows |
| `--layer` | - | Optional AnnData layer to use as expression input |
| `--model` | - | Optional model/checkpoint path for model-based annotation |

## Example Queries

- "Assign cell types to my spatial tissue spots"
- "Use Tangram to map reference data to my slide"

## Output Structure

```
output_dir/
├── report.md
├── result.json
├── processed.h5ad
├── figures/
│   ├── umap_annotation.png
│   └── spatial_annotation.png
├── tables/
│   └── annotation_summary.csv
└── reproducibility/
    ├── commands.sh
    └── environment.yml
```

## Dependencies

**Required**: scanpy, anndata, numpy, pandas, scipy, matplotlib

**Optional**:
- `tangram-sc` — Tangram deep learning mapping
- `scvi-tools` — scANVI and CellAssign

## Safety

- **Local-first**: Strict offline processing without external upload.
- **Disclaimer**: Requires SPATIALCLAW reporting structures and disclaimers.
- **Audit trail**: Hyperparameters and operational flow states are logged fully.

## Integration with Orchestrator

**Trigger conditions**:
- Automatically invoked dynamically based on tool metadata and user intent matching.

**Chaining partners**:
- `spatial-preprocessing` — QC before annotation
- `spatial-domain-identification` — Regionalization after annotation
- `spatial-cell-communication` — L-R scoring using annotated types

## Citations

- [Tangram](https://doi.org/10.1038/s41592-021-01264-7) — Biancalani et al., *Nature Methods* 2021
- [scANVI](https://doi.org/10.15252/msb.20209620) — Xu et al., *Mol Syst Biol* 2021
- [CellAssign](https://doi.org/10.1038/s41592-019-0529-1) — Zhang et al., *Nature Methods* 2019
