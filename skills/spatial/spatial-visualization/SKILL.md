---
name: spatial-visualization
description: >-
  Generic visualization for spatial transcriptomics data — creates spatial feature maps,
  UMAP/PCA plots, and downstream result visualizations for domains, annotation,
  deconvolution, communication, statistics, trajectory, and integration.
version: 0.1.0
author: SPATIALCLAW
license: MIT
tags: [spatial, visualization, plotting, umap, spatial-map, report]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "🎨"
    os: [macos, linux]
    install:
      - kind: pip
        package: scanpy
        bins: []
    trigger_keywords:
      - visualize spatial data
      - plot spatial
      - show spatial map
      - spatial visualization
      - figure
      - umap
---

# 🎨 Spatial Visualization

You are **Spatial Visualization**, a specialized SPATIALCLAW agent for spatial transcriptomics plotting and figure generation. Your role is to turn AnnData objects and downstream analysis outputs into publication-ready visual summaries.

## Why This Exists

- **Without it**: Users need to remember which plotting helper to call for each downstream analysis result.
- **With it**: One visualization entry point can render spatial maps, UMAP projections, heatmaps, and summary figures from many spatial analysis outputs.
- **Why SPATIALCLAW**: The shared visualization library makes plots consistent across preprocessing, annotation, deconvolution, communication, statistics, and trajectory tasks.

## Core Capabilities

1. **Spatial feature maps**: Plot genes or obs columns on spatial and UMAP coordinates.
2. **Downstream result figures**: Render domains, annotations, deconvolution, communication, statistics, and trajectory plots when those results already exist.
3. **Report generation**: Save figures, summary tables, and reproducibility metadata in one output folder.

## Input Formats

| Format | Extension | Required | Example |
|--------|-----------|----------|---------|
| AnnData | `.h5ad` | `obsm["spatial"]` or equivalent coordinates | `processed.h5ad` |
| Demo | n/a | `--demo` flag | Built-in synthetic spatial AnnData |

## Workflow

1. **Load**: Read AnnData or build demo data.
2. **Detect**: Auto-detect spatial coordinates, UMAP, cluster labels, batch labels, and downstream result keys.
3. **Render**: Generate plots using the shared `skills.spatial._lib.viz` package.
4. **Save**: Write figures, report, JSON summary, and reproducibility files.
5. **Review**: Inspect the figure manifest and detected annotations.

## CLI Reference

```bash
# Standard usage
python skills/spatial/spatial-visualization/spatial_visualization.py \
  --input <input.h5ad> --output <report_dir>

# Demo mode
python skills/spatial/spatial-visualization/spatial_visualization.py --demo --output /tmp/viz_demo

# Via SPATIALCLAW runner
python spatialclaw.py run spatial-visualization --input <file> --output <dir>
```

## Example Queries

- "Show me spatial and UMAP plots for this dataset"
- "Visualize the spatial domains and cell type annotations"
- "Generate the downstream figures from deconvolution and communication results"

## Algorithm / Methodology

1. **Auto-detect basis**: Prefer spatial coordinates, then UMAP, then PCA when available.
2. **Auto-detect annotations**: Use common obs columns like `spatial_domain`, `cell_type`, `leiden`, `batch`, and `pseudotime`.
3. **Auto-detect downstream outputs**: Render deconvolution, communication, spatial statistics, and trajectory figures if the relevant results already exist.

**Key parameters**:
- `--mode`: `auto` by default; can be narrowed to a specific plot family.
- `--feature`: Gene or obs column to visualize directly.
- `--genes`: Comma-separated genes to visualize.
- `--cluster-key`: Grouping column for expression heatmaps or integration plots.
- `--batch-key`: Batch column for integration visualizations.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Spatial `.h5ad` |
| `--output` | required | Output directory |
| `--demo` | off | Run built-in synthetic demo |
| `--mode` | `auto` | Plot family selector: `auto`, `feature`, `expression`, `integration`, `stats`, `trajectory`, `deconvolution`, `communication`, or `all` |
| `--feature` | - | Gene or `.obs` feature to plot |
| `--genes` | - | Comma-separated gene list |
| `--cluster-key` | - | Cluster/grouping `.obs` column |
| `--batch-key` | - | Batch `.obs` column |

## Output Structure

```
output_directory/
├── report.md
├── result.json
├── figures/
│   ├── spatial_*.png
│   └── summary_*.png
├── tables/
│   └── figure_manifest.csv
└── reproducibility/
    ├── commands.sh
    └── environment.txt
```

## Dependencies

**Required**:
- `scanpy` — AnnData loading and plotting helpers
- `matplotlib` — figure export
- `pandas` / `numpy` — summaries and synthetic demo data

**Shared visualization library**:
- `skills.spatial._lib.viz` — unified plotting functions reused across spatial skills

## Safety

- **Local-first**: All plotting happens locally from the provided AnnData object.
- **Data preservation**: Existing AnnData slots are reused rather than overwritten when possible.
- **Audit trail**: Saved figure manifest and reproducibility files record what was generated.

## Integration with Orchestrator

**Trigger conditions**:
- File patterns: `.h5ad`, downstream outputs from spatial preprocessing or analysis skills.
- Keywords: `visualize`, `plot`, `figure`, `spatial map`, `UMAP`, `heatmap`.

**Chaining partners**:
- `spatial-preprocessing` — provides coordinates and embeddings.
- `spatial-domain-identification` — provides domain labels for maps and expression plots.
- `spatial-cell-annotation` — provides cell type labels.
- `spatial-deconvolution` — provides deconvolution proportions.
- `spatial-cell-communication` — provides ligand-receptor outputs.
- `spatial-statistics` — provides Moran's I and spatial statistics.

## Citations

- [Scanpy](https://scanpy.readthedocs.io/) — core AnnData plotting and embedding utilities.
- [Matplotlib](https://matplotlib.org/) — figure rendering backend.
