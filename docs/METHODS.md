# SpatialClaw Methods Guide

This guide summarizes the spatial-only skill set and canonical CLI names.

## Preprocessing

Canonical skill: `spatial-preprocessing`

```bash
python spatialclaw.py run spatial-preprocessing \
  --input data.h5ad \
  --output output/spatial_preprocessing
```

Key parameters include `--data-type`, `--species`, `--min-genes`,
`--min-cells`, `--max-mt-pct`, `--n-top-hvg`, `--n-pcs`, `--n-neighbors`, and
`--leiden-resolution`.

## Spatial Domain Identification

Canonical skill: `spatial-domain-identification`

Methods include `leiden`, `louvain`, `spagcn`, `stagate`, `graphst`, and
`banksy` when dependencies are installed.

```bash
python spatialclaw.py run spatial-domain-identification \
  --input output/spatial_preprocessing/processed.h5ad \
  --output output/spatial_domain_identification \
  --method leiden --resolution 0.8
```

## Cell Type Annotation

Canonical skill: `spatial-cell-annotation`

Methods include marker-based annotation, Tangram, scANVI, and CellAssign.

```bash
python spatialclaw.py run spatial-cell-annotation \
  --input output/spatial_preprocessing/processed.h5ad \
  --output output/spatial_annotation \
  --method marker_based --species human
```

## Deconvolution

Canonical skill: `spatial-deconvolution`

Methods include Tangram, Stereoscope, and GraphST.

```bash
python spatialclaw.py run spatial-deconvolution \
  --input output/spatial_preprocessing/processed.h5ad \
  --output output/spatial_deconvolution \
  --method tangram \
  --reference ref.h5ad \
  --cell-type-key cell_type
```

## Spatial Statistics

Canonical skill: `spatial-statistics`

Analysis types include neighborhood enrichment, Moran's I, Geary's C, local
Moran, Getis-Ord, bivariate Moran, Ripley, co-occurrence, network properties,
and spatial centrality.

```bash
python spatialclaw.py run spatial-statistics \
  --input output/spatial_preprocessing/processed.h5ad \
  --output output/spatial_statistics \
  --analysis-type moran --n-top-genes 20
```

## Spatially Variable Genes

Canonical skill: `spatial-svg-detection`

Methods include Moran's I, SpatialDE, and FlashS.

```bash
python spatialclaw.py run spatial-svg-detection \
  --input output/spatial_preprocessing/processed.h5ad \
  --output output/spatial_svg \
  --method morans --n-top-genes 50
```

## Differential Expression

Canonical skill: `spatial-de`

```bash
python spatialclaw.py run spatial-de \
  --input output/spatial_preprocessing/processed.h5ad \
  --output output/spatial_de \
  --method wilcoxon --groupby leiden --n-top-genes 20
```

## Condition Comparison

Canonical skill: `spatial-condition-comparison`

```bash
python spatialclaw.py run spatial-condition-comparison \
  --input data.h5ad \
  --output output/spatial_condition_comparison \
  --condition-key treatment \
  --sample-key sample_id \
  --reference-condition control
```

## Cell Communication

Canonical skill: `spatial-cell-communication`

Methods include built-in ligand-receptor scoring, LIANA+, CellPhoneDB, and
FastCCC when optional Python dependencies are available.

```bash
python spatialclaw.py run spatial-cell-communication \
  --input output/spatial_preprocessing/processed.h5ad \
  --output output/spatial_cell_communication \
  --method liana --cell-type-key leiden --species human
```

## Velocity And Trajectory

Canonical skills: `spatial-velocity`, `spatial-trajectory`

```bash
python spatialclaw.py run spatial-velocity \
  --input data.h5ad \
  --output output/spatial_velocity \
  --method stochastic

python spatialclaw.py run spatial-trajectory \
  --input output/spatial_velocity/processed.h5ad \
  --output output/spatial_trajectory \
  --method dpt
```

## Enrichment

Canonical skill: `spatial-enrichment`

```bash
python spatialclaw.py run spatial-enrichment \
  --input output/spatial_de/processed.h5ad \
  --output output/spatial_enrichment \
  --analysis-type go \
  --groupby leiden \
  --organism human
```

## CNV

Canonical skill: `spatial-cnv`

```bash
python spatialclaw.py run spatial-cnv \
  --input output/spatial_preprocessing/processed.h5ad \
  --output output/spatial_cnv \
  --method infercnvpy --reference-key cell_type
```

## Multi-Sample Integration

Canonical skill: `spatial-integration`

```bash
python spatialclaw.py run spatial-integration \
  --input combined.h5ad \
  --output output/spatial_integration \
  --method harmony --batch-key batch
```

## Spatial Registration

Canonical skill: `spatial-registration`

```bash
python spatialclaw.py run spatial-registration \
  --input combined.h5ad \
  --output output/spatial_registration \
  --method paste --reference-slice slice1.h5ad
```

## Spatial Modality Integration

Canonical skill: `spatial-modality-integrate`

This skill accepts directory-based spatial platform inputs for DeepST or PearlST.

```bash
python spatialclaw.py run spatial-modality-integrate \
  --input /path/to/DLPFC/151673 \
  --output output/spatial_modality \
  --method deepst --n-domains 7 --use-gpu
```

## Spatial Paired-Modality Integration

Canonical skill: `spatial-omics-integrate`

This is a spatial skill for paired spatial modalities such as RNA plus protein
or RNA plus ATAC.

```bash
python spatialclaw.py run spatial-omics-integrate \
  --input rna.h5ad \
  --omics2 protein.h5ad \
  --output output/spatial_omics \
  --method spatialglue \
  --omics1-type rna \
  --omics2-type protein
```

## Current Registry

Use `python spatialclaw.py list` for the active registry. Public skill
documentation and routing are limited to registered CLI-capable spatial skills.
Internal helpers under `skills/spatial/_lib/` are implementation modules, not
standalone skills.

## Orchestrator

Canonical skill: `spatial-orchestrator`

```bash
python spatialclaw.py run spatial-orchestrator \
  --query "find spatially variable genes" \
  --output output/spatial_route

python spatialclaw.py run spatial-orchestrator \
  --pipeline standard \
  --input data.h5ad \
  --output output/spatial_pipeline
```
