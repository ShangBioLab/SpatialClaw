---
name: spatial-omics-integrate
description: >-
  Integrate two aligned spatial omics modalities on the same cells using
  SpatialGlue or SpaDDM.
  Standard input contract: `--input <omics1.h5ad> --omics2 <omics2.h5ad>`.
  Supports demo mode and validated modality pairs such as RNA+Protein and
  RNA+ATAC.
version: 2.1.0
author: SPATIALCLAW Team
license: MIT
tags: [spatial, multi-omics, integration, SpatialGlue, SpaDDM, RNA, protein, ATAC]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "🧬🔗"
    os: [macos, linux]
    install:
      - kind: pip
        package: SpatialGlue
        bins: []
      - kind: pip
        package: torch-geometric
        bins: []
      - kind: pip
        package: scanpy
        bins: []
    trigger_keywords:
      - spatial multi-omics
      - SpatialGlue
      - SpaDDM
      - RNA protein integration
      - RNA ATAC integration
      - multi-modal spatial
---

# 🧬🔗 Spatial Omics Integrate

You are **Spatial Omics Integrate**, a SPATIALCLAW skill for integrating two spatial omics modalities measured on the same cells or spots. Your role is to keep the multi-input contract consistent with the rest of SPATIALCLAW: the first modality enters via standard `--input`, and the second modality enters via `--omics2`, while `--method` selects the integration backend.

## Why This Exists

- **Without it**: Users have to wire together modality-specific preprocessing, graph construction, and backend-specific training code manually.
- **With it**: One skill runs the two-file workflow with either SpatialGlue or SpaDDM, writes a report, and saves a standardized integrated `.h5ad`.
- **Why SPATIALCLAW**: The skill now matches the platform-wide input pattern instead of inventing a separate primary argument contract.

## Core Capabilities

1. **Two-modality integration** with a shared CLI for `spatialglue` and `spaddm`
2. **Validated modality pairs** for RNA+Protein and RNA+ATAC
3. **Attention-weight inspection** for inter-modality contribution
4. **Integrated clustering** with method-native embeddings plus a standardized `X_spatial_omics` alias
5. **Demo mode** for quick smoke testing when real paired data is unavailable

## Input Contract

| Mode | Required CLI | Purpose |
|------|--------------|---------|
| Standard run | `--input <omics1.h5ad> --omics2 <omics2.h5ad> --output <dir>` | Real paired multi-omics data |
| Demo run | `--demo --output <dir>` | Synthetic example data |

### Critical Requirements

- Both inputs must be `.h5ad` files
- Both inputs must contain the **same number of cells/spots**
- Cells/spots should be in the **same order**; if the two inputs share the same `obs_names` but are ordered differently, SPATIALCLAW will realign omics2 to omics1 automatically
- This skill integrates **two different modalities on the same cells**, not multiple samples
- The first modality should be passed through standard `--input`
- At least one input must contain `obsm["spatial"]`; if only one modality has spatial coordinates, SPATIALCLAW copies them to the other modality

If the two inputs are not aligned, fix the alignment before running this skill.

### Interactive / LLM Routing Notes

- Keep the tool call lean: use `file_path` for the primary omics file plus common structured fields such as `method`, `platform`, `device`, `n_epochs`, and `output_dir`
- Put skill-specific options in `extra_args`, especially `--omics2`, `--omics1-type`, `--omics2-type`, `--clustering-method`, `--n-clusters`, `--n-hvg`, and `--n-latent`
- Treat `--data-type` as the canonical CLI spelling

## Supported Modalities

| CLI Value | Meaning |
|-----------|---------|
| `rna` | Spatial transcriptomics / RNA-seq |
| `protein` | ADT / CITE-seq / spatial protein measurements |
| `atac` | Spatial chromatin accessibility / ATAC-like peak matrix |

Validated combinations:

- RNA + Protein
- RNA + ATAC

Current backend coverage:

| Method | Validated pairs | Notes |
|--------|-----------------|-------|
| `spatialglue` | RNA+Protein, RNA+ATAC | Dual-attention GNN; default clustering is Leiden |
| `spaddm` | RNA+Protein, RNA+ATAC | Directional diffusion model; clustered with Leiden by default in SPATIALCLAW |

## Workflow

1. **Load**: Read omics1 from `--input` and omics2 from `--omics2`
2. **Validate**: Confirm paired cell count consistency before training
3. **Preprocess**: Apply modality-specific preprocessing for each input
4. **Construct graphs**: Build the method-specific graph inputs
5. **Integrate**: Train the selected backend and obtain joint embeddings plus attention weights
6. **Cluster and report**: Save integrated AnnData, figures, report, metrics tables, and reproducibility metadata

## CLI Reference

```bash
# RNA + Protein with SpatialGlue
spatialclaw run spatial-omics-integrate \
  --input rna.h5ad \
  --omics2 protein.h5ad \
  --method spatialglue \
  --omics1-type rna \
  --omics2-type protein \
  --n-clusters 8 \
  --output results/spatialglue_rna_protein

# RNA + ATAC with SpaDDM
spatialclaw run spatial-omics-integrate \
  --input rna.h5ad \
  --omics2 atac.h5ad \
  --method spaddm \
  --omics1-type rna \
  --omics2-type atac \
  --data-type Spatial-epigenome-transcriptome \
  --clustering-method leiden \
  --output results/spaddm_rna_atac

# Demo
spatialclaw run spatial-omics-integrate --demo --method spaddm --n-epochs 10 --output results/spaddm_demo

# Equivalent top-level invocation
python spatialclaw.py run spatial-omics-integrate \
  --input rna.h5ad \
  --omics2 protein.h5ad \
  --output results/spatialglue_rna_protein

# Direct script invocation
python skills/spatial/spatial-omics-integrate/spatial_omics_integrate.py \
  --input rna.h5ad \
  --omics2 protein.h5ad \
  --output results/spatialglue_rna_protein
```

## Parameters

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `--input` | path | - | First modality input |
| `--omics2` | path | - | Second modality input |
| `--output` | path | required | Output directory |
| `--method` | str | `spatialglue` | `spatialglue` or `spaddm` |
| `--omics1-type` | str | `rna` | `rna`, `protein`, or `atac` |
| `--omics2-type` | str | `protein` | `rna`, `protein`, or `atac` |
| `--data-type` | str | `auto` | `auto`, `10x`, `Stereo-CITE-seq`, `SPOTS`, `spatial-epigenome`, `Spatial-epigenome-transcriptome`, or `Visium CytAssist` |
| `--clustering-method` | str | `auto` | `auto`, `leiden`, or `louvain` |
| `--n-hvg` | int | 3000 | HVGs for RNA preprocessing |
| `--n-clusters` | int | auto | Target clusters; auto if omitted |
| `--n-epochs` | int | method-specific | SpatialGlue defaults to 200; SpaDDM uses its upstream data-type defaults unless overridden |
| `--n-latent` | int | 64 | Latent dimension for SpaDDM |
| `--random-seed` | int | method-specific | SpatialGlue default is 2022; SpaDDM default is 2024 |
| `--device` | str | `gpu` | Compute device. `gpu` prefers CUDA with CPU fallback; explicit values like `cpu`, `cuda`, and `cuda:0` are also accepted |
| `--demo` | flag | off | Use built-in synthetic paired data |


## Output Structure

```text
output_dir/
├── integrated_<Method>.h5ad
├── report.md
├── result.json
├── umap_spatial_clusters.png
├── attention_weights.png
├── feature_clustering_stats.png
├── workflow_summary.png
├── tables/
│   ├── integration_metrics.csv
│   └── preprocessing_summary.csv
└── reproducibility/
    ├── commands.sh
    └── environment.yml
```

Key output fields:

- Integrated clustering labels are stored in `.obs["spatial_omics_cluster"]` and also in the backend-native key such as `.obs["SpatialGlue"]` or `.obs["SpatialDDM"]`
- Joint representation is stored in `.obsm["X_spatial_omics"]` and in the backend-native embedding key
- Modality-specific latent spaces and attention weights are preserved in the output object
- SpaDDM runs also preserve `rec_omics1` / `rec_omics2` reconstruction outputs

## When to Use

Use this skill when:

- You have **two aligned spatial omics modalities**
- You want a **joint embedding and clustering**
- You need modality-specific preprocessing handled automatically
- You want to inspect the relative contribution of each modality
- You want to switch between SpatialGlue and SpaDDM without changing the basic two-file contract

Do not use this skill when:

- You need to integrate **multiple samples of the same modality**: use `spatial-modality-integrate`
- You only have one omics matrix
- The two inputs do not share the same cells/spots
- You need SpaDDM's optional cross-omics translation workflow with `moscot`: that is an upstream downstream analysis step and is not run automatically by this skill
