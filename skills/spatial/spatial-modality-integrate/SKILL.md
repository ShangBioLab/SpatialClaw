---
name: spatial-modality-integrate
description: >-
  Identify spatial domains with DeepST or PearlST. This skill is
  directory-only: use `--input /path/to/sample` for single-sample runs; use
  `--mode integration --input-list samples.txt` only with DeepST.
version: 2.2.0
author: SPATIALCLAW Team
license: MIT
tags: [spatial, DeepST, PearlST, domain-identification, integration, morphology]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "🧠🔬"
    os: [macos, linux]
    install:
      - kind: pip
        package: scanpy
        bins: []
      - kind: pip
        package: anndata
        bins: []
    trigger_keywords:
      - DeepST
      - PearlST
      - spatial integration
      - multi-sample spatial
      - tissue domains
      - morphology
---

# 🧠🔬 Spatial Modality Integrate

You are **Spatial Modality Integrate**, a SPATIALCLAW spatial analysis skill with two backends:

- **DeepST** for single-sample domain identification and same-modality multi-sample integration
- **PearlST** for single-sample spatial transcriptomics with optional Visium histology features

Your role is to keep the CLI contract consistent with other skills while preserving the real upstream input assumptions: sample directories, not loose single-file ad hoc invocation.

## Why This Exists

- **Without it**: Users have to manually translate DeepST and PearlST's repository-specific scripts, directory assumptions, and undocumented defaults into one-off notebooks.
- **With it**: One standard SPATIALCLAW skill exposes both methods through a stable CLI, reproducible outputs, and method-aware reports.
- **Why SPATIALCLAW**: The skill keeps the standard `--input` entry for single-sample runs, and only uses `--input-list` as the alternative input path for DeepST integration mode.

## Core Capabilities

1. **DeepST single-sample domain identification** from one spatial sample directory
2. **DeepST multi-sample integration** from a text file listing sample directories
3. **PearlST single-sample analysis** with PDE denoising, alpha-complex graph construction, and WARGA training
4. **Optional morphology-aware modeling** when tissue images are available
5. **Standardized reporting** via `report.md`, `metadata.json`, figures, and `.h5ad` outputs

## Input Contract

This skill is **directory-only**.

| Mode | Required CLI | Purpose |
|------|--------------|---------|
| Single sample | `--input /path/to/sample --output <dir>` | Standard SPATIALCLAW input pattern for both DeepST and PearlST |
| Integration | `--mode integration --input-list samples.txt --output <dir>` | Alternative multi-sample input pattern for **DeepST only** |

### Important Constraints

- `--input` must point to a **sample directory**, not a single file
- `--input-list` must point to a text file containing **one sample directory per line**
- Runs use real sample directories or an input-list file.
- `--method pearlst` currently supports **single-sample mode only**
- This skill is for **same-modality spatial analysis**, not RNA + protein / ATAC cross-omics integration

### Interactive / LLM Routing Notes

- Keep the tool call lean: use `file_path` for single-sample runs plus common structured fields such as `method`, `platform`, `device`, `n_epochs`, and `output_dir`
- Put skill-specific options in `extra_args`, especially `--mode integration`, `--input-list`, `--n-domains`, `--pre-epochs`, `--batch-key`, `--ground-truth`, and `--simclr-features`
- The canonical CLI names are the ones listed below

## Expected Directory Layout

Typical Visium-style input:

```text
data/DLPFC/151673/
├── filtered_feature_bc_matrix.h5
└── spatial/
    ├── tissue_positions_list.csv
    ├── tissue_hires_image.png
    └── tissue_lowres_image.png
```

Notes:

- For `Visium`, both DeepST and PearlST load the directory directly from the Visium folder layout.
- For DeepST integration, each sample listed in `--input-list` must be one such sample directory.
- PearlST is a **single-sample** method. The upstream README describes support for `10X Visium`, `Stereo-seq`, `Slide-seqV2`, `MERFISH`, and `STARmap`.
- Non-Visium PearlST inputs should still arrive as a sample directory, typically containing a `.h5ad` plus spatial metadata.
- By default, morphology features are auto-enabled when suitable tissue images are detected.
- Use `--no-morphological` to force image features off even when tissue images are present.
- For PearlST, morphology currently means **Visium histology features**. You can provide them in one of two ways:
  - Pass `--simclr-features <csv>` if you already have PearlST-compatible `simCLR_representation_resnet50.csv`
  - Or let the skill generate them from the bundled PearlST SimCLR checkpoint

## Workflow

### Single-Sample Mode

#### DeepST

1. **Load**: Resolve `--input` to a sample directory and load the spatial matrix
2. **Preprocess**: Build augmentation, graph, and PCA representations; the skill uses an HVG-first path before DeepST PCA
3. **Optional morphology**: When H&E images are available, prefer cached low-resolution image features instead of writing per-spot PNG crops
4. **Train**: Run DeepST domain identification
5. **Cluster**: Respect `--n-domains` directly when provided; otherwise estimate a target domain count from the DeepST embedding and then refine spatially
6. **Annotate**: Store embeddings and refined domain assignments in the output AnnData
7. **Report**: Save figures, `report.md`, `metadata.json`, and `<sample>_deepst.h5ad`

#### PearlST

1. **Load**: Read a single spatial sample directory
2. **Preprocess**: Filter genes, normalize, and keep **exactly 2000 HVGs**
3. **Enhance**: Run PearlST's PDE-based denoising and neighbor-aware augmentation
4. **Optional histology**: For Visium, load or generate PearlST-compatible SimCLR image features
5. **Train**: Build the alpha-complex graph and train the WARGA adversarial graph autoencoder
6. **Cluster and summarize**: Run KMeans on `PearlST_embed`, optionally compute ARI if `--ground-truth` is provided, and save `<sample>_pearlst.h5ad`

### Integration Mode

1. **Read list**: Parse `--input-list` as one sample directory per line
2. **Load each sample**: Build one AnnData object per sample directory
3. **Optional morphology**: Extract image features for each sample when all samples provide tissue images
4. **Integrate**: Run DeepST integration with batch-aware training
5. **Report**: Save `integrated_deepst.h5ad`, per-sample `.h5ad` files, figures, and summary files

## CLI Reference

```bash
# Single sample via the standard --input contract
spatialclaw run spatial-modality-integrate \
  --input data/DLPFC/151673 \
  --output results/deepst_single

# Single sample with explicit DeepST parameters
spatialclaw run spatial-modality-integrate \
  --method deepst \
  --input data/DLPFC/151673 \
  --n-domains 7 \
  --pre-epochs 500 \
  --epochs 500 \
  --output results/deepst_single

# Single sample with PearlST
spatialclaw run spatial-modality-integrate \
  --method pearlst \
  --input data/DLPFC/151673 \
  --n-domains 7 \
  --epochs 300 \
  --output results/pearlst_single

# PearlST with provided ground truth and precomputed simCLR features
spatialclaw run spatial-modality-integrate \
  --method pearlst \
  --input data/DLPFC/151673 \
  --ground-truth data/DLPFC/151673/labels.tsv \
  --simclr-features data/DLPFC/151673/simCLR_representation_resnet50.csv \
  --output results/pearlst_with_labels

# Multi-sample integration via --input-list
spatialclaw run spatial-modality-integrate \
  --method deepst \
  --mode integration \
  --input-list samples.txt \
  --batch-key batch \
  --n-domains 8 \
  --output results/deepst_integration

# Disable morphology explicitly even if tissue images exist
spatialclaw run spatial-modality-integrate \
  --input data/DLPFC/151673 \
  --no-morphological \
  --output results/deepst_no_morphology

# Equivalent top-level invocation
python spatialclaw.py run spatial-modality-integrate \
  --input data/DLPFC/151673 \
  --output results/deepst_single

# Direct script invocation
python skills/spatial/spatial-modality-integrate/spatial_modality_integrate.py \
  --input data/DLPFC/151673 \
  --output results/deepst_single
```

Example `samples.txt`:

```text
data/DLPFC/151673
data/DLPFC/151674
data/DLPFC/151675
```

## Parameters

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `--input` | path | - | Sample directory; required in `single` mode |
| `--output` | path | - | Output directory for reports, figures, and h5ad files |
| `--method` | str | `deepst` | `deepst` or `pearlst` |
| `--mode` | str | `single` | `single` or `integration` |
| `--input-list` | path | - | List of sample directories; required in `integration` mode and only used by DeepST |
| `--platform` | str | `Visium` | `Visium`, `Stereo-seq`, `Slide-seq`, `Slide-seqV2`, `MERFISH`, or `STARmap` |
| `--n-domains` | int | auto | Number of spatial domains; when set, DeepST uses it directly instead of running slower automatic resolution search |
| `--batch-key` | str | `batch` | Batch column used in DeepST integration mode |
| `--pre-epochs` | int | 500 | DeepST pretraining epochs |
| `--epochs` | int | method-specific | `500` for DeepST, `1270` upstream default for PearlST |
| `--pca-components` | int | 200 | DeepST PCA dimensions |
| `--n-top-genes` | int | 2000 | PearlST HVG count; upstream algorithm expects exactly `2000` |
| `--use-morphological` | flag | auto | Force-enable H&E morphology features |
| `--no-morphological` | flag | off | Disable H&E morphology features even when images exist |
| `--ground-truth` | path | - | Optional barcode-indexed labels for PearlST ARI evaluation |
| `--simclr-features` | path | - | Optional PearlST `simCLR_representation_resnet50.csv` |
| `--simclr-model` | path | bundled | Optional PearlST SimCLR checkpoint path |
| `--use-gpu` | flag | auto | Prefer GPU acceleration when available |
| `--no-gpu` | flag | off | Force CPU execution |
| `--device` | str | auto | Explicit device such as `cpu` or `cuda:0` |
| `--random-seed` | int | 0 | Reproducibility seed |


## Output Structure

### Single-sample run

```text
output_dir/
├── <sample_name>_deepst.h5ad         # or <sample_name>_pearlst.h5ad
├── report.md
├── metadata.json
├── simCLR_representation_resnet50.csv # PearlST only when features are generated on the fly
└── figures/
    ├── spatial_domain_identification.png
    ├── umap_spatial_comparison.png
    ├── domain_sizes.png
    ├── pearlst_pseudotime.png        # PearlST only, when pseudotime succeeds
    └── workflow_summary.png
```

### Integration run

```text
output_dir/
├── integrated_deepst.h5ad
├── <sample_1>_deepst.h5ad
├── <sample_2>_deepst.h5ad
├── report.md
├── metadata.json
└── figures/
    ├── spatial_domain_identification.png
    ├── umap_spatial_comparison.png
    ├── domain_sizes.png
    └── workflow_summary.png
```

## When to Use

Use this skill when:

- You have one or more spatial samples stored as sample directories
- You want DeepST-based tissue domain identification or PearlST-based single-sample ST analysis
- You want to integrate multiple samples of the **same modality** with DeepST
- You want morphology-aware features from tissue images without having to remember an extra flag

Do not use this skill when:

- You need to integrate **two different omics modalities** such as RNA + protein: use `spatial-omics-integrate`
- You want to run PearlST on multiple samples jointly: PearlST is not a multi-sample integration backend here
- You only have a single `.h5ad` or `.h5` file and no sample directory layout
- You want demo data: this skill does not provide any
