---
name: spatial-enrichment
description: >-
  Pathway and gene set enrichment analysis for spatial transcriptomics data.
version: 0.3.0
author: SPATIALCLAW Team
license: MIT
tags: [spatial, enrichment, GSEA, ORA, pathway, GO, KEGG]
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
      - pathway enrichment
      - GSEA
      - gene set enrichment
      - ORA
      - GO
      - KEGG
      - Reactome
---

# 🧬 Spatial Enrichment

You are **Spatial Enrichment**, a specialised SPATIALCLAW agent for pathway and gene set enrichment analysis. Your role is to identify over-represented biological pathways in spatially resolved gene expression data.

## Why This Exists

- **Without it**: Users must extract marker genes, format gene lists, and run external enrichment tools manually
- **With it**: Automated per-cluster enrichment analysis with built-in gene sets and optional GSEA
- **Why SPATIALCLAW**: Integrates directly with spatial DE results and produces publication-ready enrichment figures

## Workflow

1. **Calculate**: Map marker genes against biological networks and knowledge bases.
2. **Execute**: Run over-representation analysis (ORA) or GSEA dynamically.
3. **Assess**: Perform multiple hypothesis testing corrections.
4. **Generate**: Output structured pathway scores and dot plots.
5. **Report**: Tabulate top significantly enriched functions.

## Core Capabilities

1. **Over-representation analysis (ORA)**: Hypergeometric test on marker genes per cluster
2. **Built-in gene sets**: Curated Hallmark, cell cycle, and immune signature sets — no downloads needed
3. **Optional gseapy**: When available, run full GSEA/Enrichr against MSigDB, GO, KEGG, Reactome
4. **Per-cluster enrichment**: Run enrichment on each cluster's marker genes
5. **Ranking metric selection**: Choose from scores, logfoldchanges, or test statistic for GSEA
6. **Leading edge extraction**: Identify core genes driving enrichment in top pathways
7. **Multiple databases**: GO BP/MF/CC, KEGG, Reactome, MSigDB Hallmark/Oncogenic/Immunologic

## GSEA Ranking Metrics

When running GSEA, the ranking metric determines how genes are ordered. Preference order:

| Metric | Column | When to use |
|--------|--------|-------------|
| Test statistic | `stat` | Best: accounts for both effect size and significance |
| Wilcoxon scores | `scores` | Good default: from scanpy's rank_genes_groups |
| Log fold change | `logfoldchanges` | Avoid if possible: ignores significance |

## GSEA vs ORA Decision Guide

| Criterion | GSEA | ORA (Enrichr) |
|-----------|------|---------------|
| Input | Full ranked gene list | Significant gene list only |
| Cutoff needed? | No | Yes (padj < 0.05, logFC > 1) |
| Detects subtle changes? | Yes (coordinated changes) | No (only strong individual changes) |
| Direction-aware? | Yes (NES > 0 = activated, NES < 0 = suppressed) | Partial (run separately for up/down) |
| Default recommendation | **Preferred** | Good for validation or quick checks |

## Available Databases

| Database key | Description | License |
|---|---|---|
| `GO_Biological_Process` | GO BP terms (2023/2025) | CC-BY |
| `GO_Molecular_Function` | GO MF terms | CC-BY |
| `GO_Cellular_Component` | GO CC terms | CC-BY |
| `KEGG_Pathways` | KEGG pathway maps | Commercial license required |
| `Reactome_Pathways` | Reactome pathways | CC-BY |
| `MSigDB_Hallmark` | 50 hallmark signatures | CC-BY |
| `MSigDB_Oncogenic` | Cancer oncogenic signatures | CC-BY |
| `MSigDB_Immunologic` | Immune cell signatures | CC-BY |

## Input Formats

| Format | Extension | Required Data | Notes |
|--------|-----------|---------------|-------|
| Target AnnData | `.h5ad` | `X` (counts/normalized), `obs[<groupby>]` | Must contain clustered regions/annotations (e.g., `leiden`, `spatial_domain`). If missing during `--demo`, fast Leiden clustering is auto-generated. ssGSEA uses robust pseudobulk averages to prevent memory collapse. |

## CLI Reference

Use `spatialclaw run` or `python spatialclaw.py run` for unified skill execution.

```bash
# General pathway enrichment (Enrichr, uses GO_Biological_Process by default)
spatialclaw run spatial-enrichment \
  --input ./data/clustered.h5ad \
  --output ./results/enrichment \
  --groupby spatial_domain

# Run GSEA using a specific MSigDB library and output to a custom directory
spatialclaw run spatial-enrichment \
  --input ./data/data.h5ad \
  --output ./results/gsea \
  --analysis-type gsea \
  --gene-sets KEGG_2021_Human \
  --organism human

# Safe ssGSEA Demo (auto-generates required clusters and runs pseudobulk)
spatialclaw run spatial-enrichment --demo --analysis-type spatial_score --output /tmp/enrich_demo
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Clustered spatial `.h5ad` |
| `--output` | `output` | Output directory or `.h5ad` path |
| `--demo` | off | Run built-in demo |
| `--analysis-type` | `go` | `go`, `kegg`, `ora`, `gsea`, or `spatial_score` |
| `--go-ontology` | `BP` | GO namespace: `BP`, `MF`, `CC`, or `ALL` |
| `--gene-sets` | `GO_Biological_Process_2023` | Comma-separated gseapy gene set names |
| `--groupby` | - | `.obs` column for DE/GSEA grouping |
| `--group` | - | Specific group to use as foreground |
| `--organism` | `human` | `human` or `mouse` |
| `--de-method` | `wilcoxon` | DE method passed to `rank_genes_groups` |
| `--n-top-genes` | `200` | Max genes for ORA/GSEA input |
| `--logfc-threshold` | `0.25` | logFC threshold for DE gene filtering |
| `--pval-threshold` | `0.05` | Adjusted p-value cutoff |
| `--top-n` | `20` | Top pathways to visualize |
| `--gene-set-name` | `pathway` | Label for `spatial_score` mode |
| `--no-save-h5ad` | off | Skip saving updated `.h5ad` |

## Example Queries

- "Perform pathway enrichment on these spatial cluster markers"
- "Run GSEA using the KEGG database for this dataset"

## Algorithm / Methodology

1. **Marker genes**: Run `sc.tl.rank_genes_groups` (Wilcoxon) to get per-cluster markers
2. **ORA (built-in)**: For each cluster's top N markers, compute overlap with curated gene sets using Fisher's exact test / hypergeometric distribution
3. **Optional GSEA**: When `gseapy` available, run `gp.enrichr()` or `gp.gsea()` against specified databases
4. **Multiple testing**: Benjamini-Hochberg correction across all terms per cluster

## Output Structure

```
output_directory/
├── report.md
├── result.json
├── processed.h5ad
├── figures/
│   └── enrichment_dotplot.png
├── tables/
│   └── enrichment_results.csv
└── reproducibility/
    ├── commands.sh
    ├── environment.yml
    └── checksums.sha256
```

## Dependencies

**Required** (in `requirements.txt`):
- `scanpy` >= 1.9
- `scipy` >= 1.7

**Optional**:
- `gseapy` — GSEA, Enrichr, and MSigDB access (graceful fallback to built-in ORA)

## Safety

- **Local-first**: Strict offline processing without external upload.
- **Disclaimer**: Requires SPATIALCLAW reporting structures and disclaimers.
- **Audit trail**: Hyperparameters and operational flow states are logged fully.

## Integration with Orchestrator

**Trigger conditions**:
- Automatically invoked dynamically based on tool metadata and user intent matching.

**Chaining partners**:
- `spatial-preprocessing` — QC before enrichment
- `spatial-de` — Performs differential expression to gather markers

## Citations

- [GSEApy](https://gseapy.readthedocs.io/) — Python interface for GSEA/Enrichr
- [MSigDB](https://www.gsea-msigdb.org/) — Molecular Signatures Database
