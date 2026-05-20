---
name: spatial-cell-communication
description: >-
  Cell-cell communication analysis via ligand-receptor scoring using built-in
  scoring, LIANA, CellPhoneDB, or FastCCC.
version: 0.4.0
author: SPATIALCLAW Team
license: MIT
tags: [spatial, communication, ligand-receptor, cell-cell-interaction, liana, cellphonedb, fastccc]
metadata:
  SPATIALCLAW:
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "📡"
    os: [macos, linux]
    install:
      - kind: pip
        package: squidpy
        bins: []
    trigger_keywords:
      - cell communication
      - ligand receptor
      - cell-cell interaction
      - LIANA
      - CellPhoneDB
      - FastCCC
---

# Spatial Cell Communication

This skill identifies ligand-receptor interactions between annotated cell groups in spatial transcriptomics data.

## Methods

| Method | Input Matrix | Notes |
|--------|--------------|-------|
| `builtin` | `adata.X` log-normalized expression | Dependency-light curated ligand-receptor scoring with permutation p-values |
| `liana` | `adata.X` log-normalized expression | Optional LIANA+ consensus ranking |
| `cellphonedb` | `adata.X` log-normalized expression | Optional CellPhoneDB statistical permutation test |
| `fastccc` | `adata.X` log-normalized expression | Optional permutation-free FastCCC workflow |

## CLI

```bash
python skills/spatial/spatial-cell-communication/spatial_cell_communication.py \
  --input <preprocessed.h5ad> \
  --method builtin \
  --cell-type-key cell_type \
  --species human \
  --output <dir>

python skills/spatial/spatial-cell-communication/spatial_cell_communication.py --demo --output <dir>
spatialclaw run spatial-cell-communication --input <file.h5ad> --output <dir>
```

Allowed `--method` values: `builtin`, `liana`, `cellphonedb`, `fastccc`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Preprocessed spatial `.h5ad` |
| `--output` | required | Output directory |
| `--demo` | off | Run built-in demo |
| `--method` | `builtin` | `builtin`, `liana`, `cellphonedb`, or `fastccc` |
| `--cell-type-key` | `leiden` | `.obs` column with cell type or cluster labels |
| `--species` | `human` | Ligand-receptor database species: `human`, `mouse`, or `zebrafish` |
| `--n-perms` | `100` | Number of permutations for permutation-based methods |

## Output

```
output_directory/
├── report.md
├── result.json
├── processed.h5ad
├── figures/
│   ├── lr_ranked.png
│   ├── lr_dotplot.png
│   ├── lr_heatmap.png
│   ├── lr_chord.png
│   └── lr_spatial.png
└── tables/
    ├── lr_interactions.csv
    └── top_interactions.csv
```

If `SPATIALCLAW_EXTRA_FIGURE_FORMATS=pdf` is set, each figure is also exported
as a PDF alongside the default PNG.

## Dependencies

Required Python packages:

- `scanpy`
- `squidpy`

Optional Python packages:

- `liana`
- `cellphonedb`
- `fastccc`

## Safety

- Local-first processing.
- Reports include SPATIALCLAW disclaimers.
- Parameters and outputs are recorded for reproducibility.

## Citations

- [LIANA+](https://github.com/saezlab/liana-py) — multi-method ligand-receptor framework.
- [CellPhoneDB](https://www.cellphonedb.org/) — curated ligand-receptor database.
- [FastCCC](https://github.com/Svvord/FastCCC) — permutation-free CCC analysis.
- [Squidpy](https://squidpy.readthedocs.io/) — spatial neighborhood analysis.
