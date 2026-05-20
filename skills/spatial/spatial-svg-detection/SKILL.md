---
name: spatial-svg-detection
description: >-
  Find genes with spatially variable expression patterns using Moran's I,
  SpatialDE, or FlashS.
version: 0.3.0
author: SpatialClaw
license: MIT
tags: [spatial, SVG, spatially-variable-genes, morans, spatialde, flashs]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "🧭"
    os: [macos, linux]
    install:
      - kind: pip
        package: scanpy
        bins: []
      - kind: pip
        package: squidpy
        bins: []
    trigger_keywords:
      - spatially variable gene
      - spatial gene
      - SVG
      - SpatialDE
      - FlashS
      - spatial pattern
      - Moran
      - spatial autocorrelation
---

# Spatial SVG Detection

This skill identifies genes whose expression varies non-randomly across tissue coordinates.

## Methods

| Method | Input Matrix | Notes |
|--------|--------------|-------|
| `morans` | `adata.X` log-normalized expression | Squidpy Moran's I spatial autocorrelation |
| `spatialde` | `adata.layers["counts"]` raw counts when available | NaiveDE stabilization followed by SpatialDE |
| `flashs` | `adata.layers["counts"]` raw counts when available | Python-native randomized kernel approximation |

## CLI

```bash
python skills/spatial/spatial-svg-detection/spatial_svg_detection.py \
  --input <processed.h5ad> \
  --method morans \
  --n-top-genes 20 \
  --output <dir>

python skills/spatial/spatial-svg-detection/spatial_svg_detection.py --demo --output <dir>
spatialclaw run spatial-svg-detection --input <file.h5ad> --output <dir>
```

Allowed `--method` values: `morans`, `spatialde`, `flashs`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Processed spatial `.h5ad` |
| `--output` | required | Output directory |
| `--demo` | off | Run built-in demo |
| `--method` | `morans` | `morans`, `spatialde`, or `flashs` |
| `--n-top-genes` | `20` | Number of top SVGs to report/plot |
| `--fdr-threshold` | `0.05` | FDR cutoff for significance |

## Output

```
output_dir/
├── report.md
├── result.json
├── processed.h5ad
├── figures/
│   └── top_svg_spatial.png
└── tables/
    └── svg_results.csv
```

## Dependencies

Required Python packages:

- `scanpy`
- `squidpy`
- `matplotlib`
- `numpy`
- `pandas`

Optional Python packages:

- `SpatialDE`
- `NaiveDE`
- `flashs`

## Safety

- Local-first processing.
- Reports include SPATIALCLAW disclaimers.
- Results are stored in `adata.uns`; original data are preserved in the output copy.

## Citations

- [Squidpy](https://squidpy.readthedocs.io/) — spatial autocorrelation.
- [SpatialDE](https://doi.org/10.1038/nmeth.4636) — Svensson et al., Nature Methods 2018.
- [FlashS](https://github.com/cafferychen777/FlashS) — frequency-domain kernel testing for SVGs.
- [Moran's I](https://en.wikipedia.org/wiki/Moran%27s_I) — spatial autocorrelation statistic.
