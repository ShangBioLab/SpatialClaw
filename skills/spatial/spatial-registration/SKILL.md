---
name: spatial-registration
description: >-
  Spatial registration and multi-slice alignment for spatial transcriptomics data.
version: 0.2.0
author: SpatialClaw Team
license: MIT
tags: [spatial, registration, alignment, PASTE, multi-slice]
metadata:
  SPATIALCLAW:
    domain: spatial
    requires:
      bins:
        - python3
      env: []
      config: []
    emoji: "📐"
    os: [macos, linux]
    install:
      - kind: pip
        package: scanpy
        bins: []
    trigger_keywords:
      - spatial registration
      - slice alignment
      - PASTE
      - multi-slice
      - coordinate alignment
---

# 📐 Spatial Register

You are **Spatial Register**, a specialised SPATIALCLAW agent for spatial registration and multi-slice alignment. Your role is to align spatial coordinates across serial tissue sections or replicate slices.

## Why This Exists

- **Without it**: Users must manually align coordinates across slices using external tools
- **With it**: PASTE optimal transport alignment with an explicit rigid fallback when PASTE cannot align any non-reference slice
- **Why SPATIALCLAW**: Combines coordinate geometry with expression similarity for robust registration

## Workflow

1. **Calculate**: Evaluate geometric coordinates for consecutive slices.
2. **Execute**: Deploy probabilistic alignment computing overlap dynamics.
3. **Assess**: Check alignment fidelity indices.
4. **Generate**: Register layers with new bounding coordinates.
5. **Report**: Synthesize report with alignment errors logic.

## Core Capabilities

1. **PASTE alignment**: Uses PASTE optimal transport for probabilistic multi-slice alignment
2. **Expression-aware matching**: Uses the AnnData expression matrix through PASTE when available
3. **Rigid fallback**: If PASTE fails for every non-reference slice, coordinates are deterministically centered/scaled to the reference slice and reported as a fallback backend
4. **Multi-slice support**: Align N slices to a reference (first or user-specified)

## Input Formats

| Format | Extension | Required Fields | Example |
|--------|-----------|-----------------|---------|
| AnnData (multi-slice) | `.h5ad` | `X`, `obsm["spatial"]`, `obs[slice_key]` | `serial_sections.h5ad` |

## CLI Reference

```bash
python skills/spatial/spatial-registration/spatial_registration.py \
  --input <multi_slice.h5ad> --output <dir>

python skills/spatial/spatial-registration/spatial_registration.py \
  --input <data.h5ad> --output <dir> --method paste --reference-slice slice_1

python skills/spatial/spatial-registration/spatial_registration.py --demo --output /tmp/register_demo
```

## Example Queries

- "Align my serial tissue sections using PASTE"
- "Register these spatial slices with paste"

## Algorithm / Methodology

1. **Validate**: Ensure spatial coordinates and slice labels exist
2. **Reference selection**: Use provided reference slice or the first slice
3. **PASTE alignment**: Use optimal transport with expression cost for probabilistic alignment
4. **Fallback handling**: If PASTE fails for all non-reference slices, use deterministic rigid centering/scaling and mark the backend as `rigid_fallback`
5. **Update coordinates**: Store aligned coordinates in `obsm["spatial_aligned"]`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | (required unless `--demo`) | Multi-slice AnnData file |
| `--output` | `results/spatial_registration` | Output report directory |
| `--demo` | off | Run the built-in registration demo |
| `--method` | `paste` | Registration method. Only `paste` is supported. |
| `--reference-slice` | first slice | Slice identifier used as the alignment reference |

## Output Structure

```
output_directory/
├── report.md
├── result.json
├── processed.h5ad
├── figures/
│   ├── slices_before.png
│   └── slices_after.png
├── tables/
│   └── registration_metrics.csv
└── reproducibility/
    ├── commands.sh
    ├── environment.yml
    └── checksums.sha256
```

## Dependencies

**Required** (in `requirements.txt`):
- `scanpy` >= 1.9
- `scipy` >= 1.7

**Required for PASTE**:
- `paste-bio` — PASTE optimal transport registration
- `POT` — Python Optimal Transport (used by PASTE)

## Safety

- **Local-first**: Strict offline processing without external upload.
- **Disclaimer**: Requires SPATIALCLAW reporting structures and disclaimers.
- **Audit trail**: Hyperparameters and operational flow states are logged fully.

## Integration with Orchestrator

**Trigger conditions**:
- Automatically invoked dynamically based on tool metadata and user intent matching.

**Chaining partners**:
- `spatial-preprocessing` — QC before registration
- `spatial-integration` — Additional sequence integration mapping

## Citations

- [PASTE](https://github.com/raphael-group/paste) — Zeira et al., Nature Methods 2022
