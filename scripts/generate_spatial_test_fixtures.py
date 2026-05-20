#!/usr/bin/env python3
"""Generate extra fixtures for spatial skill testing.

This script does not run any skills. It only prepares reusable input files
for tests that need metadata, multi-sample structure, multi-slice structure,
velocity layers, or tiny image assets.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ST = ROOT / "data" / "deconvolution" / "Human_Lymph_Node" / "ST.h5ad"
DEFAULT_SC = ROOT / "data" / "deconvolution" / "Human_Lymph_Node" / "scRNA.h5ad"
OUT_DIR = ROOT / "data" / "test_fixtures" / "spatial"


def _read(path: Path) -> ad.AnnData:
    if not path.exists():
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def _ensure_counts_layer(adata: ad.AnnData) -> None:
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()


def _ensure_spatial(adata: ad.AnnData) -> None:
    if "spatial" not in adata.obsm:
        raise ValueError("Expected obsm['spatial'] in input AnnData")


def _make_categorical_labels(n: int, labels: list[str]) -> np.ndarray:
    idx = np.arange(n) % len(labels)
    return np.array([labels[i] for i in idx], dtype=object)


def build_multi_sample_raw(st_path: Path) -> Path:
    base = _read(st_path)
    _ensure_spatial(base)
    _ensure_counts_layer(base)

    copies = []
    configs = [
        ("control", "sample_a", "batch_a", (0.0, 0.0)),
        ("control", "sample_b", "batch_b", (1500.0, 0.0)),
        ("treated", "sample_c", "batch_c", (0.0, 1500.0)),
        ("treated", "sample_d", "batch_d", (1500.0, 1500.0)),
    ]

    for cond, sample_id, batch, offset in configs:
        adata = base.copy()
        adata.obs["condition"] = cond
        adata.obs["sample_id"] = sample_id
        adata.obs["batch"] = batch
        adata.obs["cell_type"] = _make_categorical_labels(
            adata.n_obs, ["B_cell", "T_cell", "Myeloid", "Stromal"]
        )
        coords = np.asarray(adata.obsm["spatial"]).copy().astype(float)
        coords[:, 0] += offset[0]
        coords[:, 1] += offset[1]
        adata.obsm["spatial"] = coords
        copies.append(adata)

    merged = ad.concat(
        copies,
        join="outer",
        label="source_dataset",
        keys=[cfg[1] for cfg in configs],
        index_unique="-",
    )
    _ensure_counts_layer(merged)
    out = OUT_DIR / "multi_sample_raw.h5ad"
    merged.write_h5ad(out)
    return out


def build_multi_slice_raw(st_path: Path) -> Path:
    base = _read(st_path)
    _ensure_spatial(base)
    _ensure_counts_layer(base)

    copies = []
    offsets = [("slice_1", (0.0, 0.0)), ("slice_2", (200.0, 50.0)), ("slice_3", (400.0, 100.0))]
    for slice_name, offset in offsets:
        adata = base.copy()
        adata.obs["slice_id"] = slice_name
        adata.obs["cell_type"] = _make_categorical_labels(
            adata.n_obs, ["B_cell", "T_cell", "Tumor", "Stromal"]
        )
        coords = np.asarray(adata.obsm["spatial"]).copy().astype(float)
        coords[:, 0] += offset[0]
        coords[:, 1] += offset[1]
        adata.obsm["spatial"] = coords
        copies.append(adata)

    merged = ad.concat(
        copies,
        join="outer",
        label="slice_source",
        keys=[name for name, _ in offsets],
        index_unique="-",
    )
    _ensure_counts_layer(merged)
    out = OUT_DIR / "multi_slice_raw.h5ad"
    merged.write_h5ad(out)
    return out


def build_velocity_fixture(st_path: Path) -> Path:
    adata = _read(st_path)
    _ensure_spatial(adata)
    counts = np.asarray(adata.X).astype(float)
    adata.layers["spliced"] = counts * 0.7
    adata.layers["unspliced"] = counts * 0.3
    _ensure_counts_layer(adata)
    adata.obs["cell_type"] = _make_categorical_labels(
        adata.n_obs, ["progenitor", "intermediate", "terminal"]
    )
    out = OUT_DIR / "velocity_fixture.h5ad"
    adata.write_h5ad(out)
    return out


def build_library_spatial_fixture(st_path: Path) -> Path:
    adata = _read(st_path)
    _ensure_spatial(adata)
    _ensure_counts_layer(adata)
    adata.obs["cell_type"] = _make_categorical_labels(
        adata.n_obs, ["B_cell", "T_cell", "Tumor", "Myeloid"]
    )
    adata.obs["condition"] = _make_categorical_labels(adata.n_obs, ["control", "treated"])
    adata.obs["response"] = _make_categorical_labels(adata.n_obs, ["responder", "non_responder"])
    adata.obs["spatial_domain"] = _make_categorical_labels(adata.n_obs, ["domain_0", "domain_1", "domain_2"])
    out = OUT_DIR / "library_spatial_fixture.h5ad"
    adata.write_h5ad(out)
    return out


def build_sc_reference_copy(sc_path: Path) -> Path:
    adata = _read(sc_path)
    if "cell_type" not in adata.obs:
        adata.obs["cell_type"] = _make_categorical_labels(
            adata.n_obs, ["B_cell", "T_cell", "Myeloid", "Stromal"]
        )
    out = OUT_DIR / "sc_reference_fixture.h5ad"
    adata.write_h5ad(out)
    return out


def build_tiny_images() -> tuple[Path, Path]:
    x = np.linspace(0, 255, 128, dtype=np.uint8)
    y = np.linspace(0, 255, 128, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    img = np.stack([xx, yy, ((xx.astype(int) + yy.astype(int)) // 2).astype(np.uint8)], axis=-1)

    png_path = OUT_DIR / "tiny_histology.png"
    tiff_path = OUT_DIR / "tiny_slide.tiff"
    Image.fromarray(img).save(png_path)
    Image.fromarray(img).save(tiff_path)
    return png_path, tiff_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    created = [
        build_multi_sample_raw(DEFAULT_ST),
        build_multi_slice_raw(DEFAULT_ST),
        build_velocity_fixture(DEFAULT_ST),
        build_library_spatial_fixture(DEFAULT_ST),
        build_sc_reference_copy(DEFAULT_SC),
    ]
    created.extend(build_tiny_images())

    print("Generated spatial test fixtures:")
    for path in created:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
