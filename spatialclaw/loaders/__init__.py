"""Spatial data loading entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SPATIAL_EXTENSIONS = {".h5ad", ".h5", ".zarr"}


def detect_spatial_format(path: str | Path) -> str:
    """Return a lightweight spatial format label from a file or directory path."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".zarr" or path.is_dir():
        return "zarr" if suffix == ".zarr" else "directory"
    if suffix in {".h5ad", ".h5"}:
        return suffix.lstrip(".")
    return "spatial"


def load_spatial_analysis_data(
    path: str | Path,
    data_type: str | None = None,
) -> Any:
    """Load spatial transcriptomics data from supported local formats."""
    from .spatial import load_spatial_data

    return load_spatial_data(Path(path), data_type)


__all__ = [
    "SPATIAL_EXTENSIONS",
    "detect_spatial_format",
    "load_spatial_analysis_data",
]
