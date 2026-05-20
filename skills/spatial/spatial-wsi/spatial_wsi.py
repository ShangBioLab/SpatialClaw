"""Whole-slide image processing and inference skill.

This skill provides WSI analysis capabilities:
- Slide loading and QC
- Tiling and patch extraction
- Tissue detection
- Patch feature extraction
- MIL-based slide classification

Python API::

    from skills.spatial._lib.wsi import run_wsi_pipeline

    result = run_wsi_pipeline("slide.svs", patch_size=256)
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.wsi import (
    SUPPORTED_FORMATS,
    load_wsi,
    tile_wsi,
    detect_tissue_regions,
    extract_patch_features,
    run_mil_inference,
    run_wsi_pipeline,
)

logger = logging.getLogger(__name__)


def run(
    slide_path: str,
    *,
    patch_size: int = 256,
    model_name: str = "resnet50",
    output_dir: str | None = None,
    run_mil: bool = False,
    n_classes: int = 2,
) -> dict:
    """Run WSI analysis pipeline.

    Parameters
    ----------
    slide_path : str
        Path to WSI file.
    patch_size : int
        Patch size for tiling.
    model_name : str
        Feature extraction model.
    output_dir : str, optional
        Output directory for patches and results.
    run_mil : bool
        Whether to run MIL classification.
    n_classes : int
        Number of classes for MIL.

    Returns
    -------
    dict
        Complete pipeline results.
    """
    logger.info("Starting WSI analysis: %s", slide_path)

    result = run_wsi_pipeline(
        slide_path,
        patch_size=patch_size,
        model_name=model_name,
        output_dir=output_dir,
    )

    if run_mil and result.get("patches", {}).get("features") is not None:
        features = result["patches"]["features"]
        mil_result = run_mil_inference(features, n_classes=n_classes)
        result["mil"] = mil_result

    return result


__all__ = [
    "run",
    "SUPPORTED_FORMATS",
    "load_wsi",
    "tile_wsi",
    "detect_tissue_regions",
    "extract_patch_features",
    "run_mil_inference",
    "run_wsi_pipeline",
]
