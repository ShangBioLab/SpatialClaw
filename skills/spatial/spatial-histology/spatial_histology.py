"""Spatial histology image analysis skill.

This skill provides histology image analysis capabilities:
- Tissue segmentation (Otsu, Watershed)
- Nucleus detection (Cellpose, StarDist, SAM)
- Cell segmentation (Cellpose, HoVer-Net)
- Patch classification

Python API::

    from skills.spatial._lib.histology import run_histology

    result = run_histology(image, task="nucleus_detection", method="cellpose")
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.histology import (
    SUPPORTED_METHODS,
    detect_nuclei,
    segment_cells,
    segment_tissue,
    run_histology,
)

logger = logging.getLogger(__name__)


def run(
    image,
    *,
    task: str = "nucleus_detection",
    method: str = "cellpose",
    **kwargs,
) -> dict:
    """Run histology analysis pipeline.

    Parameters
    ----------
    image : array or path
        Input image.
    task : str
        Analysis task: tissue_segmentation, nucleus_detection,
        cell_segmentation, patch_classification.
    method : str
        Method for the specified task.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    dict
        Results including masks, detections, and statistics.
    """
    logger.info("Starting histology analysis (task=%s, method=%s)", task, method)

    result = run_histology(image, task=task, method=method, **kwargs)

    return result


__all__ = [
    "run",
    "SUPPORTED_METHODS",
    "segment_tissue",
    "detect_nuclei",
    "segment_cells",
    "run_histology",
]
