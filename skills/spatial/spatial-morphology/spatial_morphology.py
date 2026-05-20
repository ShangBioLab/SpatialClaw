"""Morphology feature extraction and image-omics fusion skill.

This skill provides morphology analysis capabilities:
- Deep feature extraction from images
- Handcrafted feature extraction
- Image-omics fusion
- IHC/IF quantification

Python API::

    from skills.spatial._lib.morphology import run_morphology

    result = run_morphology(adata, images, method="deep_features")
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.morphology import (
    SUPPORTED_METHODS,
    extract_deep_features,
    extract_handcrafted_features,
    fuse_image_omics,
    quantify_ihc,
    run_morphology,
)

logger = logging.getLogger(__name__)


def run(
    adata,
    images: list | None = None,
    *,
    method: str = "deep_features",
    model_name: str = "resnet50",
    fusion_method: str = "concatenation",
) -> dict:
    """Run morphology analysis pipeline.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data.
    images : list, optional
        List of patch images corresponding to spots.
    method : str
        Feature extraction method.
    model_name : str
        Deep learning model for feature extraction.
    fusion_method : str
        Method for image-omics fusion.

    Returns
    -------
    dict
        Analysis results.
    """
    logger.info("Starting morphology analysis (method=%s)", method)

    result = run_morphology(
        adata,
        images=images,
        method=method,
        model_name=model_name,
        fusion_method=fusion_method,
    )

    return result


__all__ = [
    "run",
    "SUPPORTED_METHODS",
    "extract_deep_features",
    "extract_handcrafted_features",
    "fuse_image_omics",
    "quantify_ihc",
    "run_morphology",
]
