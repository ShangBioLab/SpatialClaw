"""Cross-modal translation and virtual omics generation skill.

This skill provides cross-modal translation capabilities:
- Image to transcriptomics prediction
- Image to proteomics prediction
- Super-resolution reconstruction
- Cross-slice interpolation

Python API::

    from skills.spatial._lib.translation import run_translation

    result = run_translation(adata, images, method="hist2st")
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.translation import (
    SUPPORTED_METHODS,
    predict_expression_from_histology,
    predict_proteomics_from_histology,
    reconstruct_super_resolution,
    interpolate_cross_slice,
    run_translation,
)

logger = logging.getLogger(__name__)


def run(
    adata,
    images: list | None = None,
    *,
    method: str = "hist2st",
    target_genes: list[str] | None = None,
    **kwargs,
) -> dict:
    """Run cross-modal translation pipeline.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data.
    images : list, optional
        List of histology images.
    method : str
        Translation method.
    target_genes : list, optional
        Target genes for expression prediction.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    dict
        Translation results.
    """
    logger.info("Starting cross-modal translation (method=%s)", method)

    result = run_translation(
        adata,
        images=images,
        method=method,
        target_genes=target_genes,
        **kwargs,
    )

    return result


__all__ = [
    "run",
    "SUPPORTED_METHODS",
    "predict_expression_from_histology",
    "predict_proteomics_from_histology",
    "reconstruct_super_resolution",
    "interpolate_cross_slice",
    "run_translation",
]
