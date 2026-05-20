"""Spatial regulatory network inference skill.

This skill provides regulatory network analysis in spatial context:
- TF activity inference
- Regulon identification (SCENIC, SpaGRN)
- Multi-omics regulatory inference (GLUE)
- Spatial regulatory pattern analysis

Python API::

    from skills.spatial._lib.regulation import run_regulation

    summary = run_regulation(adata, method="tf_activity", species="human")
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.regulation import (
    SUPPORTED_METHODS,
    run_regulation,
)

logger = logging.getLogger(__name__)


def run(
    adata,
    *,
    method: str = "tf_activity",
    species: str = "human",
    tf_list: list[str] | None = None,
    n_regulons: int = 50,
    atac_adata = None,
) -> dict:
    """Run spatial regulatory network inference.

    Parameters
    ----------
    adata : AnnData
        Preprocessed spatial data.
    method : str
        Regulatory inference method: spagrn, scenic, scenic_plus, glue, tf_activity.
    species : str
        Species for TF database: human or mouse.
    tf_list : list, optional
        Custom TF list for analysis.
    n_regulons : int
        Target number of regulons.
    atac_adata : AnnData, optional
        ATAC data for multi-omics methods.

    Returns
    -------
    dict
        Summary with regulon activities and statistics.
    """
    logger.info("Starting spatial regulatory analysis (method=%s)", method)

    result = run_regulation(
        adata,
        method=method,
        species=species,
        tf_list=tf_list,
        n_regulons=n_regulons,
        atac_adata=atac_adata,
    )

    return result


__all__ = [
    "run",
    "SUPPORTED_METHODS",
    "run_regulation",
]
