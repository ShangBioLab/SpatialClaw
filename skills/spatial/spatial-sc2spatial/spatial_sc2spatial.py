"""scRNA-seq to spatial transcriptomics mapping skill.

This skill provides methods for transferring information between
scRNA-seq and spatial transcriptomics data:
- Cell type label transfer
- Cell localization (Tangram, CellTrek)
- Gene expression imputation (SpaGE)
- Joint embedding (gimVI)

Python API::

    from skills.spatial._lib.sc2spatial import run_sc2spatial_mapping

    summary = run_sc2spatial_mapping(adata_sp, adata_ref, method="label_transfer")
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.sc2spatial import (
    SUPPORTED_METHODS,
    run_sc2spatial_mapping,
)

logger = logging.getLogger(__name__)


def run(
    adata_sp,
    adata_ref,
    *,
    method: str = "tangram",
    cell_type_key: str = "cell_type",
    **kwargs,
) -> dict:
    """Run scRNA-seq to spatial mapping.

    Parameters
    ----------
    adata_sp : AnnData
        Spatial transcriptomics data.
    adata_ref : AnnData
        Reference scRNA-seq data with cell type annotations.
    method : str
        Mapping method: tangram, spage, gimvi, celltrek, label_transfer.
    cell_type_key : str
        Column in adata_ref.obs containing cell type labels.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    dict
        Summary with mapping results and statistics.
    """
    logger.info("Starting scRNA-seq to spatial mapping (method=%s)", method)

    result = run_sc2spatial_mapping(
        adata_sp,
        adata_ref,
        method=method,
        cell_type_key=cell_type_key,
        **kwargs,
    )

    return result


__all__ = [
    "run",
    "SUPPORTED_METHODS",
    "run_sc2spatial_mapping",
]
