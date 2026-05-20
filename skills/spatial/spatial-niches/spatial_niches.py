"""Spatial niche identification and characterization skill.

This skill provides comprehensive niche analysis capabilities:
- Niche identification using multiple methods (CellCharter, Leiden, NiCo)
- Niche characterization by composition and markers
- Cross-condition niche comparison
- Niche trajectory analysis

Python API::

    from skills.spatial._lib.niches import run_niche_identification

    summary = run_niche_identification(adata, method="leiden_niche", cell_type_key="cell_type")
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.niches import (
    SUPPORTED_METHODS,
    characterize_niches,
    compare_niches,
    run_niche_identification,
)

logger = logging.getLogger(__name__)


def run(
    adata,
    *,
    method: str = "leiden_niche",
    cell_type_key: str = "cell_type",
    n_niches: int = 10,
    n_neighbors: int = 10,
    resolution: float = 1.0,
    characterize: bool = True,
    compare_conditions: bool = False,
    condition_key: str = "condition",
) -> dict:
    """Run spatial niche analysis pipeline.

    Parameters
    ----------
    adata : AnnData
        Preprocessed spatial data with cell type annotations.
    method : str
        Niche identification method: cellcharter, leiden_niche, nico.
    cell_type_key : str
        Column in adata.obs containing cell type labels.
    n_niches : int
        Target number of niches (for cellcharter).
    n_neighbors : int
        Number of spatial neighbors for neighborhood computation.
    resolution : float
        Resolution for Leiden-based methods.
    characterize : bool
        Whether to characterize niches after identification.
    compare_conditions : bool
        Whether to compare niches across conditions.
    condition_key : str
        Column in adata.obs containing condition labels.

    Returns
    -------
    dict
        Summary with niche labels and statistics.
    """
    logger.info("Starting spatial niche analysis (method=%s)", method)

    result = run_niche_identification(
        adata,
        method=method,
        cell_type_key=cell_type_key,
        n_niches=n_niches,
        n_neighbors=n_neighbors,
        resolution=resolution,
    )

    if characterize:
        logger.info("Characterizing niches...")
        char_result = characterize_niches(
            adata,
            niche_key="niche",
            cell_type_key=cell_type_key,
        )
        result["characterization"] = char_result

    if compare_conditions and condition_key in adata.obs.columns:
        logger.info("Comparing niches across conditions...")
        comp_result = compare_niches(
            adata,
            niche_key="niche",
            condition_key=condition_key,
        )
        result["comparison"] = comp_result

    return result


__all__ = [
    "run",
    "SUPPORTED_METHODS",
    "run_niche_identification",
    "characterize_niches",
    "compare_niches",
]
