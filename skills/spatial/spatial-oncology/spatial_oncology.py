"""Oncology-focused spatial analysis skill.

This skill provides oncology analysis capabilities:
- Tumor ecosystem mapping (core/edge/interface)
- Supervised niche discovery
- Exploratory cohort outcome association
- Response-associated pattern analysis

Outputs are research-use summaries only and are not diagnostic, prognostic, or
treatment recommendation tools.

Python API::

    from skills.spatial._lib.oncology import run_oncology

    result = run_oncology(adata, method="tumor_ecosystem", cell_type_key="cell_type")
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.oncology import (
    SUPPORTED_METHODS,
    map_tumor_ecosystem,
    discover_supervised_niches,
    predict_clinical_outcome,
    analyze_therapy_response,
    run_oncology,
)

logger = logging.getLogger(__name__)


def run(
    adata,
    *,
    method: str = "tumor_ecosystem",
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    **kwargs,
) -> dict:
    """Run oncology analysis pipeline.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data.
    method : str
        Analysis method.
    cell_type_key : str
        Column with cell type labels.
    condition_key : str
        Column with condition labels.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    dict
        Analysis results.
    """
    logger.info("Starting oncology analysis (method=%s)", method)

    result = run_oncology(
        adata,
        method=method,
        cell_type_key=cell_type_key,
        condition_key=condition_key,
        **kwargs,
    )

    return result


__all__ = [
    "run",
    "SUPPORTED_METHODS",
    "map_tumor_ecosystem",
    "discover_supervised_niches",
    "predict_clinical_outcome",
    "analyze_therapy_response",
    "run_oncology",
]
