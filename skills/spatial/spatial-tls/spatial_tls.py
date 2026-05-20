"""Tertiary Lymphoid Structures (TLS) analysis skill.

This skill provides TLS analysis capabilities:
- TLS region detection
- TLS maturity classification
- TLS ecosystem analysis
- Exploratory TLS-outcome association

Outputs are research-use cohort summaries only and are not diagnostic,
prognostic, or treatment recommendation tools.

Python API::

    from skills.spatial._lib.tls import run_tls

    result = run_tls(adata, method="tls_detection", cell_type_key="cell_type")
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.tls import (
    SUPPORTED_METHODS,
    detect_tls_regions,
    classify_tls_maturity,
    analyze_tls_ecosystem,
    associate_tls_with_outcome,
    run_tls,
)

logger = logging.getLogger(__name__)


def run(
    adata,
    *,
    method: str = "tls_detection",
    cell_type_key: str = "cell_type",
    **kwargs,
) -> dict:
    """Run TLS analysis pipeline.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data.
    method : str
        Analysis method.
    cell_type_key : str
        Column with cell type labels.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    dict
        Analysis results.
    """
    logger.info("Starting TLS analysis (method=%s)", method)

    result = run_tls(
        adata,
        method=method,
        cell_type_key=cell_type_key,
        **kwargs,
    )

    return result


__all__ = [
    "run",
    "SUPPORTED_METHODS",
    "detect_tls_regions",
    "classify_tls_maturity",
    "analyze_tls_ecosystem",
    "associate_tls_with_outcome",
    "run_tls",
]
