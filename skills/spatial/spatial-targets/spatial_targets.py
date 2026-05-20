"""Target discovery and prioritization skill.

This skill provides target discovery capabilities:
- Spatial biomarker discovery
- Target prioritization
- Drug annotation
- Research-use evidence summary generation

Outputs are exploratory research summaries only and are not therapy
recommendations or clinical decision support.

Python API::

    from skills.spatial._lib.targets import run_targets

    result = run_targets(adata, method="biomarker_discovery", domain_key="spatial_domain")
"""

from __future__ import annotations

import logging
from typing import Any

from skills.spatial._lib.targets import (
    SUPPORTED_METHODS,
    discover_biomarkers,
    prioritize_targets,
    match_drugs,
    generate_actionable_report,
    run_targets,
)

logger = logging.getLogger(__name__)


def run(
    adata,
    *,
    method: str = "biomarker_discovery",
    domain_key: str = "spatial_domain",
    **kwargs,
) -> dict:
    """Run target discovery pipeline.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data.
    method : str
        Analysis method.
    domain_key : str
        Column with domain labels.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    dict
        Analysis results.
    """
    logger.info("Starting target discovery (method=%s)", method)

    result = run_targets(
        adata,
        method=method,
        domain_key=domain_key,
        **kwargs,
    )

    return result


__all__ = [
    "run",
    "SUPPORTED_METHODS",
    "discover_biomarkers",
    "prioritize_targets",
    "match_drugs",
    "generate_actionable_report",
    "run_targets",
]
