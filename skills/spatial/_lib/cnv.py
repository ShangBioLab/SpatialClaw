"""Spatial CNV inference functions.

Provides inferCNVpy for copy number variation analysis.

Input matrix convention:
  - infercnvpy: adata.X (log-normalized) — computes log-fold-change vs reference

Usage::

    from skills.spatial._lib.cnv import run_cnv, SUPPORTED_METHODS
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from .dependency_manager import require

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("infercnvpy",)
COUNT_BASED_METHODS: tuple[str, ...] = ()


def validate_reference(adata, reference_key: str | None, reference_cat: list[str]) -> None:
    """Validate that reference key and categories exist in adata."""
    if reference_key is None:
        return
    if reference_key not in adata.obs.columns:
        raise ValueError(f"'{reference_key}' not in adata.obs. Available: {list(adata.obs.columns)}")
    avail = set(adata.obs[reference_key].unique())
    missing = set(reference_cat) - avail
    if missing:
        raise ValueError(f"Categories {sorted(missing)} not in '{reference_key}'. Available: {sorted(avail)}")


def run_infercnvpy(adata, *, reference_key: str | None = None, reference_cat: list[str] | None = None,
                   window_size: int = 100, step: int = 10, dynamic_threshold: float | None = 1.5) -> dict:
    """Infer CNV using inferCNVpy.

    Uses ``adata.X`` (log-normalized) — inferCNVpy subtracts the reference
    expression in log-space (equivalent to log-fold-change) and smooths across
    genomic windows.  The method explicitly requires normalized, log-transformed
    input per its documentation.

    Also requires gene genomic position annotations (chromosome, start, end)
    in ``adata.var`` and optionally a reference cell group in ``adata.obs``.
    """
    require("infercnvpy", feature="CNV inference")
    import infercnvpy as cnv

    req_cols = {"chromosome", "start", "end"}
    if not req_cols.issubset(adata.var.columns):
        missing = req_cols - set(adata.var.columns)
        raise ValueError(
            f"inferCNVpy requires genomic annotations. Missing adata.var columns: {list(missing)}. "
            "Please ensure gene positions are mapped before running CNV."
        )

    logger.info("Running inferCNVpy on adata.X (log-normalized), window=%d, step=%d", window_size, step)
    
    cnv.tl.infercnv(
        adata, 
        reference_key=reference_key, 
        reference_cat=reference_cat,
        window_size=window_size, 
        step=step, 
        dynamic_threshold=dynamic_threshold
    )
    
    if "cnv_leiden" not in adata.obs:
        logger.info("Computing CNV Leiden clusters required by inferCNVpy cnv_score...")
        cnv.tl.pca(adata)
        cnv.pp.neighbors(adata)
        cnv.tl.leiden(adata)

    logger.info("Computing overall CNV anomaly scores per cell...")
    cnv.tl.cnv_score(adata)

    cnv_score_col = "cnv_score" if "cnv_score" in adata.obs.columns else None
    
    if cnv_score_col:
        # Fill any NaNs that might have emerged during sliding window edge cases
        if adata.obs[cnv_score_col].isna().any():
            adata.obs[cnv_score_col] = adata.obs[cnv_score_col].fillna(0.0)
            
        mean_score = float(adata.obs[cnv_score_col].mean())
        threshold = float(adata.obs[cnv_score_col].quantile(0.9))
        high_cnv_pct = float((adata.obs[cnv_score_col] > threshold).mean() * 100)
    else:
        mean_score = 0.0
        high_cnv_pct = 0.0

    return {
        "method": "infercnvpy", 
        "n_genes": adata.n_vars,
        "mean_cnv_score": float(f"{mean_score:.4f}"), 
        "high_cnv_fraction_pct": float(f"{high_cnv_pct:.2f}"),
        "cnv_score_key": cnv_score_col,
    }


def run_cnv(adata, *, method: str = "infercnvpy", reference_key: str | None = None,
            reference_cat: list[str] | str | None = None, window_size: int = 100, step: int = 10) -> dict:
    """Run CNV inference. Returns summary dict.

    Input matrix:
      - infercnvpy: adata.X (log-normalized)
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown CNV method '{method}'. Choose from: {SUPPORTED_METHODS}")

    # Safely cast string to list for the reference categories to prevent iteration bugs
    if isinstance(reference_cat, str):
        reference_cat = [reference_cat]
    reference_cat = reference_cat or []

    validate_reference(adata, reference_key, reference_cat)

    logger.info("Starting CNV inference workflow using method '%s' (%d cells, %d genes)...", method, adata.n_obs, adata.n_vars)

    if method == "infercnvpy":
        result = run_infercnvpy(adata, reference_key=reference_key, reference_cat=reference_cat,
                                window_size=window_size, step=step)
    else:
        raise NotImplementedError(f"Handler for method '{method}' is not implemented.")
                                
    return {"n_cells": adata.n_obs, "n_genes": adata.n_vars, **result}
