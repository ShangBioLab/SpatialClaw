"""Tertiary Lymphoid Structures (TLS) detection and analysis.

Provides methods for detecting, characterizing, and analyzing TLS
in spatial omics data, including maturity assessment and
exploratory outcome association.

Supported methods:
  - tls_detection: Identify TLS regions
  - tls_typing: Classify TLS maturity and subtype
  - tls_ecosystem: Analyze TLS cellular composition
  - tls_clinical: Explore TLS association with outcome metadata

Input convention:
  - AnnData with spatial coordinates and cell type annotations
  - Optional: outcome metadata and survival data

Research-use boundary:
  - Outputs are exploratory analytical summaries only.
  - They do not diagnose disease, predict prognosis, recommend treatment, or
    provide clinical decision support.

Usage::

    from skills.spatial._lib.tls import (
        detect_tls_regions,
        analyze_tls_ecosystem,
        SUPPORTED_METHODS,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import distance

from .adata_utils import get_spatial_key, require_spatial_coords
from .dependency_manager import require

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = (
    "tls_detection", "tls_typing", "tls_ecosystem", "tls_clinical",
)


TLS_IMMUNE_MARKERS = {
    "human": {
        "B_cell": ["CD19", "CD20", "MS4A1", "CD79A", "CD79B"],
        "T_cell": ["CD3D", "CD3E", "CD4", "CD8A", "CD8B"],
        "follicular_helper": ["CXCR5", "BCL6", "PD1", "PDCD1", "ICOS"],
        "germinal_center": ["AICDA", "BCL6", "MEF2C"],
        "dendritic": ["CD21", "CD23", "FCER2", "CLEC4C"],
        "macrophage": ["CD68", "CD163", "LYZ"],
        "high_endothelial_venule": ["PNAd", "MECA79", "ICAM1", "VCAM1"],
    },
    "mouse": {
        "B_cell": ["Cd19", "Cd20", "Ms4a1", "Cd79a", "Cd79b"],
        "T_cell": ["Cd3d", "Cd3e", "Cd4", "Cd8a", "Cd8b"],
        "follicular_helper": ["Cxcr5", "Bcl6", "Pd1", "Pdcd1", "Icos"],
        "germinal_center": ["Aicda", "Bcl6", "Mef2c"],
        "dendritic": ["Cd21", "Cd23", "Fcer2", "Clec4c"],
        "macrophage": ["Cd68", "Cd163", "Lyz2"],
        "high_endothelial_venule": ["Pnad", "Icam1", "Vcam1"],
    },
}


def detect_tls_regions(
    adata,
    *,
    cell_type_key: str = "cell_type",
    b_cell_types: list[str] | None = None,
    t_cell_types: list[str] | None = None,
    min_b_cells: int = 20,
    min_t_cells: int = 10,
    neighborhood_radius: float = 100,
    species: str = "human",
) -> dict:
    """Detect TLS regions based on B/T cell co-localization.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with cell type annotations.
    cell_type_key : str
        Column with cell type labels.
    b_cell_types : list, optional
        Cell types to consider as B cells.
    t_cell_types : list, optional
        Cell types to consider as T cells.
    min_b_cells : int
        Minimum B cells for TLS candidate.
    min_t_cells : int
        Minimum T cells for TLS candidate.
    neighborhood_radius : float
        Radius for local neighborhood analysis.
    species : str
        Species for marker genes.

    Returns
    -------
    dict
        TLS detection results.
    """
    spatial_key = require_spatial_coords(adata)
    coords = adata.obsm[spatial_key]

    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in adata.obs")

    if b_cell_types is None:
        b_cell_types = _infer_b_cell_types(adata, cell_type_key)
    if t_cell_types is None:
        t_cell_types = _infer_t_cell_types(adata, cell_type_key)

    logger.info("Detecting TLS regions (B cells: %s, T cells: %s)", b_cell_types, t_cell_types)

    b_mask = adata.obs[cell_type_key].isin(b_cell_types).values
    t_mask = adata.obs[cell_type_key].isin(t_cell_types).values

    b_coords = coords[b_mask]
    t_coords = coords[t_mask]

    if len(b_coords) < min_b_cells or len(t_coords) < min_t_cells:
        logger.warning("Insufficient B/T cells for TLS detection")
        return {
            "n_tls": 0,
            "status": "insufficient_immune_cells",
            "n_b_cells": len(b_coords),
            "n_t_cells": len(t_coords),
        }

    from sklearn.cluster import DBSCAN
    b_clusters = DBSCAN(eps=neighborhood_radius, min_samples=min_b_cells).fit(b_coords)
    t_clusters = DBSCAN(eps=neighborhood_radius, min_samples=min_t_cells).fit(t_coords)

    tls_regions = []
    b_labels_unique = set(b_clusters.labels_) - {-1}
    t_labels_unique = set(t_clusters.labels_) - {-1}

    for b_label in b_labels_unique:
        b_cluster_mask = b_clusters.labels_ == b_label
        b_cluster_coords = b_coords[b_cluster_mask]
        b_center = b_cluster_coords.mean(axis=0)

        for t_label in t_labels_unique:
            t_cluster_mask = t_clusters.labels_ == t_label
            t_cluster_coords = t_coords[t_cluster_mask]
            t_center = t_cluster_coords.mean(axis=0)

            dist = np.linalg.norm(b_center - t_center)
            if dist < neighborhood_radius * 2:
                tls_center = (b_center + t_center) / 2
                tls_regions.append({
                    "center": tls_center,
                    "n_b_cells": int(b_cluster_mask.sum()),
                    "n_t_cells": int(t_cluster_mask.sum()),
                    "b_cluster": int(b_label),
                    "t_cluster": int(t_label),
                })

    adata.obs["tls_region"] = -1
    for i, tls in enumerate(tls_regions):
        center = tls["center"]
        dists = np.linalg.norm(coords - center, axis=1)
        in_tls = dists < neighborhood_radius * 1.5
        adata.obs.loc[in_tls, "tls_region"] = i

    adata.obs["tls_region"] = pd.Categorical(adata.obs["tls_region"].astype(str))

    adata.uns["tls_regions"] = tls_regions

    return {
        "n_tls": len(tls_regions),
        "tls_regions": tls_regions,
        "b_cell_types": b_cell_types,
        "t_cell_types": t_cell_types,
        "total_b_cells": int(b_mask.sum()),
        "total_t_cells": int(t_mask.sum()),
    }


def _infer_b_cell_types(adata, cell_type_key: str) -> list[str]:
    """Infer B cell types from data."""
    cell_types = adata.obs[cell_type_key].unique()
    b_keywords = ["B cell", "B-cell", "Bcell", "plasma", "memory B"]
    return [ct for ct in cell_types if any(kw in str(ct) for kw in b_keywords)]


def _infer_t_cell_types(adata, cell_type_key: str) -> list[str]:
    """Infer T cell types from data."""
    cell_types = adata.obs[cell_type_key].unique()
    t_keywords = ["T cell", "T-cell", "Tcell", "CD4", "CD8", "helper", "cytotoxic"]
    return [ct for ct in cell_types if any(kw in str(ct) for kw in t_keywords)]


def classify_tls_maturity(
    adata,
    *,
    tls_key: str = "tls_region",
    species: str = "human",
) -> dict:
    """Classify TLS maturity based on cellular composition.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with TLS labels.
    tls_key : str
        Column with TLS region labels.
    species : str
        Species for marker genes.

    Returns
    -------
    dict
        TLS maturity classification results.
    """
    if tls_key not in adata.obs.columns:
        raise ValueError(f"TLS key '{tls_key}' not found in adata.obs. Run detect_tls_regions first.")

    logger.info("Classifying TLS maturity...")

    tls_regions = adata.obs[tls_key].unique()
    tls_regions = [t for t in tls_regions if str(t) != "-1"]

    maturity_scores = {}
    markers = TLS_IMMUNE_MARKERS.get(species, TLS_IMMUNE_MARKERS["human"])

    for tls_id in tls_regions:
        mask = adata.obs[tls_key] == tls_id
        tls_adata = adata[mask]

        maturity_indicators = {}

        gc_genes = markers.get("germinal_center", [])
        gc_present = sum(1 for g in gc_genes if g in tls_adata.var_names)
        maturity_indicators["germinal_center_markers"] = gc_present

        fh_genes = markers.get("follicular_helper", [])
        fh_present = sum(1 for g in fh_genes if g in tls_adata.var_names)
        maturity_indicators["follicular_helper_markers"] = fh_present

        dc_genes = markers.get("dendritic", [])
        dc_present = sum(1 for g in dc_genes if g in tls_adata.var_names)
        maturity_indicators["dendritic_cell_markers"] = dc_present

        maturity_score = (
            0.4 * min(gc_present / max(len(gc_genes), 1), 1.0) +
            0.3 * min(fh_present / max(len(fh_genes), 1), 1.0) +
            0.3 * min(dc_present / max(len(dc_genes), 1), 1.0)
        )

        if maturity_score > 0.6:
            maturity_class = "mature"
        elif maturity_score > 0.3:
            maturity_class = "intermediate"
        else:
            maturity_class = "early"

        maturity_scores[tls_id] = {
            "maturity_score": float(maturity_score),
            "maturity_class": maturity_class,
            "indicators": maturity_indicators,
        }

    adata.uns["tls_maturity"] = maturity_scores

    maturity_dist = pd.Series([m["maturity_class"] for m in maturity_scores.values()]).value_counts().to_dict()

    return {
        "n_tls_classified": len(maturity_scores),
        "maturity_distribution": maturity_dist,
        "maturity_scores": maturity_scores,
    }


def analyze_tls_ecosystem(
    adata,
    *,
    tls_key: str = "tls_region",
    cell_type_key: str = "cell_type",
) -> dict:
    """Analyze cellular ecosystem within TLS regions.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with TLS labels.
    tls_key : str
        Column with TLS region labels.
    cell_type_key : str
        Column with cell type labels.

    Returns
    -------
    dict
        TLS ecosystem analysis results.
    """
    if tls_key not in adata.obs.columns:
        raise ValueError(f"TLS key '{tls_key}' not found in adata.obs")

    logger.info("Analyzing TLS ecosystems...")

    tls_regions = adata.obs[tls_key].unique()
    tls_regions = [t for t in tls_regions if str(t) != "-1"]

    ecosystem_analysis = {}

    for tls_id in tls_regions:
        mask = adata.obs[tls_key] == tls_id
        tls_adata = adata[mask]

        ct_composition = tls_adata.obs[cell_type_key].value_counts(normalize=True).to_dict()

        spatial_key = get_spatial_key(adata)
        if spatial_key:
            coords = tls_adata.obsm[spatial_key]
            if len(coords) > 2:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(coords)
                    area = hull.volume
                except Exception:
                    area = 0.0
            else:
                area = 0.0
        else:
            area = 0.0

        ecosystem_analysis[tls_id] = {
            "n_cells": int(mask.sum()),
            "cell_type_composition": ct_composition,
            "spatial_area": float(area),
            "cell_density": float(mask.sum() / max(area, 1)),
        }

    adata.uns["tls_ecosystem"] = ecosystem_analysis

    return {
        "n_tls_analyzed": len(ecosystem_analysis),
        "ecosystem_analysis": ecosystem_analysis,
    }


def associate_tls_with_outcome(
    adata,
    *,
    tls_key: str = "tls_region",
    outcome_key: str = "survival",
    time_key: str = "time",
    event_key: str = "event",
) -> dict:
    """Explore TLS presence association with outcome metadata.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with TLS and outcome metadata.
    tls_key : str
        Column with TLS region labels.
    outcome_key : str
        Type of outcome analysis.
    time_key : str
        Column with survival time.
    event_key : str
        Column with event indicator.

    Returns
    -------
    dict
        Exploratory TLS-outcome association results.
    """
    if tls_key not in adata.obs.columns:
        raise ValueError(f"TLS key '{tls_key}' not found in adata.obs")

    logger.info("Exploring TLS association with outcome metadata...")

    tls_present = (adata.obs[tls_key] != "-1").values

    if time_key in adata.obs.columns and event_key in adata.obs.columns:
        time = adata.obs[time_key].values
        event = adata.obs[event_key].values

        require("lifelines", feature="Survival analysis")
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        tls_positive = tls_present
        tls_negative = ~tls_present

        results = logrank_test(
            time[tls_positive], time[tls_negative],
            event_observed_A=event[tls_positive],
            event_observed_B=event[tls_negative],
        )

        kmf_tls = KaplanMeierFitter()
        kmf_tls.fit(time[tls_positive], event[tls_positive], label="TLS+")
        kmf_no_tls = KaplanMeierFitter()
        kmf_no_tls.fit(time[tls_negative], event[tls_negative], label="TLS-")

        adata.uns["tls_survival"] = {
            "logrank_p": results.p_value,
            "test_statistic": results.test_statistic,
            "tls_positive_median": kmf_tls.median_survival_time_,
            "tls_negative_median": kmf_no_tls.median_survival_time_,
        }

        return {
            "research_use_only": True,
            "interpretation": "cohort-level exploratory association; not prognostic prediction",
            "n_tls_positive": int(tls_positive.sum()),
            "n_tls_negative": int(tls_negative.sum()),
            "logrank_p_value": float(results.p_value),
            "test_statistic": float(results.test_statistic),
            "tls_positive_median_survival": float(kmf_tls.median_survival_time_) if kmf_tls.median_survival_time_ else None,
            "tls_negative_median_survival": float(kmf_no_tls.median_survival_time_) if kmf_no_tls.median_survival_time_ else None,
        }

    return {
        "research_use_only": True,
        "interpretation": "no survival data; not a clinical result",
        "n_tls_positive": int(tls_present.sum()),
        "n_tls_negative": int((~tls_present).sum()),
        "status": "no_survival_data",
    }


def run_tls(
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
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Choose from: {SUPPORTED_METHODS}")

    n_cells, n_genes = adata.n_obs, adata.n_vars
    logger.info("TLS analysis: %d cells, method=%s", n_cells, method)

    dispatch: dict[str, Any] = {
        "tls_detection": lambda: detect_tls_regions(
            adata, cell_type_key=cell_type_key, **kwargs
        ),
        "tls_typing": lambda: classify_tls_maturity(adata, **kwargs),
        "tls_ecosystem": lambda: analyze_tls_ecosystem(
            adata, cell_type_key=cell_type_key, **kwargs
        ),
        "tls_clinical": lambda: associate_tls_with_outcome(adata, **kwargs),
    }

    result = dispatch[method]()
    result["n_cells"] = n_cells
    result["n_genes"] = n_genes
    result["method"] = method

    return result
