"""Oncology-focused spatial analysis for cancer cohorts.

Provides methods for tumor ecosystem analysis, exploratory cohort outcome
association, and response-associated pattern summaries.

Supported methods:
  - tumor_ecosystem: Tumor core/edge/interface analysis
  - supervised_niche: Condition-specific niche discovery
  - clinical_prediction: Exploratory survival/outcome association
  - therapy_response: Response-associated spatial pattern summary

Input convention:
  - AnnData with spatial coordinates, cell types, and clinical metadata
  - Optional: outcome and treatment metadata

Research-use boundary:
  - Outputs are exploratory analytical summaries only.
  - They do not diagnose disease, predict individual patient outcomes, recommend
    treatment, or provide clinical decision support.

Usage::

    from skills.spatial._lib.oncology import (
        map_tumor_ecosystem,
        predict_clinical_outcome,
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
    "tumor_ecosystem", "supervised_niche", "clinical_prediction", "therapy_response",
)


def map_tumor_ecosystem(
    adata,
    *,
    tumor_cell_types: list[str] | None = None,
    cell_type_key: str = "cell_type",
    interface_width: int = 50,
) -> dict:
    """Map tumor ecosystem including core, edge, and interface regions.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with cell type annotations.
    tumor_cell_types : list, optional
        Cell types to consider as tumor. Default: common tumor types.
    cell_type_key : str
        Column in adata.obs with cell type labels.
    interface_width : int
        Width in pixels for interface zone detection.

    Returns
    -------
    dict
        Ecosystem mapping results.
    """
    spatial_key = require_spatial_coords(adata)
    coords = adata.obsm[spatial_key]

    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in adata.obs")

    if tumor_cell_types is None:
        tumor_cell_types = _infer_tumor_cell_types(adata, cell_type_key)

    logger.info("Mapping tumor ecosystem (tumor types: %s)", tumor_cell_types)

    tumor_mask = adata.obs[cell_type_key].isin(tumor_cell_types).values
    tumor_coords = coords[tumor_mask]

    if len(tumor_coords) < 10:
        logger.warning("Too few tumor cells detected")
        return {"n_tumor_cells": len(tumor_coords), "status": "insufficient_tumor"}

    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(tumor_coords)
        tumor_boundary = tumor_coords[hull.vertices]
    except Exception:
        logger.warning("Could not compute convex hull, using boundary approximation")
        tumor_boundary = tumor_coords

    from scipy.spatial import distance_matrix
    dist_to_tumor = distance_matrix(coords, tumor_coords).min(axis=1)

    core_threshold = np.percentile(dist_to_tumor[tumor_mask], 50)
    edge_threshold = interface_width

    ecosystem_labels = np.array(["stroma"] * adata.n_obs, dtype=object)
    ecosystem_labels[tumor_mask & (dist_to_tumor <= core_threshold)] = "tumor_core"
    ecosystem_labels[tumor_mask & (dist_to_tumor > core_threshold)] = "tumor_edge"
    ecosystem_labels[~tumor_mask & (dist_to_tumor <= edge_threshold)] = "interface"

    adata.obs["tumor_ecosystem"] = pd.Categorical(ecosystem_labels)

    interface_mask = ecosystem_labels == "interface"
    immune_cell_types = ["T cells", "B cells", "Macrophages", "NK cells", "Dendritic cells"]
    immune_mask = adata.obs[cell_type_key].isin(immune_cell_types).values

    interface_immune_fraction = (interface_mask & immune_mask).sum() / max(interface_mask.sum(), 1)

    return {
        "n_tumor_cells": int(tumor_mask.sum()),
        "n_core_cells": int((ecosystem_labels == "tumor_core").sum()),
        "n_edge_cells": int((ecosystem_labels == "tumor_edge").sum()),
        "n_interface_cells": int(interface_mask.sum()),
        "n_stroma_cells": int((ecosystem_labels == "stroma").sum()),
        "interface_immune_fraction": float(interface_immune_fraction),
        "tumor_cell_types": tumor_cell_types,
    }


def _infer_tumor_cell_types(adata, cell_type_key: str) -> list[str]:
    """Infer tumor cell types from data."""
    cell_types = adata.obs[cell_type_key].value_counts()
    tumor_keywords = ["tumor", "cancer", "malignant", "carcinoma", "melanoma"]

    tumor_types = []
    for ct in cell_types.index:
        if any(kw in str(ct).lower() for kw in tumor_keywords):
            tumor_types.append(ct)

    if not tumor_types:
        tumor_types = [cell_types.index[0]]

    return tumor_types


def discover_supervised_niches(
    adata,
    *,
    condition_key: str = "condition",
    cell_type_key: str = "cell_type",
    n_niches: int = 10,
    n_neighbors: int = 10,
) -> dict:
    """Discover condition-specific niches using supervised approach.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data.
    condition_key : str
        Column in adata.obs with condition labels.
    cell_type_key : str
        Column in adata.obs with cell type labels.
    n_niches : int
        Target number of niches.
    n_neighbors : int
        Number of neighbors for composition computation.

    Returns
    -------
    dict
        Supervised niche discovery results.
    """
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in adata.obs")

    logger.info("Discovering supervised niches (n_niches=%d)", n_niches)

    from .niches import _compute_neighborhood_composition
    spatial_key = get_spatial_key(adata)
    if spatial_key is None:
        raise ValueError("No spatial coordinates found")

    composition = _compute_neighborhood_composition(
        adata, cell_type_key=cell_type_key, n_neighbors=n_neighbors, spatial_key=spatial_key
    )

    adata.obsm["niche_composition"] = composition

    conditions = adata.obs[condition_key].unique()
    condition_niches = {}

    for cond in conditions:
        mask = adata.obs[condition_key] == cond
        comp_subset = composition[mask]

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(n_niches, mask.sum()), random_state=42)
        niche_labels = np.zeros(mask.sum(), dtype=int)
        niche_labels = kmeans.fit_predict(comp_subset)

        condition_niches[cond] = {
            "n_niches": len(np.unique(niche_labels)),
            "niche_sizes": pd.Series(niche_labels).value_counts().to_dict(),
        }

    adata.obs["supervised_niche"] = "unknown"
    for cond in conditions:
        mask = adata.obs[condition_key] == cond
        niche_labels = KMeans(n_clusters=n_niches, random_state=42).fit_predict(composition[mask])
        adata.obs.loc[mask, "supervised_niche"] = [f"{cond}_niche_{n}" for n in niche_labels]

    return {
        "n_conditions": len(conditions),
        "conditions": list(conditions),
        "n_niches_per_condition": condition_niches,
    }


def predict_clinical_outcome(
    adata,
    *,
    outcome_key: str = "survival",
    time_key: str = "time",
    event_key: str = "event",
    features: str = "pca",
    model_type: str = "cox",
) -> dict:
    """Explore cohort outcome associations from spatial features.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with outcome metadata.
    outcome_key : str
        Type of outcome: survival, response, subtype.
    time_key : str
        Column with survival time.
    event_key : str
        Column with event indicator.
    features : str
        Feature type: pca, niche, ecosystem.
    model_type : str
        Model: cox, random_forest, neural_net.

    Returns
    -------
    dict
        Exploratory outcome association results.
    """
    logger.info(
        "Running exploratory outcome association (outcome=%s, model=%s)",
        outcome_key,
        model_type,
    )

    if features == "pca":
        if "X_pca" not in adata.obsm:
            raise ValueError("X_pca not found in adata.obsm")
        X = adata.obsm["X_pca"]
    elif features == "niche":
        if "niche_composition" not in adata.obsm:
            raise ValueError("niche_composition not found in adata.obsm")
        X = adata.obsm["niche_composition"]
    else:
        raise ValueError(f"Unknown feature type: {features}")

    if outcome_key == "survival":
        if time_key not in adata.obs.columns:
            raise ValueError(f"Time key '{time_key}' not found in adata.obs")
        if event_key not in adata.obs.columns:
            raise ValueError(f"Event key '{event_key}' not found in adata.obs")

        time = adata.obs[time_key].values
        event = adata.obs[event_key].values

        if model_type == "cox":
            require("lifelines", feature="Cox proportional hazards model")
            from lifelines import CoxPHFitter

            df = pd.DataFrame(X, columns=[f"PC{i}" for i in range(X.shape[1])])
            df[time_key] = time
            df[event_key] = event

            cph = CoxPHFitter()
            cph.fit(df, duration_col=time_key, event_col=event_key)

            risk_scores = cph.predict_partial_hazard(df)
            adata.obs["exploratory_survival_association_score"] = risk_scores.values

            return {
                "model": "cox",
                "research_use_only": True,
                "interpretation": "cohort-level exploratory association; not a clinical prediction",
                "n_samples": len(time),
                "n_events": int(event.sum()),
                "concordance_index": cph.concordance_index_,
                "feature_importance": dict(zip(
                    [f"PC{i}" for i in range(X.shape[1])],
                    cph.params_["coef"].values
                )),
            }
        else:
            logger.warning(
                "Model type %s is not implemented; returning no predictive scores.",
                model_type,
            )
            return {
                "model": model_type,
                "status": "not_implemented",
                "research_use_only": True,
                "interpretation": "no validated model output was generated",
            }

    else:
        raise ValueError(f"Outcome type {outcome_key} not yet implemented")


def analyze_therapy_response(
    adata,
    *,
    response_key: str = "response",
    treatment_key: str = "treatment",
    cell_type_key: str = "cell_type",
) -> dict:
    """Summarize spatial patterns associated with response labels.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with treatment and response metadata.
    response_key : str
        Column with response labels (responder/non-responder).
    treatment_key : str
        Column with treatment type.
    cell_type_key : str
        Column with cell type labels.

    Returns
    -------
    dict
        Exploratory response-associated pattern results.
    """
    if response_key not in adata.obs.columns:
        raise ValueError(f"Response key '{response_key}' not found in adata.obs")

    logger.info("Summarizing response-associated spatial patterns...")

    responders = adata[adata.obs[response_key] == "responder"].copy()
    non_responders = adata[adata.obs[response_key] == "non_responder"].copy()

    response_associated_patterns = {}

    if "tumor_ecosystem" in adata.obs.columns:
        for group_name, group_data in [("responders", responders), ("non_responders", non_responders)]:
            if len(group_data) > 0:
                ecosystem_dist = group_data.obs["tumor_ecosystem"].value_counts(normalize=True).to_dict()
                response_associated_patterns[f"{group_name}_ecosystem"] = ecosystem_dist

    if cell_type_key in adata.obs.columns:
        for group_name, group_data in [("responders", responders), ("non_responders", non_responders)]:
            if len(group_data) > 0:
                ct_dist = group_data.obs[cell_type_key].value_counts(normalize=True).to_dict()
                response_associated_patterns[f"{group_name}_cell_types"] = ct_dist

    adata.uns["therapy_response_patterns"] = response_associated_patterns

    return {
        "research_use_only": True,
        "interpretation": "cohort-level exploratory association; not treatment guidance",
        "n_responders": len(responders),
        "n_non_responders": len(non_responders),
        "response_associated_patterns": response_associated_patterns,
    }


def run_oncology(
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
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Choose from: {SUPPORTED_METHODS}")

    n_cells, n_genes = adata.n_obs, adata.n_vars
    logger.info("Oncology analysis: %d cells, method=%s", n_cells, method)

    dispatch: dict[str, Any] = {
        "tumor_ecosystem": lambda: map_tumor_ecosystem(
            adata, cell_type_key=cell_type_key, **kwargs
        ),
        "supervised_niche": lambda: discover_supervised_niches(
            adata, condition_key=condition_key, cell_type_key=cell_type_key, **kwargs
        ),
        "clinical_prediction": lambda: predict_clinical_outcome(adata, **kwargs),
        "therapy_response": lambda: analyze_therapy_response(
            adata, cell_type_key=cell_type_key, **kwargs
        ),
    }

    result = dispatch[method]()
    result["n_cells"] = n_cells
    result["n_genes"] = n_genes
    result["method"] = method

    return result
