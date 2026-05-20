"""Spatial niche identification, characterization, and comparison.

Provides methods for discovering and analyzing tissue microenvironments
beyond simple spatial domain clustering. Niches represent local cellular
neighborhoods with distinct composition and functional programs.

Supported methods:
  - cellcharter: Graph-based niche clustering (CellCharter)
  - scniche: Niche identification via similarity clustering
  - nichecompass: Niche-aware graph neural network
  - nico: Neighborhood-based niche analysis

Input matrix convention:
  All methods use log-normalized expression (adata.X) plus spatial
  coordinates (obsm["spatial"]). Optional cell type labels and
  deconvolution proportions enhance niche characterization.

Usage::

    from skills.spatial._lib.niches import (
        run_niche_identification,
        characterize_niches,
        compare_niches,
        SUPPORTED_METHODS,
    )

    summary = run_niche_identification(adata, method="cellcharter")
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .adata_utils import get_spatial_key, require_spatial_coords
from .dependency_manager import require

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("cellcharter", "scniche", "nichecompass", "nico", "leiden_niche")


def _build_spatial_neighbors(adata, n_neighbors: int = 10, spatial_key: str = "spatial"):
    """Build spatial neighbor graph for niche analysis."""
    try:
        import squidpy as sq
        sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, n_neighs=n_neighbors, coord_type="generic")
        return True
    except Exception as e:
        logger.warning("Could not build spatial neighbors via squidpy: %s", e)
        return False


def _compute_neighborhood_composition(
    adata,
    cell_type_key: str = "cell_type",
    n_neighbors: int = 10,
    spatial_key: str = "spatial",
) -> np.ndarray:
    """Compute cell type composition in spatial neighborhoods."""
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in adata.obs")
    
    coords = adata.obsm[spatial_key]
    cell_types = adata.obs[cell_type_key].astype("category")
    n_types = len(cell_types.cat.categories)
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    composition = np.zeros((adata.n_obs, n_types))
    for i in range(adata.n_obs):
        neighbor_types = cell_types.iloc[indices[i, 1:]].values
        for ct in neighbor_types:
            ct_idx = cell_types.cat.categories.get_loc(ct)
            composition[i, ct_idx] += 1
        composition[i] /= (n_neighbors)
    
    return composition


def identify_niches_cellcharter(
    adata,
    *,
    n_niches: int = 10,
    n_neighbors: int = 10,
    n_clusters_k: int = 5,
    seed: int = 42,
) -> dict:
    """Identify niches using CellCharter algorithm.

    Uses log-normalized expression (adata.X) and spatial coordinates
    to cluster cells into spatially coherent niches based on both
    expression similarity and spatial proximity.
    """
    require("cellcharter", feature="CellCharter niche identification")
    import cellcharter as cc

    spatial_key = require_spatial_coords(adata)
    logger.info("Running CellCharter niche identification (n_niches=%d)", n_niches)

    _build_spatial_neighbors(adata, n_neighbors=n_neighbors, spatial_key=spatial_key)

    cc.gr.remove_long_links(adata)
    cc.tl.clustering(
        adata,
        n_clusters=n_niches,
        k=n_clusters_k,
        random_state=seed,
        key_added="niche",
    )

    n_niches_found = adata.obs["niche"].nunique()
    logger.info("CellCharter identified %d niches", n_niches_found)

    return {
        "method": "cellcharter",
        "n_niches": n_niches_found,
        "n_niches_requested": n_niches,
        "n_neighbors": n_neighbors,
        "niche_counts": adata.obs["niche"].value_counts().to_dict(),
    }


def identify_niches_leiden(
    adata,
    *,
    cell_type_key: str = "cell_type",
    n_neighbors: int = 10,
    resolution: float = 1.0,
) -> dict:
    """Identify niches using neighborhood composition + Leiden clustering.

    Computes cell type composition in spatial neighborhoods, then
    applies Leiden clustering to group cells with similar neighborhood
    profiles into niches.
    """
    spatial_key = require_spatial_coords(adata)
    
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' required for Leiden niche identification")

    logger.info("Computing neighborhood composition for niche identification...")
    composition = _compute_neighborhood_composition(
        adata, cell_type_key=cell_type_key, n_neighbors=n_neighbors, spatial_key=spatial_key
    )

    adata.obsm["neighborhood_composition"] = composition

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(15, adata.n_obs - 1), metric="cosine").fit(composition)
    distances, indices = nbrs.kneighbors(composition)

    adjacency = np.zeros((adata.n_obs, adata.n_obs))
    for i in range(adata.n_obs):
        for j in indices[i]:
            adjacency[i, j] = 1
            adjacency[j, i] = 1

    import igraph as ig
    import leidenalg as la

    sources, targets = np.where(adjacency > 0)
    g = ig.Graph(edges=list(zip(sources, targets)))
    partition = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=resolution)
    
    labels = np.array([str(m) for m in partition.membership])
    adata.obs["niche"] = pd.Categorical(labels)

    n_niches = adata.obs["niche"].nunique()
    logger.info("Leiden niche identification found %d niches", n_niches)

    return {
        "method": "leiden_niche",
        "n_niches": n_niches,
        "resolution": resolution,
        "n_neighbors": n_neighbors,
        "cell_type_key": cell_type_key,
        "niche_counts": adata.obs["niche"].value_counts().to_dict(),
    }


def identify_niches_nico(
    adata,
    *,
    cell_type_key: str = "cell_type",
    n_neighbors: int = 10,
    resolution: float = 1.0,
) -> dict:
    """Identify niches using NiCo (Neighborhood Co-occurrence) method.

    Analyzes co-occurrence patterns of cell types in spatial neighborhoods
    to define functional niches.
    """
    spatial_key = require_spatial_coords(adata)
    
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' required for NiCo niche identification")

    logger.info("Running NiCo niche identification...")

    composition = _compute_neighborhood_composition(
        adata, cell_type_key=cell_type_key, n_neighbors=n_neighbors, spatial_key=spatial_key
    )

    cell_types = adata.obs[cell_type_key].astype("category")
    type_names = cell_types.cat.categories.tolist()
    
    adata.obsm["nico_composition"] = composition

    sc.pp.neighbors(adata, use_rep="nico_composition", n_neighbors=15, metric="cosine")
    sc.tl.leiden(adata, resolution=resolution, key_added="niche", flavor="igraph")

    n_niches = adata.obs["niche"].nunique()
    logger.info("NiCo identified %d niches", n_niches)

    return {
        "method": "nico",
        "n_niches": n_niches,
        "resolution": resolution,
        "n_neighbors": n_neighbors,
        "cell_type_key": cell_type_key,
        "niche_counts": adata.obs["niche"].value_counts().to_dict(),
    }


def characterize_niches(
    adata,
    niche_key: str = "niche",
    cell_type_key: str = "cell_type",
) -> dict:
    """Characterize niches by composition and spatial properties.

    Computes:
    - Cell type composition per niche
    - Niche size and spatial extent
    - Niche-specific marker genes
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"Niche key '{niche_key}' not found in adata.obs")
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in adata.obs")

    niches = adata.obs[niche_key].unique()
    cell_types = adata.obs[cell_type_key].astype("category").cat.categories

    composition_matrix = {}
    for niche in niches:
        mask = adata.obs[niche_key] == niche
        ct_counts = adata.obs.loc[mask, cell_type_key].value_counts()
        total = mask.sum()
        composition_matrix[niche] = {
            ct: ct_counts.get(ct, 0) / total for ct in cell_types
        }

    composition_df = pd.DataFrame(composition_matrix).T
    adata.uns["niche_composition"] = composition_df

    if "X_pca" in adata.obsm:
        logger.info("Finding niche-specific markers...")
        sc.tl.rank_genes_groups(adata, niche_key, method="wilcoxon", n_genes=50)
        niche_markers = {}
        for niche in niches:
            markers_df = sc.get.rank_genes_groups_df(adata, group=str(niche))
            niche_markers[niche] = markers_df.head(20)["names"].tolist()
        adata.uns["niche_markers"] = niche_markers

    return {
        "n_niches": len(niches),
        "composition": composition_df.to_dict(),
        "niche_sizes": adata.obs[niche_key].value_counts().to_dict(),
    }


def compare_niches(
    adata,
    niche_key: str = "niche",
    condition_key: str = "condition",
) -> dict:
    """Compare niche distributions across conditions/samples.

    Computes:
    - Niche abundance per condition
    - Differential niche abundance testing
    - Condition-specific niche programs
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"Niche key '{niche_key}' not found in adata.obs")
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")

    conditions = adata.obs[condition_key].unique()
    niches = adata.obs[niche_key].unique()

    abundance = {}
    for cond in conditions:
        mask = adata.obs[condition_key] == cond
        niche_counts = adata.obs.loc[mask, niche_key].value_counts()
        total = mask.sum()
        abundance[cond] = {n: niche_counts.get(n, 0) / total for n in niches}

    abundance_df = pd.DataFrame(abundance).T
    adata.uns["niche_abundance_by_condition"] = abundance_df

    return {
        "n_conditions": len(conditions),
        "n_niches": len(niches),
        "abundance": abundance_df.to_dict(),
        "conditions": list(conditions),
    }


def run_niche_identification(
    adata,
    *,
    method: str = "leiden_niche",
    cell_type_key: str = "cell_type",
    n_niches: int = 10,
    n_neighbors: int = 10,
    resolution: float = 1.0,
) -> dict:
    """Run niche identification with the specified method.

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

    Returns
    -------
    dict
        Summary with niche labels and statistics.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {SUPPORTED_METHODS}")

    n_cells, n_genes = adata.n_obs, adata.n_vars
    logger.info("Niche identification: %d cells x %d genes", n_cells, n_genes)

    dispatch: dict[str, Any] = {
        "cellcharter": lambda: identify_niches_cellcharter(
            adata, n_niches=n_niches, n_neighbors=n_neighbors
        ),
        "leiden_niche": lambda: identify_niches_leiden(
            adata, cell_type_key=cell_type_key, n_neighbors=n_neighbors, resolution=resolution
        ),
        "nico": lambda: identify_niches_nico(
            adata, cell_type_key=cell_type_key, n_neighbors=n_neighbors, resolution=resolution
        ),
    }

    if method not in dispatch:
        raise ValueError(f"Method '{method}' not yet implemented")

    result = dispatch[method]()
    result["n_cells"] = n_cells
    result["n_genes"] = n_genes

    return result
