"""Spatial batch integration functions.

Harmony, BBKNN, Scanorama, and STAligner batch integration with mixing metrics.

Usage::

    from skills.spatial._lib.integration import run_integration, SUPPORTED_METHODS
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scanpy as sc

from .adata_utils import ensure_neighbors, ensure_pca
from .dependency_manager import require

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("harmony", "bbknn", "scanorama", "staligner")


def integrate_harmony(adata, batch_key: str) -> dict:
    """Run Harmony integration on PCA embeddings."""
    require("harmonypy", feature="Harmony batch integration")
    import harmonypy

    ensure_pca(adata)
    ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key, max_iter_harmony=20)
    corrected = ho.Z_corr
    if corrected.shape[0] != adata.n_obs and corrected.shape[1] == adata.n_obs:
        corrected = corrected.T
    adata.obsm["X_pca_harmony"] = corrected
    sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=15)
    sc.tl.umap(adata)
    return {"method": "harmony", "embedding_key": "X_pca_harmony"}


def integrate_bbknn(adata, batch_key: str) -> dict:
    """Run BBKNN batch-balanced nearest neighbours."""
    require("bbknn", feature="BBKNN batch integration")
    import bbknn

    ensure_pca(adata)
    bbknn.bbknn(adata, batch_key=batch_key)
    sc.tl.umap(adata)
    return {"method": "bbknn", "embedding_key": "X_pca"}


def integrate_scanorama(adata, batch_key: str) -> dict:
    """Run Scanorama integration via Scanpy's external API."""
    require("scanorama", feature="Scanorama batch integration")
    ensure_pca(adata)
    sc.external.pp.scanorama_integrate(
        adata, key=batch_key, basis="X_pca", adjusted_basis="X_scanorama",
    )
    sc.pp.neighbors(adata, use_rep="X_scanorama")
    sc.tl.umap(adata)
    return {"method": "scanorama", "embedding_key": "X_scanorama"}


def integrate_staligner(adata, batch_key: str) -> dict:
    """Run STAligner integration and store embedding in ``X_staligner``."""
    require("STAligner", feature="STAligner batch integration")
    import STAligner
    import pandas as pd
    from scipy import sparse

    ensure_pca(adata)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")

    # STAligner expects spatial coordinates in obsm['spatial']
    if "spatial" not in adata.obsm:
        raise ValueError(
            "STAligner requires spatial coordinates in adata.obsm['spatial']. "
            "Please ensure the data contains spatial information."
        )

    # Ensure X is sparse (STAligner expects sparse format)
    if not sparse.issparse(adata.X):
        logger.info("Converting X to sparse format required by STAligner")
        adata.X = sparse.csr_matrix(adata.X)

    # Pre-compute spatial network if not already done
    if "Spatial_Net" not in adata.uns:
        try:
            Cal_Spatial_Net = getattr(STAligner, "Cal_Spatial_Net", None)
            if Cal_Spatial_Net is None:
                from STAligner.ST_utils import Cal_Spatial_Net
            Cal_Spatial_Net(adata, rad_cutoff=150, k_cutoff=None, max_neigh=50, model='Radius', verbose=False)
        except Exception as e:
            logger.debug(f"Could not compute spatial network: {e}. Proceeding without it.")

    # Pre-compute edge list if not present
    if "edgeList" not in adata.uns and "Spatial_Net" in adata.uns:
        try:
            spatial_net = adata.uns["Spatial_Net"].copy()
            cells = np.array(adata.obs.index)
            cells_id_tran = {cell: idx for idx, cell in enumerate(cells)}
            edge_list = []
            for _, row in spatial_net.iterrows():
                cell1_idx = cells_id_tran.get(row["Cell1"])
                cell2_idx = cells_id_tran.get(row["Cell2"])
                if cell1_idx is not None and cell2_idx is not None:
                    edge_list.append([cell1_idx, cell2_idx])
            if edge_list:
                adata.uns["edgeList"] = np.array(edge_list, dtype=np.int64)
        except Exception as e:
            logger.debug(f"Could not construct edgeList: {e}")

    # STAligner expects 'batch_name' column; rename if needed
    batch_name_exists = "batch_name" in adata.obs.columns
    if not batch_name_exists and batch_key != "batch_name":
        adata.obs["batch_name"] = adata.obs[batch_key]

    train_fn = getattr(STAligner, "train_STAligner", None)
    if train_fn is None:
        raise RuntimeError(
            "Installed STAligner package does not expose train_STAligner(). "
            "Please install a compatible STAligner release."
        )

    logger.info("Running STAligner training...")
    result_obj = train_fn(adata, verbose=False)
    source = result_obj if hasattr(result_obj, "obsm") else adata

    # Clean up temporary batch_name if we created it
    if not batch_name_exists and batch_key != "batch_name" and "batch_name" in adata.obs.columns:
        adata.obs.drop("batch_name", axis=1, inplace=True)

    embedding = None
    for key in ("X_staligner", "X_STAligner", "STAligner", "staligner"):
        if key in source.obsm:
            embedding = np.asarray(source.obsm[key])
            break

    if embedding is None and isinstance(result_obj, np.ndarray):
        if result_obj.shape[0] == adata.n_obs:
            embedding = result_obj

    if embedding is None:
        raise RuntimeError(
            "STAligner ran but no embedding was found in AnnData.obsm. "
            "Expected one of: X_staligner, X_STAligner, STAligner, staligner."
        )

    adata.obsm["X_staligner"] = embedding
    sc.pp.neighbors(adata, use_rep="X_staligner", n_neighbors=15)
    sc.tl.umap(adata)
    return {"method": "staligner", "embedding_key": "X_staligner"}


def compute_batch_mixing(adata, batch_key: str) -> float:
    """Compute batch mixing entropy from the neighbor graph."""
    try:
        from scipy import sparse
        if "connectivities" not in adata.obsp:
            return 0.0
        conn = adata.obsp["connectivities"]
        if sparse.issparse(conn):
            conn = conn.toarray()
        batch_labels = adata.obs[batch_key].values
        batches = np.unique(batch_labels)
        n_batches = len(batches)
        if n_batches < 2:
            return 0.0

        entropies = []
        for i in range(adata.n_obs):
            neighbors_idx = np.where(conn[i] > 0)[0]
            if len(neighbors_idx) == 0:
                continue
            neighbor_batches = batch_labels[neighbors_idx]
            counts = np.array([np.sum(neighbor_batches == b) for b in batches])
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            entropies.append(entropy)

        max_entropy = np.log(n_batches)
        return float(np.mean(entropies) / max_entropy) if entropies else 0.0
    except Exception:
        return 0.0


def run_integration(adata, *, method: str = "harmony", batch_key: str = "batch") -> dict:
    """Run multi-sample integration. Returns summary dict."""
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not in adata.obs. Available: {list(adata.obs.columns)}")

    batches = sorted(adata.obs[batch_key].unique().tolist(), key=str)
    n_batches = len(batches)
    batch_sizes = {str(b): int((adata.obs[batch_key] == b).sum()) for b in batches}

    logger.info("Input: %d cells x %d genes, %d batches", adata.n_obs, adata.n_vars, n_batches)

    if n_batches < 2:
        raise ValueError(
            f"Only 1 batch found in '{batch_key}'. "
            "Multi-sample integration requires at least 2 batches."
        )
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown integration method '{method}'. Choose from: {SUPPORTED_METHODS}")

    if "X_pca" not in adata.obsm:
        raise ValueError(
            "X_pca not found. Run spatial-preprocessing before integration:\n"
            "  python spatialclaw.py run spatial-preprocessing --input data.h5ad --output results/"
        )
    if "X_umap" not in adata.obsm:
        ensure_neighbors(adata)
        sc.tl.umap(adata)
    umap_before = adata.obsm["X_umap"].copy()
    mixing_before = compute_batch_mixing(adata, batch_key)

    if method == "harmony":
        result = integrate_harmony(adata, batch_key)
    elif method == "bbknn":
        result = integrate_bbknn(adata, batch_key)
    elif method == "scanorama":
        result = integrate_scanorama(adata, batch_key)
    elif method == "staligner":
        try:
            result = integrate_staligner(adata, batch_key)
        except Exception as e:
            logger.warning(f"STAligner failed: {e}. Falling back to Harmony integration.")
            try:
                result = integrate_harmony(adata, batch_key)
            except Exception:
                logger.warning("Harmony failed. Trying BBKNN...")
                try:
                    result = integrate_bbknn(adata, batch_key)
                except Exception:
                    logger.warning("BBKNN failed. Trying Scanorama...")
                    try:
                        result = integrate_scanorama(adata, batch_key)
                    except Exception:
                        logger.warning("All integration methods failed. Using PCA baseline.")
                        result = {"method": "fallback", "embedding_key": "X_pca"}

    mixing_after = compute_batch_mixing(adata, batch_key)
    adata.obsm["X_umap_before_integration"] = umap_before

    if "leiden" not in adata.obs.columns:
        sc.tl.leiden(adata, resolution=1.0, flavor="igraph")

    return {
        "n_cells": adata.n_obs, "n_genes": adata.n_vars,
        "n_batches": n_batches, "batches": batches, "batch_sizes": batch_sizes,
        "method": result["method"], "embedding_key": result["embedding_key"],
        "batch_mixing_before": round(mixing_before, 4),
        "batch_mixing_after": round(mixing_after, 4),
    }
