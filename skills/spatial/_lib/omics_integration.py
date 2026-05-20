"""Shared spatial multi-omics integration backends.

This module exposes the method registry and backend runners for SpatialGlue
and SpaDDM.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData

from .dependency_manager import require

logger = logging.getLogger(__name__)

try:
    from SpatialGlue.preprocess import construct_neighbor_graph
    from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
    SPATIALGLUE_AVAILABLE = True
except ImportError:
    Train_SpatialGlue = None
    construct_neighbor_graph = None
    SPATIALGLUE_AVAILABLE = False
    logger.warning("SpatialGlue package not installed. Install via: pip install SpatialGlue")


# ---------------------------------------------------------------------------
# Supported Omics Types and Preprocessing Registry
# ---------------------------------------------------------------------------

# SpatialGlue officially supports spatial multi-omics integration.
# Currently implemented and validated:
SUPPORTED_OMICS_TYPES = {
    "rna": "RNA-seq / Spatial transcriptomics",
    "protein": "ADT / CITE-seq / spatial protein",
    "atac": "ATAC-seq / spatial chromatin accessibility",
}
SUPPORTED_MODALITY_PAIRS = (
    frozenset(("rna", "protein")),
    frozenset(("rna", "atac")),
)

SPATIALGLUE_SUPPORTED_DATATYPES = {
    "10x",
    "Stereo-CITE-seq",
    "SPOTS",
    "spatial-epigenome",
}
SPADDM_SUPPORTED_DATATYPES = {
    "10x",
    "Stereo-CITE-seq",
    "SPOTS",
    "Spatial-epigenome-transcriptome",
    "Visium CytAssist",
}
SPADDM_DEFAULTS = {
    "SPOTS": {"epochs": 600, "weight_factors": (0.5, 0.5, 0.1, 0.5)},
    "Stereo-CITE-seq": {"epochs": 700, "weight_factors": (0.5, 0.5, 0.1, 0.5)},
    "10x": {"epochs": 600, "weight_factors": (0.5, 0.5, 0.1, 0.5)},
    "Spatial-epigenome-transcriptome": {"epochs": 1200, "weight_factors": (0.5, 0.5, 0.1, 0.5)},
    "Visium CytAssist": {"epochs": 800, "weight_factors": (0.5, 0.5, 0.1, 0.5)},
}


@dataclass(frozen=True)
class MultiOmicsMethodSpec:
    key: str
    label: str
    description: str
    embedding_key: str
    cluster_key: str
    default_seed: int
    default_clustering_method: str
    supported_datatypes: frozenset[str]
    supported_pairs: tuple[frozenset[str], ...]


METHOD_SPECS: dict[str, MultiOmicsMethodSpec] = {
    "spatialglue": MultiOmicsMethodSpec(
        key="spatialglue",
        label="SpatialGlue",
        description="Dual-attention graph neural network for spatial multi-omics integration.",
        embedding_key="SpatialGlue",
        cluster_key="SpatialGlue",
        default_seed=2022,
        default_clustering_method="leiden",
        supported_datatypes=frozenset(SPATIALGLUE_SUPPORTED_DATATYPES),
        supported_pairs=SUPPORTED_MODALITY_PAIRS,
    ),
    "spaddm": MultiOmicsMethodSpec(
        key="spaddm",
        label="SpaDDM",
        description="Directional diffusion model for spatial multi-omics integration.",
        embedding_key="SpatialDDM",
        cluster_key="SpatialDDM",
        default_seed=2024,
        default_clustering_method="leiden",
        supported_datatypes=frozenset(SPADDM_SUPPORTED_DATATYPES),
        supported_pairs=SUPPORTED_MODALITY_PAIRS,
    ),
}
METHOD_ALIASES = {
    "spatialglue": "spatialglue",
    "spatial-glue": "spatialglue",
    "glue": "spatialglue",
    "spaddm": "spaddm",
    "spa-ddm": "spaddm",
    "spatialddm": "spaddm",
    "spatial-ddm": "spaddm",
}
SUPPORTED_MULTIOMICS_METHODS = tuple(METHOD_SPECS)


def normalize_multiomics_method(method: str) -> str:
    key = str(method or "spatialglue").strip().lower()
    if key not in METHOD_ALIASES:
        raise ValueError(
            f"Unknown integration method '{method}'. "
            f"Choose from: {', '.join(SUPPORTED_MULTIOMICS_METHODS)}"
        )
    return METHOD_ALIASES[key]


def get_multiomics_method_spec(method: str) -> MultiOmicsMethodSpec:
    return METHOD_SPECS[normalize_multiomics_method(method)]


def resolve_compute_device(
    device: torch.device | str | None,
    *,
    prefer_gpu: bool = True,
) -> torch.device:
    """Resolve user-facing device aliases to a concrete torch device.

    Accepted values:
    - ``None`` / ``""`` / ``"auto"``: prefer GPU when available, else CPU
    - ``"gpu"``: alias for "prefer CUDA, else CPU"
    - ``"cuda"`` / ``"cuda:0"`` / ``"cuda:1"``: explicit CUDA device
    - any other torch-recognized device string such as ``"cpu"`` or ``"mps"``
    """
    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = str(device or "").strip().lower()
        if not requested or requested == "auto":
            requested = "gpu" if prefer_gpu else "cpu"

        if requested == "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            logger.warning("GPU requested but CUDA is unavailable; falling back to CPU.")
            return torch.device("cpu")

        if requested == "cuda":
            requested = "cuda:0"

        try:
            resolved = torch.device(requested)
        except RuntimeError as exc:
            raise ValueError(
                "Unsupported device value "
                f"'{device}'. Use gpu, auto, cpu, cuda, cuda:0, or another torch-supported device."
            ) from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA device '{resolved}' was requested but CUDA is unavailable. "
            "Use device='gpu' for automatic CPU fallback or device='cpu' to force CPU."
        )
    return resolved


def _pair_key(omics_type_1: str, omics_type_2: str) -> frozenset[str]:
    return frozenset((omics_type_1, omics_type_2))


def _validate_method_pair(method: str, omics_type_1: str, omics_type_2: str) -> None:
    spec = get_multiomics_method_spec(method)
    if _pair_key(omics_type_1, omics_type_2) not in spec.supported_pairs:
        readable_pairs = sorted("+".join(sorted(p)) for p in spec.supported_pairs)
        raise ValueError(
            f"{spec.label} currently supports validated modality pairs: {', '.join(readable_pairs)}. "
            f"Received: {omics_type_1}+{omics_type_2}."
        )


def _resolve_datatype(method: str, omics_type_1: str, omics_type_2: str, datatype: str | None) -> str:
    normalized_method = normalize_multiomics_method(method)
    default_datatype = "10x"
    if _pair_key(omics_type_1, omics_type_2) == frozenset(("rna", "atac")):
        default_datatype = (
            "spatial-epigenome"
            if normalized_method == "spatialglue"
            else "Spatial-epigenome-transcriptome"
        )
    if datatype in (None, "", "auto"):
        return default_datatype

    raw = str(datatype).strip()
    lowered = raw.lower()
    if normalized_method == "spatialglue":
        alias_map = {
            "spatial-epigenome-transcriptome": "spatial-epigenome",
            "visium cytassist": "10x",
        }
        resolved = alias_map.get(lowered, raw)
        if resolved not in SPATIALGLUE_SUPPORTED_DATATYPES:
            raise ValueError(
                f"Unsupported data type '{datatype}' for SpatialGlue. "
                f"Choose from: {', '.join(sorted(SPATIALGLUE_SUPPORTED_DATATYPES))}, or 'auto'."
            )
        return resolved

    alias_map = {
        "spatial-epigenome": "Spatial-epigenome-transcriptome",
        "spatial-cite-seq": "Stereo-CITE-seq",
    }
    resolved = alias_map.get(lowered, raw)
    if resolved not in SPADDM_SUPPORTED_DATATYPES:
        raise ValueError(
            f"Unsupported data type '{datatype}' for SpaDDM. "
            f"Choose from: {', '.join(sorted(SPADDM_SUPPORTED_DATATYPES))}, or 'auto'."
        )
    return resolved


def _fix_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_paired_modalities(adata_omics1: AnnData, adata_omics2: AnnData) -> tuple[AnnData, AnnData]:
    if adata_omics1.n_obs != adata_omics2.n_obs:
        raise ValueError(
            f"Omics1 ({adata_omics1.n_obs} cells) and Omics2 ({adata_omics2.n_obs} cells) "
            "must have the same number of cells/spots."
        )
    adata_omics1 = adata_omics1.copy()
    adata_omics2 = adata_omics2.copy()
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    if adata_omics1.obs_names.equals(adata_omics2.obs_names):
        return _ensure_shared_spatial_coordinates(adata_omics1, adata_omics2)
    if set(adata_omics1.obs_names) == set(adata_omics2.obs_names):
        logger.info("Reordering omics2 to match omics1 obs_names")
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        return _ensure_shared_spatial_coordinates(adata_omics1, adata_omics2)

    raise ValueError(
        "Omics1 and Omics2 must contain the same obs_names in the same order, "
        "or the same obs_names so they can be realigned."
    )


def _ensure_shared_spatial_coordinates(
    adata_omics1: AnnData,
    adata_omics2: AnnData,
) -> tuple[AnnData, AnnData]:
    has_spatial_1 = "spatial" in adata_omics1.obsm
    has_spatial_2 = "spatial" in adata_omics2.obsm
    if not has_spatial_1 and not has_spatial_2:
        raise ValueError("At least one modality must contain adata.obsm['spatial'].")
    if has_spatial_1 and not has_spatial_2:
        logger.info("Copying spatial coordinates from omics1 to omics2")
        adata_omics2.obsm["spatial"] = np.asarray(adata_omics1.obsm["spatial"]).copy()
    elif has_spatial_2 and not has_spatial_1:
        logger.info("Copying spatial coordinates from omics2 to omics1")
        adata_omics1.obsm["spatial"] = np.asarray(adata_omics2.obsm["spatial"]).copy()
    return adata_omics1, adata_omics2


# ---------------------------------------------------------------------------
# Preprocessing Functions by Omics Type
# ---------------------------------------------------------------------------


def _compute_pca(adata: AnnData, *, n_comps: int, use_reps: str | None = None) -> np.ndarray:
    from sklearn.decomposition import PCA
    from scipy.sparse import csc_matrix, csr_matrix

    n_comps = max(2, min(n_comps, adata.n_obs - 1))
    if use_reps is not None:
        matrix = np.asarray(adata.obsm[use_reps])
    elif isinstance(adata.X, (csc_matrix, csr_matrix)):
        matrix = adata.X.toarray()
    else:
        matrix = np.asarray(adata.X)
    n_comps = min(n_comps, matrix.shape[1] - 1) if matrix.shape[1] > 1 else 1
    return PCA(n_components=n_comps).fit_transform(matrix)


def _manual_clr_normalize_each_cell(adata: AnnData) -> AnnData:
    import scipy.sparse as sp

    def seurat_clr(x: np.ndarray) -> np.ndarray:
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    matrix = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    adata.X = np.apply_along_axis(seurat_clr, 1, matrix)
    return adata


def _tfidf(matrix):
    import scipy.sparse

    idf = matrix.shape[0] / matrix.sum(axis=0)
    if scipy.sparse.issparse(matrix):
        tf = matrix.multiply(1 / matrix.sum(axis=1))
        return tf.multiply(idf)
    tf = matrix / matrix.sum(axis=1, keepdims=True)
    return tf * idf


def _compute_lsi(
    adata: AnnData,
    *,
    n_components: int = 20,
    use_highly_variable: bool | None = None,
) -> np.ndarray:
    import sklearn

    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = _tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    return X_lsi[:, 1:]


def preprocess_rna(
    adata: AnnData,
    n_hvg: int = 3000,
    target_sum: float = 1e4,
    n_pcs: int | None = None,
    scale: bool = True,
    min_cells: int = 10,
) -> dict:
    logger.info("Preprocessing RNA: %d cells x %d genes", adata.n_obs, adata.n_vars)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    n_genes_filtered = adata.n_vars

    n_top_genes = min(n_hvg, adata.n_vars)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    n_hvg_selected = int(adata.var["highly_variable"].sum())

    sc.pp.normalize_total(adata_hvg, target_sum=target_sum)
    sc.pp.log1p(adata_hvg)
    if scale:
        sc.pp.scale(adata_hvg)

    if n_pcs is None:
        n_pcs = min(50, adata_hvg.n_vars - 1, adata_hvg.n_obs - 1)
    feat_pca = _compute_pca(adata_hvg, n_comps=n_pcs)
    adata.obsm["feat"] = feat_pca

    return {
        "modality": "rna",
        "n_cells": adata.n_obs,
        "n_genes_original": n_genes_filtered,
        "n_hvg_selected": n_hvg_selected,
        "n_features_final": adata_hvg.n_vars,
        "n_pcs": int(feat_pca.shape[1]),
        "normalization": "CPM",
        "transformation": "log1p",
        "scaled": scale,
    }


def preprocess_protein(
    adata: AnnData,
    n_pcs: int | None = None,
) -> dict:
    logger.info("Preprocessing protein: %d cells x %d proteins", adata.n_obs, adata.n_vars)
    _manual_clr_normalize_each_cell(adata)
    sc.pp.scale(adata)
    if n_pcs is None:
        n_pcs = min(adata.n_vars - 1, 64, adata.n_obs - 1)
    feat_pca = _compute_pca(adata, n_comps=n_pcs)
    adata.obsm["feat"] = feat_pca

    return {
        "modality": "protein",
        "n_cells": adata.n_obs,
        "n_proteins": adata.n_vars,
        "n_pcs": int(feat_pca.shape[1]),
        "normalization": "CLR",
    }


def preprocess_atac_spatialglue(
    adata: AnnData,
    n_pcs: int | None = None,
    binarize: bool = True,
) -> dict:
    logger.info("Preprocessing ATAC for SpatialGlue: %d cells x %d peaks", adata.n_obs, adata.n_vars)
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize
    from sklearn.utils.extmath import randomized_svd

    if binarize:
        matrix = adata.X
        if sp.issparse(matrix):
            matrix = matrix.copy()
            matrix.data = (matrix.data > 0).astype(np.float32)
        else:
            matrix = (np.asarray(matrix) > 0).astype(np.float32)
        adata.X = matrix

    X_test = adata.X
    peak_counts = np.asarray(X_test.sum(axis=0)).ravel()
    peak_mask = peak_counts >= 5
    if not np.any(peak_mask):
        raise ValueError("ATAC preprocessing removed all peaks after applying the minimum count filter.")

    adata_work = adata[:, peak_mask].copy()

    matrix = adata_work.X
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        matrix.data /= row_sums[matrix.nonzero()[0]]
        col_counts = np.asarray(matrix.sum(axis=0)).ravel()
        idf = np.log(adata_work.n_obs / (col_counts + 1e-10))
        matrix = matrix.multiply(idf).tocsr()
    else:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums
        col_counts = matrix.sum(axis=0)
        matrix = matrix * np.log(adata_work.n_obs / (col_counts + 1e-10))

    if n_pcs is None:
        n_pcs = min(20, adata_work.n_vars - 1, adata_work.n_obs - 1)

    if sp.issparse(matrix):
        X_norm = normalize(matrix, norm="l1", axis=1)
        X_norm = np.log1p(X_norm.toarray() * 1e4)
    else:
        X_norm = normalize(matrix, norm="l1", axis=1)
        X_norm = np.log1p(X_norm * 1e4)

    U, _, _ = randomized_svd(X_norm, n_components=n_pcs, random_state=0)
    X_lsi = U
    X_lsi -= X_lsi.mean(axis=0, keepdims=True)
    X_lsi /= X_lsi.std(axis=0, ddof=1, keepdims=True)
    X_lsi = X_lsi[:, 1:] if X_lsi.shape[1] > 1 else X_lsi

    adata.obsm["X_lsi"] = X_lsi
    adata.obsm["feat"] = X_lsi
    return {
        "modality": "atac",
        "n_cells": adata_work.n_obs,
        "n_peaks": adata_work.n_vars,
        "n_components": int(adata.obsm["feat"].shape[1]),
        "normalization": "TF-IDF + LSI",
        "binarized": binarize,
    }

def preprocess_atac_spaddm(
    adata: AnnData,
    n_pcs: int | None = None,
) -> dict:
    logger.info("Preprocessing ATAC for SpaDDM: %d cells x %d peaks", adata.n_obs, adata.n_vars)
    if "X_lsi" not in adata.obsm:
        if n_pcs is None:
            n_pcs = min(51, adata.n_vars - 1, adata.n_obs - 1)
        adata.obsm["X_lsi"] = _compute_lsi(
            adata,
            n_components=n_pcs,
            use_highly_variable=False,
        )
    adata.obsm["feat"] = np.asarray(adata.obsm["X_lsi"]).copy()
    return {
        "modality": "atac",
        "n_cells": adata.n_obs,
        "n_peaks": adata.n_vars,
        "n_components": int(adata.obsm["feat"].shape[1]),
        "normalization": "TF-IDF + LSI",
    }


# ---------------------------------------------------------------------------
# Generic Preprocessing Dispatcher
# ---------------------------------------------------------------------------


def preprocess_omics(
    adata: AnnData,
    omics_type: Literal["rna", "protein", "atac"],
    *,
    method: str = "spatialglue",
    **kwargs,
) -> dict:
    if omics_type not in SUPPORTED_OMICS_TYPES:
        raise ValueError(
            f"Unknown omics type: {omics_type}. "
            f"Supported types: {', '.join(SUPPORTED_OMICS_TYPES.keys())}"
        )

    normalized_method = normalize_multiomics_method(method)
    if omics_type == "rna":
        return preprocess_rna(adata, **kwargs)
    if omics_type == "protein":
        return preprocess_protein(adata, **kwargs)
    if normalized_method == "spaddm":
        return preprocess_atac_spaddm(adata, **kwargs)
    return preprocess_atac_spatialglue(adata, **kwargs)


# ---------------------------------------------------------------------------
# Omics Type Auto-Detection
# ---------------------------------------------------------------------------


def detect_omics_type(adata) -> str:
    """Auto-detect omics type from AnnData metadata or structure.
    
    Detects the type of spatial omics data based on metadata and characteristics.
    Supports detection of RNA, Protein/ADT, and ATAC-seq data types.
    
    Parameters
    ----------
    adata : AnnData
        Input data
        
    Returns
    -------
    str
        Detected omics type: 'rna', 'protein', 'atac', or 'rna' (default)
        
    Note
    ----
    This detects only officially supported paired spatial modality types.
    Unsupported modality types are rejected before backend execution.
    """
    # Check modality metadata
    if 'modality' in adata.uns:
        modality = str(adata.uns['modality']).lower()
        if modality in SUPPORTED_OMICS_TYPES:
            return modality
    
    # Check for RNA-seq evidence
    if 'highly_variable' in adata.var.columns:
        return "rna"
    
    if adata.n_vars > 10000 and adata.n_obs < 100000:
        # Many features (genes), reasonable samples → likely RNA
        return "rna"
    
    # Check for Protein/ADT evidence
    if ('protein' in str(adata).lower() or
        'adt' in str(adata).lower() or
        'cite' in str(adata).lower()):
        return "protein"
    
    if adata.n_vars < 100 and 'protein' not in str(adata).lower():
        # Few features (antibodies) → likely protein
        return "protein"
    
    # Check for ATAC-seq evidence
    if 'atac' in str(adata).lower() or 'peak' in str(adata).lower():
        return "atac"
    
    if (hasattr(adata.X, 'nnz') and  # sparse matrix
        (adata.X.nnz / (adata.n_obs * adata.n_vars)) < 0.01):  # very sparse
        # Very sparse data → likely ATAC
        return "atac"
    
    logger.warning("Could not auto-detect omics type; defaulting to 'rna'")
    return "rna"


# ---------------------------------------------------------------------------
# Graph Construction and Model Training
# ---------------------------------------------------------------------------


def _feature_count_from_summary(summary: dict, omics_type: str) -> int:
    key = {
        "rna": "n_hvg_selected",
        "protein": "n_proteins",
        "atac": "n_peaks",
    }[omics_type]
    return int(summary.get(key, 0))


def _build_spaddm_graph(adata: AnnData):
    require("torch-geometric", feature="SpaDDM spatial multi-omics integration")
    from sklearn.neighbors import NearestNeighbors
    from .spaddm_model import SpaDDMGraph

    coords = np.asarray(adata.obsm["spatial"])
    datatype = adata.uns.get("_spatial_omics_datatype")
    n_neighbors = 6 if datatype in {"Stereo-CITE-seq", "Spatial-epigenome-transcriptome"} else 3
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    sources = indices[:, 0].repeat(n_neighbors)
    targets = indices[:, 1:].reshape(-1)
    self_loops = np.arange(adata.n_obs, dtype=np.int64)
    edge_index = np.vstack([
        np.concatenate([sources, self_loops]),
        np.concatenate([targets, self_loops]),
    ])
    return SpaDDMGraph(
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        feat=torch.as_tensor(np.asarray(adata.obsm["feat"]), dtype=torch.float32),
        num_nodes=adata.n_obs,
    )


def _search_resolution(
    adata: AnnData,
    *,
    method: str,
    target_clusters: int,
    use_rep: str,
    start: float = 0.1,
    end: float = 3.0,
    increment: float = 0.05,
) -> float:
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for resolution in sorted(np.arange(start, end, increment), reverse=True):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=float(resolution), key_added="_tmp_clusters")
        else:
            sc.tl.louvain(adata, random_state=0, resolution=float(resolution), key_added="_tmp_clusters")
        if adata.obs["_tmp_clusters"].nunique() == target_clusters:
            del adata.obs["_tmp_clusters"]
            return float(resolution)
    if "_tmp_clusters" in adata.obs:
        del adata.obs["_tmp_clusters"]
    raise AssertionError("Resolution is not found. Please try bigger range or smaller step!.")


def cluster_embedding(
    adata: AnnData,
    *,
    embedding_key: str,
    add_key: str,
    requested_method: str,
    default_method: str,
    n_clusters: int | None,
    random_seed: int,
) -> str:
    actual_method = requested_method.lower()
    if actual_method == "auto":
        actual_method = default_method

    if actual_method not in {"leiden", "louvain"}:
        raise ValueError("clustering_method must be one of: auto, leiden, louvain")

    resolution = 1.0
    if n_clusters is not None:
        try:
            resolution = _search_resolution(
                adata,
                method=actual_method,
                target_clusters=n_clusters,
                use_rep=embedding_key,
            )
        except AssertionError:
            resolution = max(0.1, 0.3 * np.log(max(n_clusters, 2)))

    sc.pp.neighbors(adata, n_neighbors=15, use_rep=embedding_key)
    if actual_method == "leiden":
        sc.tl.leiden(adata, random_state=random_seed, resolution=resolution, key_added=add_key)
    else:
        sc.tl.louvain(adata, random_state=random_seed, resolution=resolution, key_added=add_key)
    return actual_method


def _attach_standard_fields(
    adata_template: AnnData,
    *,
    method_spec: MultiOmicsMethodSpec,
    omics_type_1: str,
    omics_type_2: str,
    datatype: str,
    embeddings: dict[str, np.ndarray],
    preprocessing_1: dict,
    preprocessing_2: dict,
    attention: dict[str, np.ndarray | None] | None = None,
    reconstructions: dict[str, np.ndarray] | None = None,
) -> AnnData:
    adata = adata_template.copy()
    adata.obsm["emb_latent_omics1"] = np.asarray(embeddings["latent_omics1"]).copy()
    adata.obsm["emb_latent_omics2"] = np.asarray(embeddings["latent_omics2"]).copy()
    adata.obsm[method_spec.embedding_key] = np.asarray(embeddings["joint"]).copy()
    adata.obsm["X_spatial_omics"] = np.asarray(embeddings["joint"]).copy()
    adata.obsm["feat_omics1"] = np.asarray(adata_template.obsm["feat"]).copy()
    if attention and attention.get("inter_omics") is not None:
        adata.obsm["alpha"] = np.asarray(attention["inter_omics"]).copy()
    if attention and attention.get("intra_omics1") is not None:
        adata.obsm["alpha_omics1"] = np.asarray(attention["intra_omics1"]).copy()
    if attention and attention.get("intra_omics2") is not None:
        adata.obsm["alpha_omics2"] = np.asarray(attention["intra_omics2"]).copy()
    if reconstructions:
        adata.obsm["rec_omics1"] = np.asarray(reconstructions["omics1"]).copy()
        adata.obsm["rec_omics2"] = np.asarray(reconstructions["omics2"]).copy()
    if method_spec.key == "spaddm":
        adata.obsm["emb_latent_omics_1"] = np.asarray(embeddings["latent_omics1"]).copy()
        adata.obsm["emb_latent_omics_2"] = np.asarray(embeddings["latent_omics2"]).copy()
        if reconstructions:
            adata.obsm["rec_omics_1"] = np.asarray(reconstructions["omics1"]).copy()
            adata.obsm["rec_omics_2"] = np.asarray(reconstructions["omics2"]).copy()

    adata.uns["omics1_modality"] = omics_type_1
    adata.uns["omics2_modality"] = omics_type_2
    adata.uns["integration_method"] = method_spec.label
    adata.uns["integration_method_key"] = method_spec.key
    adata.uns["integration_datatype"] = datatype
    adata.uns["integration_date"] = pd.Timestamp.now().isoformat()
    adata.uns["preprocessing_omics1"] = preprocessing_1
    adata.uns["preprocessing_omics2"] = preprocessing_2
    return adata


def _run_spatialglue_backend(
    adata_omics1: AnnData,
    adata_omics2: AnnData,
    *,
    omics_type_1: str,
    omics_type_2: str,
    datatype: str,
    n_hvg: int,
    n_clusters: int | None,
    random_seed: int,
    device: torch.device,
    n_epochs: int,
    clustering_method: str,
) -> dict:
    if not SPATIALGLUE_AVAILABLE or Train_SpatialGlue is None or construct_neighbor_graph is None:
        raise ImportError("SpatialGlue not installed. Install via: pip install SpatialGlue")

    _fix_seed(random_seed)
    summary_omics1 = preprocess_omics(
        adata_omics1,
        omics_type_1,
        method="spatialglue",
        n_hvg=n_hvg,
        scale=True,
    )
    summary_omics2 = preprocess_omics(
        adata_omics2,
        omics_type_2,
        method="spatialglue",
        n_pcs=adata_omics1.obsm["feat"].shape[1] if omics_type_2 != "atac" else None,
    )

    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=datatype)
    model = Train_SpatialGlue(data, datatype=datatype, device=device, epochs=n_epochs)
    if getattr(model, "epochs", n_epochs) != n_epochs:
        model.epochs = n_epochs
    output = model.train()

    spec = METHOD_SPECS["spatialglue"]
    adata_integrated = _attach_standard_fields(
        adata_omics1,
        method_spec=spec,
        omics_type_1=omics_type_1,
        omics_type_2=omics_type_2,
        datatype=datatype,
        embeddings={
            "latent_omics1": output["emb_latent_omics1"],
            "latent_omics2": output["emb_latent_omics2"],
            "joint": output["SpatialGlue"],
        },
        preprocessing_1=summary_omics1,
        preprocessing_2=summary_omics2,
        attention={
            "inter_omics": output.get("alpha"),
            "intra_omics1": output.get("alpha_omics1"),
            "intra_omics2": output.get("alpha_omics2"),
        },
    )
    adata_integrated.obsm["feat_omics2"] = np.asarray(adata_omics2.obsm["feat"]).copy()
    actual_clustering = cluster_embedding(
        adata_integrated,
        embedding_key=spec.embedding_key,
        add_key=spec.cluster_key,
        requested_method=clustering_method,
        default_method=spec.default_clustering_method,
        n_clusters=n_clusters,
        random_seed=random_seed,
    )
    adata_integrated.obs["spatial_omics_cluster"] = pd.Categorical(
        adata_integrated.obs[spec.cluster_key].astype(str)
    )
    sc.pp.neighbors(adata_integrated, use_rep=spec.embedding_key, n_neighbors=10)
    sc.tl.umap(adata_integrated)

    return {
        "method": spec.key,
        "method_label": spec.label,
        "embedding_key": spec.embedding_key,
        "cluster_key": spec.cluster_key,
        "n_cells": adata_integrated.n_obs,
        "n_omics1_features": _feature_count_from_summary(summary_omics1, omics_type_1),
        "n_omics2_features": _feature_count_from_summary(summary_omics2, omics_type_2),
        "omics1_modality": omics_type_1,
        "omics2_modality": omics_type_2,
        "datatype": datatype,
        "device": str(device),
        "clustering_method": actual_clustering,
        "n_clusters": int(adata_integrated.obs[spec.cluster_key].nunique()),
        "random_seed": random_seed,
        "n_epochs_requested": n_epochs,
        "n_epochs_actual": int(getattr(model, "epochs", n_epochs)),
        "adata_integrated": adata_integrated,
        "embeddings": {
            "latent_omics1": output["emb_latent_omics1"],
            "latent_omics2": output["emb_latent_omics2"],
            "joint": output["SpatialGlue"],
        },
        "attention_weights": {
            "intra_omics1": output.get("alpha_omics1"),
            "intra_omics2": output.get("alpha_omics2"),
            "inter_omics": output.get("alpha"),
        },
        "preprocessing": {"omics1": summary_omics1, "omics2": summary_omics2},
        "reconstruction_metrics": {},
    }


def _run_spaddm_backend(
    adata_omics1: AnnData,
    adata_omics2: AnnData,
    *,
    omics_type_1: str,
    omics_type_2: str,
    datatype: str,
    n_hvg: int,
    n_clusters: int | None,
    random_seed: int,
    device: torch.device,
    n_epochs: int | None,
    clustering_method: str,
    n_latent: int,
) -> dict:
    require("torch-geometric", feature="SpaDDM spatial multi-omics integration")
    from .spaddm_model import SpaDDMTrainer

    _fix_seed(random_seed)
    summary_omics1 = preprocess_omics(
        adata_omics1,
        omics_type_1,
        method="spaddm",
        n_hvg=n_hvg,
        scale=(omics_type_2 == "protein"),
        n_pcs=50,
    )
    summary_omics2 = preprocess_omics(
        adata_omics2,
        omics_type_2,
        method="spaddm",
    )

    adata_omics1.uns["_spatial_omics_datatype"] = datatype
    adata_omics2.uns["_spatial_omics_datatype"] = datatype
    graph_omics_1 = _build_spaddm_graph(adata_omics1)
    graph_omics_2 = _build_spaddm_graph(adata_omics2)

    defaults = SPADDM_DEFAULTS[datatype]
    effective_epochs = n_epochs if n_epochs is not None else defaults["epochs"]
    trainer = SpaDDMTrainer(
        graph_omics_1,
        graph_omics_2,
        device=device,
        epochs=effective_epochs,
        latent_dim=n_latent,
        weight_factors=defaults["weight_factors"],
    )
    output = trainer.train()

    spec = METHOD_SPECS["spaddm"]
    adata_integrated = _attach_standard_fields(
        adata_omics1,
        method_spec=spec,
        omics_type_1=omics_type_1,
        omics_type_2=omics_type_2,
        datatype=datatype,
        embeddings={
            "latent_omics1": output["emb_omics_1"],
            "latent_omics2": output["emb_omics_2"],
            "joint": output["SpatialDDM"],
        },
        preprocessing_1=summary_omics1,
        preprocessing_2=summary_omics2,
        attention={"inter_omics": output.get("alpha")},
        reconstructions={
            "omics1": output.get("rec_omics_1"),
            "omics2": output.get("rec_omics_2"),
        },
    )
    adata_integrated.obsm["feat_omics2"] = np.asarray(adata_omics2.obsm["feat"]).copy()
    actual_clustering = cluster_embedding(
        adata_integrated,
        embedding_key=spec.embedding_key,
        add_key=spec.cluster_key,
        requested_method=clustering_method,
        default_method=spec.default_clustering_method,
        n_clusters=n_clusters,
        random_seed=random_seed,
    )
    adata_integrated.obs["spatial_omics_cluster"] = pd.Categorical(
        adata_integrated.obs[spec.cluster_key].astype(str)
    )
    sc.pp.neighbors(adata_integrated, use_rep=spec.embedding_key, n_neighbors=10)
    sc.tl.umap(adata_integrated)

    reconstruction_metrics = compute_reconstruction_metrics(
        adata_omics1.obsm.get("feat"),
        output.get("rec_omics_1"),
        adata_omics2.obsm.get("feat"),
        output.get("rec_omics_2"),
    )

    return {
        "method": spec.key,
        "method_label": spec.label,
        "embedding_key": spec.embedding_key,
        "cluster_key": spec.cluster_key,
        "n_cells": adata_integrated.n_obs,
        "n_omics1_features": _feature_count_from_summary(summary_omics1, omics_type_1),
        "n_omics2_features": _feature_count_from_summary(summary_omics2, omics_type_2),
        "omics1_modality": omics_type_1,
        "omics2_modality": omics_type_2,
        "datatype": datatype,
        "device": str(device),
        "clustering_method": actual_clustering,
        "n_clusters": int(adata_integrated.obs[spec.cluster_key].nunique()),
        "random_seed": random_seed,
        "n_epochs_requested": effective_epochs,
        "n_epochs_actual": effective_epochs,
        "adata_integrated": adata_integrated,
        "embeddings": {
            "latent_omics1": output["emb_omics_1"],
            "latent_omics2": output["emb_omics_2"],
            "joint": output["SpatialDDM"],
        },
        "attention_weights": {"inter_omics": output.get("alpha")},
        "preprocessing": {"omics1": summary_omics1, "omics2": summary_omics2},
        "reconstruction_metrics": reconstruction_metrics,
    }


def run_spatial_omics_integration(
    adata_omics1: AnnData,
    adata_omics2: AnnData,
    *,
    method: str = "spatialglue",
    omics_type_1: Literal["rna", "protein", "atac"] = "rna",
    omics_type_2: Literal["rna", "protein", "atac"] = "protein",
    datatype: str | None = "auto",
    n_hvg: int = 3000,
    n_clusters: int | None = None,
    random_seed: int | None = None,
    device: torch.device | str | None = None,
    n_epochs: int | None = None,
    clustering_method: str = "auto",
    n_latent: int = 64,
) -> dict:
    normalized_method = normalize_multiomics_method(method)
    spec = METHOD_SPECS[normalized_method]
    _validate_method_pair(normalized_method, omics_type_1, omics_type_2)
    resolved_datatype = _resolve_datatype(normalized_method, omics_type_1, omics_type_2, datatype)

    if random_seed is None:
        random_seed = spec.default_seed
    device = resolve_compute_device(device, prefer_gpu=True)

    adata_omics1, adata_omics2 = _ensure_paired_modalities(adata_omics1, adata_omics2)
    logger.info(
        "Running %s integration for %s + %s on %s",
        spec.label,
        omics_type_1,
        omics_type_2,
        resolved_datatype,
    )
    logger.info("Using device: %s", device)

    if normalized_method == "spatialglue":
        return _run_spatialglue_backend(
            adata_omics1,
            adata_omics2,
            omics_type_1=omics_type_1,
            omics_type_2=omics_type_2,
            datatype=resolved_datatype,
            n_hvg=n_hvg,
            n_clusters=n_clusters,
            random_seed=random_seed,
            device=device,
            n_epochs=n_epochs if n_epochs is not None else 200,
            clustering_method=clustering_method,
        )

    return _run_spaddm_backend(
        adata_omics1,
        adata_omics2,
        omics_type_1=omics_type_1,
        omics_type_2=omics_type_2,
        datatype=resolved_datatype,
        n_hvg=n_hvg,
        n_clusters=n_clusters,
        random_seed=random_seed,
        device=device,
        n_epochs=n_epochs,
        clustering_method=clustering_method,
        n_latent=n_latent,
    )


def run_spatialglue_integration(
    adata_omics1: AnnData,
    adata_omics2: AnnData,
    *,
    omics_type_1: Literal["rna", "protein", "atac"] = "rna",
    omics_type_2: Literal["rna", "protein", "atac"] = "protein",
    datatype: Literal["10x", "Stereo-CITE-seq", "SPOTS", "spatial-epigenome"] = "10x",
    n_hvg: int = 3000,
    n_clusters: int | None = None,
    random_seed: int = 2022,
    device: torch.device | str | None = None,
    n_epochs: int = 200,
    clustering_method: str = "leiden",
) -> dict:
    return run_spatial_omics_integration(
        adata_omics1,
        adata_omics2,
        method="spatialglue",
        omics_type_1=omics_type_1,
        omics_type_2=omics_type_2,
        datatype=datatype,
        n_hvg=n_hvg,
        n_clusters=n_clusters,
        random_seed=random_seed,
        device=device,
        n_epochs=n_epochs,
        clustering_method=clustering_method,
    )


def run_spaddm_integration(
    adata_omics1: AnnData,
    adata_omics2: AnnData,
    *,
    omics_type_1: Literal["rna", "protein", "atac"] = "rna",
    omics_type_2: Literal["rna", "protein", "atac"] = "protein",
    datatype: str | None = "auto",
    n_hvg: int = 3000,
    n_clusters: int | None = None,
    random_seed: int = 2024,
    device: torch.device | str | None = None,
    n_epochs: int | None = None,
    clustering_method: str = "auto",
    n_latent: int = 64,
) -> dict:
    return run_spatial_omics_integration(
        adata_omics1,
        adata_omics2,
        method="spaddm",
        omics_type_1=omics_type_1,
        omics_type_2=omics_type_2,
        datatype=datatype,
        n_hvg=n_hvg,
        n_clusters=n_clusters,
        random_seed=random_seed,
        device=device,
        n_epochs=n_epochs,
        clustering_method=clustering_method,
        n_latent=n_latent,
    )


# ---------------------------------------------------------------------------
# Quality Metrics
# ---------------------------------------------------------------------------


def compute_attention_balance(alpha_weights: np.ndarray | None) -> dict:
    if alpha_weights is None:
        return {}
    alpha_weights = np.asarray(alpha_weights)
    if alpha_weights.ndim != 2 or alpha_weights.shape[1] != 2:
        return {}
    avg_attention = np.mean(alpha_weights, axis=0)
    entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-10))
    max_entropy = np.log(alpha_weights.shape[1])
    entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    return {
        "avg_attention_omics1": float(avg_attention[0]),
        "avg_attention_omics2": float(avg_attention[1]),
        "attention_entropy": float(entropy),
        "attention_entropy_normalized": float(entropy_normalized),
    }


def compute_cluster_purity(
    adata,
    cluster_key: str = "spatial_omics_cluster",
    reference_key: str | None = None,
) -> float:
    if reference_key is None or reference_key not in adata.obs.columns:
        return 0.0
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(adata.obs[reference_key].values, adata.obs[cluster_key].values))


def compute_reconstruction_metrics(
    feat_omics1: np.ndarray | None,
    rec_omics1: np.ndarray | None,
    feat_omics2: np.ndarray | None,
    rec_omics2: np.ndarray | None,
) -> dict:
    metrics = {}
    if feat_omics1 is not None and rec_omics1 is not None:
        metrics["omics1_reconstruction_mse"] = float(
            np.mean((np.asarray(feat_omics1) - np.asarray(rec_omics1)) ** 2)
        )
    if feat_omics2 is not None and rec_omics2 is not None:
        metrics["omics2_reconstruction_mse"] = float(
            np.mean((np.asarray(feat_omics2) - np.asarray(rec_omics2)) ** 2)
        )
    return metrics
