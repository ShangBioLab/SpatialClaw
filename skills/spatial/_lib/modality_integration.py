"""Spatial modality analysis backends for DeepST and PearlST.

This module exposes DeepST and PearlST execution paths for spatial domain
identification and same-modality spatial integration. The PearlST path follows
the published workflow: HVG preprocessing, PDE-based augmentation,
alpha-complex graph construction, and WARGA training.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors, kneighbors_graph

try:
    import deepstkit as dt
except ImportError:
    dt = None

import scanpy as sc

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from torchvision import models, transforms
except ImportError:
    models = None
    transforms = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpatialModalityMethodSpec:
    """Method metadata used by the modality skill and registry tests."""

    key: str
    label: str
    description: str
    embedding_key: str
    cluster_key: str
    supports_integration: bool
    supports_morphology: bool
    supported_platforms: frozenset[str]
    default_epochs: int


METHOD_SPECS: dict[str, SpatialModalityMethodSpec] = {
    "deepst": SpatialModalityMethodSpec(
        key="deepst",
        label="DeepST",
        description="Graph neural network backend for spatial domain identification and multi-sample integration.",
        embedding_key="DeepST_embed",
        cluster_key="DeepST_refine_domain",
        supports_integration=True,
        supports_morphology=True,
        supported_platforms=frozenset({"Visium", "Stereo-seq", "Slide-seq", "Slide-seqV2", "MERFISH", "STARmap"}),
        default_epochs=500,
    ),
    "pearlst": SpatialModalityMethodSpec(
        key="pearlst",
        label="PearlST",
        description="PDE-enhanced adversarial graph autoencoder for single-sample spatial transcriptomics.",
        embedding_key="PearlST_embed",
        cluster_key="pred_label",
        supports_integration=False,
        supports_morphology=True,
        supported_platforms=frozenset({"Visium", "Stereo-seq", "Slide-seq", "Slide-seqV2", "MERFISH", "STARmap"}),
        default_epochs=1270,
    ),
}
METHOD_ALIASES = {
    "deepst": "deepst",
    "deep-st": "deepst",
    "pearlst": "pearlst",
    "pearl-st": "pearlst",
    "pearist": "pearlst",
    "pearlst-code": "pearlst",
}
SUPPORTED_MODALITY_METHODS = tuple(METHOD_SPECS)


def normalize_modality_method(method: str | None) -> str:
    key = str(method or "deepst").strip().lower()
    if key not in METHOD_ALIASES:
        raise ValueError(
            f"Unknown spatial modality method '{method}'. "
            f"Choose from: {', '.join(SUPPORTED_MODALITY_METHODS)}"
        )
    return METHOD_ALIASES[key]


def get_modality_method_spec(method: str | None) -> SpatialModalityMethodSpec:
    return METHOD_SPECS[normalize_modality_method(method)]


def resolve_compute_device(
    device: torch.device | str | None,
    *,
    prefer_gpu: bool = True,
) -> torch.device:
    """Resolve user-facing device aliases to a concrete torch device."""
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
                f"Unsupported device '{device}'. Use gpu, auto, cpu, cuda, or cuda:0."
            ) from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA device '{resolved}' was requested but CUDA is unavailable. "
            "Use device='gpu' for automatic CPU fallback or device='cpu' to force CPU."
        )
    return resolved


def check_pearlst_dependencies(*, use_morphological: bool = False) -> tuple[bool, list[str]]:
    """Return whether the minimal PearlST dependencies are available."""
    missing: list[str] = []
    for module_name in ("gudhi",):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if use_morphological:
        if models is None or transforms is None:
            missing.append("torchvision")
        if Image is None:
            missing.append("Pillow")
    return not missing, missing


def check_deepst_installed() -> bool:
    """Check if DeepST is installed."""
    try:
        import deepstkit
        return True
    except ImportError:
        logger.error("DeepST not installed. Install with: pip install deepstkit")
        return False


def validate_spatial_data(adata: sc.AnnData) -> bool:
    """Validate that AnnData has spatial coordinates."""
    if "spatial" not in adata.obsm:
        logger.error("AnnData must have 'spatial' in .obsm")
        return False
    if adata.obsm["spatial"].shape != (adata.shape[0], 2):
        logger.error("Spatial coordinates must be (n_obs, 2) shape")
        return False
    return True


def _normalize_deepst_domain_count(n_domains: int, n_obs: int) -> int:
    """Clamp requested DeepST domain count to a valid range for the sample."""
    if n_obs < 2:
        raise ValueError("DeepST clustering requires at least two spots.")
    target = max(2, int(n_domains))
    return min(target, n_obs)


def _estimate_deepst_domain_count(
    embeddings: np.ndarray,
    *,
    random_seed: int = 0,
    max_clusters: int = 20,
) -> int:
    """Estimate a stable domain count from DeepST embeddings."""
    n_obs = int(embeddings.shape[0])
    if n_obs <= 2:
        return 2

    max_candidate = min(max_clusters, max(2, n_obs // 10), n_obs - 1)
    scores: dict[int, float] = {}
    for n_clusters in range(2, max_candidate + 1):
        labels = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            n_init=10,
        ).fit_predict(embeddings)
        if np.unique(labels).size < 2:
            continue
        scores[n_clusters] = float(silhouette_score(embeddings, labels))

    if not scores:
        return min(7, n_obs)
    return int(max(scores, key=scores.get))


def _cluster_deepst_embedding(
    adata: sc.AnnData,
    *,
    deepst_instance: object,
    n_domains: int,
    batch_key: Optional[str] = None,
    use_obsm: str = "DeepST_embed",
    random_seed: int = 0,
) -> tuple[sc.AnnData, dict]:
    """Cluster DeepST embeddings with an exact KMeans fallback."""
    target_domains = _normalize_deepst_domain_count(n_domains, adata.n_obs)
    info = {
        "clustering_backend": "deepst-upstream",
        "clustering_fallback_used": False,
        "n_domains_requested": target_domains,
    }

    try:
        kwargs = {
            "n_domains": target_domains,
            "priori": True,
        }
        if batch_key is not None:
            kwargs["batch_key"] = batch_key
        clustered = deepst_instance._get_cluster_data(adata, **kwargs)
        domain_col = (
            "DeepST_refine_domain"
            if "DeepST_refine_domain" in clustered.obs
            else "DeepST_domain"
        )
        observed = int(pd.Series(clustered.obs[domain_col].astype(str)).nunique())
        info["n_domains_observed"] = observed
        if observed != target_domains:
            raise ValueError(
                f"DeepST returned {observed} domains after requesting {target_domains}"
            )
        return clustered, info
    except Exception as exc:
        logger.warning(
            "DeepST clustering failed or returned unstable domains (%s). "
            "Using KMeans on DeepST embeddings.",
            exc,
        )

    embeddings = np.asarray(adata.obsm[use_obsm], dtype=np.float32)
    labels = KMeans(
        n_clusters=target_domains,
        random_state=random_seed,
        n_init=10,
    ).fit_predict(embeddings)
    label_series = pd.Series(labels, index=adata.obs_names).astype(str)
    adata = adata.copy()
    adata.obs["DeepST_domain"] = pd.Categorical(label_series)
    adata.obs["DeepST_refine_domain"] = pd.Categorical(label_series)
    info.update(
        {
            "clustering_backend": "kmeans-fallback",
            "clustering_fallback_used": True,
            "n_domains_observed": int(label_series.nunique()),
        }
    )
    return adata, info


def run_deepst_spatial_domain_identification(
    adata: sc.AnnData,
    n_domains: Optional[int] = None,
    pre_epochs: int = 500,
    epochs: int = 500,
    pca_components: int = 200,
    use_morphological: bool = False,
    use_gpu: bool = True,
    random_seed: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Run DeepST for single-sample spatial domain identification.
    
    Parameters
    ----------
    adata : sc.AnnData
        Spatial transcriptomics data with .obsm['spatial']
    n_domains : int, optional
        Target number of spatial domains. If None, auto-detect.
    pre_epochs : int
        Pretraining epochs for autoencoder
    epochs : int
        Main training epochs
    pca_components : int
        PCA dimensionality (default: 200)
    use_morphological : bool
        Extract H&E morphological features if available
    use_gpu : bool
        Use GPU acceleration
    random_seed : int
        Random seed for reproducibility
    output_dir : str or Path, optional
        Output directory for intermediate files
    
    Returns
    -------
    dict
        Results dictionary with:
        - 'adata': AnnData with DeepST results
        - 'embeddings': Learned embeddings (n_obs, latent_dim)
        - 'n_domains': Number of domains detected
        - 'spatial_domain_identification': Domain assignments
    """
    if not check_deepst_installed():
        raise ImportError("DeepST not installed")
    
    if not validate_spatial_data(adata):
        raise ValueError("Invalid spatial coordinates")
    
    logger.info(f"Running DeepST single-sample analysis on {adata.shape[0]} spots × {adata.shape[1]} genes")
    
    # Set seeds
    dt.utils_func.seed_torch(seed=random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="deepst_")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize DeepST
    deepst = dt.main.run(
        save_path=str(output_dir),
        task="Identify_Domain",
        pre_epochs=pre_epochs,
        epochs=epochs,
        use_gpu=use_gpu,
    )
    
    # Data augmentation
    logger.info("Augmenting spatial data...")
    adata_aug = deepst._get_augment(
        adata.copy(),
        spatial_type="BallTree",
        use_morphological=use_morphological,
    )
    
    # Graph construction
    logger.info("Constructing spatial graph...")
    graph_dict = deepst._get_graph(
        adata_aug.obsm["spatial"],
        distType="KDTree",
    )
    
    # PCA preprocessing
    logger.info(f"PCA preprocessing to {pca_components} components...")
    data = deepst._data_process(
        adata_aug,
        pca_n_comps=pca_components,
    )
    
    # Model training
    logger.info("Training DeepST model...")
    embeddings = deepst._fit(
        data=data,
        graph_dict=graph_dict,
    )
    
    # Store embeddings
    adata.obsm["DeepST_embed"] = embeddings
    
    # Auto-detect domains if not specified
    if n_domains is None:
        logger.info("Auto-detecting number of domains from DeepST embeddings...")
        n_domains = _estimate_deepst_domain_count(
            embeddings,
            random_seed=random_seed,
        )
        logger.info(f"Auto-detected {n_domains} domains")
    else:
        n_domains = _normalize_deepst_domain_count(n_domains, adata.n_obs)
    
    # Clustering
    logger.info(f"Clustering into {n_domains} domains...")
    adata, clustering_info = _cluster_deepst_embedding(
        adata,
        deepst_instance=deepst,
        n_domains=n_domains,
        random_seed=random_seed,
    )
    
    # Get domain assignments
    domain_col = "DeepST_refine_domain" if "DeepST_refine_domain" in adata.obs else "DeepST_domain"
    spatial_domain_identification = adata.obs[domain_col].values
    
    return {
        "adata": adata,
        "embeddings": embeddings,
        "n_domains": n_domains,
        "spatial_domain_identification": spatial_domain_identification,
        "domain_column": domain_col,
        **clustering_info,
    }


def run_deepst_integration(
    adata_list: list[sc.AnnData],
    sample_ids: Optional[list[str]] = None,
    n_domains: Optional[int] = None,
    batch_key: str = "batch",
    pre_epochs: int = 500,
    epochs: int = 600,
    pca_components: int = 200,
    use_morphological: bool = False,
    use_gpu: bool = True,
    random_seed: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Run DeepST for multi-sample integration with batch correction.
    
    Parameters
    ----------
    adata_list : list of sc.AnnData
        List of spatial transcriptomics datasets with .obsm['spatial']
    sample_ids : list of str, optional
        Sample identifiers (default: auto-generated)
    n_domains : int, optional
        Target number of spatial domains
    batch_key : str
        Column name to store batch/sample ID
    pre_epochs : int
        Pretraining epochs
    epochs : int
        Main training epochs (typically longer than single-sample)
    pca_components : int
        PCA dimensionality
    use_morphological : bool
        Use H&E morphological features
    use_gpu : bool
        GPU acceleration
    random_seed : int
        Random seed
    output_dir : str or Path, optional
        Output directory
    
    Returns
    -------
    dict
        Results dictionary with:
        - 'adata_integrated': Integrated AnnData
        - 'embeddings': Integrated embeddings
        - 'n_domains': Number of domains
        - 'batch_corrected': Batch-corrected expression matrix
    """
    if not check_deepst_installed():
        raise ImportError("DeepST not installed")
    
    # Validate all samples
    for i, adata in enumerate(adata_list):
        if not validate_spatial_data(adata):
            raise ValueError(f"Sample {i} has invalid spatial coordinates")
    
    # Generate sample IDs if not provided
    if sample_ids is None:
        sample_ids = [f"Sample_{i}" for i in range(len(adata_list))]
    
    logger.info(f"Running DeepST integration on {len(adata_list)} samples")
    
    # Set seeds
    dt.utils_func.seed_torch(seed=random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="deepst_integration_")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize DeepST for integration
    integration_model = dt.main.run(
        save_path=str(output_dir),
        task="Integration",
        pre_epochs=pre_epochs,
        epochs=epochs,
        use_gpu=use_gpu,
    )
    
    # Process each sample
    processed_data = []
    spatial_graphs = []
    
    for adata, sample_id in zip(adata_list, sample_ids):
        logger.info(f"Processing sample: {sample_id}")
        
        adata_copy = adata.copy()
        adata_copy.obs[batch_key] = sample_id
        
        # Augmentation
        adata_copy = integration_model._get_augment(
            adata_copy,
            spatial_type="BallTree",
            use_morphological=use_morphological,
        )
        
        # Graph construction
        graph = integration_model._get_graph(
            adata_copy.obsm["spatial"],
            distType="KDTree",
        )
        
        processed_data.append(adata_copy)
        spatial_graphs.append(graph)
    
    # Combine samples
    logger.info("Combining samples...")
    combined_adata, combined_graph = integration_model._get_multiple_adata(
        adata_list=processed_data,
        data_name_list=sample_ids,
        graph_list=spatial_graphs,
    )
    
    # PCA preprocessing
    logger.info(f"PCA preprocessing to {pca_components} components...")
    integrated_data = integration_model._data_process(
        combined_adata,
        pca_n_comps=pca_components,
    )
    
    # Training with domain adversarial learning
    logger.info("Training integrated model with batch correction...")
    embeddings = integration_model._fit(
        data=integrated_data,
        graph_dict=combined_graph,
        domains=combined_adata.obs[batch_key].values,
        n_domains=len(sample_ids),
    )
    
    combined_adata.obsm["DeepST_embed"] = embeddings
    
    # Set default n_domains if not specified
    if n_domains is None:
        n_domains = 8
    else:
        n_domains = _normalize_deepst_domain_count(n_domains, combined_adata.n_obs)
    
    # Clustering
    logger.info(f"Clustering into {n_domains} domains...")
    combined_adata, clustering_info = _cluster_deepst_embedding(
        combined_adata,
        deepst_instance=integration_model,
        n_domains=n_domains,
        batch_key=batch_key,
        random_seed=random_seed,
    )
    
    return {
        "adata_integrated": combined_adata,
        "embeddings": embeddings,
        "n_domains": n_domains,
        "sample_ids": sample_ids,
        "batch_key": batch_key,
        "batch_corrected": combined_adata.X,
        **clustering_info,
    }


def extract_spatial_domain_identification(
    adata: sc.AnnData,
    domain_column: str = "DeepST_refine_domain",
) -> pd.DataFrame:
    """Extract spatial domain information.
    
    Parameters
    ----------
    adata : sc.AnnData
        AnnData with domain assignments
    domain_column : str
        Column name with domain assignments
    
    Returns
    -------
    pd.DataFrame
        DataFrame with spot coordinates, domain assignments, and embeddings
    """
    result_df = pd.DataFrame(
        adata.obsm["spatial"],
        columns=["x", "y"],
        index=adata.obs.index,
    )
    
    if domain_column in adata.obs:
        result_df["domain"] = adata.obs[domain_column].values
    
    if "DeepST_embed" in adata.obsm:
        embed = adata.obsm["DeepST_embed"]
        for i in range(min(3, embed.shape[1])):
            result_df[f"embed_{i}"] = embed[:, i]
    
    return result_df


def compute_domain_statistics(
    adata: sc.AnnData,
    domain_column: str = "DeepST_refine_domain",
) -> pd.DataFrame:
    """Compute statistics for each spatial domain.
    
    Parameters
    ----------
    adata : sc.AnnData
        AnnData with domain assignments
    domain_column : str
        Column with domain assignments
    
    Returns
    -------
    pd.DataFrame
        Statistics per domain (size, mean x, mean y, etc.)
    """
    if domain_column not in adata.obs:
        raise ValueError(f"Column '{domain_column}' not found in adata.obs")
    
    stats_list = []
    spatial = adata.obsm["spatial"]
    
    for domain in adata.obs[domain_column].unique():
        mask = adata.obs[domain_column] == domain
        domain_coords = spatial[mask]
        
        stats_list.append({
            "domain": domain,
            "n_spots": mask.sum(),
            "mean_x": domain_coords[:, 0].mean(),
            "mean_y": domain_coords[:, 1].mean(),
            "std_x": domain_coords[:, 0].std(),
            "std_y": domain_coords[:, 1].std(),
            "area_x": domain_coords[:, 0].max() - domain_coords[:, 0].min(),
            "area_y": domain_coords[:, 1].max() - domain_coords[:, 1].min(),
        })
    
    return pd.DataFrame(stats_list)


# ---------------------------------------------------------------------------
# PearlST backend
# ---------------------------------------------------------------------------


def _normalize_pearlst_platform(platform: str | None) -> str:
    raw = str(platform or "Visium").strip()
    lowered = raw.lower()
    alias_map = {
        "visium": "Visium",
        "st": "Visium",
        "stereo-seq": "stereoseq",
        "stereoseq": "stereoseq",
        "slide-seq": "slideseqv2",
        "slide-seqv2": "slideseqv2",
        "slideseqv2": "slideseqv2",
        "merfish": "MERFISH",
        "starmap": "STARmap",
    }
    return alias_map.get(lowered, raw)


def _set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _ensure_dense_matrix(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, "A"):
        return np.asarray(matrix.A)
    return np.asarray(matrix)


def _coefficient(values: np.ndarray, k: float) -> np.ndarray:
    return 1.0 / (1.0 + ((values / k) ** 2))


def _anisotropic_denoising(
    count: np.ndarray,
    image_loc: int,
    iterations: int,
    k: float,
    lamb: float = 0.1,
) -> np.ndarray:
    image = count[image_loc].copy()
    new_image = np.zeros(image.shape, dtype=image.dtype)
    for _ in range(iterations):
        north = image[:-2, 1:-1] - image[1:-1, 1:-1]
        south = image[2:, 1:-1] - image[1:-1, 1:-1]
        east = image[1:-1, 2:] - image[1:-1, 1:-1]
        west = image[1:-1, :-2] - image[1:-1, 1:-1]
        new_image[1:-1, 1:-1] = image[1:-1, 1:-1] + lamb * (
            _coefficient(north, k) * north
            + _coefficient(south, k) * south
            + _coefficient(east, k) * east
            + _coefficient(west, k) * west
        )
        image = new_image.copy()
    return image


def _anisotropic_diffusion(
    count: np.ndarray,
    adj_spot_index: np.ndarray,
    image_loc: int,
    iterations: int,
    k: float,
    lamb: float = 0.1,
) -> np.ndarray:
    image = count[image_loc].copy()
    for _ in range(iterations):
        north = count[adj_spot_index[image_loc][0]] - image
        south = count[adj_spot_index[image_loc][1]] - image
        east = count[adj_spot_index[image_loc][2]] - image
        west = count[adj_spot_index[image_loc][3]] - image
        image = image + lamb * (
            _coefficient(north, k) * north
            + _coefficient(south, k) * south
            + _coefficient(east, k) * east
            + _coefficient(west, k) * west
        )
    return image


def _gene_data_denoising(count: np.ndarray, *, iterations: int, k: float) -> np.ndarray:
    if count.shape[1] != 2000:
        raise ValueError(
            "PearlST upstream diffusion expects exactly 2000 highly variable genes "
            f"(received {count.shape[1]})."
        )
    image_like = count.reshape(len(count), 50, 40)
    denoised = image_like.copy()
    for idx in range(len(image_like)):
        denoised[idx] = _anisotropic_denoising(image_like, idx, iterations=iterations, k=k)
    return denoised.reshape(len(count), 2000)


def _gene_data_augmentation(
    count: np.ndarray,
    adj_spot_index: np.ndarray,
    *,
    iterations: int,
    k: float,
) -> np.ndarray:
    augmented = count.copy()
    for idx in range(len(count)):
        augmented[idx] = _anisotropic_diffusion(
            count,
            adj_spot_index=adj_spot_index,
            image_loc=idx,
            iterations=iterations,
            k=k,
        )
    return augmented


def _cal_spatial_weight(
    data: np.ndarray,
    *,
    spatial_k: int = 50,
    spatial_type: str = "BallTree",
) -> np.ndarray:
    if spatial_type == "NearestNeighbors":
        nbrs = NearestNeighbors(n_neighbors=spatial_k + 1, algorithm="ball_tree").fit(data)
        _, indices = nbrs.kneighbors(data)
    elif spatial_type == "KDTree":
        tree = KDTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    else:
        tree = BallTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    indices = indices[:, 1:]
    weights = np.zeros((data.shape[0], data.shape[0]), dtype=np.float32)
    for idx, neighbors in enumerate(indices):
        weights[idx, neighbors] = 1.0
    return weights


def _cal_gene_weight(
    data: np.ndarray,
    *,
    n_components: int = 50,
    gene_dist_type: str = "cosine",
) -> np.ndarray:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=7)
    reduced = pca.fit_transform(data)
    return (1 - pairwise_distances(reduced, metric=gene_dist_type)).astype(np.float32)


def _cal_weight_matrix(
    adata: sc.AnnData,
    *,
    platform: str,
    pd_dist_type: str = "euclidean",
    md_dist_type: str = "cosine",
    gb_dist_type: str = "correlation",
    n_components: int = 50,
    no_morphological: bool = True,
    spatial_k: int = 30,
    spatial_type: str = "KDTree",
) -> sc.AnnData:
    if platform in {"Visium", "ST"}:
        img_row = adata.obs["imagerow"]
        img_col = adata.obs["imagecol"]
        if platform == "Visium":
            array_row = adata.obs["array_row"]
            array_col = adata.obs["array_col"]
            rate = 3.0
        else:
            array_row = adata.obs_names.map(lambda x: x.split("x")[1])
            array_col = adata.obs_names.map(lambda x: x.split("x")[0])
            rate = 1.5
        reg_row = LinearRegression().fit(np.asarray(array_row).reshape(-1, 1), img_row)
        reg_col = LinearRegression().fit(np.asarray(array_col).reshape(-1, 1), img_col)
        physical_distance = pairwise_distances(
            adata.obs[["imagecol", "imagerow"]],
            metric=pd_dist_type,
        ).astype(np.float32)
        unit = float(np.hypot(reg_row.coef_, reg_col.coef_))
        physical_distance = np.where(physical_distance >= rate * unit, 0.0, 1.0).astype(np.float32)
    else:
        physical_distance = _cal_spatial_weight(
            np.asarray(adata.obsm["spatial"]),
            spatial_k=spatial_k,
            spatial_type=spatial_type,
        )

    gene_counts = _ensure_dense_matrix(adata.X).astype(np.float32)
    if platform in {"Visium", "ST", "slideseqv2", "stereoseq"}:
        gene_correlation = _cal_gene_weight(
            gene_counts,
            gene_dist_type=gb_dist_type,
            n_components=n_components,
        )
    else:
        gene_correlation = (1 - pairwise_distances(gene_counts, metric=gb_dist_type)).astype(np.float32)

    if platform in {"Visium", "ST"} and "image_feat_pca" in adata.obsm:
        morphology = (1 - pairwise_distances(np.asarray(adata.obsm["image_feat_pca"]), metric=md_dist_type)).astype(np.float32)
        morphology[morphology < 0] = 0
        adata.obsm["weights_matrix_all"] = physical_distance * gene_correlation * morphology
        if no_morphological:
            adata.obsm["weights_matrix_nomd"] = physical_distance * gene_correlation
    else:
        adata.obsm["weights_matrix_nomd"] = physical_distance * gene_correlation
    return adata


def _find_adjacent_spot(
    adata: sc.AnnData,
    *,
    neighbour_k: int = 5,
    weights: str = "weights_matrix_all",
) -> sc.AnnData:
    weights_matrix = np.asarray(adata.obsm[weights])
    weights_list: list[np.ndarray] = []
    near_spot_list: list[np.ndarray] = []
    for idx in range(adata.shape[0]):
        if weights == "physical_distance":
            current_spot = weights_matrix[idx].argsort()[-(neighbour_k + 3):][:(neighbour_k + 2)]
        else:
            current_spot = weights_matrix[idx].argsort()[-neighbour_k:][:neighbour_k - 1]
        near_spot_list.append(current_spot)
        spot_weight = weights_matrix[idx][current_spot]
        if spot_weight.sum() > 0:
            weights_list.append(spot_weight / spot_weight.sum())
        else:
            weights_list.append(spot_weight)
    adata.obsm["near_spots"] = np.asarray(near_spot_list)
    adata.obsm["adjacent_weight"] = np.asarray(weights_list)
    return adata


def _cal_weighted_near_spots(
    adata: sc.AnnData,
    *,
    platform: str,
    no_morphological: bool = False,
    weights: str = "weights_matrix_all",
) -> sc.AnnData:
    adata = _cal_weight_matrix(
        adata,
        platform=platform,
        no_morphological=no_morphological,
    )
    return _find_adjacent_spot(adata, neighbour_k=5, weights=weights)


def _default_pearlst_simclr_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "PearlST-main"
        / "SimCLR_model"
        / "128_0.5_200_64_500_model_resnet50.pth"
    )


def _normalize_feature_index(index_value: Any) -> str:
    raw = str(index_value).replace("\\", "/").rstrip("/")
    basename = raw.split("/")[-1]
    if basename.endswith("40"):
        basename = basename[:-2]
    return basename


def _prepare_visium_image_coordinates(
    adata: sc.AnnData,
    *,
    image_quality: str = "lowres",
) -> tuple[sc.AnnData, str]:
    adata = adata.copy()
    spatial_block = adata.uns.get("spatial") or {}
    if not spatial_block:
        raise ValueError("PearlST morphology for Visium requires adata.uns['spatial'].")
    library_id = next(iter(spatial_block))
    library = spatial_block[library_id]
    if image_quality not in library.get("images", {}):
        raise ValueError(
            f"Requested image quality '{image_quality}' is unavailable. "
            f"Found: {', '.join(sorted(library.get('images', {}).keys()))}"
        )
    scale_key = f"tissue_{image_quality}_scalef"
    scale = float(library["scalefactors"].get(scale_key, 1.0))
    library["use_quality"] = image_quality
    adata.obs["imagecol"] = np.asarray(adata.obsm["spatial"][:, 0], dtype=np.float32) * scale
    adata.obs["imagerow"] = np.asarray(adata.obsm["spatial"][:, 1], dtype=np.float32) * scale
    return adata, library_id


class _PearlSTSimCLRModel(nn.Module):
    """Minimal copy of PearlST's SimCLR encoder for inference."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        if models is None:
            raise ImportError("torchvision is required for PearlST image feature extraction.")
        encoder_layers = []
        for name, module in models.resnet50().named_children():
            if name == "conv1":
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                encoder_layers.append(module)
        self.f = nn.Sequential(*encoder_layers)
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        projection = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(projection, dim=-1)


def _load_pearlst_image_feature_csv(
    feature_path: Union[str, Path],
    obs_names: pd.Index,
) -> np.ndarray:
    df = pd.read_csv(feature_path, index_col=0)
    df.index = df.index.map(_normalize_feature_index)
    if not set(obs_names).issubset(set(df.index)):
        missing = list(obs_names.difference(df.index))[:5]
        raise ValueError(
            "simCLR feature CSV is missing barcodes required by the dataset. "
            f"Examples: {missing}"
        )
    return np.asarray(df.loc[obs_names], dtype=np.float32)


def _extract_pearlst_image_features(
    adata: sc.AnnData,
    *,
    model_path: Union[str, Path],
    output_dir: Path,
    batch_size: int = 64,
    crop_size: int = 40,
    target_size: int = 32,
    image_quality: str = "lowres",
    device: torch.device,
) -> tuple[np.ndarray, Path]:
    if Image is None or transforms is None:
        raise ImportError("Pillow and torchvision are required for PearlST image feature extraction.")

    adata, library_id = _prepare_visium_image_coordinates(adata, image_quality=image_quality)
    library = adata.uns["spatial"][library_id]
    image = library["images"][library["use_quality"]]
    if image.dtype in (np.float32, np.float64):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    resize_filter = getattr(Image, "Resampling", Image).BILINEAR
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    model = _PearlSTSimCLRModel(feature_dim=128)
    state_dict = torch.load(Path(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tensors: list[torch.Tensor] = []
    for image_row, image_col in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
        top = image_row - crop_size / 2
        bottom = image_row + crop_size / 2
        left = image_col - crop_size / 2
        right = image_col + crop_size / 2
        tile = pil_image.crop((left, top, right, bottom))
        tile = tile.resize((target_size, target_size), resize_filter)
        tensors.append(transform(tile))

    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(device)
            feature, _ = model(batch)
            features.append(feature.cpu().numpy())

    feature_array = np.concatenate(features, axis=0).astype(np.float32)
    output_path = output_dir / "simCLR_representation_resnet50.csv"
    pd.DataFrame(feature_array, index=adata.obs_names).to_csv(output_path)
    return feature_array, output_path


def _sparse_mx_to_torch_sparse_tensor(matrix: sp.spmatrix) -> torch.Tensor:
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    values = torch.from_numpy(matrix.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(matrix.shape))


def _preprocess_graph(adj: sp.spmatrix) -> torch.Tensor:
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.asarray(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return _sparse_mx_to_torch_sparse_tensor(adj_normalized)


def _graph_alpha(spatial_locs: np.ndarray, *, n_neighbors: int = 10) -> sp.csr_matrix:
    import gudhi
    import networkx as nx

    knn_graph = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode="distance")
    estimated_graph_cut = knn_graph.sum() / float(knn_graph.count_nonzero())
    alpha_complex = gudhi.AlphaComplex(points=spatial_locs.tolist())
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=float(estimated_graph_cut ** 2))
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from(range(len(spatial_locs)))
    for simplex, _ in skeleton:
        if len(simplex) == 2:
            initial_graph.add_edge(simplex[0], simplex[1])

    extended_graph = nx.Graph()
    extended_graph.add_nodes_from(initial_graph)
    extended_graph.add_edges_from(initial_graph.edges)
    for idx in range(len(spatial_locs)):
        if extended_graph.has_edge(idx, idx):
            extended_graph.remove_edge(idx, idx)
    return sp.csr_matrix(nx.to_scipy_sparse_array(extended_graph, format="csr"), dtype=np.float32)


class _GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, act=F.relu):
        super().__init__()
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        inputs = F.dropout(inputs, self.dropout, training=self.training)
        support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, support)
        return self.act(output)


class _InnerProductDecoder(nn.Module):
    def __init__(self, dropout: float, act=torch.sigmoid):
        super().__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.dropout(z, self.dropout, training=self.training)
        return self.act(torch.mm(z, z.t()))


class _GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim: int, hidden_dim1: int, hidden_dim2: int, dropout: float):
        super().__init__()
        self.gc1 = _GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = _GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = _InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.gc2(self.gc1(x, adj), adj)
        return self.dc(z), z


class _Regularizer(nn.Module):
    def __init__(self, in_channels: int, hidden_dim1: int, hidden_dim2: int):
        super().__init__()
        self.dc_den1 = nn.Linear(in_channels, hidden_dim1)
        self.dc_den2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dc_output = nn.Linear(hidden_dim2, 1)
        nn.init.normal_(self.dc_den1.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.dc_den2.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.dc_output.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.dc_den1.bias)
        nn.init.zeros_(self.dc_den2.bias)
        nn.init.zeros_(self.dc_output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden1 = torch.sigmoid(self.dc_den1(inputs))
        hidden2 = torch.sigmoid(self.dc_den2(hidden1))
        return self.dc_output(hidden2)


def _compute_gradient_penalty(
    regularizer: _Regularizer,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand((real_samples.size(0), 1), device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = regularizer(interpolates)
    fake = torch.ones((real_samples.shape[0], 1), device=real_samples.device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 0.01) ** 2).mean()


def _train_pearlst_warga(
    adata: sc.AnnData,
    *,
    epochs: int,
    hidden1: int,
    hidden2: int,
    reg_in_channels: int,
    reg_hidden1: int,
    reg_hidden2: int,
    gp_lambda: float,
    lr: float,
    reg_lr: float,
    dropout: float,
    graph_neighbors: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    features_np = np.asarray(adata.obsm["augment_data"], dtype=np.float32)
    features = torch.tensor(features_np, device=device)
    adj = _graph_alpha(np.asarray(adata.obsm["spatial"]), n_neighbors=graph_neighbors).astype(np.float32)
    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()
    adj_norm = _preprocess_graph(adj_orig).to(device)
    adj_label = torch.FloatTensor((adj + sp.eye(adj.shape[0], dtype=np.float32)).toarray()).to(device)
    edge_sum = float(adj.sum())
    pos_weight = float(adj.shape[0] * adj.shape[0] - edge_sum) / edge_sum
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - edge_sum) * 2)

    model = _GCNModelAE(features_np.shape[1], hidden1, hidden2, dropout).to(device)
    regularizer = _Regularizer(reg_in_channels, reg_hidden1, reg_hidden2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    regularizer_optimizer = torch.optim.Adam(regularizer.parameters(), lr=reg_lr)

    final_loss = 0.0
    for epoch_idx in range(1, epochs + 1):
        model.train()
        regularizer.train()
        predicted_adj, embedding = model(features, adj_norm)
        for _ in range(5):
            f_z = regularizer(embedding)
            random_samples = torch.normal(0.0, 1.0, [features_np.shape[0], hidden2], device=device)
            f_r = regularizer(random_samples)
            gradient_penalty = _compute_gradient_penalty(regularizer, random_samples, embedding)
            reg_loss = -f_r.mean() + f_z.mean() + gp_lambda * gradient_penalty
            regularizer_optimizer.zero_grad()
            reg_loss.backward(retain_graph=True)
            regularizer_optimizer.step()

        generator_loss = -regularizer(embedding).mean()
        recon_loss = norm * F.binary_cross_entropy_with_logits(
            predicted_adj,
            adj_label,
            pos_weight=torch.tensor(pos_weight, device=device),
        )
        loss = recon_loss + generator_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if epoch_idx % 50 == 0 or epoch_idx == epochs:
            logger.info("PearlST epoch %d/%d loss=%.5f", epoch_idx, epochs, final_loss)

    return embedding.detach().cpu().numpy(), final_loss


def _load_ground_truth_labels(
    ground_truth: Union[str, Path, pd.Series, pd.DataFrame, None],
    obs_names: pd.Index,
) -> Optional[pd.Series]:
    if ground_truth is None:
        return None
    if isinstance(ground_truth, pd.Series):
        labels = ground_truth.copy()
    elif isinstance(ground_truth, pd.DataFrame):
        if ground_truth.shape[1] == 0:
            return None
        labels = ground_truth.iloc[:, 0].copy()
    else:
        path = Path(ground_truth)
        if not path.exists():
            logger.warning(
                "Optional PearlST ground-truth label file not found: %s. "
                "Continuing without ARI evaluation.",
                path,
            )
            return None
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep, index_col=0)
        normalized_cols = {str(col).strip().lower(): col for col in df.columns}
        preferred_cols = ("ground_truth", "label", "labels", "cluster", "domain")
        matched_col = next((normalized_cols[name] for name in preferred_cols if name in normalized_cols), None)
        if matched_col is not None:
            labels = df[matched_col].copy()
        elif df.shape[1] == 1:
            labels = df.iloc[:, 0].copy()
        else:
            logger.warning(
                "Optional PearlST ground-truth file %s does not look like a label table "
                "(columns=%s). Continuing without ARI evaluation.",
                path,
                list(df.columns[:10]),
            )
            return None
    labels.index = labels.index.map(str)
    labels = labels.reindex(obs_names)
    return labels.astype("string")


def _compute_pearlst_layouts(adata: sc.AnnData, *, random_seed: int) -> None:
    embed_adata = sc.AnnData(adata.obsm["PearlST_embed"].copy())
    embed_adata.obs["pred_label"] = pd.Categorical(adata.obs["pred_label"].astype(str))
    neighbors = min(15, max(2, embed_adata.n_obs - 1))
    sc.pp.neighbors(embed_adata, n_neighbors=neighbors)
    sc.tl.umap(embed_adata, random_state=random_seed)
    sc.tl.paga(embed_adata, groups="pred_label")
    adata.obsm["PearlST_umap"] = embed_adata.obsm["X_umap"]

    if embed_adata.n_obs > 1:
        max_cell_for_subsampling = 5000
        if embed_adata.shape[0] <= max_cell_for_subsampling:
            sub_x = embed_adata.X
        else:
            selected = np.random.choice(
                np.arange(embed_adata.shape[0]),
                max_cell_for_subsampling,
                replace=False,
            )
            sub_x = embed_adata.X[selected, :]
        dist_sum = pairwise_distances(sub_x).sum(axis=1)
        embed_adata.uns["iroot"] = int(np.argmax(dist_sum))
        sc.tl.diffmap(embed_adata)
        sc.tl.dpt(embed_adata)
        adata.obs["PearlST_pseudotime"] = embed_adata.obs["dpt_pseudotime"].to_numpy()


def run_pearlst_spatial_domain_identification(
    adata: sc.AnnData,
    *,
    platform: str = "Visium",
    n_domains: Optional[int] = None,
    n_top_genes: int = 2000,
    epochs: int = 1270,
    use_morphological: bool = True,
    use_gpu: bool = True,
    device: torch.device | str | None = None,
    random_seed: int = 0,
    ground_truth: Union[str, Path, pd.Series, pd.DataFrame, None] = None,
    simclr_features_path: Optional[Union[str, Path]] = None,
    simclr_model_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    hidden1: int = 256,
    hidden2: int = 32,
    reg_in_channels: Optional[int] = None,
    reg_hidden1: int = 16,
    reg_hidden2: int = 8,
    gp_lambda: float = 5.0,
    lr: float = 0.001,
    reg_lr: float = 0.0005,
    dropout: float = 0.05,
    graph_neighbors: int = 10,
    denoise_iterations: int = 3,
    augmentation_iterations: int = 4,
    diffusion_k: float = 0.2,
    image_quality: str = "lowres",
) -> dict:
    """Run PearlST on a single spatial transcriptomics sample.

    The implementation follows the published PearlST method while avoiding the
    original GUI/Tk backend and optional dependencies such as stlearn/glob2.
    """
    if not validate_spatial_data(adata):
        raise ValueError("Invalid spatial coordinates")

    platform = _normalize_pearlst_platform(platform)
    morphology_requested = bool(use_morphological)
    morphology_for_check = morphology_requested and platform == "Visium"
    deps_ok, missing = check_pearlst_dependencies(use_morphological=morphology_for_check)
    if not deps_ok:
        raise ImportError(
            "PearlST dependencies are missing: "
            + ", ".join(sorted(set(missing)))
            + ". Install with: conda run -n sppy310 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
            + " ".join(sorted(set(missing)))
        )

    if n_top_genes != 2000:
        raise ValueError(
            "PearlST currently requires --n-top-genes 2000 because the upstream PDE "
            "augmentation reshapes each spot into a fixed 50x40 grid."
        )

    _set_random_seed(random_seed)
    resolved_device = resolve_compute_device(device, prefer_gpu=use_gpu)
    output_path = Path(output_dir or Path.cwd() / "pearlst_output").resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    adata = adata.copy()
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if adata.n_vars < n_top_genes:
        raise ValueError(
            f"PearlST requires at least {n_top_genes} genes after filtering, "
            f"but only {adata.n_vars} remain."
        )
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    if adata.n_vars != n_top_genes:
        raise ValueError(
            "PearlST preprocessing did not yield exactly 2000 HVGs; "
            f"received {adata.n_vars}."
        )

    raw_count = _ensure_dense_matrix(adata.X).astype(np.float32)
    denoised = _gene_data_denoising(raw_count, iterations=denoise_iterations, k=diffusion_k)
    adata.X = denoised

    simclr_output_path: Optional[Path] = None
    applied_morphology = morphology_requested and platform == "Visium"
    if platform == "Visium":
        adata, _ = _prepare_visium_image_coordinates(adata, image_quality=image_quality)
        if applied_morphology:
            if simclr_features_path:
                feature_matrix = _load_pearlst_image_feature_csv(simclr_features_path, adata.obs_names)
                simclr_output_path = Path(simclr_features_path)
            else:
                chosen_model = Path(simclr_model_path) if simclr_model_path else _default_pearlst_simclr_model_path()
                if not chosen_model.exists():
                    raise FileNotFoundError(
                        "PearlST SimCLR checkpoint not found. "
                        f"Expected: {chosen_model}"
                    )
                feature_matrix, simclr_output_path = _extract_pearlst_image_features(
                    adata,
                    model_path=chosen_model,
                    output_dir=output_path,
                    image_quality=image_quality,
                    device=resolved_device,
                )
            adata.obsm["image_feat"] = feature_matrix
            adata.obsm["image_feat_pca"] = feature_matrix
            adata = _cal_weighted_near_spots(
                adata,
                platform="Visium",
                no_morphological=False,
                weights="weights_matrix_all",
            )
        else:
            adata = _cal_weighted_near_spots(
                adata,
                platform="Visium",
                no_morphological=True,
                weights="weights_matrix_nomd",
            )
    else:
        if morphology_requested:
            logger.warning(
                "PearlST morphology is only supported for Visium-style inputs in this implementation. "
                "Proceeding without image features for platform %s.",
                platform,
            )
        applied_morphology = False
        adata = _cal_weighted_near_spots(
            adata,
            platform=platform,
            no_morphological=True,
            weights="weights_matrix_nomd",
        )

    augment_data = _gene_data_augmentation(
        _ensure_dense_matrix(adata.X).astype(np.float32),
        np.asarray(adata.obsm["near_spots"]),
        iterations=augmentation_iterations,
        k=diffusion_k,
    ).astype(np.float32)
    adata.obsm["augment_data"] = augment_data

    for key in (
        "weights_matrix_all",
        "weights_matrix_nomd",
        "near_spots",
        "adjacent_weight",
        "image_feat",
        "image_feat_pca",
    ):
        if key in adata.obsm:
            del adata.obsm[key]

    ground_truth_labels = _load_ground_truth_labels(ground_truth, adata.obs_names)
    if ground_truth_labels is not None:
        adata.obs["ground_truth"] = ground_truth_labels.astype("object")
        if n_domains is None:
            n_domains = int(pd.Series(ground_truth_labels.dropna()).nunique())

    if n_domains is None:
        n_domains = 7

    embedding, final_loss = _train_pearlst_warga(
        adata,
        epochs=epochs,
        hidden1=hidden1,
        hidden2=hidden2,
        reg_in_channels=reg_in_channels or hidden2,
        reg_hidden1=reg_hidden1,
        reg_hidden2=reg_hidden2,
        gp_lambda=gp_lambda,
        lr=lr,
        reg_lr=reg_lr,
        dropout=dropout,
        graph_neighbors=graph_neighbors,
        device=resolved_device,
    )
    adata.obsm["PearlST_embed"] = embedding
    kmeans = KMeans(n_clusters=n_domains, random_state=random_seed, n_init=10).fit(embedding)
    adata.obs["pred_label"] = pd.Series(kmeans.labels_, index=adata.obs_names).astype(str)

    ari = None
    if ground_truth_labels is not None:
        valid = ground_truth_labels.notna()
        if valid.any():
            ari = float(adjusted_rand_score(
                ground_truth_labels.loc[valid].astype(str),
                adata.obs.loc[valid, "pred_label"].astype(str),
            ))

    try:
        _compute_pearlst_layouts(adata, random_seed=random_seed)
    except Exception as exc:
        logger.warning("PearlST layout generation failed: %s", exc)

    return {
        "adata": adata,
        "embeddings": embedding,
        "n_domains": n_domains,
        "domain_column": "pred_label",
        "embedding_key": "PearlST_embed",
        "n_spots": adata.n_obs,
        "n_genes": adata.n_vars,
        "embedding_dim": embedding.shape[1],
        "epochs": epochs,
        "use_morphological": applied_morphology,
        "morphology_requested": morphology_requested,
        "platform": platform,
        "device": str(resolved_device),
        "random_seed": random_seed,
        "ari": ari,
        "graph_neighbors": graph_neighbors,
        "simclr_features_path": str(simclr_output_path) if simclr_output_path else None,
        "preprocessing": {
            "n_hvg_selected": adata.n_vars,
            "denoise_iterations": denoise_iterations,
            "augmentation_iterations": augmentation_iterations,
            "diffusion_k": diffusion_k,
            "final_loss": final_loss,
        },
    }
