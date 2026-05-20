#!/usr/bin/env python3
"""Spatial domain analysis using DeepST or PearlST.

This skill keeps SPATIALCLAW's directory-based input contract while exposing two
method backends:

- DeepST: single-sample domain identification or multi-sample integration
- PearlST: single-sample spatial transcriptomics with optional Visium histology

Usage:
    python spatialclaw.py run spatial-modality-integrate --input /path/to/sample --output <dir>
    python spatialclaw.py run spatial-modality-integrate --method pearlst --input /path/to/sample --output <dir>
    python spatialclaw.py run spatial-modality-integrate --mode integration --input-list <samples.txt> --output <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Required package missing: {e}")
    print("Install with: pip install scanpy matplotlib seaborn")
    sys.exit(1)

try:
    import deepstkit as dt
    from deepstkit.utils_func import refine as deepst_refine
except ImportError:
    dt = None
    deepst_refine = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from torchvision import models
except ImportError:
    models = None

from spatialclaw.common.checksums import sha256_file
from spatialclaw.common.report import (
    generate_report_footer,
    generate_report_header,
    write_result_json,
)
from skills.spatial._lib.adata_utils import store_analysis_metadata
from skills.spatial._lib.modality_integration import (
    SUPPORTED_MODALITY_METHODS,
    get_modality_method_spec,
    normalize_modality_method,
    run_pearlst_spatial_domain_identification,
)
from skills.spatial._lib.figure_io import save_figure, non_interactive_backend

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SKILL_NAME = "spatial-modality-integrate"
SKILL_VERSION = "2.2.0"
_DEEPST_IMAGE_CACHE_ROOT = _PROJECT_ROOT / "output" / ".cache" / "deepst_image_features"
_DEEPST_FAST_IMAGE_PCA_COMPONENTS = 50


def _require_deepst() -> None:
    if dt is None:
        raise ImportError(
            "DeepST backend is not installed. Install with: "
            "conda run -n sppy310 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
            "deepstkit"
        )


def _resolve_user_path(path_like: Union[str, Path]) -> Path:
    """Resolve user-provided paths relative to the project root when needed."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()

    project_candidate = (_PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate

    return path.resolve()


def _deepst_fast_image_dependencies_available() -> bool:
    return Image is not None and models is not None


def _select_deepst_image_quality(adata: sc.AnnData) -> tuple[str, np.ndarray, np.ndarray]:
    """Prefer low-resolution Visium imagery for faster morphology features."""
    spatial_block = adata.uns.get("spatial") or {}
    if not spatial_block:
        raise ValueError("DeepST morphology requires adata.uns['spatial'] image metadata.")

    library_id = next(iter(spatial_block))
    library = spatial_block[library_id]
    images = library.get("images", {})
    quality = "lowres" if "lowres" in images else ("hires" if "hires" in images else None)
    if quality is None:
        available = ", ".join(sorted(images))
        raise ValueError(f"No Visium tissue image found in adata.uns['spatial']; available keys: {available}")

    scale_key = f"tissue_{quality}_scalef"
    scale = float(library.get("scalefactors", {}).get(scale_key, 1.0))
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32) * scale
    image = np.asarray(images[quality])
    return quality, image, coords


def _deepst_image_cache_key(
    *,
    sample_name: str,
    sample_dir: Optional[Union[str, Path]],
    image_path: Optional[Path],
    quality: str,
    obs_names: pd.Index,
    pca_components: int,
    crop_size: int,
    target_size: int,
    cnn_type: str,
) -> str:
    sample_dir_str = str(_resolve_user_path(sample_dir)) if sample_dir else ""
    image_stat = image_path.stat() if image_path and image_path.exists() else None
    payload = {
        "sample_name": sample_name,
        "sample_dir": sample_dir_str,
        "image_path": str(image_path.resolve()) if image_path and image_path.exists() else "",
        "image_size": getattr(image_stat, "st_size", None),
        "image_mtime_ns": getattr(image_stat, "st_mtime_ns", None),
        "quality": quality,
        "obs_hash": hashlib.sha256("\n".join(map(str, obs_names)).encode("utf-8")).hexdigest(),
        "pca_components": pca_components,
        "crop_size": crop_size,
        "target_size": target_size,
        "cnn_type": cnn_type,
        "skill_version": SKILL_VERSION,
        "fast_image_backend": "resnet50_batched_in_memory",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _load_cached_deepst_image_features(
    cache_path: Path,
    obs_names: pd.Index,
) -> Optional[np.ndarray]:
    if not cache_path.exists():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=False)
        cached_obs_names = payload["obs_names"].astype(str)
        cached_index = pd.Index(cached_obs_names)
        current_index = pd.Index(obs_names.astype(str))
        if set(current_index) != set(cached_index):
            return None
        feature_frame = pd.DataFrame(
            payload["image_feat_pca"].astype(np.float32),
            index=cached_index,
        )
        return np.asarray(feature_frame.loc[current_index], dtype=np.float32)
    except Exception as exc:
        logger.warning("Could not read DeepST image feature cache %s: %s", cache_path, exc)
        return None


def _save_cached_deepst_image_features(
    cache_path: Path,
    obs_names: pd.Index,
    image_feat_pca: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    obs_array = np.asarray(obs_names.astype(str), dtype=f"<U{max(len(str(name)) for name in obs_names.astype(str)) if len(obs_names) else 1}")
    np.savez_compressed(
        cache_path,
        obs_names=obs_array,
        image_feat_pca=np.asarray(image_feat_pca, dtype=np.float32),
    )


def _extract_deepst_image_features_fast(
    adata: sc.AnnData,
    *,
    sample_name: str,
    sample_dir: Optional[Union[str, Path]],
    image_path: Optional[Path],
    use_gpu: bool,
    output_dir: Path,
    crop_size: int = 50,
    target_size: int = 224,
    pca_components: int = _DEEPST_FAST_IMAGE_PCA_COMPONENTS,
    cnn_type: str = "ResNet50",
) -> tuple[np.ndarray, dict]:
    """Extract DeepST morphology features without writing one PNG per spot."""
    if not _deepst_fast_image_dependencies_available():
        raise ImportError("Pillow and torchvision are required for the DeepST fast image path.")
    if cnn_type != "ResNet50":
        raise ValueError(f"Unsupported DeepST fast CNN type: {cnn_type}")

    quality, image, coords = _select_deepst_image_quality(adata)
    cache_key = _deepst_image_cache_key(
        sample_name=sample_name,
        sample_dir=sample_dir,
        image_path=image_path,
        quality=quality,
        obs_names=adata.obs_names,
        pca_components=pca_components,
        crop_size=crop_size,
        target_size=target_size,
        cnn_type=cnn_type,
    )
    cache_path = _DEEPST_IMAGE_CACHE_ROOT / f"{cache_key}.npz"
    cached = _load_cached_deepst_image_features(cache_path, adata.obs_names)
    if cached is not None:
        logger.info("Loaded cached DeepST image features for %s from %s", sample_name, cache_path)
        return cached, {
            "backend": "fast-cache",
            "image_quality": quality,
            "cache_hit": True,
            "cache_path": str(cache_path),
        }

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()

    if image.dtype in (np.float32, np.float64):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    resize_filter = getattr(Image, "Resampling", Image).BILINEAR
    batch_size = 128 if device.type == "cuda" else 32

    logger.info(
        "Extracting DeepST image features in memory for %s using %s image (%d spots, batch_size=%d, device=%s)",
        sample_name,
        quality,
        adata.n_obs,
        batch_size,
        device,
    )

    raw_features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, adata.n_obs, batch_size):
            batch_coords = coords[start:start + batch_size]
            tensors = []
            for image_col, image_row in batch_coords:
                left = image_col - crop_size / 2
                upper = image_row - crop_size / 2
                right = image_col + crop_size / 2
                lower = image_row + crop_size / 2
                tile = pil_image.crop((left, upper, right, lower))
                if target_size:
                    tile = tile.resize((target_size, target_size), resize_filter)
                tensors.append(preprocess(tile))
            batch = torch.stack(tensors, dim=0).to(device)
            raw_features.append(model(batch).cpu().numpy())

    feature_matrix = np.concatenate(raw_features, axis=0).astype(np.float32)
    n_pca = min(pca_components, feature_matrix.shape[0], feature_matrix.shape[1])
    if n_pca < 1:
        raise ValueError("DeepST image feature extraction produced no usable samples for PCA.")
    if n_pca == feature_matrix.shape[1]:
        image_feat_pca = feature_matrix
    else:
        pca = PCA(n_components=n_pca, random_state=0)
        image_feat_pca = pca.fit_transform(feature_matrix).astype(np.float32)

    _save_cached_deepst_image_features(cache_path, adata.obs_names, image_feat_pca)
    logger.info("Saved DeepST image feature cache for %s to %s", sample_name, cache_path)
    return image_feat_pca, {
        "backend": "fast-extract",
        "image_quality": quality,
        "cache_hit": False,
        "cache_path": str(cache_path),
    }


def _attach_deepst_image_features(
    adata: sc.AnnData,
    *,
    sample_name: str,
    sample_dir: Optional[Union[str, Path]],
    output_dir: Path,
    use_gpu: bool,
    deepst_instance: Optional[object] = None,
) -> tuple[sc.AnnData, dict]:
    """Attach morphology features for DeepST, preferring the faster in-memory path."""
    sample_path = _resolve_user_path(sample_dir) if sample_dir else None
    image_path, _ = _load_tissue_image(sample_path, prefer_lowres=True) if sample_path else (None, None)

    try:
        image_feat_pca, metadata = _extract_deepst_image_features_fast(
            adata,
            sample_name=sample_name,
            sample_dir=sample_path,
            image_path=image_path,
            use_gpu=use_gpu,
            output_dir=output_dir,
        )
        adata.obsm["image_feat_pca"] = image_feat_pca
        return adata, metadata
    except Exception as exc:
        logger.warning(
            "DeepST fast image feature extraction failed for %s: %s. "
            "Falling back to the DeepST crop-to-disk image feature path.",
            sample_name,
            exc,
        )
        if deepst_instance is None:
            raise
        adata = deepst_instance._get_image_crop(adata, data_name=sample_name)
        return adata, {
            "backend": "deepst-crop-to-disk",
            "image_quality": "crop-to-disk",
            "cache_hit": False,
            "cache_path": None,
        }


# ---------------------------------------------------------------------------
# Data Loading (using DeepST methods)
# ---------------------------------------------------------------------------


def _load_tissue_image(sample_path: Path, prefer_lowres: bool = True) -> tuple[Optional[Path], Optional[str]]:
    """
    Load H&E tissue image path from Visium-format directory.
    
    Searches for tissue_hires_image.png and tissue_lowres_image.png.
    Returns (image_path, image_type) or (None, None) if no image found.
    
    Parameters:
    -----------
    sample_path : Path
        Path to sample directory containing spatial/ subdirectory
    prefer_lowres : bool
        If True, prefer tissue_lowres_image.png over hires version (faster loading)
    
    Returns:
    --------
    tuple
        (image_path: Path or None, image_type: str or None)
        image_type is "hires" or "lowres"
    """
    spatial_dir = sample_path / "spatial"
    if not spatial_dir.exists():
        return None, None
    
    # Look for images in order of preference
    candidates = []
    if prefer_lowres:
        candidates = [
            (spatial_dir / "tissue_lowres_image.png", "lowres"),
            (spatial_dir / "tissue_hires_image.png", "hires"),
        ]
    else:
        candidates = [
            (spatial_dir / "tissue_hires_image.png", "hires"),
            (spatial_dir / "tissue_lowres_image.png", "lowres"),
        ]
    
    for img_path, img_type in candidates:
        if img_path.exists():
            logger.info(f"Found H&E image: {img_path.name} ({img_type})")
            return img_path, img_type
    
    logger.debug(f"No H&E tissue images found in {spatial_dir}")
    return None, None


def load_spatial_data(
    input_path: Union[str, Path],
    platform: str = "Visium",
    deepst_instance: Optional[object] = None,
) -> Optional[dict]:
    """
    Load spatial transcriptomics data using DeepST methods.
    
    Directory-only input mode:
    - input_path points directly to a sample directory
    - Example: /data/DLPFC/151673
    - Automatically resolves data_dir=/data/DLPFC and sample_id=151673
    
    Parameters:
    -----------
    input_path : str or Path
        Direct path to sample directory (e.g., /data/DLPFC/151673)
    platform : str
        Spatial platform: "Visium", "Stereo-seq", "Slide-seq", "MERFISH"
    deepst_instance : object, optional
        Initialized DeepST instance for data loading
        
    Returns:
    --------
    dict with keys:
        - "adata": AnnData object with expression and spatial coordinates
        - "sample_name": str, sample identifier
        - "input_path": str, path to H5 file
        - "platform": str, platform name
        - "input_hash": str or None, SHA256 hash of input file
        - "image_path": Path or None, path to H&E tissue image if found
        - "image_type": str or None, "hires" or "lowres" for image resolution
    """
    
    try:
        _require_deepst()
        sample_path = _resolve_user_path(input_path)
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample directory not found: {sample_path}")
        if not sample_path.is_dir():
            raise ValueError(
                "DeepST requires directory-based input. "
                f"--input must point to a sample directory, got file: {sample_path}"
            )

        data_dir = sample_path.parent
        sample_id = sample_path.name
        logger.info(
            f"Resolved sample directory {sample_path} -> "
            f"data_dir={data_dir}, sample_id={sample_id}"
        )

        # Initialize DeepST if not provided
        if deepst_instance is None:
            deepst_instance = dt.main.run(
                save_path=".",  # Temporary
                task="Identify_Domain",
                use_gpu=False,  # Don't initialize GPU yet
            )

        # Find the actual H5 file for hash computation
        h5_file_path = None

        # Use DeepST's built-in loader for Visium
        if platform.lower() == "visium":
            logger.info(f"Using DeepST's Visium loader for {sample_id}")
            adata = deepst_instance._get_adata(
                platform=platform,
                data_path=str(data_dir),
                data_name=sample_id,
                verbose=False,
            )
            logger.info(f"Loaded Visium data: {adata.shape[0]} spots × {adata.shape[1]} genes")
            logger.info(f"Spatial coordinates available: {'spatial' in adata.obsm}")

            # Find the H5 file for hashing
            h5_files = list(sample_path.glob("*_feature_bc_matrix.h5"))
            if h5_files:
                h5_file_path = h5_files[0]
        else:
            # For other platforms, use scanpy for the h5 file
            h5_files = list(sample_path.glob("*_feature_bc_matrix.h5"))
            if not h5_files:
                h5_files = list(sample_path.glob("*.h5"))

            if h5_files:
                adata = sc.read_h5(h5_files[0])
                h5_file_path = h5_files[0]
                logger.info(f"Loaded {platform} data from {h5_files[0]}")
            else:
                raise FileNotFoundError(f"No H5 file found in {sample_path}")

            # Try to load spatial coordinates if available
            coords_file = sample_path / "spatial" / "tissue_positions_list.csv"
            if coords_file.exists():
                coords_df = pd.read_csv(coords_file, header=None, index_col=0)
                # Visium format: barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
                if coords_df.shape[1] >= 4:
                    adata.obsm["spatial"] = coords_df.iloc[:, 2:4].values
                    logger.info(f"Loaded spatial coordinates from {coords_file}")

        # Load H&E tissue image if available
        image_path, image_type = _load_tissue_image(sample_path, prefer_lowres=True)

        return {
            "adata": adata,
            "sample_name": sample_id,
            "input_path": str(h5_file_path) if h5_file_path else str(sample_path),
            "platform": platform,
            "input_hash": sha256_file(h5_file_path) if h5_file_path else None,
            "image_path": image_path,
            "image_type": image_type,
        }
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def _hydrate_spatial_metadata_from_table(adata: sc.AnnData, sample_path: Path) -> sc.AnnData:
    """Backfill spatial obs columns from a simple metadata table when available."""
    metadata_candidates = [sample_path / "metadata.tsv", sample_path / "metadata.csv"]
    metadata_path = next((path for path in metadata_candidates if path.exists()), None)
    if metadata_path is None:
        return adata

    sep = "\t" if metadata_path.suffix.lower() == ".tsv" else ","
    meta = pd.read_csv(metadata_path, sep=sep, index_col=0)
    meta.index = meta.index.map(str)
    shared = adata.obs_names.intersection(meta.index)
    if shared.empty:
        return adata

    for column in ("array_row", "array_col", "imagerow", "imagecol"):
        if column in meta.columns and column not in adata.obs:
            adata.obs[column] = meta.reindex(adata.obs_names)[column]

    if "spatial" not in adata.obsm:
        if {"x", "y"}.issubset(meta.columns):
            adata.obsm["spatial"] = meta.reindex(adata.obs_names)[["x", "y"]].to_numpy()
        elif {"array_col", "array_row"}.issubset(meta.columns):
            adata.obsm["spatial"] = meta.reindex(adata.obs_names)[["array_col", "array_row"]].to_numpy()
        elif {"imagecol", "imagerow"}.issubset(meta.columns):
            adata.obsm["spatial"] = meta.reindex(adata.obs_names)[["imagecol", "imagerow"]].to_numpy()
    return adata


def load_pearlst_data(
    input_path: Union[str, Path],
    platform: str = "Visium",
) -> dict:
    """Load a single-sample PearlST input from a directory-based SPATIALCLAW input."""
    sample_path = _resolve_user_path(input_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_path}")
    if not sample_path.is_dir():
        raise ValueError(
            "spatial-modality-integrate expects --input to point to a sample directory."
        )

    sample_id = sample_path.name
    platform_key = str(platform or "Visium").strip().lower()
    image_path, image_type = _load_tissue_image(sample_path, prefer_lowres=True)
    input_file: Optional[Path] = None

    if platform_key == "visium":
        adata = sc.read_visium(sample_path)
        adata.var_names_make_unique()
        h5_files = list(sample_path.glob("*_feature_bc_matrix.h5"))
        if h5_files:
            input_file = h5_files[0]
    else:
        candidates = (
            list(sample_path.glob("*.h5ad"))
            + list(sample_path.glob("*.h5"))
            + list(sample_path.glob("*.loom"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .h5ad/.h5/.loom file found in {sample_path} for PearlST non-Visium input."
            )
        input_file = candidates[0]
        if input_file.suffix.lower() == ".h5ad":
            adata = sc.read_h5ad(input_file)
        elif input_file.suffix.lower() == ".loom":
            adata = sc.read_loom(input_file)
        else:
            try:
                adata = sc.read_h5ad(input_file)
            except Exception:
                adata = sc.read_10x_h5(input_file)
        adata.var_names_make_unique()
        adata = _hydrate_spatial_metadata_from_table(adata, sample_path)
        if "spatial" not in adata.obsm:
            raise ValueError(
                "PearlST requires spatial coordinates in adata.obsm['spatial'] for non-Visium inputs."
            )

    return {
        "adata": adata,
        "sample_name": sample_id,
        "input_path": str(input_file or sample_path),
        "platform": platform,
        "input_hash": sha256_file(input_file) if input_file and input_file.exists() else None,
        "image_path": image_path,
        "image_type": image_type,
    }


def _resolve_morphology_setting(
    requested: Optional[bool],
    sample_infos: list[dict],
    mode: str,
) -> dict:
    """Resolve morphology usage from CLI intent and tissue-image availability."""
    sample_names = [info.get("sample_name", "unknown") for info in sample_infos]
    samples_with_images = [
        info.get("sample_name", "unknown")
        for info in sample_infos
        if info.get("image_path")
    ]
    samples_without_images = [
        info.get("sample_name", "unknown")
        for info in sample_infos
        if not info.get("image_path")
    ]
    has_all_images = bool(sample_infos) and not samples_without_images

    if requested is False:
        logger.info("Morphology features explicitly disabled via --no-morphological.")
        return {
            "enabled": False,
            "requested": requested,
            "status_label": "Disabled (explicitly turned off)",
        }

    if mode == "single":
        sample_label = sample_names[0] if sample_names else "unknown"
        if requested is True:
            if samples_with_images:
                return {
                    "enabled": True,
                    "requested": requested,
                    "status_label": "Enabled (explicitly requested)",
                }
            logger.warning(
                "Requested morphology features for sample %s, but no H&E tissue image was found. "
                "Proceeding without morphological augmentation.",
                sample_label,
            )
            return {
                "enabled": False,
                "requested": requested,
                "status_label": "Disabled (no tissue image available)",
            }

        if samples_with_images:
            logger.info(
                "Auto-enabling morphology features because an H&E tissue image was detected for sample %s.",
                sample_label,
            )
            return {
                "enabled": True,
                "requested": requested,
                "status_label": "Enabled (auto-detected from tissue image)",
            }

        logger.info(
            "No H&E tissue image detected for sample %s. Morphology features remain disabled.",
            sample_label,
        )
        return {
            "enabled": False,
            "requested": requested,
            "status_label": "Disabled (no tissue image available)",
        }

    if requested is True:
        if has_all_images:
            return {
                "enabled": True,
                "requested": requested,
                "status_label": "Enabled (explicitly requested)",
            }
        if samples_with_images:
            logger.warning(
                "Requested morphology features for integration, but these samples do not contain H&E tissue images: %s. "
                "Disabling morphology so DeepST sees a consistent feature space across batches.",
                ", ".join(samples_without_images),
            )
            return {
                "enabled": False,
                "requested": requested,
                "status_label": "Disabled (not all samples have tissue images)",
            }
        logger.warning(
            "Requested morphology features for integration, but no H&E tissue images were found across samples. "
            "Disabling morphology augmentation."
        )
        return {
            "enabled": False,
            "requested": requested,
            "status_label": "Disabled (no tissue images available)",
        }

    if has_all_images:
        logger.info(
            "Auto-enabling morphology features because all %d integration samples contain H&E tissue images.",
            len(sample_infos),
        )
        return {
            "enabled": True,
            "requested": requested,
            "status_label": "Enabled (auto-detected from tissue images)",
        }

    if samples_with_images:
        logger.info(
            "Detected H&E tissue images for only a subset of integration samples (%s). "
            "Morphology remains disabled to keep DeepST inputs consistent across batches.",
            ", ".join(samples_with_images),
        )
        return {
            "enabled": False,
            "requested": requested,
            "status_label": "Disabled (not all samples have tissue images)",
        }

    logger.info("No H&E tissue images detected across integration samples. Morphology features remain disabled.")
    return {
        "enabled": False,
        "requested": requested,
        "status_label": "Disabled (no tissue images available)",
    }


# ---------------------------------------------------------------------------
# DeepST Analysis
# ---------------------------------------------------------------------------


def _prepare_deepst_expression_fast(
    adata: sc.AnnData,
    *,
    n_top_genes: int,
) -> tuple[sc.AnnData, dict]:
    """Reduce DeepST expression space before augmentation/PCA for faster runs."""
    adata = adata.copy()
    adata.var_names_make_unique()
    original_n_vars = int(adata.n_vars)

    if adata.n_vars <= n_top_genes:
        return adata, {
            "fast_hvg_applied": False,
            "n_features_original": original_n_vars,
            "n_features_selected": int(adata.n_vars),
        }

    work = adata.copy()
    sc.pp.filter_genes(work, min_cells=3)
    if work.n_vars <= n_top_genes:
        return work, {
            "fast_hvg_applied": False,
            "n_features_original": original_n_vars,
            "n_features_selected": int(work.n_vars),
        }

    hvg_probe = work.copy()
    sc.pp.normalize_total(hvg_probe, target_sum=1e4)
    sc.pp.log1p(hvg_probe)
    sc.pp.highly_variable_genes(
        hvg_probe,
        flavor="seurat",
        n_top_genes=min(n_top_genes, hvg_probe.n_vars),
    )
    keep_genes = hvg_probe.var_names[hvg_probe.var["highly_variable"].to_numpy()]
    prepared = work[:, keep_genes].copy()

    logger.info(
        "DeepST fast expression path reduced genes from %d to %d HVGs before augmentation/PCA.",
        original_n_vars,
        prepared.n_vars,
    )
    return prepared, {
        "fast_hvg_applied": True,
        "n_features_original": original_n_vars,
        "n_features_selected": int(prepared.n_vars),
    }


def _deepst_data_process_fast(
    adata: sc.AnnData,
    *,
    pca_n_comps: int,
) -> np.ndarray:
    """A lighter DeepST preprocessing path using float32 math and randomized PCA."""
    data = np.asarray(adata.obsm["augment_gene_data"], dtype=np.float32)
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    data = data / row_sums
    data = np.log1p(data).astype(np.float32, copy=False)

    gene_mean = data.mean(axis=0, keepdims=True, dtype=np.float32)
    gene_std = data.std(axis=0, keepdims=True, dtype=np.float32)
    gene_std[gene_std == 0] = 1.0
    data = (data - gene_mean) / gene_std

    n_comps = min(pca_n_comps, data.shape[0], data.shape[1])
    if n_comps < 1:
        raise ValueError("DeepST fast preprocessing cannot compute PCA with zero components.")

    pca = PCA(n_components=n_comps, svd_solver="randomized", random_state=0)
    return pca.fit_transform(data).astype(np.float32, copy=False)


def _ensure_dense_for_deepst_clustering(adata: sc.AnnData) -> sc.AnnData:
    if sp.issparse(adata.X):
        adata.X = np.asarray(adata.X.toarray(), dtype=np.float32)
    return adata


def _normalize_deepst_domain_count(n_domains: int, n_obs: int) -> int:
    """Clamp requested DeepST domain count to a valid range for the current sample size."""
    if n_obs < 2:
        raise ValueError("DeepST clustering requires at least two spots.")
    requested = max(2, int(n_domains))
    if requested > n_obs:
        logger.warning(
            "Requested n_domains=%d exceeds number of spots (%d); using %d instead.",
            requested,
            n_obs,
            n_obs,
        )
        requested = n_obs
    return requested


def _estimate_deepst_n_domains(
    embeddings: np.ndarray,
    *,
    random_seed: int = 0,
    max_clusters: int = 20,
    max_silhouette_samples: int = 2000,
) -> int:
    """Estimate a stable DeepST domain count with KMeans + silhouette on the embedding."""
    n_obs = int(embeddings.shape[0])
    if n_obs <= 2:
        return 2

    max_candidate = min(max_clusters, max(2, n_obs // 10), n_obs - 1)
    if max_candidate < 2:
        return min(7, n_obs)

    sample_indices = None
    embedding_eval = embeddings
    if n_obs > max_silhouette_samples:
        rng = np.random.default_rng(random_seed)
        sample_indices = np.sort(
            rng.choice(n_obs, size=max_silhouette_samples, replace=False)
        )
        embedding_eval = embeddings[sample_indices]

    silhouette_scores = {}
    for n in range(2, max_candidate + 1):
        labels = KMeans(
            n_clusters=n,
            random_state=random_seed,
            n_init=10,
        ).fit_predict(embeddings)
        eval_labels = labels if sample_indices is None else labels[sample_indices]
        if np.unique(eval_labels).size < 2:
            continue
        silhouette_scores[n] = float(silhouette_score(embedding_eval, eval_labels))

    if not silhouette_scores:
        fallback = min(7, n_obs)
        logger.info("DeepST automatic domain detection had no valid silhouette scores; using fallback n_domains=%d", fallback)
        return fallback

    best_n = max(silhouette_scores, key=silhouette_scores.get)
    logger.info(
        "Auto-detected %d DeepST domains using KMeans silhouette score %.4f",
        best_n,
        silhouette_scores[best_n],
    )
    return int(best_n)


def _refine_deepst_domains(
    adata: sc.AnnData,
    *,
    domain_key: str,
    output_key: str,
    batch_key: Optional[str] = None,
    shape: str = "hexagon",
) -> sc.AnnData:
    """Apply DeepST-style spatial refinement to domain labels when coordinates are available."""
    if "spatial" not in adata.obsm or deepst_refine is None:
        adata.obs[output_key] = pd.Categorical(adata.obs[domain_key].astype(str))
        return adata

    if batch_key and batch_key in adata.obs.columns:
        refined_records = []
        for batch_name in adata.obs[batch_key].astype(str).unique():
            sub = adata[adata.obs[batch_key].astype(str) == batch_name].copy()
            adj_2d = distance.cdist(
                np.asarray(sub.obsm["spatial"], dtype=np.float32),
                np.asarray(sub.obsm["spatial"], dtype=np.float32),
                "euclidean",
            )
            refined = deepst_refine(
                sub.obs_names.tolist(),
                sub.obs[domain_key].astype(str).tolist(),
                adj_2d,
                shape,
            )
            refined_records.extend(zip(sub.obs_names.tolist(), refined))
        refined_series = pd.Series(dict(refined_records), index=adata.obs_names)
    else:
        adj_2d = distance.cdist(
            np.asarray(adata.obsm["spatial"], dtype=np.float32),
            np.asarray(adata.obsm["spatial"], dtype=np.float32),
            "euclidean",
        )
        refined_series = pd.Series(
            deepst_refine(
                adata.obs_names.tolist(),
                adata.obs[domain_key].astype(str).tolist(),
                adj_2d,
                shape,
            ),
            index=adata.obs_names,
        )

    adata.obs[output_key] = pd.Categorical(refined_series.astype(str))
    return adata


def _cluster_deepst_embeddings(
    adata: sc.AnnData,
    *,
    deepst_instance: object,
    n_domains: int,
    batch_key: Optional[str] = None,
    use_obsm: str = "DeepST_embed",
    key_added: str = "DeepST_domain",
    output_key: str = "DeepST_refine_domain",
    shape: str = "hexagon",
    random_seed: int = 0,
) -> tuple[sc.AnnData, dict]:
    """Cluster DeepST embeddings, preferring the upstream path and falling back to KMeans if needed."""
    adata = _ensure_dense_for_deepst_clustering(adata)
    target_domains = _normalize_deepst_domain_count(n_domains, adata.n_obs)
    clustering_info = {
        "clustering_backend": "deepst-upstream",
        "clustering_fallback_used": False,
        "n_domains_requested": target_domains,
    }

    try:
        adata = deepst_instance._get_cluster_data(
            adata,
            n_domains=target_domains,
            priori=True,
            batch_key=batch_key,
            use_obsm=use_obsm,
            key_added=key_added,
            output_key=output_key,
            shape=shape,
        )
        resolved_key = output_key if output_key in adata.obs else key_added
        actual_n = int(pd.Series(adata.obs[resolved_key].astype(str)).nunique())
        clustering_info["n_domains_observed"] = actual_n

        if actual_n != target_domains:
            raise ValueError(
                f"DeepST returned {actual_n} domains after requesting {target_domains}"
            )
        return adata, clustering_info
    except Exception as exc:
        logger.warning(
            "DeepST upstream clustering failed or returned unstable domains (%s). "
            "Falling back to exact KMeans clustering on DeepST embeddings.",
            exc,
        )

    embeddings = np.asarray(adata.obsm[use_obsm], dtype=np.float32)
    labels = KMeans(
        n_clusters=target_domains,
        random_state=random_seed,
        n_init=10,
    ).fit_predict(embeddings)
    label_series = pd.Series(labels, index=adata.obs_names).astype(str)
    adata.obs[key_added] = pd.Categorical(label_series)
    adata = _refine_deepst_domains(
        adata,
        domain_key=key_added,
        output_key=output_key,
        batch_key=batch_key,
        shape=shape,
    )
    clustering_info.update(
        {
            "clustering_backend": "kmeans-fallback",
            "clustering_fallback_used": True,
            "n_domains_observed": int(label_series.nunique()),
        }
    )
    return adata, clustering_info


def run_deepst_single_sample(
    adata: sc.AnnData,
    sample_name: str,
    output_dir: Path,
    n_domains: Optional[int] = None,
    pre_epochs: int = 500,
    epochs: int = 500,
    pca_components: int = 200,
    use_morphological: bool = False,
    use_gpu: bool = True,
    random_seed: int = 0,
    sample_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Run DeepST for single-sample spatial domain identification.
    
    Parameters include sample_dir to optionally extract H&E image features.
    """
    _require_deepst()
    logger.info(f"Running DeepST on {sample_name}...")

    fast_expression_target = max(2000, min(3000, pca_components * 15))
    adata, expression_info = _prepare_deepst_expression_fast(
        adata,
        n_top_genes=fast_expression_target,
    )
    
    # Set random seed
    dt.utils_func.seed_torch(seed=random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Initialize DeepST
    deepst = dt.main.run(
        save_path=str(output_dir),
        task="Identify_Domain",
        pre_epochs=pre_epochs,
        epochs=epochs,
        use_gpu=use_gpu,
    )
    
    # Step 1: Optional - Extract H&E image features if morphological analysis requested
    if use_morphological and sample_dir:
        try:
            logger.info("Extracting H&E image features for %s using the DeepST fast path...", sample_name)
            adata, image_feature_info = _attach_deepst_image_features(
                adata,
                sample_name=sample_name,
                sample_dir=sample_dir,
                output_dir=output_dir,
                use_gpu=use_gpu,
                deepst_instance=deepst,
            )
            logger.info(
                "Attached DeepST image features for %s: shape=%s, backend=%s, quality=%s, cache_hit=%s",
                sample_name,
                getattr(adata.obsm.get("image_feat_pca"), "shape", "N/A"),
                image_feature_info.get("backend"),
                image_feature_info.get("image_quality"),
                image_feature_info.get("cache_hit"),
            )
        except Exception as e:
            logger.warning(
                f"Could not extract H&E image features for {sample_name}: {e}\n"
                "Proceeding without morphological augmentation."
            )
            use_morphological = False
            image_feature_info = {
                "backend": "disabled",
                "image_quality": None,
                "cache_hit": False,
                "cache_path": None,
            }
    elif use_morphological and not sample_dir:
        logger.warning(
            "Requested use_morphological=True, but sample_dir not provided. "
            "Cannot extract H&E image features. Proceeding without morphological augmentation."
        )
        use_morphological = False
        image_feature_info = {
            "backend": "disabled",
            "image_quality": None,
            "cache_hit": False,
            "cache_path": None,
        }
    else:
        image_feature_info = {
            "backend": "disabled",
            "image_quality": None,
            "cache_hit": False,
            "cache_path": None,
        }
    
    # Verify image features exist if morphological augmentation requested
    if use_morphological and "image_feat_pca" not in adata.obsm:
        logger.warning(
            "Requested use_morphological=True, but image_feat_pca not found in adata.obsm. "
            "Disabling morphological augmentation."
        )
        use_morphological = False
    
    # Data augmentation
    logger.info("Performing data augmentation...")
    adata = deepst._get_augment(
        adata,
        spatial_type="BallTree",
        use_morphological=use_morphological,
    )
    
    # Graph construction
    logger.info("Constructing spatial graph...")
    graph_dict = deepst._get_graph(
        adata.obsm["spatial"],
        distType="KDTree",
    )
    
    # Data processing (PCA)
    logger.info(
        "Performing DeepST fast preprocessing to %d PCA components on %d genes...",
        pca_components,
        adata.n_vars,
    )
    data = _deepst_data_process_fast(
        adata,
        pca_n_comps=pca_components,
    )
    
    # Model training
    logger.info("Training DeepST model...")
    deepst_embed = deepst._fit(
        data=data,
        graph_dict=graph_dict,
    )
    
    # Store embeddings
    adata.obsm["DeepST_embed"] = deepst_embed
    
    # Clustering
    if n_domains is None:
        logger.info("Auto-detecting number of DeepST domains from the embedding...")
        n_domains = _estimate_deepst_n_domains(
            deepst_embed,
            random_seed=random_seed,
        )
    else:
        n_domains = _normalize_deepst_domain_count(n_domains, adata.n_obs)

    logger.info(
        "DeepST clustering will respect the target domain count (%d) for a stable finish.",
        n_domains,
    )
    logger.info("Clustering into %d spatial domains...", n_domains)
    adata, clustering_info = _cluster_deepst_embeddings(
        adata,
        deepst_instance=deepst,
        n_domains=n_domains,
        random_seed=random_seed,
    )
    
    return {
        "adata": adata,
        "embeddings": deepst_embed,
        "n_domains": n_domains,
        "sample_name": sample_name,
        "deepst_instance": deepst,
        # Summary statistics
        "n_spots": adata.shape[0],
        "n_genes": adata.shape[1],
        "embedding_dim": deepst_embed.shape[1],
        "pre_epochs": pre_epochs,
        "epochs": epochs,
        "pca_components": pca_components,
        "use_morphological": use_morphological,
        "use_gpu": use_gpu,
        "random_seed": random_seed,
        "preprocessing": {
            "n_features_original": expression_info["n_features_original"],
            "n_hvg_selected": expression_info["n_features_selected"],
            "fast_hvg_applied": expression_info["fast_hvg_applied"],
        },
        "image_feature_backend": image_feature_info.get("backend"),
        "image_feature_quality": image_feature_info.get("image_quality"),
        "image_feature_cache_hit": image_feature_info.get("cache_hit"),
        "image_feature_cache_path": image_feature_info.get("cache_path"),
        "clustering_backend": clustering_info.get("clustering_backend"),
        "clustering_fallback_used": clustering_info.get("clustering_fallback_used"),
    }


def run_deepst_integration(
    adata_list: list[sc.AnnData],
    sample_ids: list[str],
    output_dir: Path,
    n_domains: Optional[int] = None,
    batch_key: str = "batch",
    pre_epochs: int = 500,
    epochs: int = 600,
    pca_components: int = 200,
    use_morphological: bool = False,
    use_gpu: bool = True,
    random_seed: int = 0,
) -> dict:
    """Run DeepST for multi-sample integration with batch correction."""
    _require_deepst()
    logger.info(f"Running DeepST integration on {len(adata_list)} samples...")
    
    # Set random seed
    dt.utils_func.seed_torch(seed=random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # SAFETY CHECK: Integration needs image features for every sample to keep batches comparable.
    missing_image_features = [
        sample_id
        for adata, sample_id in zip(adata_list, sample_ids)
        if "image_feat_pca" not in adata.obsm
    ]
    if use_morphological and missing_image_features:
        logger.warning(
            "Requested use_morphological=True, but image features are missing for these samples: %s. "
            "Disabling morphological augmentation so integration uses a consistent feature space.",
            ", ".join(missing_image_features),
        )
        use_morphological = False
    
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
        
        # Add batch identifier
        adata.obs[batch_key] = sample_id
        
        # Augmentation
        adata = integration_model._get_augment(
            adata,
            spatial_type="BallTree",
            use_morphological=use_morphological,
        )
        
        # Graph construction
        graph = integration_model._get_graph(
            adata.obsm["spatial"],
            distType="KDTree",
        )
        
        processed_data.append(adata)
        spatial_graphs.append(graph)
    
    # Combine multiple samples
    logger.info("Combining samples into unified dataset...")
    combined_adata, combined_graph = integration_model._get_multiple_adata(
        adata_list=processed_data,
        data_name_list=sample_ids,
        graph_list=spatial_graphs,
    )
    
    # Data processing
    logger.info(f"Performing PCA to {pca_components} components...")
    integrated_data = integration_model._data_process(
        combined_adata,
        pca_n_comps=pca_components,
    )
    
    # Model training with domain adversarial learning
    logger.info("Training integrated DeepST model with batch correction...")
    embeddings = integration_model._fit(
        data=integrated_data,
        graph_dict=combined_graph,
        domains=combined_adata.obs[batch_key].values,
        n_domains=len(sample_ids),
    )
    
    combined_adata.obsm["DeepST_embed"] = embeddings
    
    # Clustering
    if n_domains is None:
        n_domains = 8  # Default for integration
    
    logger.info(f"Clustering into {n_domains} spatial domains...")
    n_domains = _normalize_deepst_domain_count(n_domains, combined_adata.n_obs)
    combined_adata, clustering_info = _cluster_deepst_embeddings(
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
        "integration_model": integration_model,
        # Summary statistics
        "n_spots": combined_adata.shape[0],
        "n_genes": combined_adata.shape[1],
        "n_samples": len(sample_ids),
        "embedding_dim": embeddings.shape[1],
        "pre_epochs": pre_epochs,
        "epochs": epochs,
        "pca_components": pca_components,
        "use_morphological": use_morphological,
        "use_gpu": use_gpu,
        "random_seed": random_seed,
        "clustering_backend": clustering_info.get("clustering_backend"),
        "clustering_fallback_used": clustering_info.get("clustering_fallback_used"),
        "preprocessing": {
            "n_features_original": combined_adata.shape[1],
            "n_hvg_selected": combined_adata.n_vars,
        },
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def write_report(
    output_dir: Path,
    summary: dict,
    input_file: Optional[str] = None,
    params: Optional[dict] = None,
) -> None:
    """Generate a method-aware spatial modality analysis report."""

    params = params or {}
    method = summary.get("method", "deepst")
    method_spec = get_modality_method_spec(method)
    method_label = summary.get("method_label", method.upper())
    mode = summary.get("mode", "single")
    embedding_key = summary.get("embedding_key", method_spec.embedding_key)
    domain_column = summary.get("domain_column", method_spec.cluster_key)
    morphology_status = summary.get("morphology_status") or (
        "Enabled (H&E)" if summary.get("use_morphological", False) else "Disabled"
    )

    input_files_list = []
    if input_file:
        input_path = Path(input_file)
        if input_path.is_file():
            input_files_list = [input_path]

    header = generate_report_header(
        title=f"{method_label} Spatial Domain Analysis Report",
        skill_name=SKILL_NAME,
        input_files=input_files_list,
        extra_metadata={
            "Method": method_label,
            "Mode": mode,
        },
    )

    body_lines = [
        "## Executive Summary\n",
    ]
    if method == "deepst":
        body_lines.extend([
            "DeepST combines spatial structure, transcriptomic variation, and optional morphology features",
            "to identify tissue domains; in integration mode it also performs adversarial batch correction.",
        ])
    else:
        body_lines.extend([
            "PearlST is a PDE-enhanced adversarial graph autoencoder for single-sample spatial transcriptomics.",
            "This implementation follows the upstream PearlST pipeline and optionally adds Visium histology features",
            "through the bundled SimCLR image encoder.",
        ])

    body_lines.extend([
        "",
        "## Data Overview\n",
        f"- **Total Spots**: {summary['n_spots']}",
        f"- **Total Genes**: {summary['n_genes']}",
        f"- **Samples**: {summary.get('n_samples', 1)}",
        f"- **Platform**: {summary.get('platform', 'unknown')}",
        f"- **H&E Image**: {'Available (' + str(summary.get('image_type', 'unknown')) + ')' if summary.get('image_path') else 'Not available'}",
        "",
        "## Analysis Mode\n",
    ])

    if mode == "integration":
        body_lines.extend([
            "- **Type**: Multi-sample integration with batch correction",
            f"- **Number of samples**: {summary.get('n_samples', 'unknown')}",
            "- **Batch correction**: Automatic via adversarial training",
            f"- **Morphology features**: {morphology_status}",
        ])
    else:
        body_lines.extend([
            "- **Type**: Single-sample spatial domain identification",
            f"- **Morphology features**: {morphology_status}",
        ])
        if method == "pearlst":
            body_lines.append("- **PearlST constraint**: current implementation supports single-sample mode only")

    body_lines.extend([
        "",
        "## Domain Detection Results\n",
        f"- **Spatial domains identified**: {summary['n_domains']}",
        f"- **Embedding dimensionality**: {summary.get('embedding_dim', 'auto')}",
        f"- **Embedding key**: `{embedding_key}`",
        f"- **Domain column**: `{domain_column}`",
    ])
    if method == "pearlst":
        body_lines.append("- **Clustering method**: KMeans on PearlST embeddings")
        if summary.get("ari") is not None:
            body_lines.append(f"- **Adjusted Rand Index (ARI)**: {summary['ari']:.4f}")
    else:
        if summary.get("clustering_backend") == "kmeans-fallback":
            body_lines.append("- **Clustering method**: KMeans fallback on DeepST embeddings with spatial refinement")
        else:
            body_lines.append("- **Clustering method**: DeepST clustering / Leiden-style refinement")

    body_lines.extend([
        "",
        "## Method Notes\n",
    ])
    if method == "deepst":
        body_lines.extend([
            "- Spatial graph construction couples nearby spots before graph representation learning.",
            "- Optional morphology augmentation uses tissue image crops when H&E images are available.",
            "- Integration mode adds domain adversarial learning to reduce batch effects across samples.",
        ])
    else:
        body_lines.extend([
            "- PearlST first filters genes and keeps exactly 2000 highly variable genes.",
            "- It then performs PDE-based denoising and neighbor-aware augmentation of the expression matrix.",
            "- The spatial graph is built with an alpha complex using `gudhi`, matching the upstream code.",
            "- The final latent space is trained with a Wasserstein adversarial graph autoencoder (WARGA).",
            "- Histology features are currently supported for Visium inputs; other PearlST-supported platforms run without image features.",
        ])

    body_lines.extend([
        "",
        "## Preprocessing Details\n",
    ])
    if summary.get("preprocessing"):
        pp = summary["preprocessing"]
        for key, value in pp.items():
            body_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    if method == "deepst":
        body_lines.append(f"- **PCA components**: {summary.get('pca_components', 'N/A')}")
        if summary.get("image_feature_backend"):
            body_lines.append(f"- **Morphology feature backend**: {summary['image_feature_backend']}")
        if summary.get("image_feature_quality"):
            body_lines.append(f"- **Morphology image quality**: {summary['image_feature_quality']}")
        if summary.get("image_feature_cache_hit") is not None:
            body_lines.append(f"- **Morphology feature cache hit**: {summary['image_feature_cache_hit']}")
    else:
        body_lines.append(f"- **HVG target**: {params.get('n_top_genes', 2000)}")

    body_lines.extend([
        "",
        "## Training Configuration\n",
    ])
    if summary.get("pre_epochs") is not None:
        body_lines.append(f"- **Pretraining epochs**: {summary['pre_epochs']}")
    body_lines.extend([
        f"- **Main training epochs**: {summary['epochs']}",
        f"- **Compute device**: {summary.get('device', 'gpu' if summary.get('use_gpu', False) else 'cpu')}",
        f"- **Random seed**: {summary['random_seed']}",
    ])
    if params.get("simclr_features"):
        body_lines.append(f"- **Provided simCLR features**: `{params['simclr_features']}`")
    if summary.get("simclr_features_path") and not params.get("simclr_features"):
        body_lines.append(f"- **Generated simCLR features**: `{summary['simclr_features_path']}`")

    body_lines.extend([
        "",
        "## Output Representations\n",
        f"- **{embedding_key}**: learned low-dimensional representation",
        f"- **{domain_column}**: predicted spatial domain assignment",
    ])
    if method == "deepst":
        body_lines.append("- **DeepST_domain / DeepST_refine_domain**: original and refined domain labels")
    if summary.get("has_pseudotime"):
        body_lines.append("- **PearlST_pseudotime**: DPT-based pseudotemporal ordering derived from PearlST embeddings")

    body_lines.extend([
        "",
        "## Parameters Used\n",
    ])
    for key, value in params.items():
        if value is None:
            continue
        body_lines.append(f"- `{key}`: {value}")

    body_lines.extend([
        "",
        "## Downstream Analysis Recommendations\n",
        "- Validate the predicted domains against marker genes, tissue annotations, or known histology layers.",
        "- Reuse the learned embedding for neighborhood analysis, trajectory inference, or cross-sample comparison.",
        "- Inspect spatial continuity and cluster size balance before interpreting small domains biologically.",
        "",
        "## References\n",
    ])
    if method == "deepst":
        body_lines.extend([
            "- **DeepST**: https://github.com/JiangBioLab/DeepST",
            "- **Core idea**: graph neural networks with denoising autoencoder components",
        ])
    else:
        body_lines.extend([
            "- **PearlST**: https://github.com/SunXQlab/PearlST-Code",
            "- **Core idea**: PDE-enhanced augmentation plus Wasserstein adversarial graph autoencoding",
        ])
    body_lines.append("")

    footer = generate_report_footer()
    report = header + "\n".join(body_lines) + "\n" + footer
    report_path = output_dir / "report.md"
    report_path.write_text(report)
    logger.info(f"Saved report: {report_path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_figures(adata: sc.AnnData, output_dir: Path, summary: dict, mode: str = "single") -> list[str]:
    """Generate method-aware spatial domain visualizations."""
    method = summary.get("method", "deepst")
    method_label = summary.get("method_label", method.upper())
    domain_col = summary.get("domain_column")
    if not domain_col or domain_col not in adata.obs:
        for candidate in ("DeepST_refine_domain", "DeepST_domain", "pred_label"):
            if candidate in adata.obs:
                domain_col = candidate
                break
    embedding_key = summary.get("embedding_key")
    if not embedding_key or embedding_key not in adata.obsm:
        for candidate in ("DeepST_embed", "PearlST_embed"):
            if candidate in adata.obsm:
                embedding_key = candidate
                break

    figures: list[str] = []
    with non_interactive_backend():
        if "spatial" in adata.obsm and domain_col in adata.obs:
            try:
                logger.info("Creating spatial domain visualization...")
                fig, ax = plt.subplots(figsize=(10, 9))
                spatial = adata.obsm["spatial"]
                domains = pd.Categorical(adata.obs[domain_col]).codes
                scatter = ax.scatter(
                    spatial[:, 0],
                    spatial[:, 1],
                    c=domains,
                    cmap="tab20",
                    s=50,
                    alpha=0.85,
                    edgecolors="k",
                    linewidth=0.3,
                )
                ax.set_xlabel("X (spatial)")
                ax.set_ylabel("Y (spatial)")
                ax.set_title(f"{method_label} Spatial Domains ({summary['n_domains']} domains)")
                ax.set_aspect("equal")
                plt.colorbar(scatter, ax=ax, label="Domain ID")
                fig.tight_layout()
                p = save_figure(fig, output_dir, "spatial_domain_identification.png", dpi=200)
                figures.append(str(p))
            except Exception as e:
                logger.warning(f"Could not generate spatial domain figure: {e}")

        if embedding_key in adata.obsm and domain_col in adata.obs and adata.n_obs > 2:
            try:
                logger.info("Creating UMAP and spatial comparison...")
                if method == "pearlst" and "PearlST_umap" in adata.obsm:
                    umap = adata.obsm["PearlST_umap"]
                else:
                    adata_umap = adata.copy()
                    n_neighbors = max(2, min(15, adata.shape[0] - 1))
                    sc.pp.neighbors(adata_umap, use_rep=embedding_key, n_neighbors=n_neighbors)
                    sc.tl.umap(adata_umap, random_state=summary.get("random_seed", 0))
                    umap = adata_umap.obsm["X_umap"]

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                domains = pd.Categorical(adata.obs[domain_col]).codes
                scatter1 = axes[0].scatter(
                    umap[:, 0],
                    umap[:, 1],
                    c=domains,
                    cmap="tab20",
                    s=30,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.2,
                )
                axes[0].set_xlabel("UMAP 1")
                axes[0].set_ylabel("UMAP 2")
                axes[0].set_title(f"{method_label} Embeddings — UMAP")
                axes[0].set_aspect("equal")
                plt.colorbar(scatter1, ax=axes[0], label="Domain")

                if "spatial" in adata.obsm:
                    spatial = adata.obsm["spatial"]
                    scatter2 = axes[1].scatter(
                        spatial[:, 0],
                        spatial[:, 1],
                        c=domains,
                        cmap="tab20",
                        s=30,
                        alpha=0.8,
                        edgecolors="black",
                        linewidth=0.2,
                    )
                    axes[1].set_xlabel("X")
                    axes[1].set_ylabel("Y")
                    axes[1].set_title(f"{method_label} Domains — Spatial")
                    axes[1].set_aspect("equal")
                    plt.colorbar(scatter2, ax=axes[1], label="Domain")

                fig.tight_layout()
                p = save_figure(fig, output_dir, "umap_spatial_comparison.png", dpi=200)
                figures.append(str(p))
            except Exception as e:
                logger.warning(f"Could not generate UMAP figure: {e}")

        if domain_col in adata.obs:
            try:
                logger.info("Creating domain statistics...")
                domain_counts = adata.obs[domain_col].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(11, 5))
                bars = ax.bar(
                    range(len(domain_counts)),
                    domain_counts.values,
                    color=plt.cm.tab20(np.linspace(0, 1, len(domain_counts))),
                    edgecolor="black",
                    linewidth=1,
                )
                ax.set_xlabel("Domain ID", fontsize=11)
                ax.set_ylabel("Number of Spots", fontsize=11)
                ax.set_title(f"Spatial Domain Size Distribution ({domain_counts.sum()} spots total)")
                ax.set_xticks(range(len(domain_counts)))
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom", fontsize=9)
                fig.tight_layout()
                p = save_figure(fig, output_dir, "domain_sizes.png", dpi=200)
                figures.append(str(p))
            except Exception as e:
                logger.warning(f"Could not generate domain sizes figure: {e}")

        batch_key = summary.get("batch_key", "batch")
        if mode == "integration" and batch_key in adata.obs and "spatial" in adata.obsm and domain_col in adata.obs:
            try:
                logger.info("Creating batch/sample comparison...")
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                spatial = adata.obsm["spatial"]
                domains = pd.Categorical(adata.obs[domain_col]).codes
                scatter1 = axes[0].scatter(
                    spatial[:, 0],
                    spatial[:, 1],
                    c=domains,
                    cmap="tab20",
                    s=30,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.2,
                )
                axes[0].set_xlabel("X")
                axes[0].set_ylabel("Y")
                axes[0].set_title("Integrated Spatial Domains")
                axes[0].set_aspect("equal")
                plt.colorbar(scatter1, ax=axes[0], label="Domain")

                batch_codes = pd.Categorical(adata.obs[batch_key]).codes
                scatter2 = axes[1].scatter(
                    spatial[:, 0],
                    spatial[:, 1],
                    c=batch_codes,
                    cmap="Set2",
                    s=30,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.2,
                )
                axes[1].set_xlabel("X")
                axes[1].set_ylabel("Y")
                axes[1].set_title("Sample/Batch Assignment")
                axes[1].set_aspect("equal")
                plt.colorbar(scatter2, ax=axes[1], label="Sample ID")
                fig.tight_layout()
                p = save_figure(fig, output_dir, "batch_comparison.png", dpi=200)
                figures.append(str(p))
            except Exception as e:
                logger.warning(f"Could not generate batch comparison figure: {e}")

        if method == "pearlst" and "PearlST_pseudotime" in adata.obs and "spatial" in adata.obsm:
            try:
                logger.info("Creating PearlST pseudotime visualization...")
                fig, ax = plt.subplots(figsize=(8, 7))
                spatial = adata.obsm["spatial"]
                scatter = ax.scatter(
                    spatial[:, 0],
                    spatial[:, 1],
                    c=adata.obs["PearlST_pseudotime"],
                    cmap="viridis",
                    s=35,
                    alpha=0.85,
                    edgecolors="none",
                )
                ax.set_xlabel("X (spatial)")
                ax.set_ylabel("Y (spatial)")
                ax.set_title("PearlST Pseudotime")
                ax.set_aspect("equal")
                plt.colorbar(scatter, ax=ax, label="Pseudotime")
                fig.tight_layout()
                p = save_figure(fig, output_dir, "pearlst_pseudotime.png", dpi=200)
                figures.append(str(p))
            except Exception as e:
                logger.warning(f"Could not generate PearlST pseudotime figure: {e}")

        try:
            logger.info("Creating workflow summary...")
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis("off")
            workflow_lines = [
                f"{method_label} Spatial Analysis Workflow",
                "",
                f"Analysis Mode: {mode.upper()} {'(Multi-Sample Integration)' if mode == 'integration' else '(Single-Sample)'}",
                "",
                "Input Data:",
                f"  • Spots: {summary['n_spots']}",
                f"  • Genes: {summary['n_genes']}",
                f"  • Samples: {summary.get('n_samples', 1)}",
                "",
                "Key Outputs:",
                f"  • Domains: {summary['n_domains']}",
                f"  • Embedding key: {embedding_key or 'n/a'}",
                f"  • Domain column: {domain_col or 'n/a'}",
                f"  • Epochs: {summary.get('epochs', 'n/a')}",
                f"  • Morphology: {summary.get('morphology_status', 'unknown')}",
            ]
            if method == "deepst":
                workflow_lines.extend([
                    "",
                    "Pipeline:",
                    "  1. Spatial augmentation and graph construction",
                    f"  2. PCA preprocessing to {summary.get('pca_components', 'n/a')} components",
                    "  3. DeepST graph autoencoder training",
                ])
                if mode == "integration":
                    workflow_lines.append("  4. Domain adversarial batch correction")
            else:
                workflow_lines.extend([
                    "",
                    "Pipeline:",
                    "  1. HVG selection to 2000 genes",
                    "  2. PDE denoising and neighbor-aware augmentation",
                    "  3. Alpha-complex graph construction",
                    "  4. WARGA training and KMeans clustering",
                ])
                if "PearlST_pseudotime" in adata.obs:
                    workflow_lines.append("  5. UMAP / pseudotime layout generation")

            workflow_text = "\n".join(workflow_lines)
            ax.text(
                0.05,
                0.95,
                workflow_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.4, pad=1),
            )
            fig.tight_layout()
            p = save_figure(fig, output_dir, "workflow_summary.png", dpi=200)
            figures.append(str(p))
        except Exception as e:
            logger.warning(f"Could not generate workflow summary: {e}")

    return figures


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Spatial modality analysis using DeepST or PearlST"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="deepst",
        help="Backend method: deepst or pearlst",
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="single", choices=["single", "integration"],
        help="Analysis mode: single-sample or multi-sample integration"
    )
    
    # Single-sample input: full path to sample directory
    parser.add_argument(
        "--input", type=str, 
        help="Full path to sample directory (e.g., /data/DLPFC/151673)."
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        default="Visium",
        choices=["Visium", "Stereo-seq", "Slide-seq", "Slide-seqV2", "MERFISH", "STARmap"],
        help="Spatial platform type",
    )
    
    # Multi-sample inputs
    parser.add_argument(
        "--input-list",
        type=str,
        help="File listing sample directories (one per line, for integration mode)",
    )
    
    # Output
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    
    # DeepST parameters
    parser.add_argument("--pre-epochs", type=int, default=None, help="DeepST pretraining epochs")
    parser.add_argument("--epochs", type=int, default=None, help="Main training epochs")
    parser.add_argument("--pca-components", type=int, default=200, help="PCA components")
    parser.add_argument("--n-top-genes", type=int, default=2000, help="PearlST highly variable genes (must be 2000)")
    parser.add_argument("--n-domains", type=int, help="Target number of domains (auto-detect if not set)")
    parser.add_argument(
        "--use-morphological",
        dest="use_morphological",
        action="store_true",
        default=None,
        help="Force-enable H&E morphological features",
    )
    parser.add_argument(
        "--no-morphological",
        dest="use_morphological",
        action="store_false",
        help="Disable H&E morphological features even when tissue images are present",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        default=None,
        help="Prefer GPU acceleration when available",
    )
    parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Force CPU execution",
    )
    parser.add_argument("--batch-key", type=str, default="batch", help="Batch identifier column")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, help="Device (cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument("--ground-truth", type=str, help="Optional CSV/TSV file with barcode-indexed domain labels")
    parser.add_argument("--simclr-features", type=str, help="Optional precomputed PearlST simCLR feature CSV")
    parser.add_argument("--simclr-model", type=str, help="Optional PearlST SimCLR checkpoint path")
    
    args = parser.parse_args()
    args.method = normalize_modality_method(args.method)
    method_spec = get_modality_method_spec(args.method)
    if args.use_gpu is None:
        args.use_gpu = True
    for flag, value in (
        ("--pre-epochs", args.pre_epochs),
        ("--epochs", args.epochs),
        ("--pca-components", args.pca_components),
        ("--n-top-genes", args.n_top_genes),
        ("--n-domains", args.n_domains),
    ):
        if value is not None and value <= 0:
            parser.error(f"{flag} must be a positive integer.")
    if args.epochs is None:
        args.epochs = method_spec.default_epochs
    if args.pre_epochs is None:
        args.pre_epochs = 500 if args.method == "deepst" else None
    
    try:
        # ===== SETUP =====
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {"figures": [], "n_domains": 0, "samples": []}
        morphology = {
            "enabled": False,
            "requested": args.use_morphological,
            "status_label": "Disabled",
        }
        
        if args.input:
            input_path = _resolve_user_path(args.input)
            if not input_path.is_dir():
                raise ValueError(f"--input must point to a directory. Got: {args.input}")
        
        # ===== SINGLE SAMPLE MODE =====
        if args.mode == "single":
            if not args.input:
                raise ValueError(
                    "spatial-modality-integrate requires directory-based input.\n"
                    "Specify:\n"
                    "  --input /path/to/sample/directory"
                )

            loader = load_spatial_data if args.method == "deepst" else load_pearlst_data
            input_info = loader(
                input_path=args.input,
                platform=args.platform,
            )
            adata = input_info["adata"]
            sample_name = input_info["sample_name"]
            morphology = _resolve_morphology_setting(
                requested=args.use_morphological,
                sample_infos=[input_info],
                mode="single",
            )
            args.use_morphological = morphology["enabled"]
            if args.method == "pearlst" and str(args.platform).lower() != "visium" and args.use_morphological:
                args.use_morphological = False
                morphology["enabled"] = False
                morphology["status_label"] = "Disabled (PearlST histology currently requires Visium)"

            if args.method == "deepst":
                summary = run_deepst_single_sample(
                    adata=adata,
                    sample_name=sample_name,
                    output_dir=output_dir,
                    n_domains=args.n_domains,
                    pre_epochs=args.pre_epochs or 500,
                    epochs=args.epochs,
                    pca_components=args.pca_components,
                    use_morphological=args.use_morphological,
                    use_gpu=args.use_gpu,
                    random_seed=args.random_seed,
                    sample_dir=args.input,
                )
                output_h5ad = output_dir / f"{sample_name}_deepst.h5ad"
            else:
                summary = run_pearlst_spatial_domain_identification(
                    adata=adata,
                    platform=args.platform,
                    n_domains=args.n_domains,
                    n_top_genes=args.n_top_genes,
                    epochs=args.epochs,
                    use_morphological=args.use_morphological,
                    use_gpu=args.use_gpu,
                    device=args.device,
                    random_seed=args.random_seed,
                    ground_truth=_resolve_user_path(args.ground_truth) if args.ground_truth else None,
                    simclr_features_path=_resolve_user_path(args.simclr_features) if args.simclr_features else None,
                    simclr_model_path=_resolve_user_path(args.simclr_model) if args.simclr_model else None,
                    output_dir=output_dir,
                )
                output_h5ad = output_dir / f"{sample_name}_pearlst.h5ad"

            if not summary.get("use_morphological", False):
                args.use_morphological = False
                if morphology["enabled"]:
                    morphology["enabled"] = False
                    morphology["status_label"] = "Disabled (image feature extraction unavailable)"

            summary = {
                **summary,
                "method": args.method,
                "method_label": method_spec.label,
                "method_description": method_spec.description,
                "mode": "single",
                "n_samples": 1,
                "platform": input_info.get("platform"),
                "image_path": input_info.get("image_path"),
                "image_type": input_info.get("image_type"),
                "morphology_status": morphology["status_label"],
                "domain_column": summary.get("domain_column", method_spec.cluster_key),
                "embedding_key": summary.get("embedding_key", method_spec.embedding_key),
                "has_pseudotime": "adata" in summary and "PearlST_pseudotime" in summary["adata"].obs,
                "image_feature_backend": summary.get("image_feature_backend"),
                "image_feature_quality": summary.get("image_feature_quality"),
                "image_feature_cache_hit": summary.get("image_feature_cache_hit"),
            }

            figures = generate_figures(summary["adata"], output_dir, summary, mode="single")
            write_report(
                output_dir=output_dir,
                summary=summary,
                input_file=input_info.get("input_path"),
                params=vars(args),
            )

            summary["adata"].write_h5ad(output_h5ad)
            logger.info(f"Saved results to {output_h5ad}")

            results["figures"] = figures
            results["n_domains"] = summary["n_domains"]
            results["samples"] = [{"sample_name": sample_name}]
        
        # ===== Multi-Sample Integration Mode =====
        elif args.mode == "integration":
            if not method_spec.supports_integration:
                raise ValueError(
                    f"{method_spec.label} only supports --mode single in this skill. "
                    "Use DeepST for multi-sample integration."
                )
            if not args.input_list:
                raise ValueError(
                    "For integration mode, provide:\n"
                    "  --input-list <file with sample directories>"
                )

            input_list_path = _resolve_user_path(args.input_list)
            with open(input_list_path, "r") as f:
                sample_paths = [line.strip() for line in f if line.strip()]
            logger.info(f"Loading {len(sample_paths)} samples from input list: {input_list_path}")
            
            adata_list = []
            sample_ids = []
            sample_infos = []

            # Load from directory paths listed in input-list
            for sample_path in sample_paths:
                try:
                    logger.info(f"Loading sample from {sample_path}...")
                    info = load_spatial_data(
                        input_path=sample_path,
                        platform=args.platform,
                    )
                    adata_list.append(info["adata"])
                    sample_ids.append(info["sample_name"])
                    sample_infos.append(info)
                except Exception as e:
                    logger.error(f"Failed to load {sample_path}: {e}")
                    continue

            if not adata_list:
                raise ValueError("No samples loaded successfully")

            logger.info(f"Successfully loaded {len(adata_list)} samples")
            morphology = _resolve_morphology_setting(
                requested=args.use_morphological,
                sample_infos=sample_infos,
                mode="integration",
            )
            args.use_morphological = morphology["enabled"]
            image_feature_backends: dict[str, dict] = {}
            
            # Optional: Extract H&E image features for morphological augmentation
            if args.use_morphological:
                logger.info("Attempting to extract H&E image features for integration mode using the DeepST fast path...")
                missing_feature_samples = []
                
                for i, (adata, sample_id) in enumerate(zip(adata_list, sample_ids)):
                    try:
                        logger.info("Extracting H&E features for sample %s...", sample_id)
                        adata_list[i], image_feature_backends[sample_id] = _attach_deepst_image_features(
                            adata,
                            sample_name=sample_id,
                            sample_dir=sample_paths[i],
                            output_dir=output_dir,
                            use_gpu=args.use_gpu,
                            deepst_instance=None,
                        )
                        if "image_feat_pca" in adata_list[i].obsm:
                            logger.info(
                                "Successfully attached image features for %s via backend=%s quality=%s cache_hit=%s",
                                sample_id,
                                image_feature_backends[sample_id].get("backend"),
                                image_feature_backends[sample_id].get("image_quality"),
                                image_feature_backends[sample_id].get("cache_hit"),
                            )
                        else:
                            logger.warning(f"Image features not found after extraction for {sample_id}")
                            missing_feature_samples.append(sample_id)
                    except Exception as e:
                        logger.warning(f"Could not extract H&E features for {sample_id}: {e}")
                        missing_feature_samples.append(sample_id)
                
                if missing_feature_samples:
                    logger.warning(
                        "Morphology was requested, but image feature extraction did not complete for all samples (%s). "
                        "Disabling morphological augmentation for this integration run.",
                        ", ".join(sorted(set(missing_feature_samples))),
                    )
                    args.use_morphological = False
                    morphology["enabled"] = False
                    morphology["status_label"] = "Disabled (image feature extraction failed for one or more samples)"
            
            # Run integration
            summary = run_deepst_integration(
                adata_list=adata_list,
                sample_ids=sample_ids,
                output_dir=output_dir,
                n_domains=args.n_domains,
                batch_key=args.batch_key,
                pre_epochs=args.pre_epochs,
                epochs=args.epochs,
                pca_components=args.pca_components,
                use_morphological=args.use_morphological,
                use_gpu=args.use_gpu,
                random_seed=args.random_seed,
            )
            if not summary.get("use_morphological", False):
                args.use_morphological = False
                if morphology["enabled"]:
                    morphology["enabled"] = False
                    morphology["status_label"] = "Disabled (image feature extraction failed)"

            summary = {
                **summary,
                "method": args.method,
                "method_label": method_spec.label,
                "method_description": method_spec.description,
                "mode": "integration",
                "n_samples": len(sample_ids),
                "platform": args.platform,
                "image_path": sample_infos[0].get("image_path") if sample_infos else None,
                "image_type": sample_infos[0].get("image_type") if sample_infos else None,
                "morphology_status": morphology["status_label"],
                "domain_column": summary.get("domain_column", method_spec.cluster_key),
                "embedding_key": summary.get("embedding_key", method_spec.embedding_key),
                "has_pseudotime": False,
                "image_feature_backend": {
                    sample_id: image_feature_backends.get(sample_id, {}).get("backend")
                    for sample_id in sample_ids
                } if image_feature_backends else None,
            }
            
            # Generate figures
            figures = generate_figures(summary["adata_integrated"], output_dir, summary, mode="integration")

            write_report(
                output_dir=output_dir,
                summary=summary,
                input_file=None,
                params=vars(args),
            )
            
            # Save integrated results
            output_h5ad = output_dir / "integrated_deepst.h5ad"
            summary["adata_integrated"].write_h5ad(output_h5ad)
            logger.info(f"Saved integrated results to {output_h5ad}")
            
            # Save per-sample results
            for sample_id in sample_ids:
                sample_adata = summary["adata_integrated"][
                    summary["adata_integrated"].obs[args.batch_key] == sample_id
                ].copy()
                sample_output = output_dir / f"{sample_id}_deepst.h5ad"
                sample_adata.write_h5ad(sample_output)
            
            results["figures"] = figures
            results["n_domains"] = summary["n_domains"]
            results["samples"] = sample_infos
        
        # ===== SAVE METADATA =====
        metadata = {
            "skill": SKILL_NAME,
            "version": SKILL_VERSION,
            "method": args.method,
            "method_label": method_spec.label,
            "timestamp": pd.Timestamp.now().isoformat(),
            "command": " ".join(sys.argv),
            "n_domains": results["n_domains"],
            "n_samples": len(results["samples"]),
            "figures": results["figures"],
            "parameters": {
                "method": args.method,
                "platform": args.platform,
                "pre_epochs": args.pre_epochs,
                "epochs": args.epochs,
                "pca_components": args.pca_components,
                "n_top_genes": args.n_top_genes,
                "use_morphological": args.use_morphological,
                "morphology_status": morphology["status_label"],
                "morphology_requested": morphology["requested"],
                "use_gpu": args.use_gpu,
                "device": args.device,
                "ground_truth": args.ground_truth,
                "simclr_features": args.simclr_features,
                "simclr_model": args.simclr_model,
                "random_seed": args.random_seed,
            },
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Analysis complete! Results saved to {output_dir}")
        logger.info(f"Found {results['n_domains']} spatial domains")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
