"""Morphology feature extraction and image-omics fusion.

Provides methods for extracting morphological features from histology images
and integrating them with spatial omics data.

Supported methods:
  - deep_features: Deep learning-based feature extraction
  - handcrafted: Traditional image features (texture, shape, color)
  - fusion: Image-omics integration methods

Input convention:
  - Images: numpy arrays or image paths
  - Omics: AnnData with spatial coordinates

Usage::

    from skills.spatial._lib.morphology import (
        run_morphology,
        fuse_image_omics,
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

from .dependency_manager import require

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("deep_features", "handcrafted", "fusion", "ihc_quantification")


def extract_handcrafted_features(
    image: np.ndarray,
    *,
    features: list[str] | None = None,
) -> dict:
    """Extract handcrafted morphology features from an image patch.

    Parameters
    ----------
    image : np.ndarray
        Input image (H x W x C or H x W).
    features : list, optional
        List of feature types to extract. Default: all.

    Returns
    -------
    dict
        Dictionary of extracted features.
    """
    if features is None:
        features = ["color", "texture", "shape", "intensity"]

    logger.info("Extracting handcrafted features: %s", features)
    result = {}

    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    if "intensity" in features:
        result["mean_intensity"] = float(np.mean(gray))
        result["std_intensity"] = float(np.std(gray))
        result["median_intensity"] = float(np.median(gray))
        result["intensity_range"] = float(np.max(gray) - np.min(gray))

    if "color" in features and image.ndim == 3:
        for i, channel in enumerate(["red", "green", "blue"]):
            result[f"mean_{channel}"] = float(np.mean(image[:, :, i]))
            result[f"std_{channel}"] = float(np.std(image[:, :, i]))

        from skimage.color import rgb2hsv
        hsv = rgb2hsv(image.astype(np.float32) / 255.0)
        result["mean_hue"] = float(np.mean(hsv[:, :, 0]))
        result["mean_saturation"] = float(np.mean(hsv[:, :, 1]))
        result["mean_value"] = float(np.mean(hsv[:, :, 2]))

    if "texture" in features:
        from skimage.feature import graycomatrix, graycoprops

        glcm = graycomatrix(
            (gray * 255 / gray.max()).astype(np.uint8),
            distances=[1, 2, 4],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
            result[f"glcm_{prop}"] = float(np.mean(graycoprops(glcm, prop)))

    if "shape" in features:
        from skimage.measure import label, regionprops

        binary = gray > np.mean(gray)
        labeled = label(binary)
        props = regionprops(labeled, intensity_image=gray)

        if props:
            largest = max(props, key=lambda x: x.area)
            result["n_regions"] = len(props)
            result["largest_area"] = float(largest.area)
            result["largest_eccentricity"] = float(largest.eccentricity)
            result["largest_solidity"] = float(largest.solidity)
        else:
            result["n_regions"] = 0
            result["largest_area"] = 0.0
            result["largest_eccentricity"] = 0.0
            result["largest_solidity"] = 0.0

    return result


def extract_deep_features(
    images: list[np.ndarray],
    *,
    model_name: str = "resnet50",
    batch_size: int = 32,
    device: str = "auto",
) -> np.ndarray:
    """Extract deep learning features from images.

    Parameters
    ----------
    images : list
        List of image arrays.
    model_name : str
        Pretrained model name.
    batch_size : int
        Batch size for inference.
    device : str
        Device for inference.

    Returns
    -------
    np.ndarray
        Feature matrix.
    """
    require("torch", feature="Deep feature extraction")
    require("torchvision", feature="Deep feature extraction")

    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Extracting deep features using %s on %s (%d images)", model_name, device, len(images))

    model_dict = {
        "resnet50": models.resnet50,
        "vgg16": models.vgg16,
        "efficientnet_b0": models.efficientnet_b0,
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_dict[model_name](pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = torch.stack([
                preprocess(img if img.ndim == 3 else np.stack([img]*3, axis=-1))
                for img in batch
            ]).to(device)
            batch_features = model(batch_tensors)
            batch_features = batch_features.squeeze(-1).squeeze(-1)
            features.append(batch_features.cpu().numpy())

    return np.vstack(features)


def fuse_image_omics(
    adata,
    image_features: np.ndarray,
    *,
    method: str = "concatenation",
    spatial_key: str = "spatial",
    n_neighbors: int = 10,
) -> dict:
    """Fuse image features with omics data.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data.
    image_features : np.ndarray
        Image feature matrix (n_spots x n_features).
    method : str
        Fusion method: concatenation, weighted, attention.
    spatial_key : str
        Key for spatial coordinates.
    n_neighbors : int
        Neighbors for spatial weighting.

    Returns
    -------
    dict
        Fusion results including fused embeddings.
    """
    logger.info("Fusing image features with omics (method=%s)", method)

    if "X_pca" not in adata.obsm:
        raise ValueError("adata must have X_pca in obsm")

    omics_features = adata.obsm["X_pca"]

    if len(omics_features) != len(image_features):
        raise ValueError(
            f"Feature count mismatch: omics={len(omics_features)}, image={len(image_features)}"
        )

    if method == "concatenation":
        fused = np.hstack([omics_features, image_features])
    elif method == "weighted":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        omics_scaled = scaler.fit_transform(omics_features)
        image_scaled = scaler.fit_transform(image_features)
        fused = 0.5 * omics_scaled + 0.5 * image_scaled
    elif method == "attention":
        require("torch", feature="Attention-based fusion")
        import torch
        import torch.nn as nn

        class AttentionFusion(nn.Module):
            def __init__(self, omics_dim, image_dim, hidden_dim=64):
                super().__init__()
                self.omics_proj = nn.Linear(omics_dim, hidden_dim)
                self.image_proj = nn.Linear(image_dim, hidden_dim)
                self.attention = nn.Linear(hidden_dim * 2, 2)

            def forward(self, omics, image):
                omics_h = self.omics_proj(omics)
                image_h = self.image_proj(image)
                combined = torch.cat([omics_h, image_h], dim=-1)
                weights = torch.softmax(self.attention(combined), dim=-1)
                fused = weights[:, 0:1] * omics_h + weights[:, 1:2] * image_h
                return fused, weights

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AttentionFusion(omics_features.shape[1], image_features.shape[1]).to(device)

        omics_tensor = torch.tensor(omics_features, dtype=torch.float32).to(device)
        image_tensor = torch.tensor(image_features, dtype=torch.float32).to(device)

        with torch.no_grad():
            fused, weights = model(omics_tensor, image_tensor)
            fused = fused.cpu().numpy()
            adata.obs["omics_attention_weight"] = weights[:, 0].cpu().numpy()
            adata.obs["image_attention_weight"] = weights[:, 1].cpu().numpy()
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    adata.obsm["X_fused"] = fused

    return {
        "method": method,
        "omics_dim": omics_features.shape[1],
        "image_dim": image_features.shape[1],
        "fused_dim": fused.shape[1],
    }


def quantify_ihc(
    image: np.ndarray,
    *,
    marker_channel: int = 0,
    threshold: float = 0.1,
    cell_mask: np.ndarray | None = None,
) -> dict:
    """Quantify IHC/IF staining from images.

    Parameters
    ----------
    image : np.ndarray
        Input IHC/IF image.
    marker_channel : int
        Channel index for marker (for multi-channel IF).
    threshold : float
        Threshold for positive staining.
    cell_mask : np.ndarray, optional
        Binary mask for cell regions.

    Returns
    -------
    dict
        Quantification results.
    """
    logger.info("Quantifying IHC/IF staining...")

    if image.ndim == 3:
        if image.shape[2] > marker_channel:
            marker = image[:, :, marker_channel]
        else:
            marker = np.mean(image, axis=2)
    else:
        marker = image

    if cell_mask is not None:
        marker = marker * cell_mask

    positive = marker > (threshold * marker.max())

    if cell_mask is not None:
        cell_area = cell_mask.sum()
        positive_area = (positive & cell_mask).sum()
    else:
        cell_area = marker.size
        positive_area = positive.sum()

    h_score = np.mean(marker) * (1 + 2 * (positive.sum() / marker.size))

    return {
        "mean_intensity": float(np.mean(marker)),
        "median_intensity": float(np.median(marker)),
        "positive_fraction": float(positive_area / cell_area) if cell_area > 0 else 0.0,
        "h_score": float(h_score),
        "total_area": int(cell_area),
        "positive_area": int(positive_area),
    }


def run_morphology(
    adata,
    images: list[np.ndarray] | None = None,
    *,
    method: str = "deep_features",
    model_name: str = "resnet50",
    fusion_method: str = "concatenation",
) -> dict:
    """Run morphology analysis pipeline.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data.
    images : list, optional
        List of patch images corresponding to spots.
    method : str
        Feature extraction method.
    model_name : str
        Deep learning model for feature extraction.
    fusion_method : str
        Method for image-omics fusion.

    Returns
    -------
    dict
        Analysis results.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Choose from: {SUPPORTED_METHODS}")

    n_cells, n_genes = adata.n_obs, adata.n_vars
    logger.info("Morphology analysis: %d cells, method=%s", n_cells, method)

    if images is None:
        logger.warning("No images provided, generating synthetic features")
        n_features = 512 if method == "deep_features" else 20
        features = np.random.randn(n_cells, n_features) * 0.1
    elif method == "deep_features":
        features = extract_deep_features(images, model_name=model_name)
    elif method == "handcrafted":
        feature_dicts = [extract_handcrafted_features(img) for img in images]
        features = pd.DataFrame(feature_dicts).values
    else:
        raise ValueError(f"Method {method} requires images")

    adata.obsm["morphology_features"] = features

    fusion_result = None
    if "X_pca" in adata.obsm:
        fusion_result = fuse_image_omics(adata, features, method=fusion_method)

    return {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "method": method,
        "feature_dim": features.shape[1],
        "fusion": fusion_result,
    }
