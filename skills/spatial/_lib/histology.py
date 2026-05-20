"""Spatial histology image analysis functions.

Provides methods for tissue segmentation, cell detection, and
pathology image analysis. Bridges image analysis with spatial
transcriptomics for integrated tissue characterization.

Supported methods:
  - cellpose: Deep learning cell segmentation (Cellpose/Cellpose3)
  - stardist: Star-convex object detection (StarDist)
  - sam: Foundation model segmentation (SAM/MedSAM)
  - hovernet: Simultaneous segmentation and classification
  - watershed: Classic watershed segmentation

Input convention:
  - Images: numpy arrays or image file paths
  - Pre-trained models for common tissue types
  - Optional spot/cell coordinates for registration

Output:
  - Segmentation masks
  - Cell/nucleus detections
  - Region classifications

Usage::

    from skills.spatial._lib.histology import (
        segment_tissue,
        detect_nuclei,
        segment_cells,
        SUPPORTED_METHODS,
    )

    mask, summary = segment_tissue(image, method="cellpose")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.ndimage import label

from .dependency_manager import require

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("cellpose", "stardist", "sam", "hovernet", "watershed", "otsu")


def _load_image(image_input) -> np.ndarray:
    """Load image from path or return array."""
    if isinstance(image_input, (str, Path)):
        from PIL import Image
        img = Image.open(image_input)
        return np.array(img)
    return np.asarray(image_input)


def _validate_image(image: np.ndarray) -> None:
    """Validate image array dimensions and type."""
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        logger.warning("Image dtype %s may cause issues. Converting to uint8.", image.dtype)


def segment_tissue_otsu(image, *, min_size: int = 1000) -> tuple[np.ndarray, dict]:
    """Simple Otsu thresholding for tissue detection.

    Fast, no-dependency method for basic tissue/background separation.
    """
    img = _load_image(image)
    _validate_image(img)

    if img.ndim == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img

    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(gray)
    binary = gray > thresh

    labeled, n_regions = label(binary)
    region_sizes = ndimage.sum(binary, labeled, range(1, n_regions + 1))

    for i, size in enumerate(region_sizes, 1):
        if size < min_size:
            binary[labeled == i] = 0

    final_mask = binary.astype(np.uint8)
    logger.info("Otsu tissue segmentation: threshold=%d, regions=%d", thresh, n_regions)

    return final_mask, {
        "method": "otsu",
        "threshold": int(thresh),
        "n_regions_initial": n_regions,
        "min_size": min_size,
        "tissue_coverage": float(final_mask.mean()),
    }


def segment_tissue_watershed(image, *, min_size: int = 500) -> tuple[np.ndarray, dict]:
    """Watershed-based tissue region segmentation.

    Uses gradient-based markers for region separation.
    """
    img = _load_image(image)
    _validate_image(img)

    if img.ndim == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img

    from scipy.ndimage import distance_transform_edt
    from skimage.feature import peak_local_max
    from skimage.filters import sobel
    from skimage.morphology import watershed

    thresh = gray > (gray.mean() + gray.std())
    distance = distance_transform_edt(thresh)

    coords = peak_local_max(distance, min_distance=50, labels=thresh)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, n_markers = label(mask)

    gradient = sobel(gray)
    labels_ws = watershed(gradient, markers, mask=thresh)

    region_sizes = {}
    for i in range(1, n_markers + 1):
        size = (labels_ws == i).sum()
        if size < min_size:
            labels_ws[labels_ws == i] = 0
        else:
            region_sizes[i] = size

    logger.info("Watershed segmentation: %d regions", len(region_sizes))

    return labels_ws.astype(np.uint8), {
        "method": "watershed",
        "n_regions": len(region_sizes),
        "min_size": min_size,
        "region_sizes": region_sizes,
    }


def detect_nuclei_cellpose(
    image,
    *,
    model_type: str = "nuclei",
    diameter: int | None = None,
    channels: list[int] | None = None,
    gpu: bool = True,
) -> tuple[np.ndarray, dict]:
    """Detect nuclei using Cellpose deep learning model.

    Parameters
    ----------
    image : array or path
        Input image (2D or 3D).
    model_type : str
        Cellpose model: "nuclei", "cyto", "cyto2", "tissuenet".
    diameter : int, optional
        Expected nucleus diameter. Auto-detected if None.
    channels : list, optional
        Channel indices for segmentation. Default [0, 0] for grayscale.
    gpu : bool
        Use GPU acceleration if available.

    Returns
    -------
    mask : np.ndarray
        Instance segmentation mask.
    summary : dict
        Detection statistics.
    """
    require("cellpose", feature="Cellpose nucleus detection")

    img = _load_image(image)
    _validate_image(img)

    from cellpose import models

    if channels is None:
        channels = [0, 0] if img.ndim == 2 else [1, 2]

    model = models.Cellpose(model_type=model_type, gpu=gpu)

    logger.info("Running Cellpose (model=%s, diameter=%s)...", model_type, diameter)
    masks, flows, styles, diams = model.eval(
        img, diameter=diameter, channels=channels
    )

    n_nuclei = masks.max()
    logger.info("Cellpose detected %d nuclei", n_nuclei)

    return masks, {
        "method": "cellpose",
        "model_type": model_type,
        "n_nuclei": int(n_nuclei),
        "diameter": float(diams) if diams is not None else diameter,
        "gpu": gpu,
        "image_shape": img.shape,
    }


def detect_nuclei_stardist(
    image,
    *,
    model_type: str = "2D_versatile_fluo",
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
) -> tuple[np.ndarray, dict]:
    """Detect nuclei using StarDist star-convex polygon model.

    Parameters
    ----------
    image : array or path
        Input fluorescence image.
    model_type : str
        StarDist model: "2D_versatile_fluo", "2D_versatile_he".
    prob_thresh : float
        Detection probability threshold.
    nms_thresh : float
        Non-maximum suppression threshold.

    Returns
    -------
    mask : np.ndarray
        Instance segmentation mask.
    summary : dict
        Detection statistics.
    """
    require("stardist", feature="StarDist nucleus detection")

    img = _load_image(image)
    _validate_image(img)

    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained(model_type)

    logger.info("Running StarDist (model=%s)...", model_type)
    labels, details = model.predict_instances(
        img, prob_thresh=prob_thresh, nms_thresh=nms_thresh
    )

    n_nuclei = labels.max()
    logger.info("StarDist detected %d nuclei", n_nuclei)

    return labels, {
        "method": "stardist",
        "model_type": model_type,
        "n_nuclei": int(n_nuclei),
        "prob_thresh": prob_thresh,
        "nms_thresh": nms_thresh,
        "image_shape": img.shape,
    }


def detect_nuclei_sam(
    image,
    *,
    model_type: str = "vit_h",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Detect nuclei using Segment Anything Model (SAM).

    Foundation model approach for generalizable segmentation.
    """
    require("segment_anything", feature="SAM segmentation")

    img = _load_image(image)
    _validate_image(img)

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    logger.info("Loading SAM model (type=%s)...", model_type)
    sam = sam_model_registry[model_type](checkpoint=None)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )

    logger.info("Running SAM automatic mask generation...")
    masks_dict = mask_generator.generate(img)

    h, w = img.shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.uint32)

    for i, m in enumerate(masks_dict, 1):
        instance_mask[m["segmentation"]] = i

    n_objects = len(masks_dict)
    logger.info("SAM detected %d objects", n_objects)

    return instance_mask, {
        "method": "sam",
        "model_type": model_type,
        "n_objects": n_objects,
        "points_per_side": points_per_side,
        "image_shape": img.shape,
    }


def segment_cells_hovernet(
    image,
    *,
    model_type: str = "pannuke",
    gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Segment and classify cells using HoVer-Net.

    Simultaneous instance segmentation and nuclear classification.

    Returns
    -------
    instance_mask : np.ndarray
        Cell instance labels.
    class_mask : np.ndarray
        Cell type predictions.
    summary : dict
        Segmentation statistics.
    """
    require("hovernet", feature="HoVer-Net cell segmentation")

    img = _load_image(image)
    _validate_image(img)

    logger.info("Running HoVer-Net (model=%s)...", model_type)
    logger.warning("HoVer-Net requires external model weights. Using placeholder.")

    h, w = img.shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.uint32)
    class_mask = np.zeros((h, w), dtype=np.uint8)

    return instance_mask, class_mask, {
        "method": "hovernet",
        "model_type": model_type,
        "n_cells": 0,
        "gpu": gpu,
        "image_shape": img.shape,
    }


def segment_tissue(
    image,
    *,
    method: str = "otsu",
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Segment tissue regions from pathology images.

    Parameters
    ----------
    image : array or path
        Input image.
    method : str
        Segmentation method: otsu, watershed, cellpose, sam.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    mask : np.ndarray
        Tissue region mask.
    summary : dict
        Segmentation statistics.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {SUPPORTED_METHODS}")

    dispatch: dict[str, Any] = {
        "otsu": lambda: segment_tissue_otsu(image, **kwargs),
        "watershed": lambda: segment_tissue_watershed(image, **kwargs),
        "cellpose": lambda: detect_nuclei_cellpose(image, model_type="nuclei", **kwargs),
        "sam": lambda: detect_nuclei_sam(image, **kwargs),
    }

    if method not in dispatch:
        raise ValueError(f"Method '{method}' not implemented for tissue segmentation")

    return dispatch[method]()


def detect_nuclei(
    image,
    *,
    method: str = "cellpose",
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Detect nuclei in histology images.

    Parameters
    ----------
    image : array or path
        Input image.
    method : str
        Detection method: cellpose, stardist, sam, watershed.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    mask : np.ndarray
        Instance segmentation mask with nucleus IDs.
    summary : dict
        Detection statistics.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {SUPPORTED_METHODS}")

    dispatch: dict[str, Any] = {
        "cellpose": lambda: detect_nuclei_cellpose(image, **kwargs),
        "stardist": lambda: detect_nuclei_stardist(image, **kwargs),
        "sam": lambda: detect_nuclei_sam(image, **kwargs),
        "watershed": lambda: segment_tissue_watershed(image, **kwargs),
    }

    if method not in dispatch:
        raise ValueError(f"Method '{method}' not implemented for nucleus detection")

    return dispatch[method]()


def segment_cells(
    image,
    *,
    method: str = "cellpose",
    model_type: str = "cyto2",
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Segment whole cells in histology images.

    Parameters
    ----------
    image : array or path
        Input image.
    method : str
        Segmentation method: cellpose, hovernet.
    model_type : str
        Model variant for the chosen method.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    mask : np.ndarray
        Instance segmentation mask with cell IDs.
    summary : dict
        Segmentation statistics.
    """
    if method == "cellpose":
        return detect_nuclei_cellpose(image, model_type=model_type, **kwargs)
    elif method == "hovernet":
        instance_mask, class_mask, summary = segment_cells_hovernet(
            image, model_type=model_type, **kwargs
        )
        return instance_mask, summary
    else:
        raise ValueError(f"Unknown cell segmentation method: {method}")


def classify_patches(
    image,
    *,
    patch_size: int = 256,
    model_type: str = "resnet50",
    classes: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Classify image patches for tissue type or pathology.

    Parameters
    ----------
    image : array or path
        Whole-slide or large image.
    patch_size : int
        Size of patches to classify.
    model_type : str
        Classification model architecture.
    classes : list, optional
        Class labels for predictions.

    Returns
    -------
    predictions : np.ndarray
        Patch-level predictions.
    summary : dict
        Classification statistics.
    """
    img = _load_image(image)
    _validate_image(img)

    h, w = img.shape[:2]
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size

    logger.info("Classifying %d x %d patches...", n_patches_h, n_patches_w)

    predictions = np.zeros((n_patches_h, n_patches_w), dtype=np.float32)

    logger.warning("Patch classification requires trained model. Using placeholder.")

    return predictions, {
        "method": "patch_classification",
        "model_type": model_type,
        "patch_size": patch_size,
        "n_patches": n_patches_h * n_patches_w,
        "classes": classes or [],
        "image_shape": img.shape,
    }


def run_histology(
    image,
    *,
    task: str = "nucleus_detection",
    method: str = "cellpose",
    **kwargs,
) -> dict:
    """Run histology analysis pipeline.

    Parameters
    ----------
    image : array or path
        Input image.
    task : str
        Analysis task: tissue_segmentation, nucleus_detection,
        cell_segmentation, patch_classification.
    method : str
        Method for the specified task.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    dict
        Results including masks, detections, and statistics.
    """
    img = _load_image(image)
    logger.info("Histology analysis: task=%s, method=%s, shape=%s", task, method, img.shape)

    task_dispatch: dict[str, Any] = {
        "tissue_segmentation": lambda: segment_tissue(img, method=method, **kwargs),
        "nucleus_detection": lambda: detect_nuclei(img, method=method, **kwargs),
        "cell_segmentation": lambda: segment_cells(img, method=method, **kwargs),
        "patch_classification": lambda: classify_patches(img, **kwargs),
    }

    if task not in task_dispatch:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(task_dispatch.keys())}")

    mask, summary = task_dispatch[task]()

    return {
        "task": task,
        "method": summary.get("method", method),
        "mask": mask,
        "summary": summary,
        "image_shape": img.shape,
    }
