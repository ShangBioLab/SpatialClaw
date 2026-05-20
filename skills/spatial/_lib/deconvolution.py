"""Shared deconvolution helpers aligned with the registered SpatialClaw skill.

This module intentionally exposes the same method surface as
``skills/spatial/spatial-deconvolution/spatial_deconvolution.py``:
Tangram, Stereoscope, and GraphST.  It exists for internal library callers that
want AnnData-in/AnnData-out helpers without drifting away from the canonical
CLI implementation.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

import scanpy as sc

from .dependency_manager import require

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    description: str
    requires_reference: bool = True
    dependencies: tuple[str, ...] = ()
    supports_gpu: bool = False


METHOD_REGISTRY: dict[str, MethodConfig] = {
    "tangram": MethodConfig(
        name="tangram",
        description="Deep learning cell-to-spot mapping (tangram-sc)",
        dependencies=("tangram-sc",),
        supports_gpu=True,
    ),
    "stereoscope": MethodConfig(
        name="stereoscope",
        description="Two-stage probabilistic deconvolution via scvi-tools",
        dependencies=("scvi-tools", "torch"),
        supports_gpu=True,
    ),
    "graphst": MethodConfig(
        name="graphst",
        description="GraphST spatial embedding with NNLS deconvolution",
        dependencies=("GraphST", "torch"),
        supports_gpu=True,
    ),
}

SUPPORTED_METHODS = tuple(METHOD_REGISTRY.keys())
DEFAULT_METHOD = "tangram"


@lru_cache(maxsize=1)
def _skill_module() -> Any:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "spatial-deconvolution"
        / "spatial_deconvolution.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_spatialclaw_spatial_deconvolution_skill",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spatial-deconvolution skill from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _load_reference(reference_path: str, cell_type_key: str) -> "sc.AnnData":
    logger.info("Loading reference: %s", reference_path)
    adata_ref = sc.read_h5ad(reference_path)
    if cell_type_key not in adata_ref.obs.columns:
        categorical_columns = [
            column
            for column in adata_ref.obs.columns
            if adata_ref.obs[column].dtype.name in ("object", "category")
        ]
        raise ValueError(
            f"Cell type key '{cell_type_key}' not found in reference obs. "
            f"Available categorical columns: {categorical_columns}"
        )
    return adata_ref


def _common_genes(adata_sp: "sc.AnnData", adata_ref: "sc.AnnData") -> list[str]:
    common = sorted(set(adata_sp.var_names) & set(adata_ref.var_names))
    if len(common) < 1:
        raise ValueError(
            "No genes are shared between spatial and reference data. "
            "Check that both use the same gene identifier format."
        )
    return common


def _device(prefer_gpu: bool = True) -> str:
    if not prefer_gpu:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _deconv_stats(
    prop_df: pd.DataFrame,
    common_genes: list[str],
    method: str,
    device: str = "cpu",
    **extra: Any,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "method": method,
        "device": device,
        "n_spots": len(prop_df),
        "n_cell_types": prop_df.shape[1],
        "cell_types": list(prop_df.columns),
        "n_common_genes": len(common_genes),
        "mean_proportions": prop_df.mean().to_dict(),
        "dominant_types": prop_df.idxmax(axis=1).value_counts().to_dict()
        if not prop_df.empty and prop_df.shape[1]
        else {},
    }
    stats.update(extra)
    return stats


def _proportions_from_adata(adata: "sc.AnnData", method: str) -> pd.DataFrame:
    key = "deconvolution_ct_pred"
    if key not in adata.obsm:
        method_key = f"{method}_ct_pred"
        if method_key in adata.obsm:
            key = method_key
        else:
            raise RuntimeError(
                f"{method} did not store proportions in adata.obsm['deconvolution_ct_pred']"
            )

    proportions = adata.obsm[key]
    if isinstance(proportions, pd.DataFrame):
        return proportions.copy()

    values = _to_dense(proportions)
    columns = [f"CellType_{index}" for index in range(values.shape[1])]
    return pd.DataFrame(values, index=adata.obs_names, columns=columns)


def deconvolve_tangram(
    adata: "sc.AnnData",
    *,
    reference_path: str,
    cell_type_key: str = "cell_type",
    n_epochs: int = 1000,
    use_gpu: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Tangram deconvolution and return spot-level cell-type proportions."""
    require("tangram-sc", feature="Tangram deconvolution")
    adata_ref = _load_reference(reference_path, cell_type_key)
    device = _device(use_gpu)
    result = _skill_module().run_tangram_deconvolution(
        adata.copy(),
        adata_ref.copy(),
        cell_type_key,
        device=device,
        n_epochs=n_epochs,
    )
    proportions = _proportions_from_adata(result, "tangram")
    return proportions, _deconv_stats(
        proportions,
        _common_genes(adata, adata_ref),
        "tangram",
        device=device,
        n_epochs=n_epochs,
    )


def deconvolve_stereoscope(
    adata: "sc.AnnData",
    *,
    reference_path: str,
    cell_type_key: str = "cell_type",
    n_epochs: int = 200,
    use_gpu: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Stereoscope deconvolution and return spot-level cell-type proportions."""
    require("scvi-tools", feature="Stereoscope deconvolution")
    adata_ref = _load_reference(reference_path, cell_type_key)
    device = _device(use_gpu)
    result = _skill_module().run_stereoscope_deconvolution(
        adata.copy(),
        adata_ref.copy(),
        cell_type_key,
        device=device,
        n_epochs=n_epochs,
    )
    proportions = _proportions_from_adata(result, "stereoscope")
    return proportions, _deconv_stats(
        proportions,
        _common_genes(adata, adata_ref),
        "stereoscope",
        device=device,
        n_epochs=n_epochs,
    )


def deconvolve_graphst(
    adata: "sc.AnnData",
    *,
    reference_path: str,
    cell_type_key: str = "cell_type",
    n_epochs: int = 1000,
    n_hvg: int = 3000,
    use_gpu: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run GraphST-guided deconvolution and return spot-level proportions."""
    require("GraphST", feature="GraphST deconvolution")
    adata_ref = _load_reference(reference_path, cell_type_key)
    device = _device(use_gpu)
    result = _skill_module().run_graphst_deconvolution(
        adata.copy(),
        adata_ref.copy(),
        cell_type_key,
        device=device,
        n_epochs=n_epochs,
        n_hvg=n_hvg,
    )
    proportions = _proportions_from_adata(result, "graphst")
    return proportions, _deconv_stats(
        proportions,
        _common_genes(adata, adata_ref),
        "graphst",
        device=device,
        n_epochs=n_epochs,
        n_hvg=n_hvg,
    )


METHOD_DISPATCH: dict[str, Any] = {
    "tangram": deconvolve_tangram,
    "stereoscope": deconvolve_stereoscope,
    "graphst": deconvolve_graphst,
}
