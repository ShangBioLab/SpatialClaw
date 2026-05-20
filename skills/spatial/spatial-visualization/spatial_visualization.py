#!/usr/bin/env python3
"""Spatial Visualization — generic plotting for spatial transcriptomics.

This skill renders spatial feature maps and downstream analysis figures from
existing AnnData outputs produced by the other spatial skills.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spatialclaw.common.checksums import sha256_file
from spatialclaw.common.report import (
    generate_report_footer,
    generate_report_header,
    write_result_json,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SKILL_NAME = "spatial-visualization"
SKILL_VERSION = "0.1.0"


def save_figure(fig, output_dir: Path, filename: str):
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


try:
    from skills.spatial._lib.viz.params import VizParams
    from skills.spatial._lib.viz.feature import plot_features
    from skills.spatial._lib.viz.communication import plot_communication
    from skills.spatial._lib.viz.deconvolution import plot_deconvolution
    from skills.spatial._lib.viz.integration import plot_integration
    from skills.spatial._lib.viz.spatial_stats import plot_spatial_stats
    _USING_SHARED_VIZ = True
except Exception as exc:  # pragma: no cover - fallback is environment dependent
    _USING_SHARED_VIZ = False
    logger.warning("Shared spatial viz package unavailable, using local fallback plotters: %s", exc)

    @dataclass
    class VizParams:
        feature: str | list[str] | None = None
        basis: str = "spatial"
        cluster_key: str | None = None
        batch_key: str | None = None
        colormap: str = "magma"
        title: str | None = None
        figure_size: tuple[float, float] | None = None
        dpi: int = 200
        alpha: float = 0.8

    def _coords_for_basis(adata: ad.AnnData, basis: str) -> np.ndarray:
        if basis == "spatial":
            if "spatial" not in adata.obsm:
                raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")
            return np.asarray(adata.obsm["spatial"])
        if basis == "umap":
            if "X_umap" not in adata.obsm:
                raise ValueError("UMAP coordinates not found in adata.obsm['X_umap']")
            return np.asarray(adata.obsm["X_umap"])
        if basis == "pca":
            if "X_pca" not in adata.obsm:
                raise ValueError("PCA coordinates not found in adata.obsm['X_pca']")
            return np.asarray(adata.obsm["X_pca"])
        raise ValueError(f"Unsupported basis: {basis}")

    def _gene_values(adata: ad.AnnData, gene: str) -> np.ndarray:
        idx = adata.var_names.get_loc(gene)
        values = adata.X[:, idx]
        if hasattr(values, "toarray"):
            values = values.toarray().ravel()
        return np.asarray(values).ravel().astype(float)

    def _plot_colorbar(fig, ax, mappable, label: str = "") -> None:
        colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        if label:
            colorbar.set_label(label)

    def plot_features(adata: ad.AnnData, params: VizParams | None = None, *, feature: str | list[str] | None = None, basis: str | None = None):
        import matplotlib.pyplot as plt

        params = params or VizParams()
        if feature is not None:
            params.feature = feature
        if basis is not None:
            params.basis = basis
        feat = params.feature
        if isinstance(feat, list):
            feat = feat[0] if feat else None
        if feat is None:
            raise ValueError("feature is required")

        coords = _coords_for_basis(adata, params.basis)
        fig, ax = plt.subplots(figsize=params.figure_size or (10, 8), dpi=params.dpi)

        if feat in adata.var_names:
            values = _gene_values(adata, feat)
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=params.colormap, s=18, alpha=params.alpha)
            _plot_colorbar(fig, ax, scatter, label=feat)
        elif feat in adata.obs.columns:
            series = adata.obs[feat]
            if pd.api.types.is_categorical_dtype(series) or series.dtype == object:
                categories = series.astype("category").cat.categories
                palette = plt.cm.get_cmap("tab20", len(categories))
                for index, cat in enumerate(categories):
                    mask = series.astype("category") == cat
                    ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=params.alpha, label=str(cat), color=palette(index))
                ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
            else:
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=series.to_numpy(), cmap=params.colormap, s=18, alpha=params.alpha)
                _plot_colorbar(fig, ax, scatter, label=feat)
        else:
            raise ValueError(f"Feature '{feat}' not found in var_names or obs")

        ax.set_title(params.title or f"{feat} ({params.basis})")
        if params.basis == "spatial":
            ax.invert_yaxis()
        plt.tight_layout()
        return fig

    def plot_expression(adata: ad.AnnData, params: VizParams | None = None, *, subtype: str | None = None, genes: list[str] | None = None, cluster_key: str | None = None):
        import matplotlib.pyplot as plt

        params = params or VizParams()
        genes = genes or ([] if params.feature is None else (params.feature if isinstance(params.feature, list) else [params.feature]))
        if not genes:
            raise ValueError("genes are required")
        cluster_key = cluster_key or params.cluster_key
        if not cluster_key or cluster_key not in adata.obs.columns:
            raise ValueError("cluster_key is required")

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        tmp = df[genes].copy()
        tmp["group"] = adata.obs[cluster_key].astype(str).to_numpy()
        mean_df = tmp.groupby("group").mean().T
        fig, ax = plt.subplots(figsize=params.figure_size or (max(6, len(mean_df.columns) * 1.1), max(4, len(mean_df.index) * 0.35)), dpi=params.dpi)
        im = ax.imshow(mean_df.values, aspect="auto", cmap=params.colormap)
        ax.set_xticks(range(mean_df.shape[1]))
        ax.set_xticklabels(mean_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(mean_df.shape[0]))
        ax.set_yticklabels(mean_df.index)
        ax.set_title(params.title or "Expression Heatmap")
        _plot_colorbar(fig, ax, im, label="Mean expression")
        plt.tight_layout()
        return fig

    def plot_integration(adata: ad.AnnData, params: VizParams | None = None, *, subtype: str | None = None, batch_key: str | None = None, cluster_key: str | None = None):
        import matplotlib.pyplot as plt

        params = params or VizParams()
        subtype = subtype or "batch"
        coords = _coords_for_basis(adata, "umap")
        key = batch_key if subtype == "batch" else cluster_key
        if not key or key not in adata.obs.columns:
            raise ValueError("required obs key not found")
        labels = adata.obs[key].astype(str)
        categories = list(pd.unique(labels))
        palette = plt.cm.get_cmap("tab20", len(categories))
        fig, ax = plt.subplots(figsize=params.figure_size or (8, 6), dpi=params.dpi)
        for index, cat in enumerate(categories):
            mask = labels == cat
            ax.scatter(coords[mask, 0], coords[mask, 1], s=18, color=palette(index), alpha=params.alpha, label=cat)
        ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax.set_title(params.title or f"Integration ({subtype})")
        plt.tight_layout()
        return fig

    def plot_spatial_stats(adata: ad.AnnData, params: VizParams | None = None, *, subtype: str | None = None, cluster_key: str | None = None):
        import matplotlib.pyplot as plt

        params = params or VizParams()
        if "moranI" not in adata.uns:
            raise ValueError("moranI results not found")
        moran = adata.uns["moranI"].copy()
        top = moran.sort_values("I", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=params.figure_size or (8, max(4, len(top) * 0.35)), dpi=params.dpi)
        ax.barh(top.index[::-1], top["I"].iloc[::-1], color="teal")
        ax.set_title(params.title or "Top Spatially Variable Genes")
        ax.set_xlabel("Moran's I")
        plt.tight_layout()
        return fig

    def plot_deconvolution(adata: ad.AnnData, params: VizParams | None = None, *, subtype: str | None = None, method: str | None = None):
        import matplotlib.pyplot as plt

        params = params or VizParams()
        subtype = subtype or "spatial_multi"
        key = f"deconvolution_{method}" if method else next((k for k in adata.obsm.keys() if k.startswith("deconvolution_")), None)
        if key is None or key not in adata.obsm:
            raise ValueError("deconvolution results not found")
        proportions = np.asarray(adata.obsm[key])
        cell_types = list(adata.uns.get(f"{key}_cell_types", [f"CellType_{i}" for i in range(proportions.shape[1])]))
        coords = _coords_for_basis(adata, "spatial")
        if subtype == "dominant":
            dominant = np.argmax(proportions, axis=1)
            labels = np.array([cell_types[i] for i in dominant])
            cats = list(pd.unique(labels))
            palette = plt.cm.get_cmap("tab20", len(cats))
            fig, ax = plt.subplots(figsize=params.figure_size or (8, 6), dpi=params.dpi)
            for index, cat in enumerate(cats):
                mask = labels == cat
                ax.scatter(coords[mask, 0], coords[mask, 1], s=18, color=palette(index), label=cat)
            ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
            ax.set_title(params.title or f"Dominant Cell Type ({method or key})")
            ax.invert_yaxis()
            plt.tight_layout()
            return fig

        n_panels = min(len(cell_types), 12)
        n_cols = min(3, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=params.figure_size or (5 * n_cols, 4 * n_rows), dpi=params.dpi)
        axes = np.ravel(np.array(axes, dtype=object))
        for index, ct in enumerate(cell_types[:n_panels]):
            ax = axes[index]
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=proportions[:, index], cmap=params.colormap, s=16, vmin=0, vmax=1)
            ax.set_title(ct)
            ax.invert_yaxis()
        for ax in axes[n_panels:]:
            ax.axis("off")
        fig.suptitle(params.title or f"Deconvolution ({method or key})")
        plt.tight_layout()
        return fig

    def plot_communication(adata: ad.AnnData, params: VizParams | None = None, *, subtype: str | None = None, top_n: int = 20, min_cells: int = 5):
        import matplotlib.pyplot as plt

        params = params or VizParams()
        subtype = subtype or "dotplot"
        if "ccc_results" not in adata.uns:
            raise ValueError("ccc_results not found")
        results = adata.uns["ccc_results"].copy()
        if subtype == "heatmap":
            pivot = results.pivot_table(index="source", columns="target", values="lr_means", aggfunc="mean", fill_value=0)
            fig, ax = plt.subplots(figsize=params.figure_size or (8, 6), dpi=params.dpi)
            im = ax.imshow(pivot.values, cmap=params.colormap)
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            ax.set_title(params.title or "Communication Heatmap")
            _plot_colorbar(fig, ax, im, label="LR mean")
            plt.tight_layout()
            return fig

        top = results.sort_values("lr_means", ascending=False).head(top_n)
        labels = top["ligand"].astype(str) + "^" + top["receptor"].astype(str)
        fig, ax = plt.subplots(figsize=params.figure_size or (8, max(4, len(top) * 0.35)), dpi=params.dpi)
        ax.barh(labels.iloc[::-1], top["lr_means"].iloc[::-1], color="slateblue")
        ax.set_title(params.title or "Communication Dotplot")
        ax.set_xlabel("LR mean")
        plt.tight_layout()
        return fig

    def plot_trajectory(adata: ad.AnnData, params: VizParams | None = None, *, subtype: str | None = None, cluster_key: str | None = None):
        import matplotlib.pyplot as plt

        params = params or VizParams()
        key = params.feature if isinstance(params.feature, str) else "pseudotime"
        if key not in adata.obs.columns:
            raise ValueError("pseudotime column not found")
        coords = _coords_for_basis(adata, params.basis if params.basis in ("spatial", "umap", "pca") else "umap")
        fig, ax = plt.subplots(figsize=params.figure_size or (8, 6), dpi=params.dpi)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=adata.obs[key].to_numpy(), cmap=params.colormap, s=18, alpha=params.alpha)
        _plot_colorbar(fig, ax, scatter, label=key)
        ax.set_title(params.title or f"Pseudotime ({key})")
        plt.tight_layout()
        return fig


def _get_spatial_key(adata: ad.AnnData) -> str | None:
    for key in ("spatial", "X_spatial"):
        if key in adata.obsm:
            return key
    return None


def _store_analysis_metadata(adata: ad.AnnData, *, skill_name: str, method: str, params: dict) -> None:
    adata.uns[f"spatialclaw_{skill_name}"] = {
        "method": method,
        "params": params,
    }


def _ensure_spatial_basis(adata: ad.AnnData) -> str | None:
    """Ensure adata.obsm['spatial'] exists when a spatial-like key is present."""
    spatial_key = _get_spatial_key(adata)
    if spatial_key and spatial_key != "spatial":
        adata.obsm["spatial"] = np.asarray(adata.obsm[spatial_key])
    return spatial_key


def _ensure_umap(adata: ad.AnnData) -> bool:
    if "X_umap" in adata.obsm:
        return True
    try:
        import scanpy as sc
    except ImportError:
        logger.warning("scanpy is not installed; skipping UMAP computation")
        return False
    if "X_pca" in adata.obsm:
        try:
            sc.tl.umap(adata)
            return "X_umap" in adata.obsm
        except Exception as exc:
            logger.warning("Could not compute UMAP: %s", exc)
    return False


def _load_expression_plotter():
    try:
        from skills.spatial._lib.viz.expression import plot_expression
    except Exception as exc:
        logger.warning("Expression plotting unavailable: %s", exc)
        return None
    return plot_expression


def _load_trajectory_plotter():
    try:
        from skills.spatial._lib.viz.trajectory import plot_trajectory
    except Exception as exc:
        logger.warning("Trajectory plotting unavailable: %s", exc)
        return None
    return plot_trajectory


def _first_existing_column(adata: ad.AnnData, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in adata.obs.columns:
            return candidate
    return None


def _first_genes(adata: ad.AnnData, limit: int = 6) -> list[str]:
    genes = [str(gene) for gene in adata.var_names[: min(limit, adata.n_vars)]]
    if "highly_variable" in adata.var.columns:
        hvgs = [str(gene) for gene in adata.var_names[adata.var["highly_variable"]].tolist()]
        if hvgs:
            genes = hvgs[:limit]
    return genes


def _first_deconvolution_method(adata: ad.AnnData) -> str | None:
    methods = []
    for key in adata.obsm.keys():
        if key.startswith("deconvolution_"):
            methods.append(key.removeprefix("deconvolution_"))
    return methods[0] if methods else None


def _has_ccc_results(adata: ad.AnnData) -> bool:
    keys = {"ccc_results", "liana_res", "cellphonedb_results", "fastccc_results"}
    return any(key in adata.uns for key in keys)


def build_demo_adata(seed: int = 42) -> ad.AnnData:
    """Create a small synthetic AnnData object with common spatial outputs."""
    rng = np.random.default_rng(seed)
    n_obs = 96
    n_vars = 24

    counts = rng.poisson(lam=4.0, size=(n_obs, n_vars)).astype(float)
    adata = ad.AnnData(counts)
    adata.var_names = [f"Gene_{i:02d}" for i in range(1, n_vars + 1)]
    adata.obs_names = [f"Spot_{i:03d}" for i in range(1, n_obs + 1)]

    clusters = np.repeat(["Domain_A", "Domain_B", "Domain_C"], repeats=n_obs // 3)
    if len(clusters) < n_obs:
        clusters = np.concatenate([clusters, rng.choice(["Domain_A", "Domain_B", "Domain_C"], size=n_obs - len(clusters))])
    rng.shuffle(clusters)

    cell_types = rng.choice(["T_cell", "B_cell", "Myeloid", "Stromal"], size=n_obs)
    batches = rng.choice(["Batch_1", "Batch_2"], size=n_obs)
    pseudotime = np.clip(np.linspace(0, 1, n_obs) + rng.normal(0, 0.05, size=n_obs), 0, 1)

    adata.obs["spatial_domain"] = pd.Categorical(clusters)
    adata.obs["cell_type"] = pd.Categorical(cell_types)
    adata.obs["leiden"] = pd.Categorical(clusters)
    adata.obs["batch"] = pd.Categorical(batches)
    adata.obs["pseudotime"] = pseudotime

    coords = np.stack([
        np.repeat([0.0, 4.5, 9.0], repeats=n_obs // 3)[:n_obs],
        np.tile(np.linspace(0, 6, n_obs // 3), 3)[:n_obs],
    ], axis=1)
    coords += rng.normal(0, 0.35, size=coords.shape)
    adata.obsm["spatial"] = coords
    adata.obsm["X_umap"] = coords + rng.normal(0, 0.2, size=coords.shape)

    adata.var["highly_variable"] = [i < min(8, n_vars) for i in range(n_vars)]

    moran = pd.DataFrame(
        {
            "I": np.linspace(0.8, 0.05, n_vars) + rng.normal(0, 0.03, size=n_vars),
            "pval_norm": np.clip(np.linspace(0.001, 0.4, n_vars), 1e-4, 1.0),
        },
        index=adata.var_names,
    )
    adata.uns["moranI"] = moran

    deconvolution = rng.dirichlet(alpha=np.ones(4), size=n_obs)
    adata.obsm["deconvolution_demo"] = deconvolution
    adata.uns["deconvolution_demo_cell_types"] = ["T_cell", "B_cell", "Myeloid", "Stromal"]

    ccc = pd.DataFrame(
        {
            "ligand": [f"L{i}" for i in range(1, 11)],
            "receptor": [f"R{i}" for i in range(1, 11)],
            "source": rng.choice(["T_cell", "B_cell", "Myeloid"], size=10),
            "target": rng.choice(["Stromal", "T_cell", "B_cell"], size=10),
            "lr_means": rng.uniform(0.1, 1.0, size=10),
            "cellphone_pvals": rng.uniform(0.001, 0.05, size=10),
        }
    )
    adata.uns["ccc_results"] = ccc

    adata.uns["spatial_name"] = "demo_visualization"
    return adata


def _save_figure_result(fig, output_dir: Path, filename: str, figures: list[dict[str, str]]) -> None:
    path = save_figure(fig, output_dir, filename)
    figures.append({"name": filename, "path": str(path)})


def generate_figures(
    adata: ad.AnnData,
    output_dir: Path,
    *,
    feature: str | None = None,
    genes: list[str] | None = None,
    mode: str = "auto",
    cluster_key: str | None = None,
    batch_key: str | None = None,
) -> list[dict[str, str]]:
    """Generate comprehensive spatial visualization figures using scanpy.pl API."""
    import matplotlib.pyplot as plt
    import scanpy as sc

    figures: list[dict[str, str]] = []

    spatial_key = _ensure_spatial_basis(adata)
    umap_ready = _ensure_umap(adata)
    cluster_key = cluster_key or _first_existing_column(adata, ["spatial_domain", "leiden", "louvain", "cell_type", "celltype"])
    batch_key = batch_key or _first_existing_column(adata, ["batch", "sample", "library_id", "slide"])

    available_feature_columns = [
        col for col in ["spatial_domain", "cell_type", "leiden", "louvain"]
        if col in adata.obs.columns
    ]

    genes = genes or _first_genes(adata, limit=6)
    plot_feature = feature or (available_feature_columns[0] if available_feature_columns else None)

    # ===== SPATIAL EMBEDDING VISUALIZATIONS =====
    if mode in ("auto", "embedding") and spatial_key:
        try:
            logger.info("Generating spatial embedding plots using scanpy.pl...")
            # pl.spatial - native spatial visualization
            sc.pl.spatial(adata, color=cluster_key, title=f"Spatial Layout - {cluster_key}", save=f"{output_dir}/scanpy_spatial_layout.png")
            figures.append({"name": "scanpy_spatial_layout.png", "path": str(output_dir / "scanpy_spatial_layout.png")})
        except Exception as exc:
            logger.warning("Could not generate pl.spatial plot: %s", exc)

    # ===== UMAP/TSNE/PCA EMBEDDINGS =====
    if mode in ("auto", "embedding") and umap_ready:
        try:
            logger.info("Generating UMAP embedding plots...")
            # pl.umap with cluster annotation
            sc.pl.umap(adata, color=cluster_key, title=f"UMAP - {cluster_key}", save=f"{output_dir}/scanpy_umap_cluster.png")
            figures.append({"name": "scanpy_umap_cluster.png", "path": str(output_dir / "scanpy_umap_cluster.png")})
        except Exception as exc:
            logger.warning("Could not generate pl.umap plot: %s", exc)

        if batch_key:
            try:
                sc.pl.umap(adata, color=batch_key, title=f"UMAP - {batch_key}", save=f"{output_dir}/scanpy_umap_batch.png")
                figures.append({"name": "scanpy_umap_batch.png", "path": str(output_dir / "scanpy_umap_batch.png")})
            except Exception as exc:
                logger.warning("Could not generate pl.umap batch plot: %s", exc)

    # ===== PCA VISUALIZATION =====
    if mode in ("auto", "embedding") and "X_pca" in adata.obsm:
        try:
            logger.info("Generating PCA embedding plots...")
            sc.pl.pca(adata, color=cluster_key, save=f"{output_dir}/scanpy_pca_cluster.png")
            figures.append({"name": "scanpy_pca_cluster.png", "path": str(output_dir / "scanpy_pca_cluster.png")})
        except Exception as exc:
            logger.warning("Could not generate pl.pca plot: %s", exc)

    # ===== GENE EXPRESSION - HEATMAP =====
    if mode in ("auto", "expression") and cluster_key and genes:
        try:
            logger.info("Generating scanpy.pl.heatmap for gene expression...")
            sc.pl.heatmap(adata, var_names=genes, groupby=cluster_key, save=f"{output_dir}/scanpy_heatmap_genes.png", show=False)
            figures.append({"name": "scanpy_heatmap_genes.png", "path": str(output_dir / "scanpy_heatmap_genes.png")})
        except Exception as exc:
            logger.warning("Could not generate pl.heatmap: %s", exc)

    # ===== GENE EXPRESSION - MATRIXPLOT =====
    if mode in ("auto", "expression") and cluster_key and genes:
        try:
            logger.info("Generating scanpy.pl.matrixplot for gene expression...")
            sc.pl.matrixplot(adata, var_names=genes, groupby=cluster_key, save=f"{output_dir}/scanpy_matrixplot_genes.png", show=False)
            figures.append({"name": "scanpy_matrixplot_genes.png", "path": str(output_dir / "scanpy_matrixplot_genes.png")})
        except Exception as exc:
            logger.warning("Could not generate pl.matrixplot: %s", exc)

    # ===== GENE EXPRESSION - DOTPLOT =====
    if mode in ("auto", "expression") and cluster_key and genes:
        try:
            logger.info("Generating scanpy.pl.dotplot for gene expression...")
            sc.pl.dotplot(adata, var_names=genes, groupby=cluster_key, save=f"{output_dir}/scanpy_dotplot_genes.png", show=False)
            figures.append({"name": "scanpy_dotplot_genes.png", "path": str(output_dir / "scanpy_dotplot_genes.png")})
        except Exception as exc:
            logger.warning("Could not generate pl.dotplot: %s", exc)

    # ===== GENE EXPRESSION - VIOLIN PLOT =====
    if mode in ("auto", "expression") and cluster_key and genes:
        try:
            logger.info("Generating scanpy.pl.violin for gene expression...")
            sc.pl.violin(adata, keys=genes, groupby=cluster_key, save=f"{output_dir}/scanpy_violin_genes.png", show=False)
            figures.append({"name": "scanpy_violin_genes.png", "path": str(output_dir / "scanpy_violin_genes.png")})
        except Exception as exc:
            logger.warning("Could not generate pl.violin: %s", exc)

    # ===== GENE EXPRESSION - STACKED VIOLIN =====
    if mode in ("auto", "expression") and cluster_key and genes:
        try:
            logger.info("Generating scanpy.pl.stacked_violin for gene expression...")
            sc.pl.stacked_violin(adata, var_names=genes, groupby=cluster_key, save=f"{output_dir}/scanpy_stacked_violin.png", show=False)
            figures.append({"name": "scanpy_stacked_violin.png", "path": str(output_dir / "scanpy_stacked_violin.png")})
        except Exception as exc:
            logger.warning("Could not generate pl.stacked_violin: %s", exc)

    # ===== CLUSTERMAP (HIERARCHICAL CLUSTERING) =====
    if mode in ("auto", "expression") and genes and "highly_variable" not in adata.var.columns:
        try:
            logger.info("Generating scanpy.pl.clustermap...")
            subset = adata[:, genes].X
            import scipy.cluster.hierarchy as sch
            sc.pl.clustermap(adata[:, genes], figsize=(10, 8), save=f"{output_dir}/scanpy_clustermap.png", show=False)
            figures.append({"name": "scanpy_clustermap.png", "path": str(output_dir / "scanpy_clustermap.png")})
        except Exception as exc:
            logger.warning("Could not generate pl.clustermap: %s", exc)

    # ===== SCATTER PLOT (GENERIC) =====
    if mode in ("auto", "expression") and umap_ready and genes:
        try:
            logger.info("Generating scanpy.pl.scatter for genes on UMAP...")
            for gene in genes[:3]:  # limit to first 3 genes for clarity
                try:
                    sc.pl.scatter(adata, x="X_umap", y=gene, color=gene, title=f"UMAP: {gene} Expression", 
                                save=f"{output_dir}/scanpy_scatter_{gene}.png", show=False)
                    figures.append({"name": f"scanpy_scatter_{gene}.png", "path": str(output_dir / f"scanpy_scatter_{gene}.png")})
                except Exception as e:
                    logger.debug("Could not plot scatter for gene %s: %s", gene, e)
        except Exception as exc:
            logger.warning("Could not generate pl.scatter plots: %s", exc)

    # ===== RANK GENES GROUPS (DIFFERENTIAL EXPRESSION) =====
    if mode in ("auto", "differential") and cluster_key:
        try:
            logger.info("Computing and visualizing rank_genes_groups...")
            if "rank_genes_groups" not in adata.uns:
                sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon")
            
            # rank_genes_groups heatmap
            try:
                sc.pl.rank_genes_groups_heatmap(adata, n_genes=6, save=f"{output_dir}/scanpy_rank_genes_heatmap.png", show=False)
                figures.append({"name": "scanpy_rank_genes_heatmap.png", "path": str(output_dir / "scanpy_rank_genes_heatmap.png")})
            except Exception as e:
                logger.debug("Could not generate rank_genes_groups_heatmap: %s", e)
            
            # rank_genes_groups dotplot
            try:
                sc.pl.rank_genes_groups_dotplot(adata, n_genes=6, save=f"{output_dir}/scanpy_rank_genes_dotplot.png", show=False)
                figures.append({"name": "scanpy_rank_genes_dotplot.png", "path": str(output_dir / "scanpy_rank_genes_dotplot.png")})
            except Exception as e:
                logger.debug("Could not generate rank_genes_groups_dotplot: %s", e)

            # rank_genes_groups stacked violin
            try:
                sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=6, save=f"{output_dir}/scanpy_rank_genes_violin.png", show=False)
                figures.append({"name": "scanpy_rank_genes_violin.png", "path": str(output_dir / "scanpy_rank_genes_violin.png")})
            except Exception as e:
                logger.debug("Could not generate rank_genes_groups_stacked_violin: %s", e)

        except Exception as exc:
            logger.warning("Could not generate rank_genes_groups visualizations: %s", exc)

    # ===== TRAJECTORY/PAGA VISUALIZATION =====
    if mode in ("auto", "trajectory"):
        # PAGA graph
        if "paga" in adata.uns:
            try:
                logger.info("Generating PAGA visualization...")
                sc.pl.paga(adata, color=cluster_key, title="PAGA Graph", save=f"{output_dir}/scanpy_paga.png", show=False)
                figures.append({"name": "scanpy_paga.png", "path": str(output_dir / "scanpy_paga.png")})
            except Exception as exc:
                logger.warning("Could not generate pl.paga plot: %s", exc)

        # Pseudotime trajectory
        if "pseudotime" in adata.obs.columns and umap_ready:
            try:
                logger.info("Generating pseudotime trajectory plots...")
                sc.pl.umap(adata, color="pseudotime", save=f"{output_dir}/scanpy_pseudotime_umap.png", show=False)
                figures.append({"name": "scanpy_pseudotime_umap.png", "path": str(output_dir / "scanpy_pseudotime_umap.png")})
            except Exception as exc:
                logger.warning("Could not generate pseudotime UMAP: %s", exc)

    # ===== HIGHLY VARIABLE GENES =====
    if mode in ("auto", "hvg"):
        try:
            logger.info("Generating highly variable genes plot...")
            if "highly_variable" in adata.var.columns:
                sc.pl.highly_variable_genes(adata, title="Highly Variable Genes", save=f"{output_dir}/scanpy_hvg.png", show=False)
                figures.append({"name": "scanpy_hvg.png", "path": str(output_dir / "scanpy_hvg.png")})
        except Exception as exc:
            logger.warning("Could not generate highly variable genes plot: %s", exc)

    # ===== GRAPH VISUALIZATION =====
    if mode in ("auto", "embedding") and "connectivities" in adata.obsp:
        try:
            logger.info("Generating graph/connectivities visualization...")
            # Build draw_graph if not present
            if "X_draw_graph_fr" not in adata.obsm:
                try:
                    sc.tl.draw_graph(adata, prog="fr")
                except Exception as e:
                    logger.debug("Could not compute draw_graph: %s", e)
            
            if "X_draw_graph_fr" in adata.obsm:
                sc.pl.draw_graph(adata, color=cluster_key, title="Graph Layout (ForceAtlas2)", 
                               save=f"{output_dir}/scanpy_graph.png", show=False)
                figures.append({"name": "scanpy_graph.png", "path": str(output_dir / "scanpy_graph.png")})
        except Exception as exc:
            logger.warning("Could not generate graph visualization: %s", exc)

    # ===== Shared downstream plotter functions =====
    if mode in ("auto", "feature") and plot_feature:
        try:
            fig = plot_features(adata, VizParams(feature=plot_feature, basis="spatial", title=f"Spatial {plot_feature}"))
            _save_figure_result(fig, output_dir, f"spatial_feature_{plot_feature}_spatial.png", figures)
            plt.close("all")
        except Exception as exc:
            logger.debug("Could not generate spatial feature plot: %s", exc)

        if umap_ready:
            try:
                fig = plot_features(adata, VizParams(feature=plot_feature, basis="umap", title=f"UMAP {plot_feature}"))
                _save_figure_result(fig, output_dir, f"spatial_feature_{plot_feature}_umap.png", figures)
                plt.close("all")
            except Exception as exc:
                logger.debug("Could not generate UMAP feature plot: %s", exc)

    if mode in ("auto", "deconvolution"):
        method = _first_deconvolution_method(adata)
        if method:
            try:
                fig = plot_deconvolution(adata, VizParams(title=f"Deconvolution ({method})"), subtype="spatial_multi", method=method)
                _save_figure_result(fig, output_dir, f"deconvolution_{method}_spatial.png", figures)
                plt.close("all")
            except Exception as exc:
                logger.debug("Could not generate deconvolution plot: %s", exc)

            try:
                fig = plot_deconvolution(adata, VizParams(title=f"Dominant Cell Type ({method})"), subtype="dominant", method=method)
                _save_figure_result(fig, output_dir, f"deconvolution_{method}_dominant.png", figures)
                plt.close("all")
            except Exception as exc:
                logger.debug("Could not generate dominant deconvolution plot: %s", exc)

    if mode in ("auto", "communication") and _has_ccc_results(adata):
        try:
            fig = plot_communication(adata, VizParams(title="Ligand-Receptor Dotplot"), subtype="dotplot", top_n=10)
            _save_figure_result(fig, output_dir, f"communication_dotplot.png", figures)
            plt.close("all")
        except Exception as exc:
            logger.debug("Could not generate communication dotplot: %s", exc)

        try:
            fig = plot_communication(adata, VizParams(title="Ligand-Receptor Heatmap"), subtype="heatmap", top_n=10)
            _save_figure_result(fig, output_dir, f"communication_heatmap.png", figures)
            plt.close("all")
        except Exception as exc:
            logger.debug("Could not generate communication heatmap: %s", exc)

    logger.info(f"Generated {len(figures)} visualization figures using scanpy.pl API")
    return figures


def write_report(
    output_dir: Path,
    summary: dict,
    input_file: str | None,
    params: dict,
) -> None:
    header = generate_report_header(
        title="Spatial Visualization Report",
        skill_name=SKILL_NAME,
        input_files=[Path(input_file)] if input_file else None,
        extra_metadata={
            "Mode": summary.get("mode", "auto"),
            "Spatial basis": summary.get("spatial_key", ""),
        },
    )

    body_lines = [
        "## Summary\n",
        f"- **Mode**: {summary.get('mode', 'auto')}",
        f"- **Figures generated**: {summary['n_figures']}",
        f"- **Cells**: {summary['n_cells']}",
        f"- **Genes**: {summary['n_genes']}",
    ]

    if summary.get("cluster_key"):
        body_lines.append(f"- **Cluster key**: {summary['cluster_key']}")
    if summary.get("batch_key"):
        body_lines.append(f"- **Batch key**: {summary['batch_key']}")

    if summary.get("available_feature_columns"):
        body_lines.extend(["", "### Detected feature columns\n"])
        for col in summary["available_feature_columns"]:
            body_lines.append(f"- {col}")

    body_lines.extend(["", "### Generated figures\n", "| Figure | Path |", "|--------|------|"])
    for item in summary.get("figures", []):
        body_lines.append(f"| {item['name']} | {item['path']} |")

    body_lines.extend(["", "## Parameters\n"])
    for key, value in params.items():
        if value is not None:
            body_lines.append(f"- `{key}`: {value}")

    footer = generate_report_footer()
    (output_dir / "report.md").write_text(header + "\n".join(body_lines) + "\n" + footer)

    checksum = sha256_file(input_file) if input_file and Path(input_file).exists() else ""
    write_result_json(
        output_dir,
        skill=SKILL_NAME,
        version=SKILL_VERSION,
        summary=summary,
        data={"params": params, **summary},
        input_checksum=checksum,
    )

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    pd.DataFrame(summary.get("figures", [])).to_csv(tables_dir / "figure_manifest.csv", index=False)

    repro_dir = output_dir / "reproducibility"
    repro_dir.mkdir(exist_ok=True)
    cmd = f"python spatial_visualization.py --input <input.h5ad> --output {output_dir}"
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd += f" --{key.replace('_', '-')}"
        elif isinstance(value, list):
            for item in value:
                cmd += f" --{key.replace('_', '-')} {item}"
        else:
            cmd += f" --{key.replace('_', '-')} {value}"
    (repro_dir / "commands.sh").write_text(f"#!/bin/bash\n{cmd}\n")
    (repro_dir / "environment.txt").write_text("scanpy\npandas\nnumpy\nmatplotlib\n")


def run(
    adata: ad.AnnData,
    output_dir: Path,
    *,
    mode: str = "auto",
    feature: str | None = None,
    genes: list[str] | None = None,
    cluster_key: str | None = None,
    batch_key: str | None = None,
) -> dict:
    """Run the visualization workflow and write outputs to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_key = _ensure_spatial_basis(adata)
    if "X_umap" not in adata.obsm:
        _ensure_umap(adata)

    figures = generate_figures(
        adata,
        output_dir,
        mode=mode,
        feature=feature,
        genes=genes,
        cluster_key=cluster_key,
        batch_key=batch_key,
    )

    available_feature_columns = [
        col for col in ["spatial_domain", "cell_type", "leiden", "louvain", "cluster", "celltype"]
        if col in adata.obs.columns
    ]

    summary = {
        "mode": mode,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_figures": len(figures),
        "spatial_key": spatial_key or "",
        "cluster_key": cluster_key or _first_existing_column(adata, ["spatial_domain", "leiden", "louvain", "cell_type", "celltype"]),
        "batch_key": batch_key or _first_existing_column(adata, ["batch", "sample", "library_id", "slide"]),
        "available_feature_columns": available_feature_columns,
        "figures": figures,
    }

    params = {
        "mode": mode,
        "feature": feature,
        "genes": genes,
        "cluster_key": cluster_key,
        "batch_key": batch_key,
    }
    write_report(output_dir, summary, adata.uns.get("input_file"), params)
    _store_analysis_metadata(
        adata,
        skill_name=SKILL_NAME,
        method=mode,
        params=params,
    )
    return summary


def build_demo_data() -> tuple[ad.AnnData, str | None]:
    """Build demo AnnData for ``--demo``."""
    adata = build_demo_adata()
    return adata, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial Visualization — generic plotting for spatial transcriptomics")
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output", dest="output_dir", required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--mode", choices=["auto", "feature", "expression", "integration", "stats", "trajectory", "deconvolution", "communication", "all"], default="auto")
    parser.add_argument("--feature", default=None, help="Gene or obs column to visualize directly")
    parser.add_argument("--genes", nargs="*", default=None, help="Genes for expression heatmaps")
    parser.add_argument("--cluster-key", default=None)
    parser.add_argument("--batch-key", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        adata, input_file = build_demo_data()
    elif args.input_path:
        adata = ad.read_h5ad(args.input_path)
        input_file = args.input_path
    else:
        print("ERROR: Provide --input or --demo", file=sys.stderr)
        sys.exit(1)

    adata.uns["input_file"] = input_file
    summary = run(
        adata,
        output_dir,
        mode=args.mode,
        feature=args.feature,
        genes=args.genes,
        cluster_key=args.cluster_key,
        batch_key=args.batch_key,
    )

    logger.info("Generated %d figure(s) in %s", summary["n_figures"], output_dir)


if __name__ == "__main__":
    main()
