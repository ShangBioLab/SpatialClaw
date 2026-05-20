#!/usr/bin/env python3
"""Spatial multi-omics integration with multiple backends.

Supports SpatialGlue and SpaDDM behind one CLI contract for aligned spatial
RNA+protein or RNA+ATAC integration.

Usage:
    python spatialclaw.py run spatial-omics-integrate --input <omics1.h5ad> --omics2 <omics2.h5ad> --output <dir>
    python spatial_omics_integrate.py --input <omics1.h5ad> --omics2 <omics2.h5ad> --output <dir>
    python spatialclaw.py run spatial-omics-integrate --demo --output <dir>
    python spatial_omics_integrate.py --demo --output <dir>
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scanpy as sc

from spatialclaw.common.checksums import sha256_file
from spatialclaw.common.report import (
    generate_report_footer,
    generate_report_header,
    write_result_json,
)
from skills.spatial._lib.adata_utils import store_analysis_metadata
from skills.spatial._lib.omics_integration import (
    compute_attention_balance,
    get_multiomics_method_spec,
    normalize_multiomics_method,
    run_spatial_omics_integration,
)
from skills.spatial._lib.figure_io import save_figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SKILL_NAME = "spatial-omics-integrate"
SKILL_VERSION = "2.0.0"


def _format_modality_label(modality: str | None) -> str:
    """Return a concise display label for a modality key."""
    mapping = {"rna": "RNA", "protein": "Protein", "atac": "ATAC"}
    key = str(modality or "omics").lower()
    return mapping.get(key, key.upper())


def _build_preprocessing_row(modality: str, preprocess_summary: dict) -> dict:
    """Normalize modality-specific preprocessing summaries into one table row."""
    feature_keys = ["n_genes_original", "n_proteins", "n_peaks", "n_features_final"]
    selected_keys = ["n_hvg_selected", "n_features_final", "n_proteins", "n_peaks"]
    n_features_original = next(
        (preprocess_summary[k] for k in feature_keys if k in preprocess_summary),
        None,
    )
    n_features_selected = next(
        (preprocess_summary[k] for k in selected_keys if k in preprocess_summary),
        None,
    )
    n_components = preprocess_summary.get("n_pcs", preprocess_summary.get("n_components"))
    return {
        "modality": _format_modality_label(modality),
        "n_features_original": n_features_original,
        "n_features_selected": n_features_selected,
        "n_components": n_components,
        "normalization": preprocess_summary.get("normalization"),
    }


def _method_label(method: str) -> str:
    return get_multiomics_method_spec(method).label


def _output_basename(method: str) -> str:
    return f"integrated_{_method_label(method)}.h5ad"


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------


def generate_figures(
    adata,
    output_dir: Path,
    summary: dict,
    omics_info: dict | None = None,
) -> list[str]:
    """Generate multi-omics integration visualization figures."""
    figures: list[str] = []
    if omics_info is None:
        omics_info = {
            "omics1_type": summary.get("omics1_modality", "rna"),
            "omics2_type": summary.get("omics2_modality", "protein"),
        }
    omics1_label = _format_modality_label(omics_info.get("omics1_type"))
    omics2_label = _format_modality_label(omics_info.get("omics2_type"))
    method_label = summary.get("method_label", "Integration")
    cluster_key = summary.get("cluster_key", "spatial_omics_cluster")
    embedding_key = summary.get("embedding_key", "X_spatial_omics")

    if "X_umap" in adata.obsm and cluster_key in adata.obs:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            umap = adata.obsm["X_umap"]
            clusters = pd.Categorical(adata.obs[cluster_key]).codes
            scatter1 = axes[0].scatter(
                umap[:, 0], umap[:, 1], c=clusters, cmap="tab20", s=10, alpha=0.8
            )
            axes[0].set_xlabel("UMAP 1")
            axes[0].set_ylabel("UMAP 2")
            axes[0].set_title(f"{method_label} Clusters — UMAP")
            axes[0].set_aspect("equal")
            plt.colorbar(scatter1, ax=axes[0], label="Cluster")

            if "spatial" in adata.obsm:
                spatial = adata.obsm["spatial"]
                scatter2 = axes[1].scatter(
                    spatial[:, 0], spatial[:, 1], c=clusters, cmap="tab20", s=10, alpha=0.8
                )
                axes[1].set_xlabel("X")
                axes[1].set_ylabel("Y")
                axes[1].set_title(f"{method_label} Clusters — Spatial")
                axes[1].set_aspect("equal")
                plt.colorbar(scatter2, ax=axes[1], label="Cluster")
            else:
                latent_omics1 = adata.obsm.get("emb_latent_omics1")
                if latent_omics1 is not None:
                    scatter2 = axes[1].scatter(
                        latent_omics1[:, 0],
                        latent_omics1[:, 1],
                        c=clusters,
                        cmap="tab20",
                        s=10,
                        alpha=0.8,
                    )
                    axes[1].set_xlabel("Latent Dim 1")
                    axes[1].set_ylabel("Latent Dim 2")
                    axes[1].set_title(f"{omics1_label} Latent Space")
                    plt.colorbar(scatter2, ax=axes[1], label="Cluster")

            fig.tight_layout()
            p = save_figure(fig, output_dir, "umap_spatial_clusters.png")
            figures.append(str(p))
        except Exception as exc:
            logger.warning("Could not generate UMAP figure: %s", exc)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        alpha = adata.obsm.get("alpha")
        if alpha is not None and alpha.shape[1] == 2:
            df = pd.DataFrame({
                omics1_label: alpha[:, 0],
                omics2_label: alpha[:, 1],
                "Cluster": adata.obs[cluster_key].astype(str).values,
            })
            df_melted = df.melt(
                id_vars=["Cluster"], value_vars=[omics1_label, omics2_label],
                var_name="Modality", value_name="Weight"
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.violinplot(
                data=df_melted, x="Cluster", y="Weight", hue="Modality",
                split=True, inner="quart", ax=ax
            )
            ax.set_title(f"{method_label} Attention Weights by Cluster")
            ax.set_ylabel("Attention Weight")
            ax.set_xlabel("Cluster")

            fig.tight_layout()
            p = save_figure(fig, output_dir, "attention_weights.png")
            figures.append(str(p))
    except Exception as exc:
        logger.warning("Could not generate attention weights figure: %s", exc)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        modalities = [omics1_label, omics2_label]
        n_features = [summary.get("n_omics1_features", 0), summary.get("n_omics2_features", 0)]
        colors = ["#1f77b4", "#ff7f0e"]

        bars = axes[0].bar(modalities, n_features, color=colors, edgecolor="black", width=0.6)
        axes[0].set_ylabel("Number of Features")
        axes[0].set_title("Feature Selection")
        for bar, feat in zip(bars, n_features):
            height = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(feat)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        n_clusters = summary.get("n_clusters", 0)
        n_cells = summary.get("n_cells", 0)
        cluster_sizes = pd.Series(adata.obs[cluster_key]).value_counts().sort_index()
        axes[1].bar(
            range(len(cluster_sizes)),
            cluster_sizes.values,
            color=plt.cm.tab20(np.linspace(0, 1, len(cluster_sizes))),
            edgecolor="black",
        )
        axes[1].set_xlabel("Cluster ID")
        axes[1].set_ylabel("Number of Cells")
        axes[1].set_title(f"Cluster Distribution ({n_clusters} clusters, {n_cells} cells)")

        fig.tight_layout()
        p = save_figure(fig, output_dir, "feature_clustering_stats.png")
        figures.append(str(p))
    except Exception as exc:
        logger.warning("Could not generate feature/clustering stats figure: %s", exc)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        workflow_text = f"""
{method_label} Multi-Omics Integration Workflow

Input:
  • Omics1 ({omics1_label}): {summary.get('n_omics1_features', 0)} features
  • Omics2 ({omics2_label}): {summary.get('n_omics2_features', 0)} features
  • Cells: {summary.get('n_cells', 0)}

Processing:
  1. Preprocess Omics1 ({omics1_label})
  2. Preprocess Omics2 ({omics2_label})
  3. Build spatial neighbor graphs
  4. Train {method_label}
  5. Cluster with {summary.get('clustering_method', 'leiden')}

Output:
  • {summary.get('n_clusters', 0)} clusters identified
  • Joint embedding: {embedding_key}
  • Attention weights: inter-modality contribution
  • Modality-specific latent spaces preserved

Methods:
  • Method: {method_label}
  • Cluster key: {cluster_key}
  • Platform: {summary.get('datatype', '10x')}
"""

        ax.text(
            0.05,
            0.95,
            workflow_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        fig.tight_layout()
        p = save_figure(fig, output_dir, "workflow_summary.png")
        figures.append(str(p))
    except Exception as exc:
        logger.warning("Could not generate workflow summary: %s", exc)

    return figures


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def write_report(
    output_dir: Path,
    summary: dict,
    omics1_file: str | None,
    omics2_file: str | None,
    params: dict,
    omics_info: dict | None = None,
) -> None:
    """Generate comprehensive integration report."""
    if omics_info is None:
        omics_info = {
            "omics1_type": summary.get("omics1_modality", "rna"),
            "omics2_type": summary.get("omics2_modality", "protein"),
        }
    omics1_label = _format_modality_label(omics_info.get("omics1_type"))
    omics2_label = _format_modality_label(omics_info.get("omics2_type"))
    method = summary.get("method", "spatialglue")
    method_label = summary.get("method_label", _method_label(method))
    embedding_key = summary.get("embedding_key", "X_spatial_omics")
    cluster_key = summary.get("cluster_key", "spatial_omics_cluster")

    header = generate_report_header(
        title=f"{method_label} Multi-Omics Integration Report",
        skill_name=SKILL_NAME,
        input_files=[Path(f) for f in [omics1_file, omics2_file] if f],
        extra_metadata={
            "Method": method_label,
            "Integration Type": "Spatial Multi-Omics",
        },
    )

    body_lines = [
        "## Executive Summary\n",
        f"{method_label} integrates aligned spatial omics modalities into a joint representation.",
        "SPATIALCLAW standardizes the inputs, outputs, figures, and reproducibility metadata across methods.",
        "",
        "## Data Overview\n",
        f"- **Total Cells**: {summary['n_cells']}",
        f"- **Omics1 ({omics1_label} features)**: {summary['n_omics1_features']}",
        f"- **Omics2 ({omics2_label} features)**: {summary['n_omics2_features']}",
        f"- **Data Type**: {summary['datatype']}",
        "",
        "## Integration Results\n",
        f"- **Method**: {method_label}",
        f"- **Compute Device**: {summary.get('device', 'cpu')}",
        f"- **Clustering Method**: {summary['clustering_method']}",
        f"- **Embedding Key**: `{embedding_key}`",
        f"- **Cluster Key**: `{cluster_key}`",
        f"- **Number of Clusters**: {summary['n_clusters']}",
        f"- **Random Seed**: {summary['random_seed']}",
        "",
        "## Output Embeddings\n",
        f"- **emb_latent_omics1**: Latent representation for first modality ({omics1_label})",
        f"- **emb_latent_omics2**: Latent representation for second modality ({omics2_label})",
        f"- **{embedding_key}**: Joint integrated representation combining both modalities",
        "",
    ]

    if method == "spatialglue":
        body_lines.extend([
            "## Attention Mechanisms\n",
            f"- **alpha_omics1**: Intra-modality attention for {omics1_label}",
            f"- **alpha_omics2**: Intra-modality attention for {omics2_label}",
            "- **alpha**: Inter-modality attention across the two modalities",
            "",
        ])
    elif method == "spaddm":
        body_lines.extend([
            "## Method Notes\n",
            "- SpaDDM uses a directional diffusion model over modality-specific spatial graphs.",
            "- SPATIALCLAW keeps SpaDDM's latent representations and cross-modality reconstructions in the output AnnData.",
            "",
        ])

    attention_metrics = summary.get("attention_metrics", {})
    if attention_metrics:
        body_lines.extend([
            "## Attention Balance\n",
            f"- **Avg Attention (Omics1)**: {attention_metrics.get('avg_attention_omics1', 0):.4f}",
            f"- **Avg Attention (Omics2)**: {attention_metrics.get('avg_attention_omics2', 0):.4f}",
            f"- **Attention Entropy (normalized)**: {attention_metrics.get('attention_entropy_normalized', 0):.4f}",
            "",
        ])

    reconstruction_metrics = summary.get("reconstruction_metrics", {})
    if reconstruction_metrics:
        body_lines.extend([
            "## Reconstruction Metrics\n",
            f"- **Omics1 Reconstruction MSE**: {reconstruction_metrics.get('omics1_reconstruction_mse', 0):.6f}",
            f"- **Omics2 Reconstruction MSE**: {reconstruction_metrics.get('omics2_reconstruction_mse', 0):.6f}",
            "",
        ])

    body_lines.extend([
        "## Preprocessing Details\n",
        f"### Omics1 ({omics1_label})",
    ])

    if omics_info["omics1_type"] == "rna":
        body_lines.extend([
            f"- Initial genes: {summary['preprocessing']['omics1']['n_genes_original']}",
            f"- Final HVGs: {summary['preprocessing']['omics1']['n_hvg_selected']}",
            f"- Scaled: {summary['preprocessing']['omics1'].get('scaled', True)}",
            "- Workflow: filter genes -> HVG -> normalize_total -> log1p -> PCA",
        ])
    elif omics_info["omics1_type"] == "protein":
        body_lines.extend([
            f"- Features: {summary['preprocessing']['omics1'].get('n_proteins', 0)}",
            "- Workflow: CLR normalize -> scale -> PCA",
        ])
    elif omics_info["omics1_type"] == "atac":
        body_lines.extend([
            f"- Peaks: {summary['preprocessing']['omics1'].get('n_peaks', 0)}",
            f"- Workflow: {summary['preprocessing']['omics1'].get('normalization', 'TF-IDF + LSI')}",
        ])

    body_lines.extend([
        "",
        f"### Omics2 ({omics2_label})",
    ])

    if omics_info["omics2_type"] == "rna":
        body_lines.extend([
            f"- Initial genes: {summary['preprocessing']['omics2']['n_genes_original']}",
            f"- Final HVGs: {summary['preprocessing']['omics2']['n_hvg_selected']}",
            f"- Scaled: {summary['preprocessing']['omics2'].get('scaled', True)}",
            "- Workflow: filter genes -> HVG -> normalize_total -> log1p -> PCA",
        ])
    elif omics_info["omics2_type"] == "protein":
        body_lines.extend([
            f"- Features: {summary['preprocessing']['omics2'].get('n_proteins', 0)}",
            "- Workflow: CLR normalize -> scale -> PCA",
        ])
    elif omics_info["omics2_type"] == "atac":
        body_lines.extend([
            f"- Peaks: {summary['preprocessing']['omics2'].get('n_peaks', 0)}",
            f"- Workflow: {summary['preprocessing']['omics2'].get('normalization', 'TF-IDF + LSI')}",
        ])

    body_lines.extend([
        "",
        "## Parameters\n",
    ])

    for k, v in params.items():
        if k not in ["omics1_file", "omics2_file"]:
            body_lines.append(f"- `{k}`: {v}")

    if "n_epochs_requested" in summary and "n_epochs_actual" in summary:
        body_lines.extend([
            "",
            "## Training Information\n",
            f"- **Epochs Requested / Effective**: {summary['n_epochs_requested']}",
            f"- **Epochs Actual**: {summary['n_epochs_actual']}",
        ])

    body_lines.extend([
        "",
        "## Downstream Analysis\n",
        f"The integrated `{embedding_key}` representation can be used for:",
        f"- **Clustering**: Already performed using {summary['clustering_method']}",
        "- **Trajectory Analysis**: Use UMAP or neighborhood graphs for structure analysis",
        "- **DEG Analysis**: Compare expression across clusters or conditions",
        "- **Cell Type Annotation**: Match clusters to known cell types",
        "- **Cross-Modality Analysis**: Compare latent spaces and modality contributions",
        "",
    ])

    if method == "spatialglue":
        body_lines.extend([
            "## References\n",
            "- SpatialGlue: https://github.com/JinmiaoChenLab/SpatialGlue",
            "- SpatialGlue tutorials: https://spatialglue-tutorials.readthedocs.io/",
            "",
        ])
    else:
        body_lines.extend([
            "## References\n",
            "- SpaDDM: https://github.com/WHY-17/SpaDDM",
            "- Cross-omics translation in the upstream SpaDDM repository is based on moscot and is not run automatically by this skill.",
            "",
        ])

    footer = generate_report_footer()
    report = header + "\n".join(body_lines) + "\n" + footer
    (output_dir / "report.md").write_text(report)
    logger.info("Wrote report: %s", output_dir / "report.md")

    # Save result JSON
    checksums = {}
    if omics1_file and Path(omics1_file).exists():
        checksums["omics1_input"] = sha256_file(omics1_file)
    if omics2_file and Path(omics2_file).exists():
        checksums["omics2_input"] = sha256_file(omics2_file)

    write_result_json(
        output_dir,
        skill=SKILL_NAME,
        version=SKILL_VERSION,
        summary={k: v for k, v in summary.items() if k != "adata_integrated"},
        data={
            "params": params,
            "preprocessing": summary.get("preprocessing"),
            "attention_metrics": summary.get("attention_metrics"),
            "reconstruction_metrics": summary.get("reconstruction_metrics"),
        },
        input_checksum=checksums,
    )

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    metrics_df = pd.DataFrame([
        {"metric": "method", "value": method_label},
        {"metric": "n_cells", "value": summary["n_cells"]},
        {"metric": "n_omics1_features", "value": summary["n_omics1_features"]},
        {"metric": "n_omics2_features", "value": summary["n_omics2_features"]},
        {"metric": "n_clusters", "value": summary["n_clusters"]},
        {"metric": "clustering_method", "value": summary["clustering_method"]},
        {"metric": "datatype", "value": summary["datatype"]},
        {"metric": "device", "value": summary.get("device", "cpu")},
    ])
    metrics_df.to_csv(tables_dir / "integration_metrics.csv", index=False)
    logger.info("Saved metrics table")

    preprocess_df = pd.DataFrame([
        _build_preprocessing_row(omics_info["omics1_type"], summary["preprocessing"]["omics1"]),
        _build_preprocessing_row(omics_info["omics2_type"], summary["preprocessing"]["omics2"]),
    ])
    preprocess_df.to_csv(tables_dir / "preprocessing_summary.csv", index=False)

    if reconstruction_metrics:
        pd.DataFrame(
            [{"metric": k, "value": v} for k, v in reconstruction_metrics.items()]
        ).to_csv(tables_dir / "reconstruction_metrics.csv", index=False)

    repro_dir = output_dir / "reproducibility"
    repro_dir.mkdir(exist_ok=True)
    cmd_parts = [f"python spatial_omics_integrate.py \\"]
    if omics1_file and omics2_file:
        cmd_parts.extend([
            f"  --input {omics1_file} \\",
            f"  --omics2 {omics2_file} \\",
            f"  --output {output_dir}",
        ])
    else:
        cmd_parts.extend([
            "  --demo \\",
            f"  --output {output_dir}",
        ])
    for k, v in params.items():
        if v is not None and k not in ["omics1_file", "omics2_file", "output_dir"]:
            cmd_parts.append(f"  --{str(k).replace('_', '-')} {v}")

    (repro_dir / "commands.sh").write_text("#!/bin/bash\n" + " \\\n".join(cmd_parts) + "\n")

    env_lines = []
    packages = ["torch", "scanpy", "anndata", "numpy", "pandas", "scipy", "scikit-learn"]
    if method == "spatialglue":
        packages.insert(1, "SpatialGlue")
    else:
        packages.insert(1, "torch-geometric")
    for pkg in packages:
        try:
            from importlib.metadata import version as _get_version
            env_lines.append(f"{pkg}=={_get_version(pkg)}")
        except Exception:
            env_lines.append(f"{pkg}=?")
    
    (repro_dir / "environment.yml").write_text("\n".join(env_lines) + "\n")
    logger.info("Reproducibility metadata saved")


# ---------------------------------------------------------------------------
# Demo Data
# ---------------------------------------------------------------------------


def get_demo_data() -> tuple:
    """Generate synthetic demo data for testing."""
    logger.info("Generating synthetic demo data")
    
    # Use preprocessing demo as base
    preprocess_script = (
        _PROJECT_ROOT / "skills" / "spatial" / "spatial-preprocessing" / "spatial_preprocessing.py"
    )
    
    with tempfile.TemporaryDirectory(prefix="omics_integration_demo_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        result = subprocess.run(
            [sys.executable, str(preprocess_script), "--demo", "--output", str(tmp_path)],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Demo generation failed: {result.stderr}")
        
        processed = tmp_path / "processed.h5ad"
        adata_omics1 = sc.read_h5ad(processed)
    
    # Create synthetic protein data
    rng = np.random.default_rng(42)
    n_proteins = max(10, int(adata_omics1.n_vars * 0.15))
    protein_indices = rng.choice(adata_omics1.n_vars, size=n_proteins, replace=False)
    
    adata_omics2 = adata_omics1[:, protein_indices].copy()
    X_protein = adata_omics2.X.copy()
    if not isinstance(X_protein, np.ndarray):
        X_protein = X_protein.toarray()
    
    X_protein = X_protein + rng.normal(0, 0.1, X_protein.shape)
    X_protein = np.maximum(X_protein, 0)
    
    adata_omics2.X = X_protein
    adata_omics2.var["protein_names"] = [f"Protein_{i}" for i in range(adata_omics2.n_vars)]
    
    logger.info("Demo: Omics1 %d×%d, Omics2 %d×%d",
               adata_omics1.n_obs, adata_omics1.n_vars, adata_omics2.n_obs, adata_omics2.n_vars)
    
    return adata_omics1, adata_omics2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _require_positive(parser: argparse.ArgumentParser, *, flag: str, value: int | None) -> None:
    if value is not None and value <= 0:
        parser.error(f"{flag} must be a positive integer.")


def main():
    parser = argparse.ArgumentParser(
        description="Spatial multi-omics integration with SpatialGlue or SpaDDM",
    )
    parser.add_argument("--input", dest="omics1_path", help="Path to first omics h5ad (e.g., RNA)")
    parser.add_argument("--omics2", dest="omics2_path", help="Path to second omics h5ad (e.g., Protein)")
    parser.add_argument("--output", dest="output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--method",
        default="spatialglue",
        choices=["spatialglue", "spaddm"],
        help="Integration backend: spatialglue or spaddm (default: spatialglue)",
    )
    parser.add_argument("--omics1-type", default="rna",
                       choices=["rna", "protein", "atac"],
                       help="Type of first omics: rna, protein, atac (default: rna)")
    parser.add_argument("--omics2-type", default="protein",
                       choices=["rna", "protein", "atac"],
                       help="Type of second omics: rna, protein, atac (default: protein)")
    parser.add_argument(
        "--data-type",
        dest="datatype",
        default="auto",
        choices=[
            "auto",
            "10x",
            "Stereo-CITE-seq",
            "SPOTS",
            "spatial-epigenome",
            "Spatial-epigenome-transcriptome",
            "Visium CytAssist",
        ],
        help="Platform / data type (default: auto)",
    )
    parser.add_argument(
        "--clustering-method",
        default="auto",
        choices=["auto", "leiden", "louvain"],
        help="Cluster assignment method after integration (default: auto)",
    )
    parser.add_argument("--demo", action="store_true", help="Use demo data")
    parser.add_argument("--n-hvg", type=int, default=3000, help="Number of HVGs (default: 3000)")
    parser.add_argument("--n-clusters", type=int, default=None, help="Number of clusters (auto if not specified)")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed (method-specific default if omitted)")
    parser.add_argument("--n-epochs", type=int, default=None, help="Training epochs (method-specific default if omitted)")
    parser.add_argument("--n-latent", type=int, default=64, help="Latent dimension for SpaDDM (default: 64)")
    parser.add_argument(
        "--device",
        default="gpu",
        help="Compute device: gpu (default, prefers CUDA with CPU fallback), auto, cpu, cuda, cuda:0, etc.",
    )

    args = parser.parse_args()
    _require_positive(parser, flag="--n-hvg", value=args.n_hvg)
    _require_positive(parser, flag="--n-latent", value=args.n_latent)
    _require_positive(parser, flag="--n-clusters", value=args.n_clusters)
    _require_positive(parser, flag="--n-epochs", value=args.n_epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.demo:
        logger.info("Loading demo data")
        adata_omics1, adata_omics2 = get_demo_data()
        omics1_file = None
        omics2_file = None
    elif args.omics1_path and args.omics2_path:
        omics1_path = Path(args.omics1_path)
        omics2_path = Path(args.omics2_path)
        
        if not omics1_path.exists() or not omics2_path.exists():
            print("ERROR: Input files not found", file=sys.stderr)
            sys.exit(1)
        
        logger.info("Loading omics1 from %s", omics1_path)
        adata_omics1 = sc.read_h5ad(omics1_path)
        logger.info("Loading omics2 from %s", omics2_path)
        adata_omics2 = sc.read_h5ad(omics2_path)
        
        omics1_file = str(omics1_path)
        omics2_file = str(omics2_path)
    else:
        print("ERROR: Provide --input and --omics2, or use --demo", file=sys.stderr)
        sys.exit(1)

    device = args.device

    normalized_method = normalize_multiomics_method(args.method)
    method_label = _method_label(normalized_method)
    params = {
        "method": normalized_method,
        "omics1_type": args.omics1_type,
        "omics2_type": args.omics2_type,
        "datatype": args.datatype,
        "clustering_method": args.clustering_method,
        "n_hvg": args.n_hvg,
        "n_clusters": args.n_clusters,
        "random_seed": args.random_seed,
        "n_epochs": args.n_epochs,
        "n_latent": args.n_latent,
        "device": args.device,
    }

    logger.info("Starting %s integration", method_label)
    summary = run_spatial_omics_integration(
        adata_omics1,
        adata_omics2,
        method=normalized_method,
        omics_type_1=args.omics1_type,
        omics_type_2=args.omics2_type,
        datatype=args.datatype,
        n_hvg=args.n_hvg,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed,
        device=device,
        n_epochs=args.n_epochs,
        clustering_method=args.clustering_method,
        n_latent=args.n_latent,
    )

    attention_metrics = compute_attention_balance(summary["attention_weights"]["inter_omics"])
    summary["attention_metrics"] = attention_metrics

    adata_integrated = summary.pop("adata_integrated")
    summary.pop("embeddings", None)
    summary.pop("attention_weights", None)

    omics_info = {
        "omics1_type": summary.get("omics1_modality", args.omics1_type),
        "omics2_type": summary.get("omics2_modality", args.omics2_type),
    }
    figures = generate_figures(adata_integrated, output_dir, summary, omics_info)
    logger.info("Generated %d figures", len(figures))

    write_report(output_dir, summary, omics1_file, omics2_file, params, omics_info)

    store_analysis_metadata(
        adata_integrated,
        SKILL_NAME,
        method_label,
        params=params,
    )

    h5ad_path = output_dir / _output_basename(normalized_method)
    adata_integrated.write_h5ad(h5ad_path)
    logger.info("Saved integrated data: %s", h5ad_path)

    omics1_label = _format_modality_label(omics_info["omics1_type"])
    omics2_label = _format_modality_label(omics_info["omics2_type"])
    print(
        f"\n✓ {method_label} integration complete!\n"
        f"  Method: {method_label}\n"
        f"  Cells: {summary['n_cells']}\n"
        f"  Omics1 ({omics1_label}) features: {summary['n_omics1_features']}\n"
        f"  Omics2 ({omics2_label}) features: {summary['n_omics2_features']}\n"
        f"  Clusters: {summary['n_clusters']}\n"
        f"  Clustering: {summary['clustering_method']}\n"
        f"  Epochs: {summary['n_epochs_actual']}\n"
        f"  Attention balance: {attention_metrics.get('attention_entropy_normalized', 0.0):.3f}\n"
        f"  Output: {output_dir}\n"
    )


if __name__ == "__main__":
    main()
