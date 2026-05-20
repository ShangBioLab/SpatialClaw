#!/usr/bin/env python3
"""Spatial Multi-sample Integration.

Integrates multiple spatial transcriptomics samples, performs batch correction,
and produces a standard SPATIALCLAW report bundle.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spatialclaw.common.checksums import sha256_file
from spatialclaw.common.report import (
    generate_report_footer,
    generate_report_header,
    write_result_json,
)
from skills.spatial._lib.dependency_manager import is_available

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SKILL_NAME = "spatial-multi-sample-integration"
SKILL_VERSION = "0.1.0"
SUPPORTED_METHODS = ("auto", "harmony", "bbknn", "scanorama", "staligner")


def _require_scanpy():
    try:
        import scanpy as sc  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "scanpy is required for spatial integration. Install with: pip install scanpy"
        ) from exc


def _store_analysis_metadata(adata, *, method: str, params: dict) -> None:
    adata.uns[f"spatialclaw_{SKILL_NAME}"] = {
        "method": method,
        "params": params,
    }


def _compute_batch_mixing(adata, batch_key: str) -> float:
    """Neighbourhood entropy-based batch mixing score in [0, 1]."""
    try:
        from scipy import sparse

        if "connectivities" not in adata.obsp:
            return 0.0
        conn = adata.obsp["connectivities"]
        if sparse.issparse(conn):
            conn = conn.toarray()

        labels = adata.obs[batch_key].to_numpy()
        batches = np.unique(labels)
        if len(batches) < 2:
            return 0.0

        entropies = []
        for idx in range(adata.n_obs):
            nn_idx = np.where(conn[idx] > 0)[0]
            if len(nn_idx) == 0:
                continue
            nn_labels = labels[nn_idx]
            counts = np.array([(nn_labels == b).sum() for b in batches], dtype=float)
            probs = counts / max(counts.sum(), 1.0)
            probs = probs[probs > 0]
            if len(probs) == 0:
                continue
            ent = float(-(probs * np.log(probs)).sum())
            entropies.append(ent)

        if not entropies:
            return 0.0
        return float(np.mean(entropies) / np.log(len(batches)))
    except Exception:
        return 0.0


def _pick_method(requested: str) -> str:
    if requested != "auto":
        return requested
    if is_available("STAligner"):
        return "staligner"
    if is_available("harmonypy"):
        return "harmony"
    if is_available("bbknn"):
        return "bbknn"
    if is_available("scanorama"):
        return "scanorama"
    return "fallback"


def _ensure_precompute(adata):
    import scanpy as sc

    if "X_pca" not in adata.obsm:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(3000, adata.n_vars), subset=True)
        sc.pp.scale(adata, max_value=10)
        n_comps = min(50, max(2, adata.n_vars - 1))
        sc.tl.pca(adata, n_comps=n_comps)

    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)


def _run_fallback(adata):
    """No optional integration backend available: keep PCA baseline."""
    import scanpy as sc

    _ensure_precompute(adata)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    sc.tl.umap(adata)
    return {"method": "fallback", "embedding_key": "X_pca"}


def _run_selected_integration(adata, method: str, batch_key: str):
    if method == "fallback":
        return _run_fallback(adata)

    from skills.spatial._lib.integration import run_integration

    return run_integration(adata, method=method, batch_key=batch_key)


def run_multi_sample_integration(adata, *, method: str = "auto", batch_key: str = "sample") -> dict:
    import scanpy as sc

    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"Batch key '{batch_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    batches = sorted(map(str, adata.obs[batch_key].unique().tolist()))
    if len(batches) < 2:
        multi_value_cols = []
        for col in adata.obs.columns:
            try:
                n_unique = int(adata.obs[col].nunique())
            except Exception:
                continue
            if n_unique >= 2:
                multi_value_cols.append((col, n_unique))

        multi_value_cols.sort(key=lambda x: x[1], reverse=True)
        hint_cols = ", ".join([f"{c}({n})" for c, n in multi_value_cols[:8]]) or "none"
        raise ValueError(
            f"Need at least 2 batches for multi-sample integration, but '{batch_key}' has only {len(batches)} unique value(s): {batches}. "
            f"Try a different --batch-key. Candidate obs columns with >=2 unique values: {hint_cols}"
        )
    batch_sizes = {b: int((adata.obs[batch_key].astype(str) == b).sum()) for b in batches}

    _ensure_precompute(adata)
    umap_before = adata.obsm.get("X_umap", None)
    if umap_before is not None:
        adata.obsm["X_umap_before_integration"] = umap_before.copy()

    before = _compute_batch_mixing(adata, batch_key)
    selected_method = _pick_method(method)
    logger.info("Requested method=%s, selected method=%s", method, selected_method)

    result = _run_selected_integration(adata, selected_method, batch_key)

    # ensure embedding + cluster labels exist
    if "neighbors" not in adata.uns:
        use_rep = result.get("embedding_key", "X_pca")
        sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)
    if "leiden" not in adata.obs.columns:
        sc.tl.leiden(adata, resolution=1.0, flavor="igraph")

    after = _compute_batch_mixing(adata, batch_key)
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_batches": int(len(batches)),
        "batches": batches,
        "batch_sizes": batch_sizes,
        "method": result.get("method", selected_method),
        "embedding_key": result.get("embedding_key", "X_pca"),
        "batch_mixing_before": round(before, 4),
        "batch_mixing_after": round(after, 4),
    }


def _save_figure(fig, output_dir: Path, filename: str) -> Path:
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_figures(adata, output_dir: Path, summary: dict, batch_key: str) -> list[str]:
    import matplotlib.pyplot as plt

    figures: list[str] = []
    if "X_umap" in adata.obsm and batch_key in adata.obs.columns:
        coords = np.asarray(adata.obsm["X_umap"])

        fig, ax = plt.subplots(figsize=(8, 6))
        labels = adata.obs[batch_key].astype(str).to_numpy()
        uniq = sorted(pd.unique(labels))
        cmap = plt.get_cmap("tab20", len(uniq))
        for i, u in enumerate(uniq):
            mask = labels == u
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.8, label=u, color=cmap(i))
        ax.set_title("UMAP by Batch")
        ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        figures.append(str(_save_figure(fig, output_dir, "umap_by_batch.png")))

        if "leiden" in adata.obs.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            cl = adata.obs["leiden"].astype(str).to_numpy()
            clu = sorted(pd.unique(cl))
            ccmap = plt.get_cmap("tab20", len(clu))
            for i, c in enumerate(clu):
                mask = cl == c
                ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.8, label=c, color=ccmap(i))
            ax.set_title("UMAP by Cluster (Leiden)")
            ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
            figures.append(str(_save_figure(fig, output_dir, "umap_by_cluster.png")))

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [summary.get("batch_mixing_before", 0.0), summary.get("batch_mixing_after", 0.0)]
    ax.bar(["Before", "After"], vals, color=["#d9534f", "#5cb85c"], edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Batch Mixing Entropy")
    ax.set_title("Integration Quality")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    figures.append(str(_save_figure(fig, output_dir, "batch_mixing.png")))

    return figures


def write_report(output_dir: Path, summary: dict, input_file: str | None, params: dict, figures: list[str]) -> None:
    header = generate_report_header(
        title="Spatial Multi-sample Integration Report",
        skill_name=SKILL_NAME,
        input_files=[Path(input_file)] if input_file else None,
        extra_metadata={"Method": summary["method"], "Batch key": params.get("batch_key", "sample")},
    )

    lines = [
        "## Summary\n",
        f"- **Cells**: {summary['n_cells']}",
        f"- **Genes**: {summary['n_genes']}",
        f"- **Batches**: {summary['n_batches']}",
        f"- **Method**: {summary['method']}",
        f"- **Embedding**: `{summary['embedding_key']}`",
        "",
        "### Batch Mixing\n",
        f"- **Before integration**: {summary['batch_mixing_before']:.4f}",
        f"- **After integration**: {summary['batch_mixing_after']:.4f}",
        "",
        "### Batch Sizes\n",
        "| Batch | Cells |",
        "|-------|-------|",
    ]
    for b, n in summary["batch_sizes"].items():
        lines.append(f"| {b} | {n} |")

    lines += ["", "### Figures\n"]
    for fig in figures:
        lines.append(f"- {fig}")

    lines += ["", "## Parameters\n"]
    for k, v in params.items():
        lines.append(f"- `{k}`: {v}")

    footer = generate_report_footer()
    (output_dir / "report.md").write_text(header + "\n".join(lines) + "\n" + footer)

    checksum = sha256_file(input_file) if input_file and Path(input_file).exists() else ""
    write_result_json(
        output_dir,
        skill=SKILL_NAME,
        version=SKILL_VERSION,
        summary={**summary, "figures": figures},
        data={"params": params, **summary, "figures": figures},
        input_checksum=checksum,
    )

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {"metric": "batch_mixing_before", "value": summary["batch_mixing_before"]},
            {"metric": "batch_mixing_after", "value": summary["batch_mixing_after"]},
            {"metric": "n_batches", "value": summary["n_batches"]},
            {"metric": "method", "value": summary["method"]},
        ]
    ).to_csv(tables_dir / "integration_metrics.csv", index=False)

    repro = output_dir / "reproducibility"
    repro.mkdir(exist_ok=True)
    cmd = (
        "python spatial_multi_sample_integration.py "
        f"--input <input.h5ad> --output {output_dir} "
        f"--method {params['method']} --batch-key {params['batch_key']}"
    )
    (repro / "commands.sh").write_text(f"#!/bin/bash\n{cmd}\n")
    (repro / "environment.txt").write_text("scanpy\nnumpy\npandas\nmatplotlib\nSTAligner (optional)\nharmonypy (optional)\nbbknn (optional)\nscanorama (optional)\n")


def get_demo_data():
    """Create demo multi-sample data by reusing spatial-preprocessing --demo."""
    import scanpy as sc

    preprocess_script = _PROJECT_ROOT / "skills" / "spatial" / "spatial-preprocessing" / "spatial_preprocessing.py"
    if not preprocess_script.exists():
        raise FileNotFoundError(f"Cannot find demo source script: {preprocess_script}")

    with tempfile.TemporaryDirectory(prefix="spatial_multi_int_demo_") as tmp:
        tmp_path = Path(tmp)
        cmd = [sys.executable, str(preprocess_script), "--demo", "--output", str(tmp_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if result.returncode != 0:
            raise RuntimeError(f"spatial-preprocessing --demo failed:\n{result.stderr}")

        processed = tmp_path / "processed.h5ad"
        if not processed.exists():
            raise FileNotFoundError(f"Expected demo file not found: {processed}")
        adata = sc.read_h5ad(processed)

    rng = np.random.default_rng(42)
    adata.obs["sample"] = rng.choice(["sample_A", "sample_B", "sample_C"], size=adata.n_obs)
    adata.obs["sample"] = pd.Categorical(adata.obs["sample"])
    return adata, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial Multi-sample Integration")
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output", dest="output_dir", required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default="auto")
    parser.add_argument("--batch-key", default="sample")
    args = parser.parse_args()

    try:
        _require_scanpy()
        import scanpy as sc
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        adata, input_file = get_demo_data()
    elif args.input_path:
        input_path = Path(args.input_path)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        adata = sc.read_h5ad(input_path)
        input_file = str(input_path)
    else:
        print("ERROR: Provide --input or --demo", file=sys.stderr)
        sys.exit(1)

    params = {"method": args.method, "batch_key": args.batch_key}
    summary = run_multi_sample_integration(adata, method=args.method, batch_key=args.batch_key)
    figures = generate_figures(adata, output_dir, summary, batch_key=args.batch_key)
    write_report(output_dir, summary, input_file, params, figures)

    _store_analysis_metadata(adata, method=summary["method"], params=params)
    adata.write_h5ad(output_dir / "processed.h5ad")

    logger.info("Saved outputs to %s", output_dir)
    logger.info("Integration complete: method=%s, batches=%d", summary["method"], summary["n_batches"])


if __name__ == "__main__":
    main()