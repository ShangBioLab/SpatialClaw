#!/usr/bin/env python3
"""
Spatial Transcriptomics Cell-Type Deconvolution
================================================
Supported methods:
  - tangram      : Tangram (default) — deep-learning cell-to-space mapping
  - stereoscope  : Stereoscope — NB probabilistic model via scvi-tools
  - graphst      : GraphST-guided — graph-attention embedding + NNLS deconvolution

Usage (CLI):
    python spatial_deconvolution.py \
        --input  <st_data.h5ad> \
        --reference <sc_ref.h5ad> \
        --cell-type-key <column> \
        --method tangram|stereoscope|graphst \
        --output <output_dir> \
        [--no-gpu] [--n-epochs N]
"""

from __future__ import annotations

import os
import json
import argparse
import sys
import traceback
from pathlib import Path

import anndata as ad
import scanpy as sc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spatialclaw.common.checksums import sha256_file
from spatialclaw.common.report import (
    generate_report_footer,
    generate_report_header,
    write_result_json,
)
from skills.spatial._lib.adata_utils import store_analysis_metadata


SKILL_NAME = "spatial-deconvolution"
SKILL_VERSION = "0.3.0"


# ─────────────────────────────────────────────────────────────────────────────
# Tangram
# ─────────────────────────────────────────────────────────────────────────────

def run_tangram_deconvolution(
    ad_sp: ad.AnnData,
    ad_sc: ad.AnnData,
    cell_type_key: str,
    device: str = "cpu",
    n_epochs: int = 1000,
) -> ad.AnnData:
    """Tangram cluster-mode deconvolution."""
    import torch
    import tangram as tg

    if "cuda" in device and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        device = "cpu"

    ad_sc.var_names_make_unique()
    ad_sp.var_names_make_unique()

    tg.pp_adatas(ad_sc, ad_sp, genes=None)

    if len(ad_sc.uns.get("training_genes", [])) == 0:
        raise ValueError(
            "Zero overlapping genes found! "
            "Check that both datasets use the same gene identifier type (Gene Symbols)."
        )

    ad_map = tg.map_cells_to_space(
        ad_sc,
        ad_sp,
        mode="clusters",
        cluster_label=cell_type_key,
        device=device,
        num_epochs=n_epochs,
    )
    tg.project_cell_annotations(ad_map, ad_sp, annotation=cell_type_key)

    # Normalise column names (remove "/" which h5ad can't store)
    if "tangram_ct_pred" in ad_sp.obsm:
        df = ad_sp.obsm["tangram_ct_pred"]
        df.columns = df.columns.str.replace("/", "_", regex=False)
        ad_sp.obsm["tangram_ct_pred"] = df
        ad_sp.obsm["deconvolution_ct_pred"] = df.copy()

    return ad_sp


# ─────────────────────────────────────────────────────────────────────────────
# Stereoscope
# ─────────────────────────────────────────────────────────────────────────────

def run_stereoscope_deconvolution(
    ad_sp: ad.AnnData,
    ad_sc: ad.AnnData,
    cell_type_key: str,
    device: str = "cpu",
    n_epochs: int = 200,
) -> ad.AnnData:
    """
    Stereoscope deconvolution via scvi-tools.

    Requires:
        pip install scvi-tools

    Algorithm:
        1. Train RNAStereoscope on the scRNA-seq reference (learns NB parameters
           per cell type and gene).
        2. Train SpatialStereoscope on ST data conditioned on the RNA model
           (learns spot-level cell-type proportions).
        3. Proportions are stored in ad_sp.obsm['stereoscope_ct_pred'].

    Notes:
        - Both AnnData objects must carry RAW integer counts in `.X` (not log-
          normalised). If only ad_sc has raw counts in `.raw`, they are extracted
          automatically.
        - Only genes present in both datasets are used.
    """
    
    try:
        import scvi
        from scvi.external import RNAStereoscope, SpatialStereoscope
    except ImportError as exc:
        raise ImportError(
            "scvi-tools is required for Stereoscope. "
            "Install it with:  pip install scvi-tools"
        ) from exc

    import torch
    if "cuda" in device and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
    def _cuda_really_works():
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            torch.zeros(1).cuda()  # 实际触发 CUDA 初始化
            return True
        except Exception:
            return False    
    # 完全屏蔽损坏的 CUDA driver，让 lightning 看不到 GPU
    if not _cuda_really_works():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    use_gpu = "cuda" in device and _cuda_really_works()
    if not use_gpu and "cuda" in device:
        print("Warning: CUDA not functional (driver too old), falling back to CPU.")

    ad_sc = ad_sc.copy()
    ad_sp = ad_sp.copy()

    # ── Ensure raw counts in .X ──────────────────────────────────────────────
    for name, adata in [("scRNA", ad_sc), ("ST", ad_sp)]:
        if adata.raw is not None:
            import numpy as np
            import scipy.sparse as sp
            raw_X = adata.raw.X
            if sp.issparse(raw_X):
                raw_X = raw_X.toarray()
            if np.any(raw_X != raw_X.astype(int)):
                print(
                    f"Warning: {name} .raw.X contains non-integer values; "
                    "using current .X as counts."
                )
            else:
                adata.X = adata.raw.X
                adata.var = adata.raw.var.copy()
                print(f"  [{name}] Using raw counts from .raw.X")

    ad_sc.var_names_make_unique()
    ad_sp.var_names_make_unique()

    # ── Intersect genes ───────────────────────────────────────────────────────
    common_genes = list(set(ad_sc.var_names) & set(ad_sp.var_names))
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between scRNA reference and ST data.")
    print(f"  Overlapping genes for Stereoscope: {len(common_genes)}")

    ad_sc_sub = ad_sc[:, common_genes].copy()
    ad_sp_sub = ad_sp[:, common_genes].copy()

    # ── Train scRNA model ─────────────────────────────────────────────────────
    print("  Training RNAStereoscope on scRNA reference ...")
    RNAStereoscope.setup_anndata(ad_sc_sub, labels_key=cell_type_key)
    sc_model = RNAStereoscope(ad_sc_sub)
    sc_model.train(
        max_epochs=n_epochs,
        accelerator="gpu" if use_gpu else "cpu",   # ← 换成这行
        plan_kwargs={"lr": 1e-3},
    )

    # ── Train spatial model ───────────────────────────────────────────────────
    print("  Training SpatialStereoscope on ST data ...")
    SpatialStereoscope.setup_anndata(ad_sp_sub)
    st_model = SpatialStereoscope.from_rna_model(
        ad_sp_sub,
        sc_model,
        prior_weight="n_obs",
    )
    st_model.train(
        max_epochs=n_epochs,
        accelerator="gpu" if use_gpu else "cpu",   # ← 换成这行
        plan_kwargs={"lr": 1e-3},
    )

    # ── Extract proportions ───────────────────────────────────────────────────
    proportions = st_model.get_proportions()
    proportions.columns = proportions.columns.str.replace("/", "_", regex=False)

    # Align index with original ad_sp
    ad_sp.obsm["stereoscope_ct_pred"] = proportions.loc[ad_sp_sub.obs_names].values
    import pandas as pd
    ad_sp.obsm["stereoscope_ct_pred"] = pd.DataFrame(
        proportions.loc[ad_sp_sub.obs_names].values,
        index=ad_sp.obs_names,
        columns=proportions.columns,
    )
    ad_sp.obsm["deconvolution_ct_pred"] = ad_sp.obsm["stereoscope_ct_pred"].copy()

    print(
        f"  Stereoscope complete. "
        f"Cell types: {list(proportions.columns)}"
    )
    return ad_sp


# ─────────────────────────────────────────────────────────────────────────────
# GraphST-guided deconvolution
# ─────────────────────────────────────────────────────────────────────────────

def run_graphst_deconvolution(
    ad_sp: ad.AnnData,
    ad_sc: ad.AnnData,
    cell_type_key: str,
    device: str = "cpu",
    n_epochs: int = 1000,
    n_hvg: int = 3000,
) -> ad.AnnData:
    """
    GraphST-guided spatial deconvolution.

    Requires:
        pip install GraphST

    Algorithm:
        1. Run GraphST on the ST data to learn a spatially-smoothed latent
           embedding (graph attention network over the spatial neighbour graph).
        2. Build a cell-type signature matrix from the scRNA reference
           (mean log-normalised expression per cell type, restricted to HVGs
           shared with the ST data).
        3. Apply Non-Negative Least Squares (NNLS) in the signature space to
           estimate cell-type proportions for each spot.
        4. Proportions are stored in ad_sp.obsm['graphst_ct_pred'].

    Why GraphST helps deconvolution:
        GraphST's message-passing step borrows information from neighbouring
        spots, effectively denoising sparse ST profiles before the linear
        decomposition step — this improves proportion estimates in low-count
        spots compared to vanilla NNLS.
    """
    try:
        from GraphST.GraphST import GraphST
    except ImportError as exc:
        raise ImportError(
            "GraphST is required for the graphst method. "
            "Install it with:  pip install GraphST"
        ) from exc

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.optimize import nnls
    import torch

    if "cuda" in device and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        device = "cpu"

    ad_sc = ad_sc.copy()
    ad_sp = ad_sp.copy()
    ad_sc.var_names_make_unique()
    ad_sp.var_names_make_unique()

    # ── Step 1: GraphST on spatial data ──────────────────────────────────────
    print("  Running GraphST to learn spatial embeddings ...")

    sp_for_gst = ad_sp.copy()

    # Only normalise/log if data looks like raw counts (max value suggests counts)
    import numpy as np_check
    import scipy.sparse as sp_check
    x_sample = sp_for_gst.X[:100]
    if sp_check.issparse(x_sample):
        x_sample = x_sample.toarray()
    _is_log = np_check.max(x_sample) < 50  # log-normalised data rarely exceeds 50
    if not _is_log:
        sc.pp.normalize_total(sp_for_gst, target_sum=1e4)
        sc.pp.log1p(sp_for_gst)
        print("  Normalised + log1p applied to ST data.")
    else:
        print("  ST data appears already log-normalised, skipping normalisation.")

    # Spatial graph (use spatial coords if available, else PCA)
    if "spatial" in ad_sp.obsm:
        import pandas as pd_sp
        sc.pp.pca(sp_for_gst, n_comps=min(50, sp_for_gst.n_vars - 1))
        sc.pp.neighbors(sp_for_gst, n_neighbors=6)
    else:
        sc.pp.pca(sp_for_gst, n_comps=min(50, sp_for_gst.n_vars - 1))
        sc.pp.neighbors(sp_for_gst)

    model = GraphST(sp_for_gst, device=device)

    # GraphST.train() API varies by version — try known parameter names
    try:
        import inspect
        sig = inspect.signature(model.train)
        train_params = set(sig.parameters.keys())
        if "n_epochs" in train_params:
            sp_for_gst = model.train(n_epochs=n_epochs)
        elif "num_epochs" in train_params:
            sp_for_gst = model.train(num_epochs=n_epochs)
        else:
            sp_for_gst = model.train()
    except Exception:
        sp_for_gst = model.train()

    # The latent representation is in sp_for_gst.obsm['emb']
    emb_key = "emb" if "emb" in sp_for_gst.obsm else "X_pca"
    latent = sp_for_gst.obsm[emb_key]   # (n_spots, latent_dim)
    print(f"  GraphST embedding shape: {latent.shape}")

    # ── Step 2: Build signature matrix from scRNA ─────────────────────────────
    print("  Building cell-type signature matrix ...")

    sc_proc = ad_sc.copy()
    sc.pp.normalize_total(sc_proc, target_sum=1e4)
    sc.pp.log1p(sc_proc)

    common_genes = list(set(sc_proc.var_names) & set(ad_sp.var_names))
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between scRNA reference and ST data.")

    # Select HVGs from the intersection
    sc_sub = sc_proc[:, common_genes].copy()
    sp_sub = ad_sp[:, common_genes].copy()
    sc.pp.normalize_total(sp_sub, target_sum=1e4)
    sc.pp.log1p(sp_sub)

    # HVG selection (top n_hvg shared genes)
    try:
        sc.pp.highly_variable_genes(sc_sub, n_top_genes=min(n_hvg, len(common_genes)))
        hvg_mask = sc_sub.var["highly_variable"]
        hvg_genes = sc_sub.var_names[hvg_mask].tolist()
    except Exception:
        hvg_genes = common_genes  # fallback: all common genes

    print(f"  Using {len(hvg_genes)} HVGs for deconvolution")

    sc_hvg = sc_sub[:, hvg_genes]
    sp_hvg = sp_sub[:, hvg_genes]

    # Mean expression per cell type  →  signature matrix  (n_hvg × n_celltypes)
    cell_types = sorted(ad_sc.obs[cell_type_key].unique())
    sig_matrix = np.zeros((len(hvg_genes), len(cell_types)))
    for j, ct in enumerate(cell_types):
        mask = ad_sc.obs[cell_type_key] == ct
        expr = sc_hvg[mask].X
        if sp.issparse(expr):
            expr = expr.toarray()
        sig_matrix[:, j] = expr.mean(axis=0)

    # ── Step 3: NNLS deconvolution on (GraphST-smoothed) spot expression ──────
    print("  Running NNLS deconvolution on GraphST-smoothed profiles ...")

    # Re-project GraphST latent → gene-space via decoder (if available)
    # Fallback: use PCA-smoothed expression from the latent embedding
    if hasattr(model, "decoder") or "X_recon" in sp_for_gst.obsm:
        recon_key = "X_recon" if "X_recon" in sp_for_gst.obsm else None
        if recon_key:
            recon = sp_for_gst.obsm[recon_key]  # reconstructed expression
            recon_sp = ad.AnnData(X=recon, obs=ad_sp.obs, var=sp_for_gst.var)
            recon_sp = recon_sp[:, hvg_genes] if all(g in recon_sp.var_names for g in hvg_genes) else None
        else:
            recon_sp = None
    else:
        recon_sp = None

    if recon_sp is not None:
        spot_matrix = recon_sp.X
        if sp.issparse(spot_matrix):
            spot_matrix = spot_matrix.toarray()
        print("  Using GraphST reconstructed expression for deconvolution.")
    else:
        # Fallback: use raw log-normalised expression
        spot_matrix = sp_hvg.X
        if sp.issparse(spot_matrix):
            spot_matrix = spot_matrix.toarray()
        print(
            "  Note: GraphST decoder output not available; "
            "using log-normalised expression with GraphST embedding stored."
        )

    # NNLS per spot
    n_spots = spot_matrix.shape[0]
    proportions = np.zeros((n_spots, len(cell_types)))
    for i in range(n_spots):
        coef, _ = nnls(sig_matrix, spot_matrix[i])
        total = coef.sum()
        proportions[i] = coef / total if total > 0 else coef

    prop_df = pd.DataFrame(proportions, index=ad_sp.obs_names, columns=cell_types)
    prop_df.columns = prop_df.columns.str.replace("/", "_", regex=False)

    ad_sp.obsm["graphst_ct_pred"] = prop_df
    ad_sp.obsm["deconvolution_ct_pred"] = prop_df.copy()

    # Also store the GraphST latent embedding for downstream use
    ad_sp.obsm["X_graphst"] = latent

    print(
        f"  GraphST deconvolution complete. "
        f"Cell types: {list(prop_df.columns)}"
    )
    return ad_sp


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

METHOD_DEFAULTS = {
    "tangram":      {"n_epochs": 1000,   "obsm_key": "tangram_ct_pred"},
    "stereoscope":  {"n_epochs": 200, "obsm_key": "stereoscope_ct_pred"},
    "graphst":      {"n_epochs": 1000,   "obsm_key": "graphst_ct_pred"},
}

SUPPORTED_METHODS = list(METHOD_DEFAULTS.keys())


def spatial_deconvolution(
    spatial_path: str,
    sc_ref_path: str,
    output_path: str,
    cell_type_key: str,
    method: str = "tangram",
    device: str = "cpu",
    n_epochs: int | None = None,
) -> dict:
    """
    Primary callable interface for the skill (used by AI agent and CLI).

    Parameters
    ----------
    spatial_path  : Path to spatial transcriptomics .h5ad
    sc_ref_path   : Path to single-cell reference .h5ad
    output_path   : Output .h5ad file path
    cell_type_key : Column in sc_ref obs containing cell type labels
    method        : One of 'tangram', 'stereoscope', 'graphst'
    device        : 'cpu', 'cuda', or 'cuda:0'
    n_epochs      : Override default training epochs (None = use method default)
    """
    method = method.lower().strip()
    if method not in SUPPORTED_METHODS:
        return {
            "status": "error",
            "message": (
                f"Unknown method '{method}'. "
                f"Supported: {', '.join(SUPPORTED_METHODS)}"
            ),
        }

    if not os.path.exists(spatial_path):
        return {"status": "error", "message": f"Spatial data not found: {spatial_path}"}
    if not os.path.exists(sc_ref_path):
        return {"status": "error", "message": f"scRNA reference not found: {sc_ref_path}"}

    try:
        print(f"Loading data ...")
        ad_sp = sc.read_h5ad(spatial_path).copy()
        ad_sc = sc.read_h5ad(sc_ref_path).copy()

        if cell_type_key not in ad_sc.obs.columns:
            return {
                "status": "error",
                "message": (
                    f"Column '{cell_type_key}' not found in scRNA .obs. "
                    f"Available columns: {list(ad_sc.obs.columns)}"
                ),
            }

        epochs = n_epochs if n_epochs is not None else METHOD_DEFAULTS[method]["n_epochs"]
        print(f"Running {method} deconvolution (n_epochs={epochs}, device={device}) ...")

        if method == "tangram":
            ad_sp = run_tangram_deconvolution(ad_sp, ad_sc, cell_type_key, device=device, n_epochs=epochs)
        elif method == "stereoscope":
            ad_sp = run_stereoscope_deconvolution(ad_sp, ad_sc, cell_type_key, device=device, n_epochs=epochs)
        elif method == "graphst":
            ad_sp = run_graphst_deconvolution(ad_sp, ad_sc, cell_type_key, device=device, n_epochs=epochs)

        store_analysis_metadata(
            ad_sp,
            SKILL_NAME,
            method,
            params={
                "method": method,
                "reference": sc_ref_path,
                "cell_type_key": cell_type_key,
                "device": device,
                "n_epochs": epochs,
            },
        )

        # Save
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ad_sp.write_h5ad(output_path)

        obsm_key = METHOD_DEFAULTS[method]["obsm_key"]
        n_celltypes = 0
        cell_types = []
        if obsm_key in ad_sp.obsm:
            import pandas as pd
            obj = ad_sp.obsm[obsm_key]
            if hasattr(obj, "columns"):
                cell_types = list(obj.columns)
                n_celltypes = len(cell_types)
            elif hasattr(obj, "shape"):
                n_celltypes = obj.shape[1]

        return {
            "status": "success",
            "method": method,
            "n_spots": int(ad_sp.n_obs),
            "n_genes": int(ad_sp.n_vars),
            "n_reference_cells": int(ad_sc.n_obs),
            "n_reference_genes": int(ad_sc.n_vars),
            "n_celltypes": int(n_celltypes),
            "cell_types": cell_types,
            "obsm_key": obsm_key,
            "output_path": str(output_path),
            "message": (
                f"{method} deconvolution completed. "
                f"Result saved to {output_path}. "
                f"Cell type proportions stored in .obsm['{obsm_key}'] and .obsm['deconvolution_ct_pred']. "
                f"Detected {n_celltypes} cell types: {cell_types}."
            ),
        }

    except Exception as exc:
        traceback.print_exc()
        return {
            "status": "error",
            "method": method,
            "message": f"Error during {method} deconvolution: {exc}",
        }


def write_report(
    output_dir: Path,
    result: dict,
    *,
    input_file: str,
    reference_file: str,
    params: dict,
) -> None:
    """Write standard SpatialClaw report.md and result.json files."""
    summary = {
        "status": result.get("status", "error"),
        "method": result.get("method", params.get("method", "unknown")),
        "n_spots": result.get("n_spots", 0),
        "n_genes": result.get("n_genes", 0),
        "n_reference_cells": result.get("n_reference_cells", 0),
        "n_reference_genes": result.get("n_reference_genes", 0),
        "n_celltypes": result.get("n_celltypes", 0),
        "cell_types": result.get("cell_types", []),
        "obsm_key": result.get("obsm_key", ""),
        "output_path": result.get("output_path", str(output_dir / "processed.h5ad")),
    }

    header = generate_report_header(
        title="Spatial Deconvolution Report",
        skill_name=SKILL_NAME,
        input_files=[Path(input_file), Path(reference_file)],
        extra_metadata={
            "Method": str(summary["method"]),
            "Status": str(summary["status"]),
            "Cell types": str(summary["n_celltypes"]),
        },
    )

    body_lines = [
        "## Summary\n",
        f"- **Status**: {summary['status']}",
        f"- **Method**: {summary['method']}",
        f"- **Spatial spots**: {summary['n_spots']}",
        f"- **Spatial genes**: {summary['n_genes']}",
        f"- **Reference cells**: {summary['n_reference_cells']}",
        f"- **Reference genes**: {summary['n_reference_genes']}",
        f"- **Cell types detected**: {summary['n_celltypes']}",
        f"- **Output h5ad**: `{Path(str(summary['output_path'])).name}`",
        "",
        "### Output Keys\n",
        f"- Method-specific proportions: `{summary['obsm_key'] or '<method>_ct_pred'}`",
        "- Unified proportions: `deconvolution_ct_pred`",
    ]

    cell_types = summary.get("cell_types") or []
    if cell_types:
        body_lines.extend(["", "### Cell Types\n"])
        for cell_type in cell_types:
            body_lines.append(f"- {cell_type}")

    if result.get("message"):
        body_lines.extend(["", "### Execution Message\n", result["message"]])

    body_lines.extend(["", "## Parameters\n"])
    for key, value in params.items():
        body_lines.append(f"- `{key}`: {value}")

    report = header + "\n".join(body_lines) + "\n" + generate_report_footer()
    (output_dir / "report.md").write_text(report)

    checksum = sha256_file(input_file) if input_file and Path(input_file).exists() else ""
    write_result_json(
        output_dir,
        skill=SKILL_NAME,
        version=SKILL_VERSION,
        summary=summary,
        data={"params": params, "result": result},
        input_checksum=checksum,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial Transcriptomics Cell-Type Deconvolution"
    )
    parser.add_argument("--input",         help="Path to spatial .h5ad")
    parser.add_argument("--reference",     help="Path to scRNA reference .h5ad")
    parser.add_argument("--cell-type-key", dest="cell_type_key",
                        help="Cell type column in reference .obs")
    parser.add_argument("--method",        default="tangram",
                        choices=SUPPORTED_METHODS,
                        help="Deconvolution method (default: tangram)")
    parser.add_argument("--output",        default="output",
                        help="Output directory or .h5ad path")
    parser.add_argument("--n-epochs",      dest="n_epochs", type=int, default=None,
                        help="Training epochs (overrides method default)")
    parser.add_argument("--no-gpu",        action="store_true", help="Force CPU")
    args = parser.parse_args()

    missing = [
        flag for flag, value in (
            ("--input", args.input),
            ("--reference", args.reference),
            ("--cell-type-key", args.cell_type_key),
        )
        if not value
    ]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    out_path = args.output
    if not out_path.endswith(".h5ad"):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "processed.h5ad")

    device = "cpu" if args.no_gpu else "cuda:0"

    result = spatial_deconvolution(
        spatial_path=args.input,
        sc_ref_path=args.reference,
        output_path=out_path,
        cell_type_key=args.cell_type_key,
        method=args.method,
        device=device,
        n_epochs=args.n_epochs,
    )

    params = {
        "method": args.method,
        "reference": args.reference,
        "cell_type_key": args.cell_type_key,
        "n_epochs": args.n_epochs,
        "device": device,
        "no_gpu": args.no_gpu,
    }
    write_report(
        Path(out_path).parent,
        result,
        input_file=args.input,
        reference_file=args.reference,
        params=params,
    )

    print(json.dumps(result, indent=2))
    if result.get("status") != "success":
        sys.exit(1)
