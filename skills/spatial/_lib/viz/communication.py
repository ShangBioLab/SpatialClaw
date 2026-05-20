"""Cell-cell communication visualization for SpatialClaw.

Sub-types:
- ``dotplot`` — L-R pair significance dot plot (liana style, default)
- ``heatmap`` — sender × receiver interaction strength heatmap
- ``chord`` — sender ↔ receiver communication ribbons / arcs
- ``lr_ranked`` — top ligand-receptor interactions with explicit labels
- ``spatial`` — spatial L-R score maps on tissue

Requires ``liana`` for ``dotplot``/``heatmap``; otherwise falls back to
custom matplotlib implementations.

Adapted from ChatSpatial visualization/cell_comm.py — removed async/ToolContext.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from .core import (
    VizParams,
    _require_spatial_coords,
    auto_spot_size,
    get_categorical_cmap,
    get_category_colors,
    setup_multi_panel_figure,
)

logger = logging.getLogger(__name__)


def _format_group_label(value: Any) -> str:
    """Render communication groups as human-readable domain labels.

    Numeric labels such as ``0`` or ``3`` are displayed as ``Domain 0`` and
    ``Domain 3`` so users do not confuse them with LR indices.
    """
    text = str(value)
    try:
        number = float(text)
    except Exception:
        return text
    if number.is_integer():
        return f"Domain {int(number)}"
    return text


# ---------------------------------------------------------------------------
# Data retrieval helpers
# ---------------------------------------------------------------------------


def _find_ccc_results(adata: Any) -> tuple[pd.DataFrame, str]:
    """Auto-detect CCC results in adata.uns and return (df, method)."""
    # Keys written by spatial-cell-communication skill
    candidates = [
        ("builtin_results", "builtin"),
        ("liana_res", "liana"),
        ("cellphonedb_results", "cellphonedb"),
        ("fastccc_results", "fastccc"),
        ("ccc_results", str(adata.uns.get("ccc_method", "communication"))),
    ]
    for key, method in candidates:
        if key in adata.uns:
            val = adata.uns[key]
            if isinstance(val, pd.DataFrame) and len(val) > 0:
                return val, method
    raise ValueError(
        "No cell-communication results found. Run spatial-cell-communication first.\n"
        "Expected one of: ccc_results, liana_res, cellphonedb_results, "
        "fastccc_results in adata.uns."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_communication(
    adata: Any,
    params: Optional[VizParams] = None,
    *,
    subtype: Optional[str] = None,
    top_n: int = 20,
    min_cells: int = 5,
) -> plt.Figure:
    """Cell-cell communication visualization.

    Args:
        adata: AnnData object with CCC results.
        params: :class:`~skills.spatial._lib.viz.params.VizParams`.
        subtype: ``"dotplot"`` (default), ``"heatmap"``, or ``"spatial"``.
        top_n: Number of top LR pairs to show.
        min_cells: Minimum cells per group for filtering.

    Returns:
        :class:`matplotlib.figure.Figure`
    """
    if params is None:
        params = VizParams()
    if subtype:
        params.subtype = subtype

    st = params.subtype or "dotplot"
    logger.info("Creating CCC visualization: %s", st)

    results, method = _find_ccc_results(adata)
    logger.info("Using CCC results from method: %s (%d rows)", method, len(results))

    dispatch = {
        "dotplot": _plot_dotplot,
        "heatmap": _plot_heatmap,
        "chord": _plot_chord,
        "lr_ranked": _plot_lr_ranked,
        "spatial": _plot_spatial,
    }
    if st not in dispatch:
        raise ValueError(
            f"Unknown communication subtype '{st}'. "
            "Choose: 'dotplot', 'heatmap', 'chord', 'lr_ranked', or 'spatial'."
        )
    return dispatch[st](adata, params, results, method, top_n=top_n)


# ---------------------------------------------------------------------------
# Dotplot  (liana-style or custom)
# ---------------------------------------------------------------------------


def _plot_dotplot(
    adata: Any, params: VizParams,
    results: pd.DataFrame, method: str, *, top_n: int = 20,
) -> plt.Figure:
    """L-R pair dotplot: dot size = -log10(p-value), colour = mean expression."""
    try:
        import liana as li
        if hasattr(li, "pl") and hasattr(li.pl, "dotplot"):
            li.pl.dotplot(
                adata=adata, colour="lr_means", size="cellphone_pvals",
                inverse_size=True, top_n=top_n,
                figure_size=params.figure_size or (10, 8),
            )
            fig = plt.gcf()
            if params.title:
                fig.suptitle(params.title)
            return fig
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("LIANA dotplot failed (%s); falling back to custom impl", exc)

    # ---- Custom matplotlib implementation ----
    return _custom_lr_dotplot(adata, params, results, method, top_n=top_n)


def _custom_lr_dotplot(
    adata: Any, params: VizParams,
    results: pd.DataFrame, method: str, *, top_n: int,
) -> plt.Figure:
    """Custom matplotlib LR dotplot when liana is unavailable."""
    # Detect LR pair column
    lr_col = next(
        (c for c in ("ligand_receptor", "lr_pair", "interaction_name") if c in results.columns),
        None,
    )
    if lr_col is None and "ligand" in results.columns and "receptor" in results.columns:
        results = results.copy()
        results["lr_pair"] = results["ligand"] + "_" + results["receptor"]
        lr_col = "lr_pair"

    if lr_col is None:
        raise ValueError(f"Cannot identify LR pair column in results: {results.columns.tolist()}")

    # Score column
    score_col = next(
        (c for c in ("lr_means", "magnitude_rank", "specificity_rank", "mean")
         if c in results.columns),
        results.columns[2] if len(results.columns) > 2 else None,
    )
    # P-value column
    pval_col = next(
        (c for c in ("cellphone_pvals", "pvalue", "pval", "p_val") if c in results.columns),
        None,
    )

    df = results.copy()
    if score_col:
        df["_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    else:
        df["_score"] = 1.0

    top_pairs = (
        df.groupby(lr_col)["_score"].mean()
        .nlargest(top_n).index.tolist()
    )
    df = df[df[lr_col].isin(top_pairs)]

    # Pivot for heatmap-style dotplot
    source_col = next(
        (c for c in ("source", "sender", "cell_type_1") if c in df.columns), None
    )
    target_col = next(
        (c for c in ("target", "receiver", "cell_type_2") if c in df.columns), None
    )

    if source_col is None or target_col is None:
        # Fall back to simple bar chart
        return _lr_barplot(df, lr_col, "_score", params, method)

    pivot = df.pivot_table(
        index=lr_col, columns=source_col, values="_score", aggfunc="mean"
    ).fillna(0)
    pivot = pivot.loc[top_pairs[:min(top_n, len(pivot))]]
    pivot.columns = [_format_group_label(col) for col in pivot.columns]

    n_pairs, n_sources = pivot.shape
    figsize = params.figure_size or (max(6, n_sources * 1.2), max(6, n_pairs * 0.35))
    fig, ax = plt.subplots(figsize=figsize, dpi=params.dpi)

    cmap = plt.cm.get_cmap(params.colormap or "RdBu_r")
    norm = plt.Normalize(vmin=pivot.values.min(), vmax=pivot.values.max())

    for i, pair in enumerate(pivot.index):
        for j, source in enumerate(pivot.columns):
            val = pivot.loc[pair, source]
            color = cmap(norm(val))
            size = max(20, val * 200)
            ax.scatter(j, i, c=[color], s=size, alpha=0.85)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index), fontsize=8)
    ax.set_title(params.title or f"LR Pair Dotplot ({method})", fontsize=13)
    ax.set_xlabel("Source cell type")
    ax.set_ylabel("LR Pair")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=score_col or "Score")
    plt.tight_layout()
    return fig


def _lr_barplot(
    df: pd.DataFrame, lr_col: str, score_col: str,
    params: VizParams, method: str,
) -> plt.Figure:
    top = df.groupby(lr_col)[score_col].mean().nlargest(20)
    figsize = params.figure_size or (8, max(4, len(top) * 0.35))
    fig, ax = plt.subplots(figsize=figsize, dpi=params.dpi)
    ax.barh(range(len(top)), top.values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(list(top.index), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean score")
    ax.set_title(params.title or f"Top LR Pairs ({method})", fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Heatmap (sender × receiver)
# ---------------------------------------------------------------------------


def _plot_heatmap(
    adata: Any, params: VizParams,
    results: pd.DataFrame, method: str, *, top_n: int = 20,
) -> plt.Figure:
    """Interaction strength heatmap: sender × receiver."""
    source_col = next(
        (c for c in ("source", "sender", "cell_type_1") if c in results.columns), None
    )
    target_col = next(
        (c for c in ("target", "receiver", "cell_type_2") if c in results.columns), None
    )
    if source_col is None or target_col is None:
        raise ValueError(
            "sender/receiver columns not found in CCC results. "
            f"Available columns: {results.columns.tolist()}"
        )

    score_col = next(
        (c for c in ("lr_means", "magnitude_rank", "mean") if c in results.columns),
        results.columns[2] if len(results.columns) > 2 else None,
    )
    results = results.copy()
    results["_s"] = pd.to_numeric(results[score_col], errors="coerce").fillna(0)

    pivot = results.pivot_table(
        index=source_col, columns=target_col, values="_s", aggfunc="sum"
    ).fillna(0)
    pivot.index = [_format_group_label(idx) for idx in pivot.index]
    pivot.columns = [_format_group_label(col) for col in pivot.columns]

    import seaborn as sns

    figsize = params.figure_size or (
        max(6, len(pivot.columns) * 0.8),
        max(5, len(pivot.index) * 0.7),
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=params.dpi)
    sns.heatmap(
        pivot, cmap=params.colormap or "YlOrRd",
        ax=ax, linewidths=0.5,
        cbar_kws={"label": score_col or "Interaction strength"},
    )
    ax.set_title(
        params.title or f"Domain-to-Domain Communication Heatmap ({method})",
        fontsize=13,
    )
    ax.set_xlabel("Target (Receiver)")
    ax.set_ylabel("Source (Sender)")
    plt.tight_layout()
    return fig


def _plot_chord(
    adata: Any, params: VizParams,
    results: pd.DataFrame, method: str, *, top_n: int = 20,
) -> plt.Figure:
    """Communication chord-like arc diagram using source-target aggregated weights."""
    source_col = next(
        (c for c in ("source", "sender", "cell_type_1") if c in results.columns), None
    )
    target_col = next(
        (c for c in ("target", "receiver", "cell_type_2") if c in results.columns), None
    )
    if source_col is None or target_col is None:
        raise ValueError(
            "sender/receiver columns not found in CCC results. "
            f"Available columns: {results.columns.tolist()}"
        )

    score_col = next(
        (c for c in ("score", "lr_means", "magnitude_rank", "mean") if c in results.columns),
        results.columns[2] if len(results.columns) > 2 else None,
    )
    if score_col is None:
        raise ValueError("Could not determine a score column for communication chord plot.")

    agg = (
        results.assign(_score=pd.to_numeric(results[score_col], errors="coerce").fillna(0))
        .groupby([source_col, target_col], as_index=False)["_score"]
        .sum()
        .sort_values("_score", ascending=False)
        .head(max(top_n, 1))
    )
    labels = sorted(set(agg[source_col].astype(str)) | set(agg[target_col].astype(str)))
    if not labels:
        raise ValueError("No source-target communication pairs available for chord plot.")

    display_labels = {_label: _format_group_label(_label) for _label in labels}
    figsize = params.figure_size or (11, 11)
    fig, ax = plt.subplots(figsize=figsize, dpi=params.dpi)
    ax.set_title(
        params.title or f"Domain-to-Domain Communication Chord ({method})",
        fontsize=13,
        pad=20,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(labels), endpoint=False)
    position_map = {
        label: np.array([np.cos(angle), np.sin(angle)])
        for label, angle in zip(labels, angles)
    }
    colors = get_category_colors(len(labels), get_categorical_cmap(len(labels)))
    color_map = {label: colors[i] for i, label in enumerate(labels)}

    outer_circle = Circle((0.0, 0.0), radius=1.0, facecolor="none", edgecolor="#d8dee9", linewidth=1.0)
    ax.add_patch(outer_circle)
    for label in labels:
        x, y = position_map[label]
        ax.scatter([x], [y], s=220, color=color_map[label], edgecolors="white", linewidth=1.2, zorder=3)
        lx, ly = 1.16 * x, 1.16 * y
        ha = "left" if lx > 0.05 else "right" if lx < -0.05 else "center"
        ax.text(lx, ly, display_labels[label], ha=ha, va="center", fontsize=10, fontweight="semibold")

    max_score = float(agg["_score"].max()) if not agg.empty else 1.0
    for _, row in agg.iterrows():
        src = str(row[source_col])
        tgt = str(row[target_col])
        score = float(row["_score"])
        start = position_map[src]
        end = position_map[tgt]
        width = 1.2 + 8.0 * (score / max_score if max_score > 0 else 0.0)

        if src == tgt:
            loop_center = 1.08 * start
            loop = Circle(
                tuple(loop_center),
                radius=0.10,
                fill=False,
                linewidth=width * 0.45,
                edgecolor=color_map[src],
                alpha=0.55,
            )
            ax.add_patch(loop)
            continue

        curve = np.vstack([
            start,
            np.array([0.0, 0.0]),
            end,
        ])
        t = np.linspace(0.0, 1.0, 100)
        pts = (
            ((1 - t) ** 2)[:, None] * curve[0]
            + (2 * (1 - t) * t)[:, None] * curve[1]
            + (t ** 2)[:, None] * curve[2]
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            linewidth=width,
            alpha=0.35,
            color=color_map[src],
            solid_capstyle="round",
        )

    ax.text(
        0.0,
        -1.28,
        "Edge width encodes aggregated sender->receiver communication strength",
        ha="center",
        va="center",
        fontsize=9,
        color="#4c566a",
    )

    return fig


def _plot_lr_ranked(
    adata: Any, params: VizParams,
    results: pd.DataFrame, method: str, *, top_n: int = 20,
) -> plt.Figure:
    """Ranked barplot with explicit ligand-receptor and source-target labels."""
    df = results.copy()
    required = {"ligand", "receptor"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"LR-ranked plot requires ligand/receptor columns; got {df.columns.tolist()}"
        )

    source_col = next(
        (c for c in ("source", "sender", "cell_type_1") if c in df.columns), None
    )
    target_col = next(
        (c for c in ("target", "receiver", "cell_type_2") if c in df.columns), None
    )
    score_col = next(
        (c for c in ("score", "lr_means", "magnitude_rank", "mean") if c in df.columns),
        None,
    )
    if score_col is None:
        raise ValueError("Could not determine a score column for ranked LR plot.")

    df["_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    df["_label"] = (
        df["ligand"].astype(str)
        + " -> "
        + df["receptor"].astype(str)
        + "\n"
        + df[source_col].map(_format_group_label).astype(str)
        + " -> "
        + df[target_col].map(_format_group_label).astype(str)
    )
    top = df.nlargest(max(top_n, 1), "_score").iloc[::-1]

    figsize = params.figure_size or (12, max(6, 0.42 * len(top)))
    fig, ax = plt.subplots(figsize=figsize, dpi=params.dpi)
    bars = ax.barh(range(len(top)), top["_score"].values, color="steelblue", alpha=0.9)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["_label"].tolist(), fontsize=8)
    ax.set_xlabel(score_col)
    ax.set_title(params.title or f"Top Ligand-Receptor Interactions ({method})", fontsize=13)

    for bar, (_, row) in zip(bars, top.iterrows()):
        pvalue = row["pvalue"] if "pvalue" in row and pd.notna(row["pvalue"]) else None
        suffix = f" | p={pvalue:.2e}" if pvalue is not None else ""
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {row['_score']:.3f}{suffix}",
            va="center",
            fontsize=7,
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spatial LR score map
# ---------------------------------------------------------------------------


def _plot_spatial(
    adata: Any, params: VizParams,
    results: pd.DataFrame, method: str, *, top_n: int = 5,
) -> plt.Figure:
    """Spatial maps of LR pair scores (requires spatial scores in obsm)."""
    score_key = next(
        (k for k in adata.obsm.keys() if "ccc" in k.lower() or "spatial_scores" in k),
        None,
    )
    if score_key is None:
        raise ValueError(
            "Spatial CCC scores not found in adata.obsm. "
            "Run spatial-cell-communication with spatial=True first."
        )

    scores = np.asarray(adata.obsm[score_key])
    n_pairs = min(top_n, scores.shape[1])

    # Get pair names if stored
    pair_names_key = f"{score_key}_names"
    if pair_names_key in adata.uns:
        pair_names = list(adata.uns[pair_names_key])[:n_pairs]
    else:
        pair_names = [f"LR_{i}" for i in range(n_pairs)]

    coords = _require_spatial_coords(adata)
    spot_size = auto_spot_size(adata, params.spot_size, basis="spatial")

    fig, axes = setup_multi_panel_figure(
        n_panels=n_pairs,
        params=params,
        default_title=f"Spatial LR Scores ({method})",
    )

    for i in range(n_pairs):
        ax = axes[i]
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=scores[:, i], cmap=params.colormap or "viridis",
            s=spot_size, alpha=params.alpha,
        )
        if params.show_colorbar:
            plt.colorbar(sc, ax=ax)
        ax.set_title(pair_names[i], fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    plt.tight_layout()
    return fig
