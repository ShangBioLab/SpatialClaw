#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import seaborn as sns


ROOT = Path("/dugaoyuan/AgentAndClaw/Claws/SpatialClaw-pri-casestudy-worktree")
COMM_DIR = ROOT / "output/casestudy_smoke_gpt54/M1_communication"
LABEL_DIR = ROOT / "output/casestudy_repro_gpt54/M1/07_domain_labeled"
OUT_DIR = ROOT / "output/casestudy_repro_gpt54/M1/08_communication_labeled"


def _load_mapping() -> dict[str, dict[str, str]]:
    df = pd.read_csv(LABEL_DIR / "domain_label_mapping.csv", dtype={"domain": str})
    mapping: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        mapping[str(row["domain"])] = {
            "domain_label": str(row["domain_label"]),
            "putative_label": str(row["putative_label"]),
        }
    return mapping


def _add_labels(df: pd.DataFrame, mapping: dict[str, dict[str, str]]) -> pd.DataFrame:
    out = df.copy()
    out["source"] = out["source"].astype(str)
    out["target"] = out["target"].astype(str)
    out["source_domain_label"] = out["source"].map(lambda x: mapping[x]["domain_label"])
    out["target_domain_label"] = out["target"].map(lambda x: mapping[x]["domain_label"])
    out["source_putative_label"] = out["source"].map(lambda x: mapping[x]["putative_label"])
    out["target_putative_label"] = out["target"].map(lambda x: mapping[x]["putative_label"])
    return out


def _aggregate(df: pd.DataFrame, label_order: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(
            ["source_domain_label", "target_domain_label", "source_putative_label", "target_putative_label"],
            as_index=False,
        )
        .agg(
            n_lr_pairs=("ligand", "count"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            min_pvalue=("pvalue", "min"),
        )
        .sort_values(["n_lr_pairs", "mean_score"], ascending=[False, False])
    )
    agg["source_domain_label"] = pd.Categorical(agg["source_domain_label"], categories=label_order, ordered=True)
    agg["target_domain_label"] = pd.Categorical(agg["target_domain_label"], categories=label_order, ordered=True)
    return agg.sort_values(["source_domain_label", "target_domain_label"])


def _plot_heatmap(agg: pd.DataFrame, label_order: list[str], out_base: Path) -> None:
    pivot = agg.pivot_table(
        index="source_domain_label",
        columns="target_domain_label",
        values="n_lr_pairs",
        fill_value=0,
    ).reindex(index=label_order, columns=label_order)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Number of significant LR pairs"},
        ax=ax,
    )
    ax.set_title("M1 Domain-to-Domain Communication (Semantic Labels)", fontsize=13)
    ax.set_xlabel("Target domain")
    ax.set_ylabel("Source domain")
    plt.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_chord_like(agg: pd.DataFrame, label_order: list[str], out_base: Path) -> None:
    top_edges = agg.sort_values(["n_lr_pairs", "mean_score"], ascending=[False, False]).head(24).copy()
    node_angles = {
        label: 2 * math.pi * i / len(label_order) for i, label in enumerate(label_order)
    }
    node_xy = {
        label: (math.cos(theta), math.sin(theta)) for label, theta in node_angles.items()
    }
    palette = sns.color_palette("tab10", n_colors=len(label_order))
    node_colors = {label: palette[i] for i, label in enumerate(label_order)}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    for label in label_order:
        x, y = node_xy[label]
        ax.scatter([x], [y], s=500, color=node_colors[label], zorder=3)
        ax.text(
            x * 1.16,
            y * 1.16,
            label,
            ha="center",
            va="center",
            fontsize=9,
        )

    max_pairs = max(top_edges["n_lr_pairs"].max(), 1)
    for _, row in top_edges.iterrows():
        src = str(row["source_domain_label"])
        tgt = str(row["target_domain_label"])
        sx, sy = node_xy[src]
        tx, ty = node_xy[tgt]
        width = 0.8 + 4.5 * float(row["n_lr_pairs"]) / max_pairs
        rad = 0.22 if src != tgt else 0.45
        edge = FancyArrowPatch(
            (sx, sy),
            (tx, ty),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=node_colors[src],
            alpha=0.45,
        )
        ax.add_patch(edge)

    ax.set_title("M1 Communication Chord-like View (Semantic Labels)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_summary(relabeled_top: pd.DataFrame, agg: pd.DataFrame, out_path: Path) -> None:
    top_pairs = agg.sort_values(["n_lr_pairs", "mean_score"], ascending=[False, False]).head(10)
    lines = [
        "# M1 Communication Rephrasing Summary",
        "",
        "## Main pattern",
        "",
        "High-scoring interactions are concentrated around the `ECM-rich` and `stromal-like` domains,",
        "especially toward `adipose-like`, `tumor-like`, and `vascular-like` regions.",
        "",
        "## Representative relabeled interactions",
        "",
    ]
    for _, row in relabeled_top.head(12).iterrows():
        lines.append(
            f"- `{row['source_domain_label']} -> {row['target_domain_label']}`: "
            f"`{row['ligand']} -> {row['receptor']}` (score={row['score']:.4f}, p={row['pvalue']:.2e})"
        )
    lines.extend(["", "## Aggregated domain-domain pairs", ""])
    for _, row in top_pairs.iterrows():
        lines.append(
            f"- `{row['source_domain_label']} -> {row['target_domain_label']}`: "
            f"{int(row['n_lr_pairs'])} LR pairs, mean score {row['mean_score']:.4f}, "
            f"min p-value {row['min_pvalue']:.2e}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = OUT_DIR / "tables"
    figures_dir = OUT_DIR / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    mapping = _load_mapping()
    label_order = [mapping[str(i)]["domain_label"] for i in sorted(int(k) for k in mapping)]

    lr = pd.read_csv(COMM_DIR / "tables/lr_interactions.csv")
    top = pd.read_csv(COMM_DIR / "tables/top_interactions.csv")
    lr_relabeled = _add_labels(lr, mapping)
    top_relabeled = _add_labels(top, mapping)

    agg = _aggregate(lr_relabeled, label_order)

    lr_relabeled.to_csv(tables_dir / "lr_interactions_relabeled.csv", index=False)
    top_relabeled.to_csv(tables_dir / "top_interactions_relabeled.csv", index=False)
    agg.to_csv(tables_dir / "domain_communication_strength_by_label.csv", index=False)

    _plot_heatmap(agg, label_order, figures_dir / "lr_heatmap_labeled")
    _plot_chord_like(agg, label_order, figures_dir / "lr_chord_labeled")
    _write_summary(top_relabeled, agg, OUT_DIR / "communication_rephrase_summary.md")


if __name__ == "__main__":
    main()
