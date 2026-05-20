#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/dugaoyuan/AgentAndClaw/Claws/SpatialClaw-pri-casestudy-worktree")
INPUT_H5AD = ROOT / "output/casestudy_repro_gpt54/M1/02_domains/processed.h5ad"
ANNOT_CSV = ROOT / "output/casestudy_repro_gpt54/M1/06_domain_annotation/domain_annotation_summary.csv"
OUT_DIR = ROOT / "output/casestudy_repro_gpt54/M1/07_domain_labeled"


def _load_mapping(csv_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mapping[str(row["domain"])] = str(row["putative_label"])
    return mapping


def _write_mapping_table(mapping: dict[str, str], out_path: Path) -> None:
    rows = [
        {"domain": domain, "domain_label": f"{domain} {label}", "putative_label": label}
        for domain, label in sorted(mapping.items(), key=lambda kv: int(kv[0]))
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _plot_spatial(adata: ad.AnnData, out_base: Path) -> None:
    coords = adata.obsm["spatial"]
    labels = adata.obs["domain_label"].astype("category")
    categories = [str(cat) for cat in labels.cat.categories]
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(categories))]
    color_map = dict(zip(categories, colors))
    point_colors = [color_map[str(label)] for label in labels.astype(str).tolist()]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=12, linewidths=0, alpha=0.9)
    ax.set_title("M1 Spatial Domains with Semantic Labels", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.invert_yaxis()

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[cat], label=cat, markersize=7)
        for cat in categories
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping = _load_mapping(ANNOT_CSV)
    adata = ad.read_h5ad(INPUT_H5AD)

    domain_series = adata.obs["spatial_domain"].astype(str)
    adata.obs["domain_putative_label"] = domain_series.map(mapping).fillna("unknown")
    adata.obs["domain_label"] = domain_series + " " + adata.obs["domain_putative_label"].astype(str)

    out_h5ad = OUT_DIR / "processed_with_domain_labels.h5ad"
    adata.write_h5ad(out_h5ad)

    _write_mapping_table(mapping, OUT_DIR / "domain_label_mapping.csv")
    _plot_spatial(adata, OUT_DIR / "m1_spatial_domains_labeled")

    report = OUT_DIR / "domain_label_summary.md"
    report.write_text(
        "\n".join(
            [
                "# M1 Domain Label Summary",
                "",
                "- Input AnnData: `02_domains/processed.h5ad`",
                "- Annotation source: `06_domain_annotation/domain_annotation_summary.csv`",
                f"- Output AnnData: `{out_h5ad.name}`",
                "",
                "## Labels",
                "",
            ]
            + [f"- `{domain}` -> `{label}`" for domain, label in sorted(mapping.items(), key=lambda kv: int(kv[0]))]
            + [
                "",
                "## Outputs",
                "",
                "- `processed_with_domain_labels.h5ad`",
                "- `domain_label_mapping.csv`",
                "- `m1_spatial_domains_labeled.png`",
                "- `m1_spatial_domains_labeled.pdf`",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
