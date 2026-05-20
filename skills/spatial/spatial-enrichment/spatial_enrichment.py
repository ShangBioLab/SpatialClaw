import os
import json
import argparse
import sys
import traceback
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spatialclaw.common.checksums import sha256_file
from spatialclaw.common.report import (
    generate_report_footer,
    generate_report_header,
    write_result_json,
)

SKILL_NAME = "spatial-enrichment"
SKILL_VERSION = "0.2.0"

# ──────────────────────────────────────────────
# 可选依赖：gseapy（GO / KEGG / GSEA）
# ──────────────────────────────────────────────
try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False

# ══════════════════════════════════════════════
# 核心工具函数
# ══════════════════════════════════════════════

def _ensure_output(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _write_standard_outputs(
    output_dir: str,
    result: dict,
    input_path: str | None,
    params: dict,
) -> None:
    """Write standardized report.md and result.json for CLI runs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    status = result.get("status", "unknown")
    analysis_type = result.get("analysis_type", params.get("analysis_type", ""))
    header = generate_report_header(
        title="Spatial Enrichment Report",
        skill_name=SKILL_NAME,
        input_files=[Path(input_path)] if input_path else None,
        extra_metadata={
            "Status": status,
            "Analysis type": str(analysis_type),
        },
    )

    body_lines = [
        "## Summary\n",
        f"- **Status**: {status}",
        f"- **Analysis type**: {analysis_type}",
    ]
    if "n_clusters" in result:
        body_lines.append(f"- **Clusters**: {result['n_clusters']}")
    if "n_pathways" in result:
        body_lines.append(f"- **Pathways reported**: {result['n_pathways']}")
    if result.get("message"):
        body_lines.append(f"- **Message**: {result['message']}")
    if result.get("outputs"):
        body_lines.extend(["", "## Outputs\n"])
        body_lines.extend(f"- `{item}`" for item in result["outputs"])

    body_lines.extend(["", "## Parameters\n"])
    for key, value in params.items():
        body_lines.append(f"- `{key}`: {value}")

    (output_path / "report.md").write_text(
        header + "\n".join(body_lines) + "\n" + generate_report_footer()
    )

    checksum = sha256_file(input_path) if input_path and Path(input_path).exists() else ""
    write_result_json(
        output_path,
        skill=SKILL_NAME,
        version=SKILL_VERSION,
        summary=result,
        data={"params": params, **result},
        input_checksum=checksum,
    )


def _get_gene_list(
    adata: ad.AnnData,
    method: str,
    groupby: str | None,
    group: str | None,
    reference: str,
    n_top: int,
    logfc_threshold: float,
    pval_threshold: float,
) -> tuple[list[str], pd.DataFrame | None]:
    """
    从 AnnData 中提取基因列表。
    支持三种来源：
      - rank_genes_groups 已有结果
      - 重新跑 scanpy rank_genes_groups
      - 直接用高变基因 (HVG)
    返回 (gene_list, de_df_or_None)
    """
    # ── 方案 A：使用已有的 rank_genes_groups ──
    if "rank_genes_groups" in adata.uns and groupby is None:
        try:
            de_result = sc.get.rank_genes_groups_df(adata, group=group)
            filtered = de_result[
                (de_result["logfoldchanges"] >= logfc_threshold) &
                (de_result["pvals_adj"] <= pval_threshold)
            ].sort_values("scores", ascending=False)
            if filtered.empty:
                print("[WARN] No genes passed thresholds; using top ranked genes instead")
                filtered = de_result.sort_values("scores", ascending=False).head(n_top)
            gene_list = filtered["names"].head(n_top).tolist()
            print(f"[INFO] Using existing rank_genes_groups results: {len(gene_list)} genes")
            return gene_list, filtered
        except Exception as e:
            print(f"[WARN] Could not use existing rank_genes_groups: {e}")

    # ── 方案 B：重新跑 DE ──
    if groupby and groupby in adata.obs.columns:
        print(f"[INFO] Running rank_genes_groups on '{groupby}' ...")
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, use_raw=False)
        grp = group if group else adata.obs[groupby].unique()[0]
        de_result = sc.get.rank_genes_groups_df(adata, group=str(grp))
        filtered = de_result[
            (de_result["logfoldchanges"] >= logfc_threshold) &
            (de_result["pvals_adj"] <= pval_threshold)
        ].sort_values("scores", ascending=False)
        if filtered.empty:
            print("[WARN] No genes passed thresholds; using top ranked genes instead")
            filtered = de_result.sort_values("scores", ascending=False).head(n_top)
        gene_list = filtered["names"].head(n_top).tolist()
        print(f"[INFO] DE analysis done: {len(gene_list)} significant genes")
        return gene_list, filtered

    # ── 方案 C：高变基因 ──
    if "highly_variable" in adata.var.columns:
        gene_list = adata.var[adata.var["highly_variable"]].index.tolist()[:n_top]
        print(f"[INFO] No DE info found; using top {len(gene_list)} HVGs")
        return gene_list, None

    # ── 方案 D：所有基因（最后兜底）──
    gene_list = adata.var_names.tolist()[:n_top]
    print(f"[WARN] Falling back to top {len(gene_list)} genes by index")
    return gene_list, None


# ══════════════════════════════════════════════
# 分析模块
# ══════════════════════════════════════════════

def run_ora(
    gene_list: list[str],
    gene_sets: list[str],
    organism: str,
    output_dir: str,
    background: list[str] | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Over-Representation Analysis (ORA) via gseapy.enrichr"""
    if not HAS_GSEAPY:
        raise ImportError("gseapy is required for ORA. Install: pip install gseapy")

    print(f"[INFO] Running ORA with gene sets: {gene_sets}")
    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism=organism,
        background=background,
        outdir=os.path.join(output_dir, "ora_raw"),
        verbose=False,
    )
    df = enr.results
    df = df.sort_values("Adjusted P-value").head(top_n * 3)

    # ── 保存表格 ──
    csv_path = os.path.join(output_dir, "ora_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] ORA results saved → {csv_path}")

    # ── 可视化 ──
    plot_df = df.head(top_n).copy()
    plot_df["-log10(padj)"] = -np.log10(plot_df["Adjusted P-value"].clip(lower=1e-300))
    plot_df["Term"] = plot_df["Term"].str[:60]

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.35)))
    bars = ax.barh(
        plot_df["Term"][::-1],
        plot_df["-log10(padj)"][::-1],
        color=plt.cm.RdYlBu_r(
            np.linspace(0.2, 0.8, len(plot_df))
        )[::-1],
    )
    ax.axvline(x=-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p=0.05")
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=12)
    ax.set_title(f"ORA Top {top_n} Pathways", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "ora_dotplot.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] ORA plot saved → {fig_path}")

    return df


def run_gsea(
    adata: ad.AnnData,
    groupby: str,
    gene_sets: list[str],
    organism: str,
    output_dir: str,
    method: str = "wilcoxon",
    top_n: int = 20,
) -> pd.DataFrame:
    """GSEA (pre-ranked) via gseapy"""
    if not HAS_GSEAPY:
        raise ImportError("gseapy is required for GSEA. Install: pip install gseapy")

    print(f"[INFO] Running GSEA on '{groupby}' ...")

    # 构建 ranked gene list（log2FC × -log10pval）
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, use_raw=False)
    groups = adata.obs[groupby].unique().tolist()
    all_results = []

    for grp in groups:
        try:
            de_df = sc.get.rank_genes_groups_df(adata, group=str(grp))
            de_df["rank_score"] = de_df["logfoldchanges"] * (
                -np.log10(de_df["pvals"].clip(lower=1e-300))
            )
            ranked = de_df[["names", "rank_score"]].dropna()
            ranked = ranked.sort_values("rank_score", ascending=False)

            gsea_res = gp.prerank(
                rnk=ranked.set_index("names")["rank_score"],
                gene_sets=gene_sets,
                organism=organism,
                outdir=os.path.join(output_dir, f"gsea_{grp}"),
                min_size=5,
                max_size=1000,
                permutation_num=100,
                verbose=False,
            )
            res_df = gsea_res.res2d.copy()
            res_df["group"] = grp
            all_results.append(res_df)
        except Exception as e:
            print(f"[WARN] GSEA failed for group '{grp}': {e}")

    if not all_results:
        raise RuntimeError("GSEA produced no results for any group.")

    combined = pd.concat(all_results, ignore_index=True)
    csv_path = os.path.join(output_dir, "gsea_results.csv")
    combined.to_csv(csv_path, index=False)
    print(f"[INFO] GSEA results saved → {csv_path}")

    # ── 可视化：每组 top NES ──
    for grp in groups:
        grp_df = combined[combined["group"] == grp].copy()
        if grp_df.empty:
            continue
        grp_df["NES"] = pd.to_numeric(grp_df["NES"], errors="coerce")
        grp_df = grp_df.dropna(subset=["NES"]).sort_values("NES", ascending=False)
        top_pos = grp_df.head(top_n // 2)
        top_neg = grp_df.tail(top_n // 2)
        plot_df = pd.concat([top_pos, top_neg]).drop_duplicates()
        plot_df["Term"] = plot_df["Term"].str[:55]

        colors = ["#d73027" if v > 0 else "#4575b4" for v in plot_df["NES"]]
        fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.38)))
        ax.barh(plot_df["Term"][::-1], plot_df["NES"][::-1], color=colors[::-1])
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=12)
        ax.set_title(f"GSEA — {grp}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        safe_grp = str(grp).replace("/", "_").replace(" ", "_")
        fig_path = os.path.join(output_dir, f"gsea_barplot_{safe_grp}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] GSEA plot saved → {fig_path}")

    return combined


def run_go_analysis(
    gene_list: list[str],
    organism: str,
    output_dir: str,
    go_ontology: str = "BP",
    top_n: int = 20,
) -> pd.DataFrame:
    """GO enrichment (BP / MF / CC) via gseapy.enrichr"""
    if not HAS_GSEAPY:
        raise ImportError("gseapy is required. Install: pip install gseapy")

    go_set_map = {
        "BP": "GO_Biological_Process_2023",
        "MF": "GO_Molecular_Function_2023",
        "CC": "GO_Cellular_Component_2023",
        "ALL": ["GO_Biological_Process_2023",
                "GO_Molecular_Function_2023",
                "GO_Cellular_Component_2023"],
    }
    gene_sets = go_set_map.get(go_ontology.upper(), go_set_map["BP"])
    if isinstance(gene_sets, str):
        gene_sets = [gene_sets]

    return run_ora(gene_list, gene_sets, organism, output_dir, top_n=top_n)


def run_kegg_analysis(
    gene_list: list[str],
    organism: str,
    output_dir: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """KEGG pathway enrichment via gseapy.enrichr"""
    if not HAS_GSEAPY:
        raise ImportError("gseapy is required. Install: pip install gseapy")

    kegg_sets = ["KEGG_2021_Human"] if organism.lower() in ("human", "homo sapiens") \
        else ["KEGG_2019_Mouse"]
    return run_ora(gene_list, kegg_sets, organism, output_dir, top_n=top_n)


def run_spatial_enrichment_score(
    adata: ad.AnnData,
    gene_sets_file: str | None,
    gene_set_name: str,
    gene_list: list[str],
    output_dir: str,
) -> ad.AnnData:
    """
    用 scanpy score_genes 计算每个 spot 的通路活性得分，
    并叠加到空间坐标上可视化。
    """
    score_key = f"score_{gene_set_name.replace(' ', '_')[:30]}"
    sc.tl.score_genes(adata, gene_list=gene_list, score_name=score_key)
    print(f"[INFO] Pathway activity score saved to adata.obs['{score_key}']")

    # 尝试空间可视化
    if "spatial" in adata.obsm:
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            coords = adata.obsm["spatial"]
            scores = adata.obs[score_key].values
            sc_plot = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=scores, cmap="RdYlBu_r", s=10, alpha=0.85
            )
            plt.colorbar(sc_plot, ax=ax, label="Activity Score")
            ax.set_title(f"Spatial Pathway Activity\n{gene_set_name[:50]}", fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            safe_name = gene_set_name.replace("/", "_").replace(" ", "_")[:40]
            fig_path = os.path.join(output_dir, f"spatial_score_{safe_name}.png")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Spatial score plot saved → {fig_path}")
        except Exception as e:
            print(f"[WARN] Spatial plot failed: {e}")

    return adata


# ══════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════

def spatial_enrichment_main(
    input_path: str,
    output_dir: str,
    analysis_type: str = "go",
    go_ontology: str = "BP",
    gene_sets: str = "GO_Biological_Process_2023",
    groupby: str | None = None,
    group: str | None = None,
    organism: str = "human",
    de_method: str = "wilcoxon",
    n_top_genes: int = 200,
    logfc_threshold: float = 0.25,
    pval_threshold: float = 0.05,
    top_n: int = 20,
    gene_set_name: str = "pathway",
    save_h5ad: bool = True,
) -> dict:
    try:
        if not os.path.exists(input_path):
            return {"status": "error", "message": f"Input file not found: {input_path}"}

        _ensure_output(output_dir)
        print(f"[INFO] Loading {input_path} ...")
        adata = sc.read_h5ad(input_path).copy()

        # 规范化表达矩阵（如果没有 log 化）
        if adata.X is not None:
            import scipy.sparse as sp
            mat = adata.X.toarray() if sp.issparse(adata.X) else adata.X
            if mat.max() > 30:
                print("[INFO] Data appears raw counts — normalizing & log1p ...")
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)

        gene_set_list = [g.strip() for g in gene_sets.split(",")]
        analysis_type = analysis_type.lower()

        results_summary = {
            "status": "success",
            "analysis_type": analysis_type,
            "output_dir": output_dir,
            "outputs": [],
        }
        if groupby and groupby in adata.obs.columns:
            results_summary["n_clusters"] = int(adata.obs[groupby].nunique())

        # ── GSEA（需要 groupby）──
        if analysis_type == "gsea":
            if not groupby:
                return {"status": "error",
                        "message": "GSEA requires --groupby (e.g. leiden, cell_type). "
                                   "Please provide a cluster column."}
            df = run_gsea(adata, groupby, gene_set_list, organism, output_dir,
                          method=de_method, top_n=top_n)
            results_summary["outputs"].append("gsea_results.csv")
            results_summary["n_pathways"] = len(df)

        # ── GO ORA ──
        elif analysis_type == "go":
            gene_list, de_df = _get_gene_list(
                adata, de_method, groupby, group,
                "rest", n_top_genes, logfc_threshold, pval_threshold
            )
            if de_df is not None:
                de_df.to_csv(os.path.join(output_dir, "de_genes.csv"), index=False)
                results_summary["outputs"].append("de_genes.csv")

            df = run_go_analysis(gene_list, organism, output_dir,
                                 go_ontology=go_ontology, top_n=top_n)
            results_summary["outputs"] += ["ora_results.csv", "ora_dotplot.png"]
            results_summary["n_pathways"] = len(df)

        # ── KEGG ORA ──
        elif analysis_type == "kegg":
            gene_list, de_df = _get_gene_list(
                adata, de_method, groupby, group,
                "rest", n_top_genes, logfc_threshold, pval_threshold
            )
            if de_df is not None:
                de_df.to_csv(os.path.join(output_dir, "de_genes.csv"), index=False)
                results_summary["outputs"].append("de_genes.csv")

            df = run_kegg_analysis(gene_list, organism, output_dir, top_n=top_n)
            results_summary["outputs"] += ["ora_results.csv", "ora_dotplot.png"]
            results_summary["n_pathways"] = len(df)

        # ── ORA（自定义 gene sets）──
        elif analysis_type == "ora":
            gene_list, de_df = _get_gene_list(
                adata, de_method, groupby, group,
                "rest", n_top_genes, logfc_threshold, pval_threshold
            )
            if de_df is not None:
                de_df.to_csv(os.path.join(output_dir, "de_genes.csv"), index=False)
            df = run_ora(gene_list, gene_set_list, organism, output_dir, top_n=top_n)
            results_summary["outputs"] += ["ora_results.csv", "ora_dotplot.png"]
            results_summary["n_pathways"] = len(df)

        # ── 空间 Score（不需要 gseapy）──
        elif analysis_type == "spatial_score":
            gene_list, _ = _get_gene_list(
                adata, de_method, groupby, group,
                "rest", n_top_genes, logfc_threshold, pval_threshold
            )
            adata = run_spatial_enrichment_score(
                adata, None, gene_set_name, gene_list, output_dir
            )
            safe = gene_set_name.replace("/", "_").replace(" ", "_")[:40]
            results_summary["outputs"].append(f"spatial_score_{safe}.png")

        else:
            return {"status": "error",
                    "message": f"Unknown analysis_type '{analysis_type}'. "
                               f"Choose from: go, kegg, ora, gsea, spatial_score"}

        # ── 保存更新后的 h5ad ──
        if save_h5ad:
            h5_path = os.path.join(output_dir, "processed.h5ad")
            if os.path.exists(h5_path):
                os.remove(h5_path)
            adata.write_h5ad(h5_path)
            results_summary["outputs"].append("processed.h5ad")
            print(f"[INFO] Updated h5ad saved → {h5_path}")

        results_summary["message"] = (
            f"Enrichment analysis ({analysis_type}) completed. "
            f"Outputs in: {output_dir}"
        )
        return results_summary

    except Exception as e:
        traceback.print_exc()
        return {"status": "error",
                "message": f"Enrichment analysis failed: {str(e)}"}


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial Enrichment Analysis (GO / KEGG / GSEA / ORA / Spatial Score)"
    )
    parser.add_argument("--input",        type=str,
                        help="Path to input .h5ad file")
    parser.add_argument("--output",       type=str, default="output",
                        help="Output directory")
    parser.add_argument("--analysis-type", dest="analysis_type", type=str,
                        default="go",
                        choices=["go", "kegg", "ora", "gsea", "spatial_score"],
                        help="Analysis type (default: go)")
    parser.add_argument("--go-ontology",  dest="go_ontology", type=str,
                        default="BP", choices=["BP", "MF", "CC", "ALL"],
                        help="GO ontology namespace (default: BP)")
    parser.add_argument("--gene-sets",    dest="gene_sets", type=str,
                        default="GO_Biological_Process_2023",
                        help="Comma-separated gseapy gene set names for ORA/GSEA")
    parser.add_argument("--groupby",      type=str, default=None,
                        help="obs column for DE / GSEA grouping (e.g. leiden, cell_type)")
    parser.add_argument("--group",        type=str, default=None,
                        help="Specific group within groupby to use as foreground")
    parser.add_argument("--organism",     type=str, default="human",
                        help="Organism: human or mouse (default: human)")
    parser.add_argument("--de-method",    dest="de_method", type=str,
                        default="wilcoxon",
                        help="DE method for rank_genes_groups (default: wilcoxon)")
    parser.add_argument("--n-top-genes",  dest="n_top_genes", type=int,
                        default=200,
                        help="Max genes to use as input for ORA (default: 200)")
    parser.add_argument("--logfc-threshold", dest="logfc_threshold",
                        type=float, default=0.25,
                        help="logFC threshold for DE gene filtering (default: 0.25)")
    parser.add_argument("--pval-threshold",  dest="pval_threshold",
                        type=float, default=0.05,
                        help="Adjusted p-value cutoff for DE (default: 0.05)")
    parser.add_argument("--top-n",        dest="top_n", type=int, default=20,
                        help="Top N pathways to visualize (default: 20)")
    parser.add_argument("--gene-set-name", dest="gene_set_name", type=str,
                        default="pathway",
                        help="Label for spatial_score mode")
    parser.add_argument("--no-save-h5ad", dest="save_h5ad",
                        action="store_false",
                        help="Skip saving updated .h5ad")
    parser.add_argument("--demo",         action="store_true",
                        help="Run demo with mock data")

    args = parser.parse_args()

    if args.demo:
        print("[DEMO] Generating mock spatial AnnData ...")
        import scipy.sparse as sp
        n_spots, n_genes = 200, 500
        rng = np.random.default_rng(42)
        X = sp.csr_matrix(rng.negative_binomial(5, 0.5, (n_spots, n_genes)).astype(np.float32))
        gene_names = [f"Gene{i}" for i in range(n_genes)]
        obs = pd.DataFrame({
            "leiden": rng.choice(["0", "1", "2"], n_spots).tolist(),
        }, index=[f"spot{i}" for i in range(n_spots)])
        var = pd.DataFrame(index=gene_names)
        adata_demo = ad.AnnData(X=X, obs=obs, var=var)
        adata_demo.obsm["spatial"] = rng.uniform(0, 1000, (n_spots, 2))
        sc.pp.normalize_total(adata_demo, target_sum=1e4)
        sc.pp.log1p(adata_demo)
        sc.pp.highly_variable_genes(adata_demo, n_top_genes=200)
        demo_path = "/tmp/demo_spatial.h5ad"
        adata_demo.write_h5ad(demo_path)
        args.input = demo_path
        args.groupby = "leiden"
        args.analysis_type = "spatial_score"
        print(f"[DEMO] Mock data saved to {demo_path}")

    out_path = args.output
    if not out_path.endswith(".h5ad"):
        os.makedirs(out_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    result = spatial_enrichment_main(
        input_path=args.input,
        output_dir=out_path,
        analysis_type=args.analysis_type,
        go_ontology=args.go_ontology,
        gene_sets=args.gene_sets,
        groupby=args.groupby,
        group=args.group,
        organism=args.organism,
        de_method=args.de_method,
        n_top_genes=args.n_top_genes,
        logfc_threshold=args.logfc_threshold,
        pval_threshold=args.pval_threshold,
        top_n=args.top_n,
        gene_set_name=args.gene_set_name,
        save_h5ad=args.save_h5ad,
    )

    params = {
        "analysis_type": args.analysis_type,
        "go_ontology": args.go_ontology,
        "gene_sets": args.gene_sets,
        "groupby": args.groupby,
        "group": args.group,
        "organism": args.organism,
        "de_method": args.de_method,
        "n_top_genes": args.n_top_genes,
        "logfc_threshold": args.logfc_threshold,
        "pval_threshold": args.pval_threshold,
        "top_n": args.top_n,
        "gene_set_name": args.gene_set_name,
        "save_h5ad": args.save_h5ad,
    }
    _write_standard_outputs(out_path, result, args.input, params)

    print(json.dumps(result, indent=2, default=str))
    if result.get("status") != "success":
        raise SystemExit(1)
