"""Spatial cell-cell communication analysis functions.

Provides a built-in ligand-receptor scorer plus LIANA, CellPhoneDB, and FastCCC
for ligand-receptor analysis.

Includes pathway-level aggregation and signaling role classification
(sender, receiver, mediator, influencer) from community best practices.

Input matrix convention:
  All CCC methods use log-normalized expression (adata.X), NOT raw counts.
  These methods compute mean L-R co-expression scores, permutation statistics,
  or consensus rankings on continuous expression values.

  - builtin:     adata.X (log-normalized); curated L-R co-expression scoring
  - liana:       adata.X (log-normalized); auto-maps gene IDs to symbols when needed
  - cellphonedb: adata.X (log-normalized); do NOT use z-scored/scaled matrix
  - fastccc:     adata.X (log-normalized); standard CCC mode

Usage::

    from skills.spatial._lib.communication import run_communication, SUPPORTED_METHODS
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import scanpy as sc

from .adata_utils import get_spatial_key
from .dependency_manager import require

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("builtin", "liana", "cellphonedb", "fastccc")

# All CCC methods use log-normalized expression, not raw counts.
# This is because they compute mean expression scores or rank-based statistics
# on continuous values — not count-based probabilistic models.
NORMALIZED_METHODS = SUPPORTED_METHODS
GENE_SYMBOL_COLUMNS = ("SYMBOL", "symbol", "gene_symbol", "gene_symbols", "feature_name", "gene_name")
_ENSEMBL_ID_RE = re.compile(r"^(ENSG|ENSMUSG|ENSDARG)\d+(?:\.\d+)?$")

_BUILTIN_LR_PAIRS = {
    "human": (
        ("ADIPOQ", "ADIPOR1"), ("ADIPOQ", "ADIPOR2"),
        ("AGT", "AGTR1"), ("AGT", "AGTR2"),
        ("ANGPT1", "TEK"), ("ANGPT2", "TEK"),
        ("APLN", "APLNR"), ("ADM", "CALCRL"),
        ("ALCAM", "CD6"), ("ANXA1", "FPR1"),
        ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"),
        ("VEGFA", "KDR"), ("VEGFA", "FLT1"),
        ("CXCL12", "CXCR4"), ("CCL2", "CCR2"),
        ("IL6", "IL6R"), ("TNF", "TNFRSF1A"),
        ("EGF", "EGFR"), ("FGF2", "FGFR1"),
        ("PDGFA", "PDGFRA"), ("NOTCH1", "JAG1"),
        ("MIF", "CD74"), ("COL1A1", "ITGA1"),
    ),
    "mouse": (
        ("Adipoq", "Adipor1"), ("Adipoq", "Adipor2"),
        ("Agt", "Agtr1a"), ("Agt", "Agtr2"),
        ("Angpt1", "Tek"), ("Angpt2", "Tek"),
        ("Apln", "Aplnr"), ("Adm", "Calcrl"),
        ("Alcam", "Cd6"), ("Anxa1", "Fpr1"),
        ("Tgfb1", "Tgfbr1"), ("Tgfb1", "Tgfbr2"),
        ("Vegfa", "Kdr"), ("Vegfa", "Flt1"),
        ("Cxcl12", "Cxcr4"), ("Ccl2", "Ccr2"),
        ("Il6", "Il6ra"), ("Tnf", "Tnfrsf1a"),
        ("Egf", "Egfr"), ("Fgf2", "Fgfr1"),
        ("Pdgfa", "Pdgfra"), ("Notch1", "Jag1"),
        ("Mif", "Cd74"), ("Col1a1", "Itga1"),
    ),
}


def _looks_like_ensembl_ids(index: pd.Index, *, sample_size: int = 50) -> bool:
    """Heuristic check for Ensembl-style feature identifiers."""
    if len(index) == 0:
        return False
    sample = [str(v) for v in index[: min(len(index), sample_size)] if pd.notna(v)]
    if not sample:
        return False
    matches = sum(1 for value in sample if _ENSEMBL_ID_RE.match(value))
    return (matches / len(sample)) >= 0.8


def _build_feature_index(values: pd.Series, fallback: pd.Index) -> pd.Index:
    """Create feature names from a metadata column, falling back to current names."""
    fallback_arr = fallback.astype(str).to_numpy(dtype=object, copy=False)
    cleaned = values.astype("string").fillna("").str.strip()
    invalid = cleaned.eq("") | cleaned.str.lower().isin({"nan", "none", "<na>"})
    renamed = np.where(invalid.to_numpy(), fallback_arr, cleaned.astype(str).to_numpy())
    return pd.Index(renamed)


def _get_liana_resource_genes(li, resource_name: str) -> set[str]:
    """Return the ligand/receptor gene set for the selected LIANA resource."""
    resource = li.resource.select_resource(resource_name)
    genes = set(resource["ligand"].astype(str)).union(set(resource["receptor"].astype(str)))
    return {gene for gene in genes if gene and gene.lower() not in {"nan", "none", "<na>"}}


def _expression_frame(adata, genes: list[str]) -> pd.DataFrame:
    """Return dense expression for selected genes as cells × genes."""
    from scipy import sparse

    X = adata[:, genes].X
    if sparse.issparse(X):
        X = X.toarray()
    return pd.DataFrame(np.asarray(X), index=adata.obs_names, columns=genes)


def _resolve_builtin_pairs(adata, species: str) -> list[tuple[str, str]]:
    """Resolve curated L-R pairs against the input feature names."""
    pairs = _BUILTIN_LR_PAIRS.get(species.lower(), _BUILTIN_LR_PAIRS["human"])
    feature_lookup = {str(name).upper(): str(name) for name in adata.var_names}
    resolved: list[tuple[str, str]] = []
    for ligand, receptor in pairs:
        ligand_name = feature_lookup.get(ligand.upper())
        receptor_name = feature_lookup.get(receptor.upper())
        if ligand_name and receptor_name:
            resolved.append((ligand_name, receptor_name))
    return resolved


def _run_builtin(
    adata, *, cell_type_key: str = "leiden", species: str = "human", n_perms: int = 100,
) -> pd.DataFrame:
    """Run a deterministic built-in L-R co-expression scorer.

    The method computes mean ligand expression in source groups multiplied by
    mean receptor expression in target groups, then estimates an empirical
    p-value by shuffling group labels. It is dependency-light and intended as
    the default local scorer; external packages remain available when selected.
    """
    pairs = _resolve_builtin_pairs(adata, species)
    columns = ["ligand", "receptor", "source", "target", "score", "pvalue"]
    if not pairs:
        logger.warning("No built-in L-R pairs matched input features for species=%s", species)
        empty = pd.DataFrame(columns=columns)
        adata.uns["builtin_results"] = empty.copy()
        adata.uns["ccc_results"] = empty.copy()
        adata.uns["ccc_method"] = "builtin"
        return empty

    genes = sorted({gene for pair in pairs for gene in pair})
    expr = _expression_frame(adata, genes)
    labels = adata.obs[cell_type_key].astype(str)
    cell_types = sorted(labels.unique().tolist(), key=str)
    group_means = expr.groupby(labels, observed=True).mean()

    records = []
    for ligand, receptor in pairs:
        for source in cell_types:
            ligand_mean = float(group_means.loc[source, ligand])
            if ligand_mean <= 0:
                continue
            for target in cell_types:
                receptor_mean = float(group_means.loc[target, receptor])
                score = ligand_mean * receptor_mean
                if score <= 0:
                    continue
                records.append({
                    "ligand": ligand,
                    "receptor": receptor,
                    "source": source,
                    "target": target,
                    "score": float(score),
                    "pvalue": 1.0,
                })

    if not records:
        empty = pd.DataFrame(columns=columns)
        adata.uns["builtin_results"] = empty.copy()
        adata.uns["ccc_results"] = empty.copy()
        adata.uns["ccc_method"] = "builtin"
        return empty

    df = pd.DataFrame(records)
    if n_perms > 0 and len(cell_types) > 1:
        rng = np.random.default_rng(42)
        exceed = np.zeros(len(df), dtype=np.int64)
        label_values = labels.to_numpy(copy=True)
        observed = df["score"].to_numpy(dtype=float)
        for _ in range(n_perms):
            shuffled = pd.Series(rng.permutation(label_values), index=expr.index)
            perm_means = expr.groupby(shuffled, observed=True).mean()
            for idx, row in df.iterrows():
                null_score = (
                    float(perm_means.loc[row["source"], row["ligand"]])
                    * float(perm_means.loc[row["target"], row["receptor"]])
                )
                if null_score >= observed[idx]:
                    exceed[idx] += 1
        df["pvalue"] = (exceed + 1) / (n_perms + 1)

    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).round(6)
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce").fillna(1.0).round(6)
    df = df.sort_values(["score", "pvalue"], ascending=[False, True]).reset_index(drop=True)
    adata.uns["builtin_results"] = df.copy()
    adata.uns["ccc_results"] = df.copy()
    adata.uns["ccc_method"] = "builtin"
    return df[columns].copy()


def _prepare_liana_adata(adata, *, li, resource_name: str) -> tuple:
    """Prepare a LIANA-safe AnnData copy and choose the best gene identifier source."""
    resource_genes = _get_liana_resource_genes(li, resource_name)
    current_names = pd.Index(adata.var_names.astype(str))
    best_names = current_names
    best_source = "var_names"
    best_overlap = len(resource_genes & set(current_names))

    for column in GENE_SYMBOL_COLUMNS:
        if column not in adata.var.columns:
            continue
        candidate = _build_feature_index(adata.var[column], current_names)
        overlap = len(resource_genes & set(candidate.astype(str)))
        if overlap > best_overlap:
            best_names = candidate
            best_source = f"var[{column}]"
            best_overlap = overlap

    overlap_ratio = best_overlap / max(len(resource_genes), 1)
    if overlap_ratio < 0.02:
        hints = []
        if _looks_like_ensembl_ids(current_names):
            hints.append("detected Ensembl-style feature IDs in var_names")
        available_cols = [col for col in GENE_SYMBOL_COLUMNS if col in adata.var.columns]
        if available_cols:
            hints.append(f"available gene-id columns: {', '.join(available_cols)}")
        hint_text = f" ({'; '.join(hints)})" if hints else ""
        raise ValueError(
            "Too few LIANA resource genes overlap with the input features: "
            f"{best_overlap}/{len(resource_genes)} matched using {best_source}{hint_text}. "
            "Provide species-matched gene symbols or map var_names to a supported symbol column."
        )

    work_adata = adata.copy()
    if best_source != "var_names":
        work_adata.var_names = best_names
        work_adata.var_names_make_unique()

    return work_adata, best_source, best_overlap, len(resource_genes)


def _run_liana(adata, *, cell_type_key: str = "leiden", species: str = "human", n_perms: int = 100) -> pd.DataFrame:
    """Run LIANA+ multi-method consensus ranking.

    Uses ``adata.X`` (log-normalized) for scoring.  The input is copied so
    feature identifiers can be remapped to a symbol column when ``var_names``
    are Ensembl IDs.  ``adata.raw`` is intentionally ignored here because the
    current preprocessing pipeline stores raw counts in ``.raw``.
    """
    li = require("liana", feature="LIANA+ cell communication")

    # LIANA defaults to human 'consensus'. For mouse, we explicitly use 'mouseconsensus'
    resource_name = "mouseconsensus" if species.lower() == "mouse" else "consensus"

    work_adata, gene_id_source, overlap_count, resource_gene_count = _prepare_liana_adata(
        adata, li=li, resource_name=resource_name,
    )
    use_raw = False
    logger.info(
        "Running LIANA+ rank_aggregate on adata.X (log-normalized) "
        "(n_perms=%d, resource=%s, gene_ids=%s, overlap=%d/%d) ...",
        n_perms, resource_name, gene_id_source, overlap_count, resource_gene_count,
    )

    li.mt.rank_aggregate(
        work_adata,
        groupby=cell_type_key,
        use_raw=use_raw,
        n_perms=n_perms,
        resource_name=resource_name,
        verbose=True
    )

    if "liana_res" not in work_adata.uns or work_adata.uns["liana_res"].empty:
        logger.warning("LIANA+ returned empty results. Check if %s L-R genes are expressed.", species)
        return pd.DataFrame(columns=["ligand", "receptor", "source", "target", "score", "pvalue"])

    # Preserve downstream plotting/reporting behaviour on the original object.
    adata.uns["liana_res"] = work_adata.uns["liana_res"].copy()
    adata.uns["ccc_results"] = adata.uns["liana_res"].copy()
    df = adata.uns["liana_res"].copy()

    col_map = {}
    if "ligand_complex" in df.columns: col_map["ligand_complex"] = "ligand"
    if "receptor_complex" in df.columns: col_map["receptor_complex"] = "receptor"
    if "sender" in df.columns and "source" not in df.columns: col_map["sender"] = "source"
    if "receiver" in df.columns and "target" not in df.columns: col_map["receiver"] = "target"
    if col_map: df = df.rename(columns=col_map)

    # Invert magnitude rank (0 is best -> 1.0 is best) for consistent score interpretation
    if "magnitude_rank" in df.columns: df["score"] = 1.0 - df["magnitude_rank"]
    elif "lr_means" in df.columns: df["score"] = df["lr_means"]
    else: df["score"] = 0.0

    # specificity_rank aggregates cellphonedb p-values and others (0 is most specific)
    if "specificity_rank" in df.columns: df["pvalue"] = df["specificity_rank"]
    else: df["pvalue"] = 0.5

    for col in ["ligand", "receptor", "source", "target", "score", "pvalue"]:
        if col not in df.columns: df[col] = ""

    return df[["ligand", "receptor", "source", "target", "score", "pvalue"]].copy().sort_values("score", ascending=False).reset_index(drop=True)


def _run_cellphonedb(adata, *, cell_type_key: str = "leiden", species: str = "human", n_perms: int = 1000) -> pd.DataFrame:
    """Run CellPhoneDB statistical method.

    Uses ``adata.X`` (log-normalized) — CellPhoneDB requires log-normalized
    expression data for scoring interactions.  Do NOT pass z-scored or scaled
    matrices, as transforms that convert zeros to non-zero values will corrupt
    the interaction scoring (CellPhoneDB v5 docs explicitly warn about this).
    """
    cpdb = require("cellphonedb", feature="CellPhoneDB cell communication")
    from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
    from pathlib import Path
    import tempfile as _tf

    if species != "human":
        raise ValueError("CellPhoneDB supports human data only.")

    cpdb_db_path = None
    try:
        import cellphonedb
        cpdb_pkg_dir = Path(cellphonedb.__file__).parent
        for candidate in [cpdb_pkg_dir / "src" / "core" / "data" / "cellphonedb.zip", cpdb_pkg_dir / "data" / "cellphonedb.zip"]:
            if candidate.exists():
                cpdb_db_path = str(candidate); break
    except Exception: pass

    with _tf.TemporaryDirectory(prefix="cpdb_") as tmp:
        tmp_path = Path(tmp)
        meta_df = pd.DataFrame({"Cell": adata.obs_names, "cell_type": adata.obs[cell_type_key].values})
        meta_df.to_csv(tmp_path / "meta.tsv", sep="\t", index=False)
        
        # Optimize memory during matrix extraction
        X_T = adata.X.T
        counts_df = pd.DataFrame(X_T.toarray() if hasattr(X_T, "toarray") else X_T, index=adata.var_names, columns=adata.obs_names)
        counts_df.to_csv(tmp_path / "counts.tsv", sep="\t")

        logger.info("Running CellPhoneDB statistical analysis (n_perms=%d, outdir=%s)...", n_perms, tmp_path)
        result = cpdb_statistical_analysis_method.call(
            cpdb_file_path=cpdb_db_path, meta_file_path=str(tmp_path / "meta.tsv"),
            counts_file_path=str(tmp_path / "counts.tsv"), counts_data="hgnc_symbol",
            output_path=str(tmp_path), iterations=n_perms, threshold=0.1,
            threads=4
        )

    # CellPhoneDB versions return either tuple-like or dict-like result objects.
    if isinstance(result, tuple):
        means_df = result[1] if len(result) > 1 else None
        pvalues_df = result[2] if len(result) > 2 else None
    elif isinstance(result, dict):
        means_df = result.get("means_result", result.get("means"))
        pvalues_df = result.get("pvalues_result", result.get("pvalues"))
    else:
        means_df = None
        pvalues_df = None
        
    if means_df is None or means_df.empty:
        logger.warning("CellPhoneDB returned empty results. No interactions met the minimum expression threshold.")
        return pd.DataFrame(columns=["ligand", "receptor", "source", "target", "score", "pvalue"])

    records = []
    for _, row in means_df.iterrows():
        pair = str(row.get("interacting_pair", ""))
        # Modern CellPhoneDB sets use '_' for ligand-receptor pairs, older versions used '|'
        parts = pair.split("_") if "_" in pair else pair.split("|")
        ligand, receptor = (parts[0] if len(parts) >= 1 else pair), (parts[1] if len(parts) >= 2 else "")
        
        # Dynamically identify cell type pair columns (e.g., 'T_cell|B_cell') to cleanly bypass prepended metadata
        for col in means_df.columns:
            if "|" not in col or col == "interacting_pair":
                continue
                
            src_tgt = str(col).split("|")
            if len(src_tgt) != 2:
                continue
                
            score = float(row.get(col, 0) or 0)
            if score < 1e-6: 
                continue
                
            source, target = src_tgt[0], src_tgt[1]
            pval = float(pvalues_df.loc[row.name, col]) if pvalues_df is not None and col in pvalues_df.columns and row.name in pvalues_df.index else 1.0
            
            records.append({
                "ligand": ligand, "receptor": receptor, "source": source, "target": target, 
                "score": float(f"{score:.4f}"), "pvalue": float(f"{pval:.4f}")
            })

    df = pd.DataFrame(records)
    return df.sort_values("score", ascending=False).reset_index(drop=True) if not df.empty else df


def _run_fastccc(adata, *, cell_type_key: str = "leiden", species: str = "human") -> pd.DataFrame:
    """Run FastCCC — FFT-based communication without permutation testing.

    Uses ``adata.X`` (log-normalized) in standard CCC mode — FastCCC
    benchmarks use the same log-transformed data as CellPhoneDB for
    comparable scoring.  In reference-based mode (not yet implemented here),
    query input could be raw counts with internal rank-based preprocessing.
    """
    require("fastccc", feature="FastCCC cell communication")
    import fastccc
    
    if species != "human": 
        raise ValueError("FastCCC currently supports human data only.")
        
    logger.info("Running FastCCC analysis (FFT-based, no permutations)...")
    try:
        result = fastccc.run(adata, groupby=cell_type_key)
    except Exception as e:
        logger.error("FastCCC execution failed: %s", e)
        return pd.DataFrame(columns=["ligand", "receptor", "source", "target", "score", "pvalue"])

    if result is None or (hasattr(result, "empty") and result.empty):
        logger.warning("FastCCC returned empty results.")
        return pd.DataFrame(columns=["ligand", "receptor", "source", "target", "score", "pvalue"])
        
    df = pd.DataFrame(result)
    
    # Map vendor-specific columns to SpatialClaw standardized keys
    col_map = {"ligand_complex": "ligand", "receptor_complex": "receptor", "sender": "source", "receiver": "target"}
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns: 
            df = df.rename(columns={old: new})
            
    df["score"] = df.get("lr_mean", df.get("score", 0.0))
    df["pvalue"] = df.get("pvalue", 0.0)
    
    for col in ["ligand", "receptor", "source", "target", "score", "pvalue"]:
        if col not in df.columns: 
            df[col] = ""
            
    # Safely cast metrics to standard datatypes to prevent downstream schema breaks
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).round(4)
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce").fillna(1.0).round(4)
    
    return df[["ligand", "receptor", "source", "target", "score", "pvalue"]].copy().sort_values("score", ascending=False).reset_index(drop=True)


def aggregate_by_pathway(lr_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate L-R interactions by signaling pathway.

    Groups interactions by source-target cell type pairs and computes
    pathway-level statistics: total interaction count, mean score,
    and top ligand-receptor pair per pathway.

    Returns a DataFrame with columns: source, target, n_interactions,
    mean_score, top_ligand, top_receptor.
    """
    if lr_df.empty or "source" not in lr_df.columns or "target" not in lr_df.columns:
        return pd.DataFrame()

    grouped = lr_df.groupby(["source", "target"], observed=True)
    records = []
    for (src, tgt), grp in grouped:
        best = grp.loc[grp["score"].idxmax()] if "score" in grp.columns and not grp["score"].isna().all() else grp.iloc[0]
        records.append({
            "source": src,
            "target": tgt,
            "n_interactions": len(grp),
            "mean_score": float(grp["score"].mean()) if "score" in grp.columns else 0.0,
            "top_ligand": best.get("ligand", ""),
            "top_receptor": best.get("receptor", ""),
        })

    return pd.DataFrame(records).sort_values("mean_score", ascending=False).reset_index(drop=True)


def classify_signaling_roles(lr_df: pd.DataFrame) -> pd.DataFrame:
    """Classify each cell type's signaling role.

    Computes four role scores per cell type:
    - **Sender**: Total outgoing interaction strength (sum of scores as source)
    - **Receiver**: Total incoming interaction strength (sum of scores as target)
    - **Hub**: Combined sender + receiver (highly connected)
    - **Dominant role**: 'sender', 'receiver', or 'balanced'

    Returns a DataFrame with columns: cell_type, sender_score, receiver_score,
    hub_score, dominant_role, n_outgoing, n_incoming.
    """
    if lr_df.empty:
        return pd.DataFrame()

    all_types = set()
    if "source" in lr_df.columns:
        all_types.update(lr_df["source"].unique())
    if "target" in lr_df.columns:
        all_types.update(lr_df["target"].unique())

    records = []
    for ct in sorted(all_types, key=str):
        out_mask = lr_df["source"] == ct if "source" in lr_df.columns else pd.Series(False, index=lr_df.index)
        in_mask = lr_df["target"] == ct if "target" in lr_df.columns else pd.Series(False, index=lr_df.index)

        sender_score = float(lr_df.loc[out_mask, "score"].sum()) if "score" in lr_df.columns else 0.0
        receiver_score = float(lr_df.loc[in_mask, "score"].sum()) if "score" in lr_df.columns else 0.0
        n_out = int(out_mask.sum())
        n_in = int(in_mask.sum())
        hub_score = sender_score + receiver_score

        if sender_score > receiver_score * 1.5:
            role = "sender"
        elif receiver_score > sender_score * 1.5:
            role = "receiver"
        else:
            role = "balanced"

        records.append({
            "cell_type": str(ct),
            "sender_score": round(sender_score, 4),
            "receiver_score": round(receiver_score, 4),
            "hub_score": round(hub_score, 4),
            "dominant_role": role,
            "n_outgoing": n_out,
            "n_incoming": n_in,
        })

    return pd.DataFrame(records).sort_values("hub_score", ascending=False).reset_index(drop=True)


def run_communication(adata, *, method: str = "builtin", cell_type_key: str = "leiden", species: str = "human", n_perms: int = 100) -> dict:
    """Run cell-cell communication analysis.

    All methods use ``adata.X`` (log-normalized expression).  Do not pass
    raw counts or z-scored matrices.  Cell type labels must be present in
    ``adata.obs[cell_type_key]``.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {SUPPORTED_METHODS}")
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not in adata.obs")

    n_cells, n_genes = adata.n_obs, adata.n_vars
    cell_types = sorted(adata.obs[cell_type_key].unique().tolist(), key=str)

    dispatch = {
        "builtin": lambda: _run_builtin(adata, cell_type_key=cell_type_key, species=species, n_perms=n_perms),
        "liana": lambda: _run_liana(adata, cell_type_key=cell_type_key, species=species, n_perms=n_perms),
        "cellphonedb": lambda: _run_cellphonedb(adata, cell_type_key=cell_type_key, species=species, n_perms=n_perms),
        "fastccc": lambda: _run_fastccc(adata, cell_type_key=cell_type_key, species=species),
    }
    lr_df = dispatch[method]()
    sig_df = lr_df[lr_df["pvalue"] < 0.05] if not lr_df.empty else lr_df

    # Pathway-level aggregation and signaling role classification
    pathway_df = aggregate_by_pathway(sig_df if not sig_df.empty else lr_df)
    roles_df = classify_signaling_roles(sig_df if not sig_df.empty else lr_df)

    return {
        "n_cells": n_cells, "n_genes": n_genes, "n_cell_types": len(cell_types),
        "cell_types": cell_types, "cell_type_key": cell_type_key, "method": method,
        "species": species, "n_interactions_tested": len(lr_df), "n_significant": len(sig_df),
        "lr_df": lr_df, "top_df": lr_df.head(50) if not lr_df.empty else lr_df,
        "pathway_df": pathway_df, "signaling_roles_df": roles_df,
    }
