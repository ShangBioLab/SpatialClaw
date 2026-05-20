"""Target discovery and prioritization from spatial findings.

Provides methods for summarizing spatial omics discoveries into exploratory
biomarkers, research target candidates, and drug annotation references.

Supported methods:
  - biomarker_discovery: Identify spatial biomarkers
  - target_prioritization: Rank research target candidates
  - drug_matching: Annotate targets with static drug references
  - actionable_report: Generate research-use evidence summary

Input convention:
  - AnnData with spatial analysis results (domains, niches, DE results)
  - Optional: external knowledge bases (Open Targets, ChEMBL)

Research-use boundary:
  - Outputs are exploratory analytical summaries only.
  - They do not recommend therapies, determine clinical actionability, diagnose
    disease, or provide clinical decision support.

Usage::

    from skills.spatial._lib.targets import (
        discover_biomarkers,
        prioritize_targets,
        SUPPORTED_METHODS,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .dependency_manager import require

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = (
    "biomarker_discovery", "target_prioritization", "drug_matching", "actionable_report",
)


def discover_biomarkers(
    adata,
    *,
    domain_key: str = "spatial_domain",
    condition_key: str | None = None,
    n_top_genes: int = 50,
    species: str = "human",
) -> dict:
    """Discover spatial biomarkers from domain/niche-specific genes.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with domain labels.
    domain_key : str
        Column with domain/niche labels.
    condition_key : str, optional
        Column with condition labels for comparison.
    n_top_genes : int
        Number of top genes per domain.
    species : str
        Species for gene annotation.

    Returns
    -------
    dict
        Biomarker discovery results.
    """
    import scanpy as sc

    if domain_key not in adata.obs.columns:
        raise ValueError(f"Domain key '{domain_key}' not found in adata.obs")

    logger.info("Discovering biomarkers from %s", domain_key)

    domains = adata.obs[domain_key].unique()

    if "rank_genes_groups" not in adata.uns or adata.uns["rank_genes_groups"]["params"]["groupby"] != domain_key:
        sc.tl.rank_genes_groups(adata, domain_key, method="wilcoxon", n_genes=n_top_genes)

    biomarkers = {}
    for domain in domains:
        markers_df = sc.get.rank_genes_groups_df(adata, group=str(domain))
        top_markers = markers_df.head(n_top_genes)

        biomarkers[domain] = {
            "genes": top_markers["names"].tolist(),
            "scores": top_markers["scores"].tolist() if "scores" in top_markers.columns else [],
            "pvals": top_markers["pvals"].tolist() if "pvals" in top_markers.columns else [],
            "logfoldchanges": top_markers["logfoldchanges"].tolist() if "logfoldchanges" in top_markers.columns else [],
        }

    if condition_key is not None and condition_key in adata.obs.columns:
        condition_biomarkers = {}
        conditions = adata.obs[condition_key].unique()

        for cond in conditions:
            mask = adata.obs[condition_key] == cond
            adata_sub = adata[mask].copy()

            if adata_sub.n_obs > 10:
                sc.tl.rank_genes_groups(adata_sub, domain_key, method="wilcoxon", n_genes=n_top_genes)
                condition_biomarkers[cond] = {}
                for domain in domains:
                    if domain in adata_sub.obs[domain_key].unique():
                        markers_df = sc.get.rank_genes_groups_df(adata_sub, group=str(domain))
                        condition_biomarkers[cond][domain] = markers_df.head(n_top_genes)["names"].tolist()

        biomarkers["condition_specific"] = condition_biomarkers

    adata.uns["spatial_biomarkers"] = biomarkers

    return {
        "n_domains": len(domains),
        "domains": list(domains),
        "n_biomarkers_per_domain": {d: len(b["genes"]) for d, b in biomarkers.items() if d != "condition_specific"},
        "species": species,
    }


def prioritize_targets(
    adata,
    *,
    biomarker_key: str = "spatial_biomarkers",
    criteria: list[str] | None = None,
    species: str = "human",
) -> dict:
    """Prioritize research target candidates from biomarkers.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with biomarker results.
    biomarker_key : str
        Key in adata.uns for biomarker results.
    criteria : list, optional
        Prioritization criteria: druggability, novelty, literature_relevance.
    species : str
        Species for target annotation.

    Returns
    -------
    dict
        Target prioritization results.
    """
    if criteria is None:
        criteria = ["druggability", "spatial_specificity", "clinical_relevance"]

    logger.info("Prioritizing targets using criteria: %s", criteria)

    if biomarker_key not in adata.uns:
        raise ValueError(f"Biomarker key '{biomarker_key}' not found in adata.uns. Run discover_biomarkers first.")

    biomarkers = adata.uns[biomarker_key]

    all_genes = []
    for domain, data in biomarkers.items():
        if domain == "condition_specific":
            continue
        all_genes.extend(data.get("genes", []))

    gene_counts = pd.Series(all_genes).value_counts()

    druggable_targets = _get_druggable_genes(species)
    literature_genes = _get_literature_relevant_genes(species)

    target_scores = []
    for gene, count in gene_counts.items():
        score = {
            "gene": gene,
            "spatial_frequency": count,
            "druggability": 1.0 if gene in druggable_targets else 0.0,
            "literature_relevance": 1.0 if gene in literature_genes else 0.0,
        }

        score["priority_score"] = (
            0.4 * score["spatial_frequency"] / gene_counts.max() +
            0.4 * score["druggability"] +
            0.2 * score["literature_relevance"]
        )

        target_scores.append(score)

    target_df = pd.DataFrame(target_scores).sort_values("priority_score", ascending=False)
    adata.uns["prioritized_targets"] = target_df

    return {
        "n_candidates": len(target_df),
        "n_druggable": int(target_df["druggability"].sum()),
        "n_literature_relevant": int(target_df["literature_relevance"].sum()),
        "top_targets": target_df.head(20)["gene"].tolist(),
        "criteria": criteria,
        "research_use_only": True,
    }


def _get_druggable_genes(species: str) -> set:
    """Return set of known druggable genes."""
    common_druggable = {
        "EGFR", "ERBB2", "ALK", "ROS1", "BRAF", "KRAS", "NRAS",
        "PIK3CA", "MTOR", "AKT1", "CDK4", "CDK6", "PARP1",
        "PDGFRA", "KIT", "FLT3", "VEGFA", "VEGFR2", "FGFR1", "FGFR2",
        "MET", "RET", "NTRK1", "NTRK2", "NTRK3", "IDH1", "IDH2",
        "BCL2", "MCL1", "BRD4", "HDAC1", "HDAC2", "DNMT3A",
        "JAK1", "JAK2", "STAT3", "BTK", "SYK", "PI3K",
    }
    return common_druggable


def _get_literature_relevant_genes(species: str) -> set:
    """Return genes commonly discussed in oncology and immunology literature."""
    literature_genes = {
        "TP53", "BRCA1", "BRCA2", "MSH2", "MSH6", "MLH1", "PMS2",
        "PTEN", "RB1", "APC", "KRAS", "NRAS", "BRAF", "EGFR",
        "ERBB2", "ALK", "ROS1", "MET", "RET", "NTRK1",
        "PD1", "PDL1", "CTLA4", "LAG3", "TIM3",
        "CD19", "CD20", "CD22", "BCMA",
    }
    return literature_genes


def match_drugs(
    adata,
    *,
    target_key: str = "prioritized_targets",
    n_top_targets: int = 20,
    species: str = "human",
) -> dict:
    """Match prioritized targets to candidate drugs.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with prioritized targets.
    target_key : str
        Key in adata.uns for target results.
    n_top_targets : int
        Number of top targets to match.
    species : str
        Species for drug matching.

    Returns
    -------
    dict
        Drug matching results.
    """
    logger.info("Matching drugs for top %d targets", n_top_targets)

    if target_key not in adata.uns:
        raise ValueError(f"Target key '{target_key}' not found in adata.uns. Run prioritize_targets first.")

    target_df = adata.uns[target_key]
    top_targets = target_df.head(n_top_targets)["gene"].tolist()

    drug_matches = {}
    known_drugs = _get_known_drug_targets()

    for target in top_targets:
        if target in known_drugs:
            drug_matches[target] = known_drugs[target]
        else:
            drug_matches[target] = {
                "approved_drugs": [],
                "investigational_drugs": [],
                "repurposing_candidates": _suggest_repurposing(target),
            }

    adata.uns["drug_matches"] = drug_matches

    return {
        "n_targets_matched": len(drug_matches),
        "n_with_approved_drugs": sum(1 for d in drug_matches.values() if d.get("approved_drugs")),
        "matches": drug_matches,
    }


def _get_known_drug_targets() -> dict:
    """Return dictionary of known drug-target matches."""
    return {
        "EGFR": {
            "approved_drugs": ["erlotinib", "gefitinib", "afatinib", "osimertinib"],
            "investigational_drugs": [],
        },
        "ERBB2": {
            "approved_drugs": ["trastuzumab", "pertuzumab", "lapatinib", "neratinib"],
            "investigational_drugs": [],
        },
        "ALK": {
            "approved_drugs": ["crizotinib", "ceritinib", "alectinib", "lorlatinib"],
            "investigational_drugs": [],
        },
        "BRAF": {
            "approved_drugs": ["vemurafenib", "dabrafenib"],
            "investigational_drugs": [],
        },
        "KRAS": {
            "approved_drugs": ["sotorasib", "adagrasib"],
            "investigational_drugs": [],
        },
        "PIK3CA": {
            "approved_drugs": ["alpelisib"],
            "investigational_drugs": [],
        },
        "MTOR": {
            "approved_drugs": ["everolimus", "temsirolimus"],
            "investigational_drugs": [],
        },
        "CDK4": {
            "approved_drugs": ["palbociclib", "ribociclib", "abemaciclib"],
            "investigational_drugs": [],
        },
        "PARP1": {
            "approved_drugs": ["olaparib", "rucaparib", "niraparib"],
            "investigational_drugs": [],
        },
        "MET": {
            "approved_drugs": ["crizotinib", "capmatinib", "tepotinib"],
            "investigational_drugs": [],
        },
        "RET": {
            "approved_drugs": ["selpercatinib", "pralsetinib"],
            "investigational_drugs": [],
        },
        "NTRK1": {
            "approved_drugs": ["larotrectinib", "entrectinib"],
            "investigational_drugs": [],
        },
        "JAK1": {
            "approved_drugs": ["ruxolitinib", "tofacitinib"],
            "investigational_drugs": [],
        },
        "JAK2": {
            "approved_drugs": ["ruxolitinib", "fedratinib"],
            "investigational_drugs": [],
        },
        "BTK": {
            "approved_drugs": ["ibrutinib", "acalabrutinib", "zanubrutinib"],
            "investigational_drugs": [],
        },
        "BCL2": {
            "approved_drugs": ["venetoclax"],
            "investigational_drugs": [],
        },
        "BRD4": {
            "approved_drugs": [],
            "investigational_drugs": ["birabresib", "azd5153"],
        },
    }


def _suggest_repurposing(target: str) -> list:
    """Return static drug annotation references for exploratory review."""
    repurposing_db = {
        "STAT3": ["napabucasin", "bardoxolone"],
        "HDAC1": ["vorinostat", "panobinostat"],
        "DNMT3A": ["azacitidine", "decitabine"],
        "SYK": ["fostamatinib"],
    }
    return repurposing_db.get(target, [])


def generate_actionable_report(
    adata,
    *,
    biomarker_key: str = "spatial_biomarkers",
    target_key: str = "prioritized_targets",
    drug_key: str = "drug_matches",
) -> dict:
    """Generate a research-use evidence summary from all findings.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data with all analysis results.
    biomarker_key : str
        Key for biomarker results.
    target_key : str
        Key for target results.
    drug_key : str
        Key for drug matching results.

    Returns
    -------
    dict
        Exploratory evidence summary.
    """
    logger.info("Generating research-use evidence summary...")

    report = {
        "summary": {},
        "biomarkers": {},
        "targets": {},
        "drugs": {},
        "research_follow_up": [],
        "research_use_only": True,
        "interpretation": "exploratory summary; not clinical actionability or treatment guidance",
    }

    if biomarker_key in adata.uns:
        biomarkers = adata.uns[biomarker_key]
        report["biomarkers"] = {
            "n_domains": len([k for k in biomarkers.keys() if k != "condition_specific"]),
            "total_unique_genes": len(set(
                g for d in biomarkers.values() if isinstance(d, dict) and "genes" in d
                for g in d["genes"]
            )),
        }

    if target_key in adata.uns:
        targets = adata.uns[target_key]
        report["targets"] = {
            "n_candidates": len(targets),
            "top_5_targets": targets.head(5)["gene"].tolist() if hasattr(targets, "head") else [],
        }

    if drug_key in adata.uns:
        drugs = adata.uns[drug_key]
        report["drugs"] = {
            "n_targets_with_matches": len(drugs),
            "approved_drugs_available": [
                t for t, d in drugs.items() if d.get("approved_drugs")
            ],
        }

    report["research_follow_up"] = [
        "Validate top biomarkers in independent cohort",
        "Review druggable target annotations in external knowledge bases",
        "Prioritize orthogonal experimental validation before any translational claim",
        "Compare immune-rich niches with study-specific response metadata where available",
    ]

    adata.uns["actionable_report"] = report

    return report


def run_targets(
    adata,
    *,
    method: str = "biomarker_discovery",
    domain_key: str = "spatial_domain",
    **kwargs,
) -> dict:
    """Run target discovery pipeline.

    Parameters
    ----------
    adata : AnnData
        Spatial omics data.
    method : str
        Analysis method.
    domain_key : str
        Column with domain labels.
    **kwargs
        Method-specific parameters.

    Returns
    -------
    dict
        Analysis results.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Choose from: {SUPPORTED_METHODS}")

    n_cells, n_genes = adata.n_obs, adata.n_vars
    logger.info("Target analysis: %d cells, method=%s", n_cells, method)

    dispatch: dict[str, Any] = {
        "biomarker_discovery": lambda: discover_biomarkers(
            adata, domain_key=domain_key, **kwargs
        ),
        "target_prioritization": lambda: prioritize_targets(adata, **kwargs),
        "drug_matching": lambda: match_drugs(adata, **kwargs),
        "actionable_report": lambda: generate_actionable_report(adata, **kwargs),
    }

    result = dispatch[method]()
    result["n_cells"] = n_cells
    result["n_genes"] = n_genes
    result["method"] = method

    return result
