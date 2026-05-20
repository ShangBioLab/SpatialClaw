"""Spatial-analysis-specific utilities for SpatialClaw skills.

This package contains all reusable analysis logic for the spatial domain,
organized by analysis type. Each skill script should import its core
functions from the corresponding _lib module.

Modules
-------
adata_utils      : AnnData helper functions (spatial key detection, metadata)
annotation       : Cell type annotation (marker-based, Tangram, scANVI, CellAssign)
cnv              : Copy number variation inference (inferCNVpy)
communication    : Cell-cell communication (built-in, LIANA, CellPhoneDB, FastCCC)
condition_comparison : Pseudobulk condition comparison (PyDESeq2, Wilcoxon)
de               : Differential expression (rank_genes_groups, PyDESeq2)
deconvolution    : Cell type deconvolution (Python-native methods)
dependency_manager : Lazy dependency import helpers
domain_identification : Spatial domain identification (6 algorithms)
enrichment       : Pathway enrichment (Enrichr, GSEA, ssGSEA)
exceptions       : Domain-specific exception classes
svg_detection    : Spatially variable gene detection (Moran's, SpatialDE, FlashS)
integration      : Batch integration (Harmony, BBKNN, Scanorama)
loader           : Multi-platform spatial data loader
preprocessing    : QC, normalization, and embedding pipeline
registration     : Multi-slice spatial registration (PASTE)
statistics       : Spatial statistics (10 analysis types)
trajectory       : Trajectory inference (DPT, CellRank)
velocity         : RNA velocity (scVelo, veloVI)
viz              : Unified visualization package
figure_io        : Figure saving utilities

Downstream spatial task modules
-------------------------------
histology        : Histology tissue, nucleus, and cell image analysis
morphology       : Morphology feature extraction and image-expression fusion
niches           : Spatial niche and tissue microenvironment analysis
oncology         : Tumor ecosystem and clinical spatial analysis helpers
regulation       : Spatial TF activity and regulatory network inference
sc2spatial       : Reference-to-spatial mapping and label transfer
targets          : Spatial biomarker and therapeutic target prioritization
tls              : Tertiary lymphoid structure detection and characterization
translation      : Cross-modal image-to-expression and super-resolution helpers
wsi              : Whole-slide image loading, tiling, and MIL helpers
"""
