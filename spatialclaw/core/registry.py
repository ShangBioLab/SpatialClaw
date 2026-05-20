"""SpatialClaw spatial skill registry.

The registry intentionally exposes only spatial analysis skills.  Older
multi-domain registries were removed during the spatial-only architecture
cleanup so CLI discovery cannot point at deleted skill trees.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from spatialclaw.core.lazy_metadata import LazySkillMetadata
from spatialclaw.paths import SKILLS_DIR

logger = logging.getLogger(__name__)

SPATIAL_SKILLS_DIR = SKILLS_DIR / "spatial"

API_ONLY_SPATIAL_SKILLS = frozenset(
    {
        "spatial-histology",
        "spatial-morphology",
        "spatial-niches",
        "spatial-oncology",
        "spatial-regulation",
        "spatial-sc2spatial",
        "spatial-targets",
        "spatial-tls",
        "spatial-translation",
        "spatial-wsi",
    }
)


class SpatialSkillRegistry:
    """Manage stable spatial skill definitions and spatial-only discovery."""

    def __init__(self):
        self.skills = _SPATIAL_SKILLS.copy()
        self.domains = _SPATIAL_DOMAINS.copy()
        self.domains["spatial"]["skill_count"] = len(self.skills)
        self._loaded = False
        self.lazy_skills: dict[str, LazySkillMetadata] = {}

    def load_all(self, skills_dir: Path | None = None) -> None:
        """Load hardcoded spatial skills and CLI-capable spatial additions."""
        if self._loaded:
            return

        target_dir = _resolve_spatial_scan_dir(skills_dir)
        if not target_dir.exists():
            return

        for skill_path in sorted(target_dir.iterdir()):
            if not _is_discoverable_skill_dir(skill_path):
                continue

            script_path = _discover_cli_script(skill_path)
            if script_path is None:
                continue

            already_registered = any(
                info.get("script") == script_path for info in self.skills.values()
            )
            if already_registered:
                continue

            alias = _canonical_skill_alias(skill_path.name)
            self.skills[alias] = {
                "domain": "spatial",
                "alias": alias,
                "script": script_path,
                "demo_args": ["--demo"],
                "description": f"Spatial analysis skill: {alias}",
                "allowed_extra_flags": set(),
                "saves_h5ad": False,
            }
            logger.debug("Discovered spatial skill: %s", alias)

        self.domains["spatial"]["skill_count"] = len(self.skills)
        self._loaded = True

    def load_lightweight(self, skills_dir: Path | None = None) -> None:
        """Load SKILL.md frontmatter for spatial skills only."""
        target_dir = _resolve_spatial_scan_dir(skills_dir)
        if not target_dir.exists():
            return

        for skill_path in sorted(target_dir.iterdir()):
            if not _is_metadata_skill_dir(skill_path):
                continue

            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue

            lazy = LazySkillMetadata(skill_path)
            skill_key = _canonical_skill_alias(skill_path.name)
            self.lazy_skills[skill_key] = lazy


def _resolve_spatial_scan_dir(skills_dir: Path | None) -> Path:
    if skills_dir is None:
        return SPATIAL_SKILLS_DIR
    skills_dir = Path(skills_dir)
    if skills_dir.name == "spatial":
        return skills_dir
    return skills_dir / "spatial"


def _is_discoverable_skill_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and not path.name.startswith((".", "__", "_"))
        and path.name not in API_ONLY_SPATIAL_SKILLS
    )


def _is_metadata_skill_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith((".", "__", "_"))


def _canonical_skill_alias(directory_name: str) -> str:
    directory_name = directory_name.lower()
    for alias, info in _SPATIAL_SKILLS.items():
        script = info.get("script")
        if isinstance(script, Path) and script.parent.name.lower() == directory_name:
            return alias
    return directory_name


def _discover_cli_script(skill_path: Path) -> Path | None:
    """Return the conventional script only when it has a CLI entrypoint."""
    script_path = skill_path / f"{skill_path.name.replace('-', '_')}.py"
    if not script_path.exists():
        return None
    try:
        source = script_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if "argparse.ArgumentParser" not in source or "__main__" not in source:
        return None
    return script_path


_SPATIAL_DOMAINS = {
    "spatial": {
        "name": "Spatial Transcriptomics",
        "primary_data_types": ["h5ad", "h5", "zarr"],
        "skill_count": 0,
    },
}


_SPATIAL_SKILLS: dict[str, dict[str, Any]] = {
    "spatial-preprocessing": {
        "domain": "spatial",
        "alias": "spatial-preprocessing",
        "script": SPATIAL_SKILLS_DIR / "spatial-preprocessing" / "spatial_preprocessing.py",
        "demo_args": ["--demo"],
        "description": "Spatial data QC, normalization, HVG, PCA/UMAP, Leiden clustering",
        "allowed_extra_flags": {
            "--data-type",
            "--species",
            "--min-genes",
            "--min-cells",
            "--max-mt-pct",
            "--max-genes",
            "--tissue",
            "--n-top-hvg",
            "--n-pcs",
            "--n-neighbors",
            "--leiden-resolution",
            "--resolutions",
        },
        "saves_h5ad": True,
    },
    "spatial-domain-identification": {
        "domain": "spatial",
        "alias": "spatial-domain-identification",
        "script": SPATIAL_SKILLS_DIR / "spatial-domain-identification" / "spatial_domain_identification.py",
        "demo_args": ["--demo"],
        "description": "Tissue region/niche identification (Leiden, Louvain, SpaGCN, STAGATE, GraphST, BANKSY)",
        "allowed_extra_flags": {
            "--method",
            "--n-domains",
            "--resolution",
            "--spatial-weight",
            "--rad-cutoff",
            "--k-nn",
            "--lambda-param",
            "--refine",
        },
        "saves_h5ad": True,
    },
    "spatial-cell-annotation": {
        "domain": "spatial",
        "alias": "spatial-cell-annotation",
        "script": SPATIAL_SKILLS_DIR / "spatial-cell-annotation" / "spatial_cell_annotation.py",
        "demo_args": ["--demo"],
        "description": "Cell type annotation (marker_based, Tangram, scANVI, CellAssign)",
        "allowed_extra_flags": {
            "--method",
            "--reference",
            "--cell-type-key",
            "--cluster-key",
            "--species",
            "--batch-key",
            "--layer",
            "--model",
        },
        "path_extra_flags": {"--reference", "--model"},
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-deconvolution": {
        "domain": "spatial",
        "alias": "spatial-deconvolution",
        "script": SPATIAL_SKILLS_DIR / "spatial-deconvolution" / "spatial_deconvolution.py",
        "description": "Spatial cell-type deconvolution (Tangram, Stereoscope, GraphST)",
        "allowed_extra_flags": {"--method", "--reference", "--cell-type-key", "--n-epochs", "--no-gpu"},
        "path_extra_flags": {"--reference"},
        "llm_argument_schema": {
            "method": {
                "cli_flag": "--method",
                "allowed_values": ("tangram", "stereoscope", "graphst"),
                "value_aliases": {
                    "graph-st": "graphst",
                    "graph st": "graphst",
                },
                "drop_invalid": True,
            },
        },
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-statistics": {
        "domain": "spatial",
        "alias": "spatial-statistics",
        "script": SPATIAL_SKILLS_DIR / "spatial-statistics" / "spatial_statistics.py",
        "demo_args": ["--demo"],
        "description": "Spatial statistics (Moran's I, Geary's C, Getis-Ord Gi*, Ripley, neighborhood enrichment)",
        "allowed_extra_flags": {
            "--analysis-type",
            "--cluster-key",
            "--genes",
            "--n-top-genes",
        },
        "multi_value_extra_flags": {"--genes"},
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-svg-detection": {
        "domain": "spatial",
        "alias": "spatial-svg-detection",
        "script": SPATIAL_SKILLS_DIR / "spatial-svg-detection" / "spatial_svg_detection.py",
        "demo_args": ["--demo"],
        "description": "Spatially variable genes (Moran's I, SpatialDE, FlashS)",
        "allowed_extra_flags": {"--method", "--n-top-genes", "--fdr-threshold"},
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-de": {
        "domain": "spatial",
        "alias": "spatial-de",
        "script": SPATIAL_SKILLS_DIR / "spatial-de" / "spatial_de.py",
        "demo_args": ["--demo"],
        "description": "Differential expression (Wilcoxon, t-test, PyDESeq2 pseudobulk)",
        "allowed_extra_flags": {
            "--groupby",
            "--group1",
            "--group2",
            "--method",
            "--n-top-genes",
        },
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-condition-comparison": {
        "domain": "spatial",
        "alias": "spatial-condition-comparison",
        "script": SPATIAL_SKILLS_DIR / "spatial-condition-comparison" / "spatial_condition_comparison.py",
        "demo_args": ["--demo"],
        "description": "Condition comparison with pseudobulk DESeq2-style statistics",
        "allowed_extra_flags": {
            "--method",
            "--condition-key",
            "--sample-key",
            "--reference-condition",
        },
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-cell-communication": {
        "domain": "spatial",
        "alias": "spatial-cell-communication",
        "script": SPATIAL_SKILLS_DIR / "spatial-cell-communication" / "spatial_cell_communication.py",
        "demo_args": ["--demo"],
        "description": "Cell-cell communication (built-in, LIANA+, CellPhoneDB, FastCCC)",
        "allowed_extra_flags": {"--method", "--species", "--cell-type-key", "--n-perms"},
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-velocity": {
        "domain": "spatial",
        "alias": "spatial-velocity",
        "script": SPATIAL_SKILLS_DIR / "spatial-velocity" / "spatial_velocity.py",
        "demo_args": ["--demo"],
        "description": "RNA velocity and cellular dynamics (scVelo, VeloVI)",
        "allowed_extra_flags": {"--method"},
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-trajectory": {
        "domain": "spatial",
        "alias": "spatial-trajectory",
        "script": SPATIAL_SKILLS_DIR / "spatial-trajectory" / "spatial_trajectory.py",
        "demo_args": ["--demo"],
        "description": "Trajectory inference (CellRank, Palantir, DPT)",
        "allowed_extra_flags": {"--method", "--root-cell", "--n-states"},
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-enrichment": {
        "domain": "spatial",
        "alias": "spatial-enrichment",
        "script": SPATIAL_SKILLS_DIR / "spatial-enrichment" / "spatial_enrichment.py",
        "demo_args": ["--demo"],
        "description": "Pathway enrichment analysis: GO, KEGG, GSEA, ORA, spatial activity score",
        "allowed_extra_flags": {
            "--analysis-type",
            "--go-ontology",
            "--gene-sets",
            "--groupby",
            "--group",
            "--organism",
            "--de-method",
            "--n-top-genes",
            "--logfc-threshold",
            "--pval-threshold",
            "--top-n",
            "--gene-set-name",
            "--no-save-h5ad",
        },
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-cnv": {
        "domain": "spatial",
        "alias": "spatial-cnv",
        "script": SPATIAL_SKILLS_DIR / "spatial-cnv" / "spatial_cnv.py",
        "demo_args": ["--demo"],
        "description": "Copy number variation inference (inferCNVpy)",
        "allowed_extra_flags": {"--method", "--reference-key", "--reference-cat", "--window-size", "--step"},
        "multi_value_extra_flags": {"--reference-cat"},
        "requires_preprocessed": True,
        "saves_h5ad": True,
    },
    "spatial-integration": {
        "domain": "spatial",
        "alias": "spatial-integration",
        "script": SPATIAL_SKILLS_DIR / "spatial-integration" / "spatial_integration.py",
        "demo_args": ["--demo"],
        "description": "Spatial multi-sample integration (Harmony, BBKNN, Scanorama, scVI)",
        "allowed_extra_flags": {"--method", "--batch-key"},
        "saves_h5ad": True,
    },
    "spatial-omics-integrate": {
        "domain": "spatial",
        "alias": "spatial-omics-integrate",
        "script": SPATIAL_SKILLS_DIR / "spatial-omics-integrate" / "spatial_omics_integrate.py",
        "demo_args": ["--demo"],
        "description": "Spatial paired-modality integration using SpatialGlue or SpaDDM",
        "requires_input": True,
        "input_mode": "file",
        "required_extra_inputs": [
            {
                "flag": "--omics2",
                "kind": "file",
                "label": "secondary spatial modality file",
            }
        ],
        "input_examples": {
            "primary": "rna.h5ad",
            "--omics2": "protein.h5ad",
            "usage": "--input rna.h5ad --omics2 protein.h5ad --method spatialglue",
        },
        "allowed_extra_flags": {
            "--method",
            "--omics2",
            "--omics1-type",
            "--omics2-type",
            "--data-type",
            "--clustering-method",
            "--n-hvg",
            "--n-clusters",
            "--n-epochs",
            "--n-latent",
            "--random-seed",
            "--device",
        },
        "llm_argument_schema": {
            "method": {
                "cli_flag": "--method",
                "allowed_values": ("spatialglue", "spaddm"),
                "value_aliases": {
                    "spatial-glue": "spatialglue",
                    "glue": "spatialglue",
                    "spa-ddm": "spaddm",
                    "spatialddm": "spaddm",
                    "spatial-ddm": "spaddm",
                },
                "drop_invalid": True,
            },
            "platform": {
                "cli_flag": "--data-type",
                "allowed_values": (
                    "auto",
                    "10x",
                    "Stereo-CITE-seq",
                    "SPOTS",
                    "spatial-epigenome",
                    "Spatial-epigenome-transcriptome",
                    "Visium CytAssist",
                ),
                "value_aliases": {
                    "spatial": "auto",
                    "visium": "auto",
                },
                "drop_invalid": True,
            },
        },
        "saves_h5ad": True,
    },
    "spatial-modality-integrate": {
        "domain": "spatial",
        "alias": "spatial-modality-integrate",
        "script": SPATIAL_SKILLS_DIR / "spatial-modality-integrate" / "spatial_modality_integrate.py",
        "description": "Spatial domain identification with DeepST or PearlST from directory-based inputs",
        "requires_input": True,
        "allows_alternative_inputs": True,
        "alternative_input_flags": {"--input-list"},
        "input_mode": "directory",
        "input_examples": {
            "primary": "/data/DLPFC/151673",
            "--input-list": ["data/DLPFC/151673", "data/DLPFC/151674"],
        },
        "allowed_extra_flags": {
            "--input-list",
            "--mode",
            "--method",
            "--platform",
            "--n-domains",
            "--n-top-genes",
            "--batch-key",
            "--pre-epochs",
            "--epochs",
            "--pca-components",
            "--use-morphological",
            "--no-morphological",
            "--ground-truth",
            "--simclr-features",
            "--simclr-model",
            "--device",
            "--random-seed",
            "--use-gpu",
            "--no-gpu",
        },
        "path_extra_flags": {"--ground-truth", "--simclr-features", "--simclr-model"},
        "llm_argument_schema": {
            "method": {
                "cli_flag": "--method",
                "allowed_values": ("deepst", "pearlst"),
                "value_aliases": {
                    "deep-st": "deepst",
                    "pearl-st": "pearlst",
                    "pearist": "pearlst",
                },
                "drop_invalid": True,
            },
            "n_epochs": {
                "cli_flag": "--epochs",
            },
            "platform": {
                "cli_flag": "--platform",
                "allowed_values": (
                    "Visium",
                    "Stereo-seq",
                    "Slide-seq",
                    "Slide-seqV2",
                    "MERFISH",
                    "STARmap",
                ),
                "value_aliases": {
                    "slide-seqv2": "Slide-seqV2",
                    "slideseqv2": "Slide-seqV2",
                    "stereoseq": "Stereo-seq",
                },
                "drop_invalid": True,
            },
        },
        "saves_h5ad": True,
    },
    "spatial-registration": {
        "domain": "spatial",
        "alias": "spatial-registration",
        "script": SPATIAL_SKILLS_DIR / "spatial-registration" / "spatial_registration.py",
        "demo_args": ["--demo"],
        "description": "Spatial registration / slice alignment (PASTE)",
        "allowed_extra_flags": {"--method", "--reference-slice"},
        "path_extra_flags": {"--reference-slice"},
        "saves_h5ad": True,
    },
    "spatial-multi-sample-integration": {
        "domain": "spatial",
        "alias": "spatial-multi-sample-integration",
        "script": SPATIAL_SKILLS_DIR / "spatial-multi-sample-integration" / "spatial_multi_sample_integration.py",
        "demo_args": ["--demo"],
        "description": "Spatial multi-sample integration and batch alignment",
        "allowed_extra_flags": {"--method", "--batch-key"},
        "saves_h5ad": True,
    },
    "spatial-visualization": {
        "domain": "spatial",
        "alias": "spatial-visualization",
        "script": SPATIAL_SKILLS_DIR / "spatial-visualization" / "spatial_visualization.py",
        "demo_args": ["--demo"],
        "description": "Spatial visualization and publication-ready plots",
        "allowed_extra_flags": {"--mode", "--feature", "--genes", "--cluster-key", "--batch-key"},
        "multi_value_extra_flags": {"--genes"},
        "saves_h5ad": False,
    },
    "spatial-orchestrator": {
        "domain": "spatial",
        "alias": "spatial-orchestrator",
        "script": SPATIAL_SKILLS_DIR / "spatial-orchestrator" / "spatial_orchestrator.py",
        "demo_args": ["--demo"],
        "description": "Spatial query routing and spatial pipeline orchestration",
        "allowed_extra_flags": {"--query", "--pipeline", "--list-skills", "--timeout"},
        "input_optional_flags": {"--query", "--list-skills"},
        "saves_h5ad": False,
    },
}


_GLOBAL_LLM_ARGUMENT_SCHEMA: dict[str, dict[str, Any]] = {
    "output_dir": {
        "aliases": ("save_dir", "save_path", "output_path"),
    },
    "method": {
        "cli_flag": "--method",
    },
    "n_epochs": {
        "cli_flag": "--n-epochs",
        "min_value": 1,
        "drop_invalid": True,
    },
    "device": {
        "cli_flag": "--device",
    },
    "platform": {
        "aliases": ("data_type",),
        "cli_flag": "--data-type",
    },
}


def _has_llm_arg_value(value: Any) -> bool:
    return value is not None and value != ""


def _merge_llm_argument_schema(skill_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    merged = {
        key: dict(spec)
        for key, spec in _GLOBAL_LLM_ARGUMENT_SCHEMA.items()
    }
    for canonical, override in (skill_info.get("llm_argument_schema") or {}).items():
        base = dict(merged.get(canonical, {}))
        base_aliases = list(base.get("aliases", ()))
        override_aliases = list(override.get("aliases", ()))
        if base_aliases or override_aliases:
            base["aliases"] = tuple(dict.fromkeys(base_aliases + override_aliases))
        for key, value in override.items():
            if key == "aliases":
                continue
            base[key] = value
        merged[canonical] = base
    return merged


def _resolve_llm_arg_value(args: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name not in args:
            continue
        value = args.get(name)
        if isinstance(value, str):
            value = value.strip()
        if _has_llm_arg_value(value):
            return value
    return None


def _normalize_llm_arg_value(value: Any, spec: dict[str, Any]) -> Any:
    if isinstance(value, str):
        value = value.strip()

    value_aliases = spec.get("value_aliases") or {}
    if isinstance(value, str) and value_aliases:
        if value in value_aliases:
            value = value_aliases[value]
        else:
            lower_aliases = {
                str(alias).lower(): normalized
                for alias, normalized in value_aliases.items()
            }
            value = lower_aliases.get(value.lower(), value)

    allowed_values = spec.get("allowed_values")
    if allowed_values:
        allowed_lookup = {str(item).lower(): item for item in allowed_values}
        if isinstance(value, str) and value.lower() in allowed_lookup:
            value = allowed_lookup[value.lower()]
        if value not in set(allowed_values) and spec.get("drop_invalid", False):
            return None

    min_value = spec.get("min_value")
    max_value = spec.get("max_value")
    if min_value is not None or max_value is not None:
        numeric_value = value
        if isinstance(value, str):
            try:
                numeric_value = int(value)
            except ValueError:
                try:
                    numeric_value = float(value)
                except ValueError:
                    numeric_value = value
        if isinstance(numeric_value, (int, float)):
            if min_value is not None and numeric_value < min_value:
                return None if spec.get("drop_invalid", False) else value
            if max_value is not None and numeric_value > max_value:
                return None if spec.get("drop_invalid", False) else value
            value = numeric_value

    return value


def _skill_allows_llm_cli_flag(skill_info: dict[str, Any], cli_flag: str) -> bool:
    allowed_extra_flags = skill_info.get("allowed_extra_flags")
    if not allowed_extra_flags:
        return False
    return cli_flag in allowed_extra_flags


def normalize_skill_llm_args(skill_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM tool-call arguments using spatial registry metadata."""
    skill_info = registry.skills.get(skill_name, {}) if skill_name else {}
    schema = _merge_llm_argument_schema(skill_info)
    normalized = dict(args)

    for canonical, spec in schema.items():
        aliases = list(spec.get("aliases", ()))
        value = _resolve_llm_arg_value(normalized, [canonical, *aliases])
        if value is None:
            continue
        for alias in aliases:
            normalized.pop(alias, None)
        value = _normalize_llm_arg_value(value, spec)
        if value is None:
            normalized.pop(canonical, None)
            continue
        normalized[canonical] = value

    return normalized


def build_skill_llm_cli_args(skill_name: str, args: dict[str, Any]) -> list[str]:
    """Translate normalized LLM args into skill-supported CLI flags."""
    skill_info = registry.skills.get(skill_name, {}) if skill_name else {}
    schema = _merge_llm_argument_schema(skill_info)
    cli_args: list[str] = []

    for canonical, spec in schema.items():
        cli_flag = spec.get("cli_flag")
        if not cli_flag or not _skill_allows_llm_cli_flag(skill_info, cli_flag):
            continue

        value = args.get(canonical)
        if isinstance(value, str):
            value = value.strip()
        if not _has_llm_arg_value(value):
            continue

        if isinstance(value, bool):
            if value:
                cli_args.append(cli_flag)
            continue

        cli_args.extend([cli_flag, str(value)])

    return cli_args


registry = SpatialSkillRegistry()
