"""Dependency tier helpers for the spatial-only SpatialClaw runtime."""

from __future__ import annotations

import importlib
import importlib.util

DOMAIN_TIERS = {
    "scanpy": "core",
    "anndata": "core",
    "squidpy": "core",
    "numpy": "core",
    "pandas": "core",
    "scipy": "core",
    "scikit-learn": "core",
    "SpaGCN": "spatial-domain-identification",
    "torch": "spatial-domain-identification",
    "torch_geometric": "spatial",
    "scvi": "spatial",
    "tangram": "spatial",
    "GraphST": "spatial",
    "cellrank": "spatial",
    "palantir": "spatial",
    "SpatialDE": "spatial",
    "esda": "spatial",
    "libpysal": "spatial",
    "pysal": "spatial",
    "scvelo": "spatial-velocity",
    "infercnvpy": "spatial-cnv",
    "gseapy": "spatial-enrichment",
    "liana": "spatial-cell-communication",
    "cellphonedb": "spatial-cell-communication",
    "fastccc": "spatial-cell-communication",
    "harmonypy": "spatial-integration",
    "bbknn": "spatial-integration",
    "scanorama": "spatial-integration",
    "ot": "spatial-registration",
    "paste": "spatial-registration",
    "pydeseq2": "spatial-condition-comparison",
}


def check_dependencies(skill_name: str, required_packages: list[str]) -> bool:
    """Validate Python dependencies and report the matching spatial extra."""
    missing_packages = []
    tiers_needed = set()

    for pkg in required_packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing_packages.append(pkg)
            tiers_needed.add(DOMAIN_TIERS.get(pkg, "full"))

    if missing_packages:
        tiers_str = ",".join(sorted(tiers_needed))
        if len(tiers_needed) == 1 and "full" not in tiers_needed:
            install_cmd = f'pip install -e ".[{next(iter(tiers_needed))}]"'
        else:
            install_cmd = f'pip install -e ".[{tiers_str}]"'

        raise ImportError(
            "\n[SpatialClaw Environment Error]\n"
            f"Skill '{skill_name}' requires missing optional dependencies: "
            f"{', '.join(missing_packages)}.\n"
            "Install them with:\n"
            f"    {install_cmd}\n"
        )

    return True


def get_installed_tiers() -> dict[str, bool]:
    """Return spatial dependency tiers and whether representative packages exist."""
    return {
        "core": importlib.util.find_spec("scanpy") is not None,
        "spatial-domain-identification": (
            importlib.util.find_spec("SpaGCN") is not None
            and importlib.util.find_spec("torch") is not None
        ),
        "spatial": importlib.util.find_spec("scvi") is not None,
        "spatial-cell-annotation": (
            importlib.util.find_spec("tangram") is not None
            or importlib.util.find_spec("scvi") is not None
        ),
        "spatial-deconvolution": (
            importlib.util.find_spec("tangram") is not None
            or importlib.util.find_spec("scvi") is not None
            or importlib.util.find_spec("GraphST") is not None
        ),
        "spatial-trajectory": (
            importlib.util.find_spec("cellrank") is not None
            or importlib.util.find_spec("palantir") is not None
        ),
        "spatial-svg-detection": importlib.util.find_spec("SpatialDE") is not None,
        "spatial-statistics": (
            importlib.util.find_spec("esda") is not None
            or importlib.util.find_spec("libpysal") is not None
        ),
        "spatial-condition-comparison": importlib.util.find_spec("pydeseq2") is not None,
        "spatial-velocity": importlib.util.find_spec("scvelo") is not None,
        "spatial-cnv": importlib.util.find_spec("infercnvpy") is not None,
        "spatial-enrichment": importlib.util.find_spec("gseapy") is not None,
        "spatial-cell-communication": importlib.util.find_spec("liana") is not None,
        "spatial-integration": (
            importlib.util.find_spec("harmonypy") is not None
            or importlib.util.find_spec("bbknn") is not None
        ),
        "spatial-registration": importlib.util.find_spec("paste") is not None,
        "banksy": importlib.util.find_spec("pybanksy") is not None,
    }
