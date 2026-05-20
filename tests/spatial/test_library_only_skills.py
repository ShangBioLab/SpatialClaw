import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

ROOT = Path(__file__).resolve().parents[2]
BASE_ST = ROOT / "data" / "deconv" / "Human_Lymph_Node" / "ST.h5ad"
BASE_SC = ROOT / "data" / "deconv" / "Human_Lymph_Node" / "scRNA.h5ad"
FIXTURE_DIR = ROOT / "data" / "test_fixtures" / "spatial"
TEST_FILE = ROOT / "tests" / "spatial" / "test_library_only_skills.py"

sys.path.insert(0, str(ROOT))


@lru_cache(maxsize=None)
def _load_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def wrapper_modules():
    return {
        "niches": _load_module(
            "spatial_niches_wrapper",
            "skills/spatial/spatial-niches/spatial_niches.py",
        ),
        "regulation": _load_module(
            "spatial_regulation_wrapper",
            "skills/spatial/spatial-regulation/spatial_regulation.py",
        ),
        "sc2spatial": _load_module(
            "spatial_sc2spatial_wrapper",
            "skills/spatial/spatial-sc2spatial/spatial_sc2spatial.py",
        ),
        "histology": _load_module(
            "spatial_histology_wrapper",
            "skills/spatial/spatial-histology/spatial_histology.py",
        ),
        "wsi": _load_module(
            "spatial_wsi_wrapper",
            "skills/spatial/spatial-wsi/spatial_wsi.py",
        ),
        "morphology": _load_module(
            "spatial_morphology_wrapper",
            "skills/spatial/spatial-morphology/spatial_morphology.py",
        ),
        "translation": _load_module(
            "spatial_translation_wrapper",
            "skills/spatial/spatial-translation/spatial_translation.py",
        ),
        "oncology": _load_module(
            "spatial_oncology_wrapper",
            "skills/spatial/spatial-oncology/spatial_oncology.py",
        ),
        "targets": _load_module(
            "spatial_targets_wrapper",
            "skills/spatial/spatial-targets/spatial_targets.py",
        ),
        "tls": _load_module(
            "spatial_tls_wrapper",
            "skills/spatial/spatial-tls/spatial_tls.py",
        ),
    }


def _make_categorical_labels(n: int, labels: list[str]) -> np.ndarray:
    idx = np.arange(n) % len(labels)
    return np.array([labels[i] for i in idx], dtype=object)


def _write_synthetic_library_fixtures(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260506)
    n_spots = 80
    n_ref_cells = 90
    n_genes = 80
    genes = [f"Gene{i:03d}" for i in range(n_genes)]

    spatial_domains = _make_categorical_labels(n_spots, ["domain_0", "domain_1", "domain_2", "domain_3"])
    spatial_x = rng.poisson(lam=3.0, size=(n_spots, n_genes)).astype(float)
    for domain_idx, domain in enumerate(sorted(set(spatial_domains))):
        mask = spatial_domains == domain
        start = domain_idx * 8
        spatial_x[mask, start:start + 8] += 6.0

    cell_types = _make_categorical_labels(n_spots, ["B cell", "T cell", "Tumor", "Myeloid"])
    cell_types[:20] = "B cell"
    cell_types[20:32] = "T cell"

    spatial_adata = ad.AnnData(
        X=spatial_x,
        obs=pd.DataFrame(
            {
                "cell_type": cell_types,
                "condition": _make_categorical_labels(n_spots, ["control", "treated"]),
                "response": _make_categorical_labels(n_spots, ["responder", "non_responder"]),
                "spatial_domain": spatial_domains,
            },
            index=[f"spot_{i:03d}" for i in range(n_spots)],
        ),
        var=pd.DataFrame(index=genes),
    )
    spatial_adata.layers["counts"] = spatial_x.copy()
    grid_x, grid_y = np.meshgrid(np.arange(10), np.arange(8))
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(float)
    coords[:32] *= 5.0
    coords[32:] = coords[32:] * 20.0 + 500.0
    spatial_adata.obsm["spatial"] = coords
    spatial_adata.obsm["X_pca"] = rng.normal(size=(n_spots, 12))

    ref_x = rng.poisson(lam=3.0, size=(n_ref_cells, n_genes)).astype(float)
    ref_cell_types = _make_categorical_labels(n_ref_cells, ["B cell", "T cell", "Tumor", "Myeloid"])
    for type_idx, cell_type in enumerate(["B cell", "T cell", "Tumor", "Myeloid"]):
        mask = ref_cell_types == cell_type
        start = type_idx * 8
        ref_x[mask, start:start + 8] += 5.0

    reference_adata = ad.AnnData(
        X=ref_x,
        obs=pd.DataFrame(
            {"cell_type": ref_cell_types},
            index=[f"cell_{i:03d}" for i in range(n_ref_cells)],
        ),
        var=pd.DataFrame(index=genes),
    )
    reference_adata.layers["counts"] = ref_x.copy()

    x = np.linspace(0, 255, 128, dtype=np.uint8)
    y = np.linspace(0, 255, 128, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    image = np.stack([xx, yy, ((xx.astype(int) + yy.astype(int)) // 2).astype(np.uint8)], axis=-1)

    from PIL import Image

    library_spatial = output_dir / "library_spatial_fixture.h5ad"
    sc_reference = output_dir / "sc_reference_fixture.h5ad"
    tiny_histology = output_dir / "tiny_histology.png"
    tiny_slide = output_dir / "tiny_slide.tiff"
    spatial_adata.write_h5ad(library_spatial)
    reference_adata.write_h5ad(sc_reference)
    Image.fromarray(image).save(tiny_histology)
    Image.fromarray(image).save(tiny_slide)

    return {
        "library_spatial": library_spatial,
        "sc_reference": sc_reference,
        "tiny_histology": tiny_histology,
        "tiny_slide": tiny_slide,
    }


@pytest.fixture(scope="session")
def generated_fixtures(tmp_path_factory):
    if not BASE_ST.exists() or not BASE_SC.exists():
        return _write_synthetic_library_fixtures(
            tmp_path_factory.mktemp("spatial_library_only_fixtures")
        )

    generator = _load_module(
        "generate_spatial_test_fixtures_module",
        "scripts/generate_spatial_test_fixtures.py",
    )
    generator.OUT_DIR.mkdir(parents=True, exist_ok=True)

    library_spatial = FIXTURE_DIR / "library_spatial_fixture.h5ad"
    if not library_spatial.exists():
        library_spatial = generator.build_library_spatial_fixture(BASE_ST)

    sc_reference = FIXTURE_DIR / "sc_reference_fixture.h5ad"
    if not sc_reference.exists():
        sc_reference = generator.build_sc_reference_copy(BASE_SC)

    tiny_histology = FIXTURE_DIR / "tiny_histology.png"
    tiny_slide = FIXTURE_DIR / "tiny_slide.tiff"
    if not tiny_histology.exists() or not tiny_slide.exists():
        tiny_histology, tiny_slide = generator.build_tiny_images()

    return {
        "library_spatial": library_spatial,
        "sc_reference": sc_reference,
        "tiny_histology": tiny_histology,
        "tiny_slide": tiny_slide,
    }


@pytest.fixture()
def spatial_adata(generated_fixtures):
    return ad.read_h5ad(generated_fixtures["library_spatial"]).copy()


@pytest.fixture()
def sc_reference_adata(generated_fixtures):
    return ad.read_h5ad(generated_fixtures["sc_reference"]).copy()


def test_spatial_niches_smoke(wrapper_modules, spatial_adata):
    pytest.importorskip("scanpy")

    result = wrapper_modules["niches"].run(
        spatial_adata,
        method="leiden_niche",
        cell_type_key="cell_type",
        characterize=False,
    )

    assert result["method"] == "leiden_niche"
    assert "niche" in spatial_adata.obs
    assert spatial_adata.obs["niche"].nunique() > 0


def test_spatial_regulation_smoke(wrapper_modules, spatial_adata):
    tf_list = [str(gene) for gene in spatial_adata.var_names[: min(5, spatial_adata.n_vars)]]

    result = wrapper_modules["regulation"].run(
        spatial_adata,
        method="tf_activity",
        tf_list=tf_list,
    )

    assert result["method"] == "tf_activity"
    assert "tf_activities" in spatial_adata.obsm
    assert spatial_adata.obsm["tf_activities"].shape[0] == spatial_adata.n_obs


def test_spatial_sc2spatial_smoke(wrapper_modules, spatial_adata, sc_reference_adata):
    pytest.importorskip("scanpy")

    result = wrapper_modules["sc2spatial"].run(
        spatial_adata,
        sc_reference_adata,
        method="label_transfer",
        cell_type_key="cell_type",
    )

    assert result["method"] == "label_transfer"
    assert "transferred_cell_type" in spatial_adata.obs
    assert "transfer_confidence" in spatial_adata.obs


def test_spatial_histology_smoke(wrapper_modules, generated_fixtures):
    result = wrapper_modules["histology"].run(
        str(generated_fixtures["tiny_histology"]),
        task="tissue_segmentation",
        method="otsu",
    )

    assert result["task"] == "tissue_segmentation"
    assert result["method"] == "otsu"
    assert result["mask"] is not None


@pytest.mark.xfail(
    strict=False,
    reason="WSI smoke test depends on an OpenSlide-compatible backend and slide reader semantics.",
)
def test_spatial_wsi_smoke(wrapper_modules, generated_fixtures, tmp_path):
    pytest.importorskip("openslide")

    result = wrapper_modules["wsi"].run(
        str(generated_fixtures["tiny_slide"]),
        patch_size=64,
        output_dir=str(tmp_path / "wsi"),
    )

    assert "slide_info" in result
    assert "patches" in result


def test_spatial_morphology_smoke(wrapper_modules, spatial_adata):
    result = wrapper_modules["morphology"].run(
        spatial_adata,
        images=None,
        method="deep_features",
    )

    assert result["method"] == "deep_features"
    assert "morphology_features" in spatial_adata.obsm
    assert spatial_adata.obsm["morphology_features"].shape[0] == spatial_adata.n_obs


def test_spatial_translation_smoke(wrapper_modules, spatial_adata):
    result = wrapper_modules["translation"].run(
        spatial_adata,
        images=None,
        method="hist2st",
    )

    assert result["method"] == "hist2st"
    assert "predicted_expression" in spatial_adata.obsm
    assert spatial_adata.obsm["predicted_expression"].shape[0] == spatial_adata.n_obs


def test_spatial_oncology_smoke(wrapper_modules, spatial_adata):
    result = wrapper_modules["oncology"].run(
        spatial_adata,
        method="tumor_ecosystem",
        cell_type_key="cell_type",
        condition_key="condition",
    )

    assert "n_tumor_cells" in result
    if result.get("status") != "insufficient_tumor":
        assert "tumor_ecosystem" in spatial_adata.obs


def test_spatial_targets_smoke(wrapper_modules, spatial_adata):
    pytest.importorskip("scanpy")

    result = wrapper_modules["targets"].run(
        spatial_adata,
        method="biomarker_discovery",
        domain_key="spatial_domain",
    )

    assert result["method"] == "biomarker_discovery"
    assert "spatial_biomarkers" in spatial_adata.uns


def test_spatial_targets_research_summary_language(wrapper_modules, spatial_adata):
    spatial_adata.uns["spatial_biomarkers"] = {
        "domain_0": {"genes": ["EGFR", "Gene001"]},
        "domain_1": {"genes": ["Gene001", "Gene002"]},
    }
    wrapper_modules["targets"].run(spatial_adata, method="target_prioritization")
    report = wrapper_modules["targets"].run(spatial_adata, method="actionable_report")

    assert report["research_use_only"] is True
    assert "research_follow_up" in report
    assert "recommendations" not in report
    assert "clinical actionability" in report["interpretation"]


def test_spatial_tls_smoke(wrapper_modules, spatial_adata):
    result = wrapper_modules["tls"].run(
        spatial_adata,
        method="tls_detection",
        cell_type_key="cell_type",
    )

    assert "n_tls" in result
    assert "tls_region" in spatial_adata.obs


def test_spatial_oncology_placeholder_outputs_are_not_random_predictions(wrapper_modules, spatial_adata):
    spatial_adata.obs["time"] = np.arange(spatial_adata.n_obs) + 1
    spatial_adata.obs["event"] = np.arange(spatial_adata.n_obs) % 2

    result = wrapper_modules["oncology"].run(
        spatial_adata,
        method="clinical_prediction",
        outcome_key="survival",
        model_type="random_forest",
        features="pca",
        time_key="time",
        event_key="event",
    )

    assert result["status"] == "not_implemented"
    assert result["research_use_only"] is True
    assert "survival_risk_score" not in spatial_adata.obs
