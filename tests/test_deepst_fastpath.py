import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SKILL_PATH = ROOT / "skills" / "spatial" / "spatial-modality-integrate" / "spatial_modality_integrate.py"


def load_skill_module():
    spec = importlib.util.spec_from_file_location("spatial_modality_integrate_test", SKILL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {SKILL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_visium_like_adata() -> ad.AnnData:
    obs = pd.DataFrame(index=pd.Index(["spot_a", "spot_b"], dtype="string"))
    var = pd.DataFrame(index=pd.Index(["g1", "g2"], dtype="string"))
    adata = ad.AnnData(np.ones((2, 2), dtype=np.float32), obs=obs, var=var)
    adata.obsm["spatial"] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    adata.uns["spatial"] = {
        "library": {
            "images": {
                "lowres": np.zeros((8, 8, 3), dtype=np.uint8),
                "hires": np.ones((16, 16, 3), dtype=np.uint8),
            },
            "scalefactors": {
                "tissue_lowres_scalef": 0.5,
                "tissue_hires_scalef": 2.0,
            },
        }
    }
    return adata


def make_cluster_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        index=pd.Index(
            ["spot_a", "spot_b", "spot_c", "spot_d", "spot_e", "spot_f"],
            dtype="string",
        )
    )
    var = pd.DataFrame(index=pd.Index(["g1", "g2"], dtype="string"))
    adata = ad.AnnData(np.ones((6, 2), dtype=np.float32), obs=obs, var=var)
    adata.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [10.0, 10.0],
            [10.0, 11.0],
            [11.0, 10.0],
        ],
        dtype=np.float32,
    )
    adata.obsm["DeepST_embed"] = np.array(
        [
            [-5.0, -5.0],
            [-5.1, -4.9],
            [-4.9, -5.2],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.8, 5.2],
        ],
        dtype=np.float32,
    )
    return adata


def test_select_deepst_image_quality_prefers_lowres():
    module = load_skill_module()
    adata = make_visium_like_adata()

    quality, image, coords = module._select_deepst_image_quality(adata)

    assert quality == "lowres"
    assert image.shape == (8, 8, 3)
    assert np.allclose(coords, np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32))


def test_deepst_image_feature_cache_round_trip(tmp_path):
    module = load_skill_module()
    cache_path = tmp_path / "deepst_cache.npz"
    obs_names = pd.Index(["spot_a", "spot_b"], dtype="string")
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    module._save_cached_deepst_image_features(cache_path, obs_names, features)
    loaded = module._load_cached_deepst_image_features(
        cache_path,
        pd.Index(["spot_b", "spot_a"], dtype="string"),
    )

    assert loaded is not None
    assert loaded.shape == (2, 2)
    assert np.allclose(loaded, np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32))


def test_estimate_deepst_n_domains_prefers_two_clear_clusters():
    module = load_skill_module()
    adata = make_cluster_adata()

    estimated = module._estimate_deepst_n_domains(
        adata.obsm["DeepST_embed"],
        random_seed=0,
        max_clusters=5,
    )

    assert estimated == 2


def test_cluster_deepst_embeddings_falls_back_when_upstream_fails():
    module = load_skill_module()
    adata = make_cluster_adata()

    class FailingDeepST:
        def _get_cluster_data(self, *args, **kwargs):
            raise ValueError("Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)")

    clustered, info = module._cluster_deepst_embeddings(
        adata,
        deepst_instance=FailingDeepST(),
        n_domains=2,
        random_seed=0,
    )

    assert info["clustering_backend"] == "kmeans-fallback"
    assert info["clustering_fallback_used"] is True
    assert clustered.obs["DeepST_domain"].astype(str).nunique() == 2
    assert "DeepST_refine_domain" in clustered.obs


def test_cluster_deepst_embeddings_falls_back_when_upstream_misses_target_count():
    module = load_skill_module()
    adata = make_cluster_adata()

    class WrongCountDeepST:
        def _get_cluster_data(self, adata, **kwargs):
            adata = adata.copy()
            adata.obs["DeepST_domain"] = pd.Categorical(["0"] * adata.n_obs)
            adata.obs["DeepST_refine_domain"] = pd.Categorical(["0"] * adata.n_obs)
            return adata

    clustered, info = module._cluster_deepst_embeddings(
        adata,
        deepst_instance=WrongCountDeepST(),
        n_domains=2,
        random_seed=0,
    )

    assert info["clustering_backend"] == "kmeans-fallback"
    assert clustered.obs["DeepST_refine_domain"].astype(str).nunique() == 2
