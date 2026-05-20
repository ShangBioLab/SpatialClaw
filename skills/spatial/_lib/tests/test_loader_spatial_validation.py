import numpy as np
import pytest

import anndata as ad

from skills.spatial._lib.adata_utils import require_spatial_coords
from skills.spatial._lib.exceptions import DataError
from skills.spatial._lib import loader
from skills.spatial._lib.loader import load_spatial_data


def _write_h5ad(path, coords=None):
    adata = ad.AnnData(np.ones((3, 2), dtype=float))
    if coords is not None:
        adata.obsm["spatial"] = coords
    adata.write_h5ad(path)
    return path


def test_load_spatial_data_requires_spatial_by_default(tmp_path):
    path = _write_h5ad(tmp_path / "sample.h5ad")

    with pytest.raises(DataError, match="No spatial coordinates"):
        load_spatial_data(path)


def test_load_spatial_data_allows_spatial_opt_out(tmp_path):
    path = _write_h5ad(tmp_path / "sample.h5ad")

    adata = load_spatial_data(path, require_spatial=False)

    assert adata.n_obs == 3


def test_load_spatial_data_rejects_bad_spatial_shape(monkeypatch, tmp_path):
    class VarNames:
        is_unique = True

    class BadShapeAdata:
        n_obs = 3
        n_vars = 2
        var_names = VarNames()
        obsm = {"spatial": np.ones((2, 2), dtype=float)}
        uns = {}

    def fake_loader(_path):
        return BadShapeAdata()

    monkeypatch.setitem(loader._LOADERS, "bad_shape", fake_loader)

    with pytest.raises(DataError, match="one row per observation"):
        load_spatial_data(tmp_path, data_type="bad_shape")


def test_load_spatial_data_rejects_non_finite_spatial(tmp_path):
    coords = np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 2.0]])
    path = _write_h5ad(tmp_path / "sample.h5ad", coords)

    with pytest.raises(DataError, match="non-finite"):
        load_spatial_data(path)


def test_require_spatial_coords_returns_valid_spatial_key():
    adata = ad.AnnData(np.ones((3, 2), dtype=float))
    adata.obsm["spatial"] = np.ones((3, 2), dtype=float)

    assert require_spatial_coords(adata) == "spatial"
