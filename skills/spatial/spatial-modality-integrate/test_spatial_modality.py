"""Tests for spatial-modality-integration skill using DeepST."""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

try:
    import scanpy as sc
    import deepstkit as dt
except ImportError:
    pytest.skip("Required packages not installed", allow_module_level=True)

from skills.spatial._lib.modality_integration import (
    run_deepst_spatial_domain_identification,
    run_deepst_integration,
    validate_spatial_data,
    extract_spatial_domain_identification,
    compute_domain_statistics,
)


def create_demo_adata(n_obs=200, n_vars=500):
    """Create a simple AnnData for testing."""
    np.random.seed(42)
    
    # Expression matrix
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    X = X / X.sum(axis=1, keepdims=True) * 10000
    X = np.log1p(X)
    
    # Create AnnData
    adata = sc.AnnData(
        X,
        obs=pd.DataFrame(index=[f"Spot_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
    )
    
    # Add spatial coordinates
    grid_size = int(np.ceil(np.sqrt(n_obs)))
    spatial_coords = np.zeros((n_obs, 2))
    spatial_coords[:, 0] = np.repeat(np.arange(grid_size), grid_size)[:n_obs]
    spatial_coords[:, 1] = np.tile(np.arange(grid_size), grid_size)[:n_obs]
    adata.obsm["spatial"] = spatial_coords
    
    return adata


def test_validate_spatial_data():
    """Test spatial data validation."""
    adata = create_demo_adata()
    assert validate_spatial_data(adata) is True
    
    # Remove spatial coordinates
    del adata.obsm["spatial"]
    assert validate_spatial_data(adata) is False


def test_single_sample_analysis():
    """Test single-sample spatial domain identification."""
    adata = create_demo_adata(n_obs=100, n_vars=300)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_deepst_spatial_domain_identification(
            adata=adata,
            n_domains=5,
            pre_epochs=10,  # Very few for fast testing
            epochs=10,
            pca_components=50,
            use_morphological=False,
            use_gpu=False,
            random_seed=42,
            output_dir=tmpdir,
        )
    
    assert "adata" in result
    assert "embeddings" in result
    assert "n_domains" in result
    assert result["n_domains"] == 5
    assert "DeepST_embed" in result["adata"].obsm
    assert "DeepST_refine_domain" in result["adata"].obs or "DeepST_domain" in result["adata"].obs


def test_multi_sample_integration():
    """Test multi-sample integration."""
    adata_list = [create_demo_adata(n_obs=50, n_vars=300) for _ in range(2)]
    sample_ids = ["Sample_A", "Sample_B"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_deepst_integration(
            adata_list=adata_list,
            sample_ids=sample_ids,
            n_domains=4,
            pre_epochs=10,
            epochs=10,
            pca_components=50,
            use_morphological=False,
            use_gpu=False,
            random_seed=42,
            output_dir=tmpdir,
        )
    
    assert "adata_integrated" in result
    assert "embeddings" in result
    assert "n_domains" in result
    assert result["n_domains"] == 4
    assert "batch" in result["adata_integrated"].obs or result["batch_key"] in result["adata_integrated"].obs


def test_extract_spatial_domain_identification():
    """Test spatial domain extraction."""
    adata = create_demo_adata(n_obs=100, n_vars=300)
    adata.obs["test_domain"] = np.random.randint(0, 5, adata.shape[0])
    adata.obsm["DeepST_embed"] = np.random.randn(adata.shape[0], 50)
    
    result_df = extract_spatial_domain_identification(adata, domain_column="test_domain")
    
    assert result_df.shape[0] == adata.shape[0]
    assert "x" in result_df.columns
    assert "y" in result_df.columns
    assert "domain" in result_df.columns


def test_compute_domain_statistics():
    """Test domain statistics computation."""
    adata = create_demo_adata(n_obs=100, n_vars=300)
    adata.obs["test_domain"] = np.random.randint(0, 5, adata.shape[0])
    
    stats = compute_domain_statistics(adata, domain_column="test_domain")
    
    assert stats.shape[0] > 0
    assert "n_spots" in stats.columns
    assert "mean_x" in stats.columns
    assert "mean_y" in stats.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
