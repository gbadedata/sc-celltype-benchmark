"""Tests for data_loader module.

Uses synthetic data and temporary files to test loading logic
without downloading real datasets.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pytest
import scipy.sparse as sp

from src.data_loader import download_pbmc3k, get_dataset, load_h5ad


class TestLoadH5ad:
    """Test h5ad loading."""

    def test_load_existing_h5ad(self, tmp_path: Path, synthetic_adata: ad.AnnData) -> None:
        """Loading a valid h5ad file returns an AnnData."""
        filepath = tmp_path / "test.h5ad"
        synthetic_adata.write_h5ad(filepath)

        result = load_h5ad(filepath)
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == synthetic_adata.n_obs
        assert result.n_vars == synthetic_adata.n_vars

    def test_load_missing_file_raises(self) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_h5ad("/nonexistent/path/fake.h5ad")

    def test_preserves_obs_columns(self, tmp_path: Path, synthetic_adata: ad.AnnData) -> None:
        """Loading preserves observation metadata columns."""
        filepath = tmp_path / "test.h5ad"
        synthetic_adata.write_h5ad(filepath)

        result = load_h5ad(filepath)
        assert "celltype_ground_truth" in result.obs.columns

    def test_preserves_sparse_matrix(self, tmp_path: Path, synthetic_adata: ad.AnnData) -> None:
        """Loading preserves sparse matrix format."""
        filepath = tmp_path / "test.h5ad"
        synthetic_adata.write_h5ad(filepath)

        result = load_h5ad(filepath)
        assert sp.issparse(result.X)


class TestGetDataset:
    """Test the main get_dataset entry point."""

    def test_get_dataset_from_h5ad(self, tmp_path: Path, synthetic_adata: ad.AnnData) -> None:
        """get_dataset loads an h5ad file when given explicit path."""
        filepath = tmp_path / "test.h5ad"
        synthetic_adata.write_h5ad(filepath)

        result = get_dataset(filepath=filepath)
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == 200

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        """get_dataset raises ValueError for unsupported formats."""
        filepath = tmp_path / "data.csv"
        filepath.write_text("a,b,c\n1,2,3\n")

        with pytest.raises(ValueError, match="Unsupported file format"):
            get_dataset(filepath=filepath)


class TestDownloadPbmc3k:
    """Test PBMC 3k download caching logic."""

    def test_returns_cached_h5ad(self, tmp_path: Path, synthetic_adata: ad.AnnData) -> None:
        """download_pbmc3k returns cached file without re-downloading."""
        cache_path = tmp_path / "pbmc3k_raw.h5ad"
        synthetic_adata.write_h5ad(cache_path)

        result = download_pbmc3k(dest_dir=tmp_path)
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == synthetic_adata.n_obs

    def test_creates_dest_dir(self, tmp_path: Path) -> None:
        """download_pbmc3k creates destination directory if missing."""
        nested = tmp_path / "a" / "b" / "c"
        # Will try to download (and may fail), but directory should be created
        try:
            download_pbmc3k(dest_dir=nested)
        except Exception:
            pass
        assert nested.exists()
