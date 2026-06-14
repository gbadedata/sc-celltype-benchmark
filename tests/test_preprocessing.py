"""Tests for preprocessing module.

Tests normalisation, HVG selection, scaling, and PCA using
synthetic data. Validates biological expectations and mathematical
properties of each step.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from src.preprocessing import (
    PreprocessingReport,
    normalize,
    run_preprocessing,
    scale_and_pca,
    select_hvg,
)
from src.quality_control import run_qc


@pytest.fixture
def qc_adata(synthetic_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData that has passed QC — ready for preprocessing."""
    adata, _ = run_qc(synthetic_adata)
    return adata


class TestNormalize:
    """Test library-size normalisation."""

    def test_raw_is_saved(self, qc_adata: ad.AnnData) -> None:
        """Raw counts are saved to adata.raw before transformation."""
        result = normalize(qc_adata)
        assert result.raw is not None
        assert result.raw.X is not None

    def test_raw_counts_unchanged(self, qc_adata: ad.AnnData) -> None:
        """Raw counts in .raw match the original input matrix exactly."""
        original_sum = float(qc_adata.X.sum())
        result = normalize(qc_adata)
        raw_sum = float(result.raw.X.sum())
        # .raw is set via adata.copy() before normalisation — sums must match
        assert abs(raw_sum - original_sum) < 1.0

    def test_log1p_applied(self, qc_adata: ad.AnnData) -> None:
        """After normalisation, values are log-transformed (max < 15 for PBMC-like data)."""
        result = normalize(qc_adata)
        X = result.X.toarray() if sp.issparse(result.X) else result.X
        # log1p(10000) ≈ 9.2 — values should be bounded
        assert X.max() < 15.0

    def test_no_negative_values(self, qc_adata: ad.AnnData) -> None:
        """Normalised counts are non-negative (log1p preserves sign)."""
        result = normalize(qc_adata)
        X = result.X.toarray() if sp.issparse(result.X) else result.X
        assert np.all(X >= 0)

    def test_cell_count_unchanged(self, qc_adata: ad.AnnData) -> None:
        """Normalisation does not remove cells."""
        n_cells_before = qc_adata.n_obs
        result = normalize(qc_adata)
        assert result.n_obs == n_cells_before


class TestSelectHvg:
    """Test highly variable gene selection."""

    def test_hvg_column_added(self, qc_adata: ad.AnnData) -> None:
        """highly_variable column is added to .var."""
        adata = normalize(qc_adata)
        result = select_hvg(adata)
        assert "highly_variable" in result.var.columns

    def test_correct_number_of_hvgs(self, qc_adata: ad.AnnData) -> None:
        """Number of HVGs does not exceed n_top_genes setting."""
        from config.settings import settings
        adata = normalize(qc_adata)
        result = select_hvg(adata)
        n_hvg = int(result.var["highly_variable"].sum())
        # May be less than n_top_genes if fewer genes are available
        assert n_hvg <= settings.n_top_genes
        assert n_hvg > 0

    def test_all_genes_retained(self, qc_adata: ad.AnnData) -> None:
        """HVG selection marks genes but does not remove them."""
        adata = normalize(qc_adata)
        n_genes_before = adata.n_vars
        result = select_hvg(adata)
        assert result.n_vars == n_genes_before

    def test_hvg_flag_is_boolean(self, qc_adata: ad.AnnData) -> None:
        """highly_variable column is boolean dtype."""
        adata = normalize(qc_adata)
        result = select_hvg(adata)
        assert result.var["highly_variable"].dtype == bool


class TestScaleAndPca:
    """Test scaling and PCA."""

    def test_pca_embedding_shape(self, qc_adata: ad.AnnData) -> None:
        """PCA embedding has correct shape: n_cells x n_pcs."""
        from config.settings import settings
        adata = normalize(qc_adata)
        adata = select_hvg(adata)
        result = scale_and_pca(adata)
        assert "X_pca" in result.obsm
        n_pcs_actual = result.obsm["X_pca"].shape[1]
        # n_pcs capped by min(n_cells, n_hvg) - 1
        assert n_pcs_actual <= settings.n_pcs
        assert n_pcs_actual > 0

    def test_variance_ratio_stored(self, qc_adata: ad.AnnData) -> None:
        """PCA variance ratios are stored in uns."""
        adata = normalize(qc_adata)
        adata = select_hvg(adata)
        result = scale_and_pca(adata)
        assert "pca" in result.uns
        assert "variance_ratio" in result.uns["pca"]

    def test_variance_ratio_sums_to_lte_one(self, qc_adata: ad.AnnData) -> None:
        """Variance ratios sum to at most 1.0."""
        adata = normalize(qc_adata)
        adata = select_hvg(adata)
        result = scale_and_pca(adata)
        total_var = float(result.uns["pca"]["variance_ratio"].sum())
        assert 0 < total_var <= 1.0


class TestRunPreprocessing:
    """Test the full preprocessing pipeline."""

    def test_returns_adata_and_report(self, qc_adata: ad.AnnData) -> None:
        """run_preprocessing returns (AnnData, PreprocessingReport)."""
        result, report = run_preprocessing(qc_adata)
        assert isinstance(result, ad.AnnData)
        assert isinstance(report, PreprocessingReport)

    def test_report_fields_consistent(self, qc_adata: ad.AnnData) -> None:
        """Report fields are internally consistent with the output AnnData."""
        adata, report = run_preprocessing(qc_adata)
        assert report.n_cells == adata.n_obs
        assert report.n_hvg <= report.n_genes_total
        assert report.n_hvg > 0

    def test_pca_embedding_present(self, qc_adata: ad.AnnData) -> None:
        """Output AnnData has PCA embedding."""
        adata, _ = run_preprocessing(qc_adata)
        assert "X_pca" in adata.obsm

    def test_raw_preserved(self, qc_adata: ad.AnnData) -> None:
        """Raw counts are preserved in .raw after full preprocessing."""
        adata, _ = run_preprocessing(qc_adata)
        assert adata.raw is not None

    def test_variance_explained_positive(self, qc_adata: ad.AnnData) -> None:
        """Cumulative variance explained is a positive percentage."""
        _, report = run_preprocessing(qc_adata)
        assert 0 < report.pct_variance_explained <= 100.0

    def test_cell_count_unchanged(self, qc_adata: ad.AnnData) -> None:
        """Preprocessing does not remove cells."""
        n_cells = qc_adata.n_obs
        adata, report = run_preprocessing(qc_adata)
        assert adata.n_obs == n_cells
        assert report.n_cells == n_cells
